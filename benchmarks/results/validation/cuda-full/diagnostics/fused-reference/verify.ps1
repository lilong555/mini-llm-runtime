param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.status -cne 'completed_numeric_failure' -or
    $evidence.numerical_gate_passed -ne $false -or $evidence.full_corpus_contract -ne $false -or
    $evidence.reference_flash_attention -ne $true) { throw '融合参照失败证据状态无效。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-fused-verify-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    if ($evidence.source.snapshot.path -cne 'source-snapshot.zip' -or $evidence.source.state_file -cne 'source-state.json' -or
        $evidence.diagnostic_source.snapshot.path -cne 'reference-diagnostic-source/source-snapshot.zip' -or
        $evidence.diagnostic_source.state_file -cne 'reference-diagnostic-source/source-state.json') { throw '源码路径无效。' }
    $snapshot = Join-Path $Directory $evidence.diagnostic_source.snapshot.path
    if ((Get-FileHash $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
        $evidence.diagnostic_source.snapshot.sha256) { throw '诊断源码快照摘要不符。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('scripts/Benchmark-Common.ps1','scripts/analyze_cuda_validation.py',
            'tests/data/qwen3_validation_cases.json')) {
            $entry = $zip.GetEntry($name)
            if ($null -eq $entry) { throw "诊断快照缺少验证工具：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,(Join-Path $temporary ([IO.Path]::GetFileName($name))))
        }
    } finally { $zip.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    foreach ($source in @($evidence.source,$evidence.diagnostic_source)) {
        if ((Get-LowerSha256 (Resolve-BenchmarkArtifact $Directory $source.state_file)) -cne $source.worktree_state_sha256 -or
            (Get-LowerSha256 (Resolve-BenchmarkArtifact $Directory $source.snapshot.path)) -cne $source.snapshot.sha256) {
            throw '诊断源码身份不符。'
        }
        Test-BenchmarkSourceArchive $Directory $source
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $evidence.artifacts) {
        if (-not $names.Add($item.path)) { throw '诊断产物重复。' }
        $path = Resolve-BenchmarkArtifact $Directory $item.path
        if ((Get-LowerSha256 $path) -cne $item.sha256 -or (Get-Item -LiteralPath $path -Force).Length -ne $item.size_bytes) {
            throw "诊断产物摘要不符：$($item.path)"
        }
    }
    foreach ($name in @('real-model/validation-summary.json','real-model/full-validation.json','numerical-audit.json',
        'reference-diagnostic/summary.json','reference-diagnostic/attention-reference.json','full-model-command.json',
        'reference-diagnostic-source.json','environment.json')) {
        if (-not $names.Contains($name)) { throw "缺少诊断证据：$name" }
    }
    $lines = @(& python3 (Join-Path $temporary 'analyze_cuda_validation.py') --directory (Join-Path $Directory 'real-model') `
        --contract (Join-Path $temporary 'qwen3_validation_cases.json') 2>&1)
    if ($LASTEXITCODE -ne 1) { throw '融合参照未保留原始数值失败。' }
    $audit = ($lines -join "`n") | ConvertFrom-Json
    if ($audit.status -cne 'numeric_failed' -or $audit.passed -ne $false -or $audit.totals.numeric_failures -ne 2 -or
        $audit.totals.teacher_comparisons -ne 11760 -or $audit.totals.generation_comparisons -ne 768 -or
        $audit.totals.argmax_divergences -ne 0 -or $audit.by_reference.cpu.numeric_failures -ne 0) {
        throw '融合参照数值失败范围不符。'
    }
    $saved = Get-Content -Raw (Join-Path $Directory 'numerical-audit.json') | ConvertFrom-Json
    if ((ConvertTo-Json -Depth 32 -Compress $audit) -cne (ConvertTo-Json -Depth 32 -Compress $saved)) {
        throw '融合参照复核结果不符。'
    }
    $diagnostic = Get-Content -Raw (Join-Path $Directory 'reference-diagnostic/attention-reference.json') | ConvertFrom-Json
    if ($diagnostic.hypothesis_supported -ne $true -or $diagnostic.fused_failures -ne 2 -or
        $diagnostic.unfused_failures -ne 0 -or $diagnostic.cpu_failures -ne 0 -or
        ($diagnostic.original_score_matches -join ',') -cne '32,32,32' -or $diagnostic.rows.Count -ne 32) {
        throw 'attention 参照重放证据不符。'
    }
    foreach ($row in $diagnostic.rows) {
        foreach ($backend in @('cpu','llama_unfused')) {
            $c = $row.comparisons.$backend
            if ($c.passed -ne $true -or $c.all_finite -ne $true -or $c.rmse -ge 0.05 -or
                $c.max_absolute -ge 0.5 -or $c.cosine -lt 0.9999 -or
                $c.near_tie -ne ($c.reference_margin -le 2*$c.max_absolute) -or
                (-not $c.near_tie -and -not $c.argmax_equal)) { throw '非融合参照或 CPU 重放未通过。' }
        }
    }
    [pscustomobject]@{Status='completed_numeric_failure'; Purpose='archive_revalidation'
        NumericalGatePassed=$false; NumericFailures=2; ReferenceDiagnosticPassed=$true; Artifacts=$names.Count}
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
