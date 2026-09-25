param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 8 numerical' -or
    $evidence.status -cne 'passed' -or $evidence.full_corpus_contract -ne $true -or
    $evidence.performance_baseline -ne $false -or $evidence.gpu_serving -ne $false -or
    $evidence.reference_attention -cne 'unfused') {
    throw '证据范围或状态无效。'
}
if ($evidence.source.snapshot.path -cne 'source-snapshot.zip' -or $evidence.source.state_file -cne 'source-state.json' -or
    $evidence.verification_source.snapshot.path -cne 'verification-source/source-snapshot.zip' -or
    $evidence.verification_source.state_file -cne 'verification-source/source-state.json') { throw '源码路径无效。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-full-verify-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $snapshot = Join-Path $Directory $evidence.verification_source.snapshot.path
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
        $evidence.verification_source.snapshot.sha256) { throw '验证工具源码快照摘要不符。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('scripts/Benchmark-Common.ps1','scripts/Test-CtestEvidence.ps1',
            'scripts/analyze_cuda_validation.py','tests/data/qwen3_validation_cases.json')) {
            $entry = $zip.GetEntry($name)
            if ($null -eq $entry) { throw "快照缺少验证工具或契约：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,(Join-Path $temporary ([IO.Path]::GetFileName($name))))
        }
    } finally { $zip.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    foreach ($source in @($evidence.source,$evidence.verification_source)) {
        if ((Get-LowerSha256 (Resolve-BenchmarkArtifact $Directory $source.state_file)) -cne $source.worktree_state_sha256 -or
            (Get-LowerSha256 (Resolve-BenchmarkArtifact $Directory $source.snapshot.path)) -cne $source.snapshot.sha256) {
            throw '源码清单或 ZIP 摘要不符。'
        }
        Test-BenchmarkSourceArchive $Directory $source
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $evidence.artifacts) {
        if (-not $names.Add($item.path)) { throw '证据路径重复。' }
        $path = Resolve-BenchmarkArtifact $Directory $item.path
        if ((Get-LowerSha256 $path) -cne $item.sha256 -or (Get-Item -LiteralPath $path -Force).Length -ne $item.size_bytes) {
            throw "证据摘要或长度不符：$($item.path)"
        }
    }
    foreach ($name in @('source.json','source-state.json','source-snapshot.zip','environment.json','build-boundaries.json',
        'verification-source/source-state.json','verification-source/source-snapshot.zip',
        'own-ctest.xml','cpu-ctest.xml','core-ctest.xml','upstream-ctest.xml','model-cpu.json','http-cpu.json',
        'full-model.txt','full-model-command.json','real-model/input.json','real-model/validation-contract.json',
        'real-model/validation-summary.json','real-model/full-validation.json','real-model/memory-plan.json',
        'real-model/weight-plan.json','numerical-audit.json','reference-mode-comparison.json','compare-runs.ps1',
        'negative-checks.json','short-model/validation-summary.json',
        'report-fixtures.txt','diagnostics/fused-reference/report-fixtures-initial.txt',
        'diagnostics/fused-reference/fixture-reference-argument.py','diagnostics/fused-reference/evidence.json')) {
        if (-not $names.Contains($name)) { throw "缺少必需证据：$name" }
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.SuiteExecutions -ne 36 -or $ctest.CaseExecutions -ne 807) {
        throw 'CTest 数量或状态不符。'
    }
    $model = Get-Content -Raw (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $http.status -cne 'passed' -or $http.passed -ne 8 -or
        $http.server_after.gpu -ne $false -or $http.server_after.backend -cne 'minillm' -or
        $http.server_after.active_requests -ne 0) { throw 'CPU 模型或 HTTP 回归未通过。' }
    $command = Get-Content -Raw (Join-Path $Directory 'full-model-command.json') | ConvertFrom-Json
    if ($command.exit_code -ne 0 -or $command.status -cne 'passed' -or $command.arguments[2] -cne 'model-full-check') {
        throw '全量模型命令未完成。'
    }
    $boundary = Get-Content -Raw (Join-Path $Directory 'build-boundaries.json') | ConvertFrom-Json
    if ($boundary.own_cache.MINILLM_ENABLE_CUDA -cne 'ON' -or $boundary.own_cache.GGML_CUDA -cne 'OFF' -or
        $boundary.cli_cpu_runtime_symbols.Count -ne 0 -or $boundary.full_validation_binary_unchanged -ne $true) {
        throw '执行归属或实模型二进制身份不符。'
    }
    foreach ($binary in $boundary.binaries) {
        if ($binary.configuration -ceq 'cpu' -and ($binary.dynamic_dependencies -join "`n") -match 'lib(cudart|cublas|cuda|nvrtc)') {
            throw 'CPU 产品依赖 CUDA。'
        }
    }
    $negative = Get-Content -Raw (Join-Path $Directory 'negative-checks.json') | ConvertFrom-Json
    if ($negative.status -cne 'passed' -or $negative.existing_directory.exit_code -ne 1 -or
        $negative.existing_directory.unchanged -ne $true -or $negative.wrong_checkpoint.exit_code -ne 1 -or
        $negative.wrong_checkpoint.report.passed -ne $false) { throw '报告保护或 checkpoint 拒绝反例未通过。' }
    $short = Get-Content -Raw (Join-Path $Directory 'short-model/validation-summary.json') | ConvertFrom-Json
    if ($short.status -cne 'passed' -or $short.passed -ne $true -or $short.logits_comparisons -ne 128 -or
        $short.golden_cases -ne 6 -or $short.full_corpus_contract -ne $false) { throw '默认短模式回归未通过。' }
    $fused = & (Join-Path $Directory 'diagnostics/fused-reference/verify.ps1')
    if ($fused.Status -cne 'completed_numeric_failure' -or $fused.NumericalGatePassed -ne $false -or
        $fused.ReferenceDiagnosticPassed -ne $true) { throw '原始融合参照失败证据不完整。' }
    $lines = @(& python3 (Join-Path $temporary 'analyze_cuda_validation.py') --directory (Join-Path $Directory 'real-model') `
        --contract (Join-Path $temporary 'qwen3_validation_cases.json') 2>&1)
    if ($LASTEXITCODE -ne 0) { $lines | Write-Output; throw '完整数值报告未通过独立复核。' }
    $audit = ($lines -join "`n") | ConvertFrom-Json
    $saved = Get-Content -Raw (Join-Path $Directory 'numerical-audit.json') | ConvertFrom-Json
    if ((ConvertTo-Json -Depth 32 -Compress $audit) -cne (ConvertTo-Json -Depth 32 -Compress $saved)) {
        throw '独立复核结果与记录不符。'
    }
    $comparison = & (Join-Path $Directory 'compare-runs.ps1')
    $savedComparison = Get-Content -Raw (Join-Path $Directory 'reference-mode-comparison.json') | ConvertFrom-Json
    if ($comparison.product_sources_equal -ne $true -or $comparison.dependency_commit_equal -ne $true -or
        $comparison.contract_bytes_equal -ne $true -or $comparison.input_bytes_equal -ne $true -or
        (ConvertTo-Json -Depth 32 -Compress $comparison) -cne (ConvertTo-Json -Depth 32 -Compress $savedComparison)) {
        throw '参照模式对照的输入、源码或记录不符。'
    }
    $full = Get-Content -Raw (Join-Path $Directory 'real-model/full-validation.json') | ConvertFrom-Json
    foreach ($item in @($full.canonical)+@($full.teacher_forcing)+@($full.generation)) {
        if (-not $names.Contains("real-model/$($item.file)")) { throw '数值报告未纳入产物清单。' }
    }
    foreach ($corpus in @('zh','en','repeated','special')) {
        foreach ($length in @(16,128,1536)) {
            foreach ($backend in @('cpu','llama_f32')) {
                if (-not $names.Contains("real-model/generation/$backend-$corpus-l$length-g32.json")) {
                    throw '生成参照未纳入产物清单。'
                }
            }
        }
    }
    [pscustomobject]@{Status='passed'; Purpose='archive_revalidation'; Artifacts=$names.Count
        SuiteExecutions=$ctest.SuiteExecutions; CaseExecutions=$ctest.CaseExecutions; TeacherCases=240
        TeacherComparisons=11760; NaturalGenerationCases=12; FullCorpusContract=$true; PerformanceBaseline=$false; GpuServing=$false}
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
