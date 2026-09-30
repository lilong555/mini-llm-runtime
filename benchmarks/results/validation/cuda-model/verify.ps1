param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 7' -or
    $evidence.status -cne 'passed' -or $evidence.complete_gpu_model -ne $true -or
    $evidence.full_corpus_contract -ne $false -or $evidence.gpu_serving -ne $false) { throw '证据范围或状态无效。' }
if ($evidence.source.snapshot.path -cne 'source-snapshot.zip' -or
    $evidence.source.state_file -cne 'source-state.json') { throw '源码路径无效。' }
$snapshot = Join-Path $Directory 'source-snapshot.zip'
if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
    $evidence.source.snapshot.sha256) { throw '源码快照摘要不符。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-model-verify-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('Benchmark-Common.ps1','Test-CtestEvidence.ps1')) {
            $entry = $zip.GetEntry("scripts/$name")
            if ($null -eq $entry) { throw "快照缺少验证工具：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,(Join-Path $temporary $name))
        }
    } finally { $zip.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    if ((Get-LowerSha256 (Join-Path $Directory 'source-state.json')) -cne
        $evidence.source.worktree_state_sha256) { throw '源码清单摘要不符。' }
    Test-BenchmarkSourceArchive $Directory $evidence.source
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($artifact in $evidence.artifacts) {
        if (-not $names.Add($artifact.path)) { throw '证据路径重复。' }
        $path = Resolve-BenchmarkArtifact $Directory $artifact.path
        if ((Get-LowerSha256 $path) -cne $artifact.sha256 -or
            (Get-Item -LiteralPath $path -Force).Length -ne $artifact.size_bytes) { throw "证据摘要不符：$($artifact.path)" }
    }
    foreach ($name in @('source-state.json','source-snapshot.zip','own-ctest.xml','cpu-ctest.xml','core-ctest.xml',
        'upstream-ctest.xml','memcheck.txt','racecheck.txt','synccheck.txt','model-memcheck.txt',
        'model-cpu.json','http-cpu.json','build-boundaries.json','real-model/validation-summary.json',
        'real-model/input.json','real-model/model-validation.json','real-model/weight-plan.json',
        'real-model-memcheck/validation-summary.json','real-model-memcheck/model-validation.json',
        'cli-0.json','cli-1.json','cli-2.json','negative-checks.json')) {
        if (-not $names.Contains($name)) { throw "缺少必需证据：$name" }
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.SuiteExecutions -ne 31 -or
        $ctest.CaseExecutions -ne 749) { throw 'CTest 数量或状态不符。' }
    $memcheck = Get-Content -Raw (Join-Path $Directory 'memcheck.txt')
    if ([regex]::Matches($memcheck,'ERROR SUMMARY: 0 errors').Count -ne 5 -or
        [regex]::Matches($memcheck,'0 bytes leaked in 0 allocations').Count -ne 5 -or
        $memcheck -notmatch '8/8 tests passed') { throw '设备单测 memcheck 未通过。' }
    $realMemcheck = Get-Content -Raw (Join-Path $Directory 'model-memcheck.txt')
    if ($realMemcheck -notmatch '2/2 tests passed' -or $realMemcheck -notmatch 'ERROR SUMMARY: 0 errors' -or
        $realMemcheck -notmatch '0 bytes leaked in 0 allocations') { throw '完整模型 memcheck 未通过。' }
    $sync = Get-Content -Raw (Join-Path $Directory 'synccheck.txt')
    $race = Get-Content -Raw (Join-Path $Directory 'racecheck.txt')
    if ($sync -notmatch '8/8 tests passed' -or $sync -notmatch 'ERROR SUMMARY: 0 errors' -or
        $race -notmatch '8/8 tests passed' -or $race -notmatch '0 hazards displayed \(0 errors, 0 warnings\)') {
        throw '运行时同步或竞争检查未通过。'
    }
    $model = Get-Content -Raw (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $http.status -cne 'passed' -or
        $http.passed -ne 8 -or $http.server_after.gpu -ne $false -or $http.server_after.backend -cne 'minillm' -or
        $http.server_after.active_requests -ne 0) { throw 'CPU 回归未通过。' }
    foreach ($label in @('real-model','real-model-memcheck')) {
        $summary = Get-Content -Raw (Join-Path $Directory "$label/validation-summary.json") | ConvertFrom-Json
        $report = Get-Content -Raw (Join-Path $Directory "$label/model-validation.json") | ConvertFrom-Json
        if ($summary.passed -ne $true -or $summary.complete_gpu_model -ne $true -or
            $summary.full_corpus_contract -ne $false -or $summary.logits_comparisons -ne 128 -or $summary.golden_cases -ne 6 -or
            $summary.model_sha256 -cne $evidence.model_sha256 -or $summary.reference_sha256 -cne $evidence.reference_sha256 -or
            $report.comparisons.Count -ne 128 -or $report.golden.Count -ne 6 -or $report.configurations.Count -ne 2) {
            throw '完整模型摘要无效。'
        }
        foreach ($c in $report.comparisons) {
            if ($c.passed -ne $true -or $c.all_finite -ne $true -or $c.rmse -ge 0.05 -or $c.max_absolute -ge 0.5 -or
                $c.cosine -lt 0.9999 -or $c.actual_sha256 -notmatch '^[0-9a-f]{64}$' -or
                $c.reference_sha256 -notmatch '^[0-9a-f]{64}$' -or
                $c.near_tie -ne ($c.reference_margin -le 2*$c.max_absolute) -or
                (-not $c.near_tie -and -not $c.argmax_equal)) { throw '完整模型数值契约未通过。' }
        }
        foreach ($c in $report.golden) {
            if ($c.passed -ne $true -or $c.actual.Count -ne 8 -or ($c.actual -join ',') -cne ($c.expected -join ',')) {
                throw '模型金标准未通过。'
            }
        }
        foreach ($c in $report.configurations) {
            if ($c.passed -ne $true -or $c.max_sequences -notin @(1,4) -or $c.max_model_len -ne 2048 -or
                $c.batch_tokens -ne 128 -or $c.steady_project_allocation_calls -ne 0 -or $c.steady_project_release_calls -ne 0 -or
                $c.before.weight_h2d_bytes -ne $c.after.weight_h2d_bytes -or $c.before.rope_h2d_bytes -ne $c.after.rope_h2d_bytes -or
                $c.after.intermediate_h2d_bytes -ne 0 -or $c.after.intermediate_d2h_bytes -ne 0 -or
                $c.after.post_launch_failures -ne 0 -or $c.after.live_kv_tokens -ne 0 -or $c.after.state -cne 'ready' -or
                $c.after.owned_device_bytes -ne $c.after.resident.total_owned_bytes -or $c.after.owned_device_allocations -ne 4) {
                throw '完整模型数据路径或生命周期不符。'
            }
            if ($c.max_sequences -eq 4 -and $c.after.owned_device_bytes -ne 3449229312) { throw 'S=4 显存计划不符。' }
        }
    }
    $contract = Get-Content -Raw (Join-Path $Directory 'real-model/validation-contract.json') | ConvertFrom-Json
    for ($i=0; $i -lt 3; ++$i) {
        $cli = Get-Content -Raw (Join-Path $Directory "cli-$i.json") | ConvertFrom-Json
        if ($cli.status -cne 'passed' -or $cli.backend -cne 'minillm-cuda' -or
            $cli.model_sha256 -cne $evidence.model_sha256 -or $cli.arithmetic.source_weight_dtype -cne 'Q8_0' -or
            $cli.arithmetic.device_weight_dtype -cne 'F32' -or
            ($cli.token_ids -join ',') -cne ($contract.stable_greedy[$i].expected_token_ids -join ',') -or
            $cli.steady_project_allocation_calls -ne 0 -or $cli.steady_project_release_calls -ne 0 -or
            $cli.after.debug_d2h_bytes -ne 0 -or $cli.after.token_d2h_bytes -ne 32 -or $cli.after.status_d2h_bytes -ne 64 -or
            $cli.before.weight_h2d_bytes -ne $cli.after.weight_h2d_bytes -or $cli.after_clear.live_kv_tokens -ne 0 -or
            $cli.after.state -cne 'ready' -or $cli.after.post_launch_failures -ne 0) { throw 'CLI 生成或数据路径不符。' }
    }
    $boundary = Get-Content -Raw (Join-Path $Directory 'build-boundaries.json') | ConvertFrom-Json
    $own = @($boundary.configurations | Where-Object { $_.configuration -ceq 'own' })[0]
    if ($own.cache.GGML_CUDA -cne 'OFF' -or $own.cache.MINILLM_ENABLE_CUDA -cne 'ON' -or
        $boundary.cli_cpu_runtime_symbols.Count -ne 0) { throw '构建或执行归属不符。' }
    $negative = Get-Content -Raw (Join-Path $Directory 'negative-checks.json') | ConvertFrom-Json
    if ($negative.status -cne 'passed' -or $negative.existing_cli_report.unchanged -ne $true -or
        $negative.existing_model_directory.unchanged -ne $true -or $negative.wrong_checkpoint.passed -ne $false) {
        throw '拒绝和报告保护反例未通过。'
    }
    [pscustomobject]@{Status='passed'; Purpose='archive_revalidation'; Artifacts=$names.Count
        SuiteExecutions=$ctest.SuiteExecutions; CaseExecutions=$ctest.CaseExecutions; CompleteGpuModel=$true
        FullCorpusContract=$false; GpuServing=$false}
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
