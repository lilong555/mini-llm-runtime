param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 8 micro and numerical validation' -or
    $evidence.status -cne 'passed' -or $evidence.performance_baseline -ne $false -or $evidence.gpu_serving -ne $false) {
    throw '证据范围或状态无效。'
}
if ($evidence.source.state_file -cne 'source-state.json' -or
    $evidence.source.snapshot.path -cne 'source-snapshot.zip') { throw '源码路径无效。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-micro-verify-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $snapshot = Join-Path $Directory 'source-snapshot.zip'
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
        $evidence.source.snapshot.sha256) { throw '源码 ZIP 摘要不符。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('scripts/Benchmark-Common.ps1', 'scripts/Test-CtestEvidence.ps1',
            'scripts/analyze_cuda_validation.py', 'scripts/analyze_cuda_micro.py', 'scripts/analyze_cuda_benchmark.py',
            'tests/data/qwen3_validation_cases.json', 'benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json')) {
            $entry = $zip.GetEntry($name)
            if ($null -eq $entry) { throw "快照缺少复核工具或契约：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $temporary ([IO.Path]::GetFileName($name))))
        }
    } finally { $zip.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    Test-BenchmarkSourceArchive $Directory $evidence.source
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $evidence.artifacts) {
        if (-not $names.Add($item.path)) { throw '证据路径重复。' }
        $path = Resolve-BenchmarkArtifact $Directory $item.path
        if ((Get-LowerSha256 $path) -cne $item.sha256 -or (Get-Item -LiteralPath $path -Force).Length -ne $item.size_bytes) {
            throw "证据摘要或长度不符：$($item.path)"
        }
    }
    foreach ($name in @('source.json','source-state.json','source-snapshot.zip','environment.json','build-boundaries.json',
        'own-ctest.xml','cpu-ctest.xml','core-ctest.xml','upstream-ctest.xml','ctest-audit.json','model-cpu.json','http-cpu.json',
        'full-model.txt','full-model-command.json','numerical-source-identity.json','numerical-audit.json',
        'real-model/input.json','real-model/validation-contract.json','real-model/validation-summary.json',
        'real-model/full-validation.json','real-model/weight-plan.json','real-model/memory-plan.json',
        'layer-memcheck.txt','layer-racecheck.txt','layer-synccheck.txt',
        'layer-memcheck-command.json','layer-racecheck-command.json','layer-synccheck-command.json','negative-checks.json',
        'model-preflight/manifest.json','model-preflight/source-state.json','model-preflight/source-snapshot.zip',
        'model-preflight/collection-status.json',
        'diagnostics/functional-report.json','diagnostics/functional-check.json','verify.ps1')) {
        if (-not $names.Contains($name)) { throw "缺少必需证据：$name" }
    }
    $source = Get-Content -Raw (Join-Path $Directory 'source.json') | ConvertFrom-Json
    if ((ConvertTo-Json -Depth 16 -Compress $source) -cne (ConvertTo-Json -Depth 16 -Compress $evidence.source)) {
        throw '源码身份记录不符。'
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.Reports -ne 4 -or $ctest.SuiteExecutions -ne 50 -or
        $ctest.CaseExecutions -ne 926) { throw 'CTest 数量或状态不符。' }
    $model = Get-Content -Raw (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $model.reference_gpu_layers -ne 0 -or
        $http.status -cne 'passed' -or $http.passed -ne 8 -or $http.server_after.gpu -ne $false -or
        $http.server_after.backend -cne 'minillm' -or $http.server_after.active_requests -ne 0) {
        throw 'CPU 模型或 HTTP 回归未通过。'
    }
    foreach ($tool in @('memcheck','racecheck','synccheck')) {
        $command = Get-Content -Raw (Join-Path $Directory "layer-$tool-command.json") | ConvertFrom-Json
        $log = Get-Content -Raw (Join-Path $Directory "layer-$tool.txt")
        if ($command.status -cne 'passed' -or $command.exit_code -ne 0 -or
            $command.command -cne 'compute-sanitizer' -or $command.arguments[1] -cne $tool -or
            $log -notmatch '(?m)^9/9 tests passed\r?$' -or $log.Contains('[FAIL]')) { throw "CUDA $tool 未通过。" }
        if ($tool -eq 'racecheck') {
            if ($log -notmatch 'RACECHECK SUMMARY: 0 hazards displayed \(0 errors, 0 warnings\)') {
                throw 'CUDA racecheck 存在错误或警告。'
            }
        } elseif ($log -notmatch 'ERROR SUMMARY: 0 errors') { throw "CUDA $tool 存在错误。" }
        if ($tool -eq 'memcheck' -and $log -notmatch 'LEAK SUMMARY: 0 bytes leaked in 0 allocations') {
            throw 'CUDA memcheck 存在泄漏。'
        }
    }
    $environment = Get-Content -Raw (Join-Path $Directory 'environment.json') | ConvertFrom-Json
    $boundary = Get-Content -Raw (Join-Path $Directory 'build-boundaries.json') | ConvertFrom-Json
    $identity = Get-Content -Raw (Join-Path $Directory 'numerical-source-identity.json') | ConvertFrom-Json
    if ($boundary.own_cache.MINILLM_ENABLE_CUDA -cne 'ON' -or $boundary.own_cache.GGML_CUDA -cne 'OFF' -or
        $boundary.micro_cpu_or_upstream_forward_symbols.Count -ne 0 -or $identity.status -cne 'passed' -or
        $identity.before_after_equal -ne $true -or $identity.binary_sha256 -cne $environment.full_validation_binary_sha256 -or
        $identity.source_state_sha256 -cne $evidence.source.worktree_state_sha256) { throw '执行归属或数值编译身份不符。' }
    foreach ($binary in $boundary.binaries) {
        if ($binary.configuration -ceq 'cpu' -and ($binary.dynamic_dependencies -join "`n") -match 'lib(cudart|cublas|cuda|nvrtc)') {
            throw 'CPU 产品依赖 CUDA。'
        }
        if ($binary.name -ceq 'minillm-cuda-model-tests' -and $binary.sha256 -cne $identity.binary_sha256) {
            throw '数值测试二进制记录不一致。'
        }
        if ($binary.name -ceq 'mini-cuda-kernel-bench' -and $binary.sha256 -cne $environment.micro_binary_sha256) {
            throw '微基准二进制记录不一致。'
        }
    }
    $command = Get-Content -Raw (Join-Path $Directory 'full-model-command.json') | ConvertFrom-Json
    if ($command.status -cne 'passed' -or $command.exit_code -ne 0 -or $command.arguments[2] -cne 'model-full-check') {
        throw '全量数值执行未完成。'
    }
    $lines = @(& python3 (Join-Path $temporary 'analyze_cuda_validation.py') `
        --directory (Join-Path $Directory 'real-model') --contract (Join-Path $temporary 'qwen3_validation_cases.json') 2>&1)
    if ($LASTEXITCODE -ne 0) { $lines | Write-Output; throw '完整数值证据复核失败。' }
    $numeric = ($lines -join "`n") | ConvertFrom-Json
    $saved = Get-Content -Raw (Join-Path $Directory 'numerical-audit.json') | ConvertFrom-Json
    if ((ConvertTo-Json -Depth 32 -Compress $numeric) -cne (ConvertTo-Json -Depth 32 -Compress $saved)) {
        throw '数值复核记录不符。'
    }
    $lines = @(& python3 (Join-Path $temporary 'analyze_cuda_micro.py') `
        --report (Join-Path $Directory 'diagnostics/functional-report.json') `
        --input (Join-Path $temporary 'qwen3-cuda-micro-v0.json') 2>&1)
    if ($LASTEXITCODE -ne 0) { $lines | Write-Output; throw 'micro 功能报告复核失败。' }
    $functional = ($lines -join "`n") | ConvertFrom-Json
    $functionalRecord = Get-Content -Raw (Join-Path $Directory 'diagnostics/functional-check.json') | ConvertFrom-Json
    if ($functional.case_count -ne 375 -or $functional.samples -ne 1875 -or
        $functionalRecord.performance_trial -ne $false -or $null -ne $functionalRecord.binary_identity_at_run) {
        throw '功能检查不能冒充带身份的独立性能 trial。'
    }
    $preflight = Get-Content -Raw (Join-Path $Directory 'model-preflight/manifest.json') | ConvertFrom-Json
    $preflightState = Get-Content -Raw (Join-Path $Directory 'model-preflight/collection-status.json') | ConvertFrom-Json
    if ($preflight.numerical_evidence.validation_binary_sha256 -cne $identity.binary_sha256 -or
        $preflight.numerical_evidence.repository_path -cne 'benchmarks/results/validation/cuda-micro' -or
        $preflightState.status -cne 'preflight_only' -or $preflightState.planned_reports -ne 70 -or
        $preflightState.reports.Count -ne 0) { throw '模型基准预检的数值来源或范围不符。' }
    Test-BenchmarkSourceArchive (Join-Path $Directory 'model-preflight') $preflight.source
    $negative = Get-Content -Raw (Join-Path $Directory 'negative-checks.json') | ConvertFrom-Json
    if ($negative.status -cne 'passed' -or $negative.checks.Count -ne 4) { throw '拒绝反例不完整。' }
    foreach ($check in $negative.checks) {
        if ($check.exit_code -ne 1 -or $check.unchanged -ne $true) { throw '拒绝反例没有无损失败。' }
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
        SuiteExecutions=$ctest.SuiteExecutions; CaseExecutions=$ctest.CaseExecutions
        MicroFunctionalCases=375
        FullNumericComparisons=($numeric.totals.teacher_comparisons + $numeric.totals.generation_comparisons)
        PerformanceBaseline=$false; GpuServing=$false}
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
