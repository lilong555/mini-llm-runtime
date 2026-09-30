param([string]$Directory = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 8 model benchmark contracts' -or
    $evidence.status -cne 'passed' -or $evidence.performance_baseline -ne $false -or
    $evidence.gpu_serving -ne $false -or $evidence.source.state_file -cne 'preflight/source-state.json' -or
    $evidence.source.snapshot.path -cne 'preflight/source-snapshot.zip' -or
    $evidence.verification_source.state_file -cne 'final-preflight/source-state.json' -or
    $evidence.verification_source.snapshot.path -cne 'final-preflight/source-snapshot.zip') { throw '验收范围或源码路径不符。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-benchmark-verify-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    $snapshot = Join-Path $Directory $evidence.verification_source.snapshot.path
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
        $evidence.verification_source.snapshot.sha256) { throw '源码快照摘要不符。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('scripts/Benchmark-Common.ps1', 'scripts/Test-CtestEvidence.ps1',
            'scripts/analyze_cuda_benchmark.py')) {
            $entry = $zip.GetEntry($name)
            if ($null -eq $entry) { throw "源码缺少验证工具：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,
                (Join-Path $temporary ([IO.Path]::GetFileName($name))))
        }
    } finally { $zip.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    foreach ($source in @($evidence.source, $evidence.verification_source)) {
        if ((Get-LowerSha256 (Resolve-BenchmarkArtifact $Directory $source.state_file)) -cne
            $source.worktree_state_sha256 -or
            (Get-LowerSha256 (Resolve-BenchmarkArtifact $Directory $source.snapshot.path)) -cne
            $source.snapshot.sha256) { throw '源码状态或 ZIP 摘要不符。' }
        Test-BenchmarkSourceArchive $Directory $source
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $evidence.artifacts) {
        if (-not $names.Add($item.path)) { throw 'artifact 路径重复。' }
        $path = Resolve-BenchmarkArtifact $Directory $item.path
        if ((Get-LowerSha256 $path) -cne $item.sha256 -or
            (Get-Item -LiteralPath $path -Force).Length -ne $item.size_bytes) { throw "artifact 摘要或长度不符：$($item.path)" }
    }
    foreach ($name in @('preflight/manifest.json', 'preflight/source-state.json', 'preflight/source-snapshot.zip',
        'preflight/input.json', 'preflight/collection-status.json', 'preflight/preflight-environment.json',
        'preflight/validation-summary.json', 'preflight/numerical-source-state.json',
        'real-processes/cpu8.json', 'real-processes/cpu16.json', 'real-processes/cuda.json',
        'preflight/cuda-aa_cuda-t0-p0.json', 'own-ctest.xml', 'cpu-ctest.xml', 'core-ctest.xml', 'upstream-ctest.xml',
        'model-cpu.json', 'http-cpu.json', 'short-model/validation-summary.json', 'build-boundaries.json',
        'diagnostics/initial-cli-fixture.txt', 'diagnostics/fixture-output-path.py', 'negative-checks.json',
        'final-preflight/manifest.json', 'final-preflight/source-state.json', 'final-preflight/source-snapshot.zip',
        'final-preflight/numerical-environment.json', 'final-preflight/collection-status.json')) {
        if (-not $names.Contains($name)) { throw "缺少必需证据：$name" }
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.Reports -ne 4 -or
        $ctest.SuiteExecutions -ne 43 -or $ctest.CaseExecutions -ne 875) { throw 'CTest 覆盖范围不符。' }
    $expectedCounts = @{ own = 16; cpu = 10; core = 7; upstream = 10 }
    foreach ($key in $expectedCounts.Keys) {
        $xml = [xml](Get-Content -Raw (Join-Path $Directory "$key-ctest.xml"))
        if ([int]$xml.testsuite.tests -ne $expectedCounts[$key]) { throw "CTest suite 数量不符：$key" }
    }
    $model = Get-Content -Raw (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    $short = Get-Content -Raw (Join-Path $Directory 'short-model/validation-summary.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $http.status -cne 'passed' -or
        $http.passed -ne 8 -or $http.server_after.backend -cne 'minillm' -or $http.server_after.gpu -ne $false -or
        $http.server_after.active_requests -ne 0 -or $short.status -cne 'passed' -or $short.passed -ne $true -or
        $short.logits_comparisons -ne 128 -or $short.golden_cases -ne 6) { throw 'CPU/HTTP 或 CUDA 短模型回归未通过。' }
    $boundary = Get-Content -Raw (Join-Path $Directory 'build-boundaries.json') | ConvertFrom-Json
    if ($boundary.single_process_binary_sha256 -cne $boundary.post_checks_binary_sha256 -or
        ($boundary.cpu_dynamic_dependencies -join "`n") -match 'lib(cudart|cublas|cuda|nvrtc)') {
        throw '基准二进制身份变化或 CPU 产品依赖 CUDA。'
    }
    foreach ($configuration in $boundary.configurations) {
        $own = if ($configuration.configuration -ceq 'own') { 'ON' } else { 'OFF' }
        $upstream = if ($configuration.configuration -ceq 'upstream') { 'ON' } else { 'OFF' }
        $llama = if ($configuration.configuration -ceq 'core') { 'OFF' } else { 'ON' }
        if ($configuration.MINILLM_ENABLE_CUDA -cne $own -or $configuration.LLMSERVE_CUDA -cne $upstream -or
            $configuration.LLMSERVE_WITH_LLAMA -cne $llama -or $configuration.LLMSERVE_REQUIRE_TEST_TOOLS -cne 'ON') {
            throw '构建边界不符。'
        }
    }
    $manifestPath = Join-Path $Directory 'preflight/manifest.json'
    $manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
    $collection = Get-Content -Raw (Join-Path $Directory 'preflight/collection-status.json') | ConvertFrom-Json
    if ($collection.status -cne 'preflight_only' -or $collection.planned_reports -ne 70 -or
        $collection.reports.Count -ne 0 -or $manifest.reports.Count -ne 70 -or
        $manifest.binary.sha256 -cne $boundary.single_process_binary_sha256) { throw '预检被错误标记为完整采集。' }
    $planText = @(& python3 (Join-Path $temporary 'analyze_cuda_benchmark.py') --schedule)
    if ($LASTEXITCODE -ne 0) { throw '不能重建固定采集顺序。' }
    $plan = ($planText -join "`n") | ConvertFrom-Json
    if ((ConvertTo-Json -Depth 32 -Compress $plan.reports) -cne (ConvertTo-Json -Depth 32 -Compress $manifest.reports) -or
        (ConvertTo-Json -Depth 32 -Compress $plan.statistics) -cne (ConvertTo-Json -Depth 32 -Compress $manifest.statistics)) {
        throw '固定顺序或统计参数不符。'
    }
    $first = $null
    $forwards = 0
    foreach ($file in @('real-processes/cpu8.json', 'real-processes/cpu16.json',
        'real-processes/cuda.json', 'preflight/cuda-aa_cuda-t0-p0.json')) {
        $reportPath = Join-Path $Directory $file
        $lines = @(& python3 (Join-Path $temporary 'analyze_cuda_benchmark.py') --report $reportPath `
            --input (Join-Path $Directory 'preflight/input.json') 2>&1)
        if ($LASTEXITCODE -ne 0) { $lines | Write-Output; throw "模型基准报告复核失败：$file" }
        $audit = ($lines -join "`n") | ConvertFrom-Json
        if ($audit.cases.PSObject.Properties.Name.Count -ne 12 -or $audit.forwards -ne 585 -or
            $audit.input_tokens -ne 30125 -or $audit.logits_rows -ne 400) { throw '完整 workload 计数不符。' }
        $forwards += $audit.forwards
        if ($null -eq $first) { $first = $audit }
        foreach ($name in $audit.cases.PSObject.Properties.Name) {
            if ($audit.cases.$name.samples.Count -ne 3 -or
                (ConvertTo-Json -Compress $audit.cases.$name.token_ids) -cne
                (ConvertTo-Json -Compress $first.cases.$name.token_ids)) { throw "样本数或跨后端 token 不一致：$name" }
        }
    }
    $bound = Get-Content -Raw (Join-Path $Directory 'preflight/cuda-aa_cuda-t0-p0.json') | ConvertFrom-Json
    if ($bound.run_identity.manifest_sha256 -cne (Get-LowerSha256 $manifestPath) -or
        $bound.run_identity.binary_sha256 -cne $manifest.binary.sha256 -or
        $bound.run_identity.source_state_sha256 -cne $manifest.source.worktree_state_sha256 -or
        (ConvertTo-Json -Depth 32 -Compress $bound.process) -cne (ConvertTo-Json -Depth 32 -Compress $manifest.reports[4])) {
        throw '真实 manifest 进程的绑定不符。'
    }
    $negative = Get-Content -Raw (Join-Path $Directory 'negative-checks.json') | ConvertFrom-Json
    if ($negative.status -cne 'passed' -or $negative.incomplete_baseline.exit_code -ne 1 -or
        $negative.existing_directory.exit_code -ne 1 -or $negative.existing_directory.unchanged -ne $true -or
        $negative.numerical_binary_identity.exit_code -ne 1) {
        throw '预检或旧归档保护反例未通过。'
    }
    $finalManifest = Get-Content -Raw (Join-Path $Directory 'final-preflight/manifest.json') | ConvertFrom-Json
    $numericEnvironment = Get-Content -Raw (Join-Path $Directory 'final-preflight/numerical-environment.json') | ConvertFrom-Json
    if ($finalManifest.numerical_evidence.validation_binary_sha256 -cne $numericEnvironment.full_validation_binary_sha256 -or
        $finalManifest.numerical_evidence.validation_binary_sha256 -cne
        '4b31b3d60cf3b5d79fcbec054a234ad273eee67a03d661a7a40bde2ea08c518f' -or
        $finalManifest.binary.sha256 -cne $manifest.binary.sha256) { throw '数值编译身份或基准 binary 不符。' }
    [pscustomobject]@{ Status = 'passed'; Artifacts = $names.Count; SuiteExecutions = $ctest.SuiteExecutions
        CaseExecutions = $ctest.CaseExecutions; RealProcesses = 4; Forwards = $forwards
        MeasuredRepetitions = 144; PerformanceBaseline = $false; GpuServing = $false }
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
