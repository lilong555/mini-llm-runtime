param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 6' -or
    $evidence.status -cne 'passed' -or $evidence.complete_gpu_model -ne $false) { throw '证据范围或状态无效。' }
if ($evidence.source.snapshot.path -cne 'source-snapshot.zip' -or
    $evidence.source.state_file -cne 'source-state.json') { throw '源码路径无效。' }
$snapshot = Join-Path $Directory 'source-snapshot.zip'
if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
    $evidence.source.snapshot.sha256) { throw '源码快照摘要不符。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-layer-verify-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('Benchmark-Common.ps1', 'Test-CtestEvidence.ps1')) {
            $entry = $zip.GetEntry("scripts/$name")
            if ($null -eq $entry) { throw "快照缺少验证工具：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $temporary $name))
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
    foreach ($name in @('source-state.json', 'source-snapshot.zip', 'own-ctest.xml', 'cpu-ctest.xml',
        'core-ctest.xml', 'upstream-ctest.xml', 'memcheck.txt', 'racecheck.txt', 'synccheck.txt',
        'model-cpu.json', 'http-cpu.json', 'real-storage/validation-summary.json', 'real-storage/memory-plan.json',
        'real-layer/layer-validation.json', 'real-layer/validation-summary.json',
        'real-layer-memcheck/layer-validation.json', 'real-layer-memcheck.txt', 'build-boundaries.json',
        'diagnostics/fp16-boundary.txt', 'diagnostics/identity.json', 'diagnostics/source-snapshot.zip')) {
        if (-not $names.Contains($name)) { throw "缺少必需证据：$name" }
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.SuiteExecutions -ne 30 -or
        $ctest.CaseExecutions -ne 741) { throw 'CTest 数量或状态不符。' }
    $memcheck = Get-Content -Raw -LiteralPath (Join-Path $Directory 'memcheck.txt')
    if ([regex]::Matches($memcheck, 'ERROR SUMMARY: 0 errors').Count -ne 4 -or
        [regex]::Matches($memcheck, '0 bytes leaked in 0 allocations').Count -ne 4 -or
        [regex]::Matches($memcheck, '11/11 tests passed').Count -ne 3 -or
        $memcheck -notmatch '7/7 tests passed') { throw '设备单测 memcheck 未通过。' }
    $realMemcheck = Get-Content -Raw -LiteralPath (Join-Path $Directory 'real-layer-memcheck.txt')
    if ($realMemcheck -notmatch '8/8 tests passed' -or $realMemcheck -notmatch 'ERROR SUMMARY: 0 errors' -or
        $realMemcheck -notmatch '0 bytes leaked in 0 allocations') { throw '真实层 memcheck 未通过。' }
    $sync = Get-Content -Raw -LiteralPath (Join-Path $Directory 'synccheck.txt')
    $race = Get-Content -Raw -LiteralPath (Join-Path $Directory 'racecheck.txt')
    if ($sync -notmatch '7/7 tests passed' -or $sync -notmatch 'ERROR SUMMARY: 0 errors' -or
        $race -notmatch '7/7 tests passed' -or
        $race -notmatch '0 hazards displayed \(0 errors, 0 warnings\)') { throw '同步或竞争检查未通过。' }
    $model = Get-Content -Raw -LiteralPath (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw -LiteralPath (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    $storage = Get-Content -Raw -LiteralPath (Join-Path $Directory 'real-storage/validation-summary.json') | ConvertFrom-Json
    $memory = Get-Content -Raw -LiteralPath (Join-Path $Directory 'real-storage/memory-plan.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $http.status -cne 'passed' -or
        $http.passed -ne 8 -or $http.server_after.gpu -ne $false -or
        $http.server_after.backend -cne 'minillm' -or $http.server_after.active_requests -ne 0 -or
        $storage.passed -ne $true -or $storage.complete_gpu_model -ne $false -or
        $storage.gemm_cases -ne 88 -or $storage.unique_tensors -ne 310 -or
        $memory.total_bytes -ne 3449229312 -or $memory.rope_bytes -ne 1048576) {
        throw 'CPU 或真实权重回归未通过。'
    }
    foreach ($label in @('real-layer', 'real-layer-memcheck')) {
        $summary = Get-Content -Raw -LiteralPath (Join-Path $Directory "$label/validation-summary.json") | ConvertFrom-Json
        $layer = Get-Content -Raw -LiteralPath (Join-Path $Directory "$label/layer-validation.json") | ConvertFrom-Json
        if ($summary.passed -ne $true -or $summary.complete_gpu_model -ne $false -or $summary.layer_cases -ne 6 -or
            $layer.cases.Count -ne 6 -or $layer.model_sha256 -cne $evidence.model_sha256) { throw '真实层摘要无效。' }
        foreach ($case in $layer.cases) {
            if ($case.passed -ne $true -or $case.rmse -ge 0.05 -or $case.max_absolute -ge 0.5 -or
                $case.cosine -lt 0.9999 -or $case.shared_qkv.passed -ne $true -or
                $case.shared_qkv.atol -ne 0.0002 -or $case.shared_qkv.rtol -ne 0.0002 -or
                $case.project_allocation_calls -ne 0 -or $case.project_release_calls -ne 0) {
                throw '真实层数值或分配契约未通过。'
            }
        }
    }
    $diagnostic = Get-Content -Raw -LiteralPath (Join-Path $Directory 'diagnostics/identity.json') | ConvertFrom-Json
    Test-BenchmarkSourceArchive (Join-Path $Directory 'diagnostics') $diagnostic.source
    if ($diagnostic.status -cne 'failed') { throw '失败诊断不能重新标记为成功。' }
    [pscustomobject]@{Status='passed'; Purpose='archive_revalidation'; Artifacts=$names.Count
        SuiteExecutions=$ctest.SuiteExecutions; CaseExecutions=$ctest.CaseExecutions; CompleteGpuModel=$false}
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
