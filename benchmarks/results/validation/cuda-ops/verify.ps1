param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 5' -or
    $evidence.status -cne 'passed' -or $evidence.complete_gpu_model -ne $false) { throw '证据范围或状态无效。' }
if ($evidence.source.snapshot.path -cne 'source-snapshot.zip' -or
    $evidence.source.state_file -cne 'source-state.json') { throw '源码路径无效。' }
$snapshot = Join-Path $Directory 'source-snapshot.zip'
if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne
    $evidence.source.snapshot.sha256) { throw '源码快照摘要不符。' }
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-ops-verify-' + [guid]::NewGuid().ToString('N'))
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
        'model-cpu.json', 'http-cpu.json', 'real-storage/validation-summary.json',
        'real-storage/weight-plan.json', 'real-storage/matrix-validation.json', 'build-boundaries.json')) {
        if (-not $names.Contains($name)) { throw "缺少必需证据：$name" }
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.SuiteExecutions -ne 29 -or
        $ctest.CaseExecutions -ne 733) { throw 'CTest 数量或状态不符。' }
    foreach ($name in @('memcheck', 'synccheck')) {
        $text = Get-Content -Raw -LiteralPath (Join-Path $Directory "$name.txt")
        if ($text -notmatch '11/11 tests passed' -or $text -notmatch 'ERROR SUMMARY: 0 errors') { throw "$name 未通过。" }
        if ($name -ceq 'memcheck' -and $text -notmatch '0 bytes leaked in 0 allocations') { throw '存在设备泄漏。' }
    }
    $race = Get-Content -Raw -LiteralPath (Join-Path $Directory 'racecheck.txt')
    if ($race -notmatch '11/11 tests passed' -or $race -notmatch '0 hazards displayed \(0 errors, 0 warnings\)') { throw 'racecheck 未通过。' }
    $model = Get-Content -Raw -LiteralPath (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw -LiteralPath (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    $storage = Get-Content -Raw -LiteralPath (Join-Path $Directory 'real-storage/validation-summary.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $http.status -cne 'passed' -or
        $http.passed -ne 8 -or $http.server_after.gpu -ne $false -or
        $http.server_after.backend -cne 'minillm' -or $http.server_after.active_requests -ne 0 -or
        $storage.passed -ne $true -or $storage.complete_gpu_model -ne $false -or
        $storage.gemm_cases -ne 88 -or $storage.unique_tensors -ne 310) {
        throw 'CPU 或真实权重回归未通过。'
    }
    [pscustomobject]@{Status='passed'; Purpose='archive_revalidation'; Artifacts=$names.Count
        SuiteExecutions=$ctest.SuiteExecutions; CaseExecutions=$ctest.CaseExecutions; CompleteGpuModel=$false}
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
