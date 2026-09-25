param([string]$Directory = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
function Require([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
function Read-Report([string]$Name) {
    Get-Content -Raw -LiteralPath (Join-Path $Directory $Name) | ConvertFrom-Json
}
$evidence = Read-Report 'evidence.json'
Require ($evidence.schema_version -eq 1 -and $evidence.scope -ceq 'CUDA-VS-001 Step 4' -and
    $evidence.status -ceq 'passed') '证据类型、版本或状态无效。'
Require ($evidence.source.snapshot.path -ceq 'source-snapshot.zip' -and
    $evidence.source.state_file -ceq 'source-state.json') '源码路径无效。'
$snapshot = Join-Path $Directory 'source-snapshot.zip'
Require ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -ceq
    $evidence.source.snapshot.sha256) '源码快照摘要不符。'

$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-storage-verify-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    # 只借用本次快照中的既有验证工具，不依赖原采集工作区。
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('Benchmark-Common.ps1', 'Test-CtestEvidence.ps1')) {
            $entry = $archive.GetEntry("scripts/$name")
            Require ($null -ne $entry) "源码快照缺少验证工具：$name"
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $temporary $name))
        }
    } finally { $archive.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    Require ((Get-LowerSha256 (Join-Path $Directory $evidence.source.state_file)) -ceq
        $evidence.source.worktree_state_sha256) '源码清单摘要不符。'
    Test-BenchmarkSourceArchive $Directory $evidence.source
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $evidence.artifacts) {
        Require ($seen.Add($item.path)) "证据路径重复：$($item.path)"
        $path = Resolve-BenchmarkArtifact $Directory $item.path
        Require ((Get-LowerSha256 $path) -ceq $item.sha256 -and
            (Get-Item -LiteralPath $path -Force).Length -eq $item.size_bytes) "证据摘要或尺寸不符：$($item.path)"
    }
    foreach ($name in @('source-state.json', 'source-snapshot.zip', 'build-boundaries.json',
        'own-ctest.xml', 'cpu-ctest.xml', 'core-ctest.xml', 'upstream-ctest.xml',
        'real/validation-summary.json', 'real/weight-plan.json', 'real/matrix-validation.json',
        'real/memory-plan.json', 'real/copy-allocation-summary.json', 'real/environment.json',
        'real/validation-contract.json', 'memcheck/validation-summary.json', 'memcheck/weight-plan.json',
        'memcheck/matrix-validation.json', 'memcheck/memory-plan.json',
        'memcheck-unit.txt', 'memcheck-real.txt', 'model-cpu.json', 'http-cpu.json', 'negative-checks.json')) {
        Require ($seen.Contains($name)) "缺少必需证据：$name"
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    Require ($ctest.Status -ceq 'passed' -and $ctest.SuiteExecutions -eq 28 -and
        $ctest.CaseExecutions -eq 722) 'CTest 数量或状态不符。'
    foreach ($prefix in @('real', 'memcheck')) {
        $summary = Read-Report "$prefix/validation-summary.json"
        $weights = Read-Report "$prefix/weight-plan.json"
        $memory = Read-Report "$prefix/memory-plan.json"
        $matrix = Read-Report "$prefix/matrix-validation.json"
        $contract = Read-Report "$prefix/validation-contract.json"
        Require ($summary.passed -eq $true -and $summary.complete_gpu_model -eq $false -and
            $summary.unique_tensors -eq 310 -and $summary.gemm_cases -eq 88 -and
            $summary.released_project_allocations -eq 4 -and
            $summary.model_sha256 -ceq $contract.model.sha256) '实模型存储验证范围或状态不符。'
        Require ($weights.records.Count -eq 311 -and $weights.verified_unique_bytes -eq 2384199680) '权重证据不完整。'
        Require ($memory.S -eq 4 -and $memory.Lmax -eq 2048 -and $memory.B -eq 128 -and
            $memory.kv_reservation_bytes -eq 939524096 -and $memory.total_bytes -eq 3448180736 -and
            $memory.total_bytes -le $memory.allowed_bytes) '目标配置或预算不符。'
        Require ($matrix.cases.Count -eq 88 -and $matrix.atol -eq 0.0002 -and
            $matrix.rtol -eq 0.0002) '矩阵用例或数值阈值不符。'
        foreach ($case in $matrix.cases) {
            Require ($case.passed -eq $true -and $case.checked_elements -eq $case.M * $case.N -and
                $case.project_allocation_calls -eq 0 -and $case.project_release_calls -eq 0) '矩阵验收失败。'
        }
    }
    foreach ($name in @('weight-plan.json', 'matrix-validation.json', 'copy-allocation-summary.json')) {
        Require ((Get-LowerSha256 (Join-Path $Directory "real/$name")) -ceq
            (Get-LowerSha256 (Join-Path $Directory "memcheck/$name"))) "普通执行与 memcheck 结果不符：$name"
    }
    foreach ($name in @('memcheck-unit.txt', 'memcheck-real.txt')) {
        $text = Get-Content -Raw -LiteralPath (Join-Path $Directory $name)
        Require ($text -match '(?m)^========= ERROR SUMMARY: 0 errors\r?$' -and
            $text -match '(?m)^========= LEAK SUMMARY: 0 bytes leaked in 0 allocations\r?$') '内存检查未通过。'
    }
    $model = Read-Report 'model-cpu.json'
    $http = Read-Report 'http-cpu.json'
    $negative = Read-Report 'negative-checks.json'
    Require ($model.status -ceq 'passed' -and $model.passed -eq 13 -and
        $http.status -ceq 'passed' -and $http.passed -eq 8 -and
        $http.server_after.backend -ceq 'minillm' -and $http.server_after.gpu -eq $false -and
        $http.server_after.active_requests -eq 0) 'CPU 模型或 HTTP 回归未通过。'
    Require ($negative.reuse_rejected -eq $true -and $negative.existing_files_unchanged -eq $true -and
        $negative.wrong_model_rejected -eq $true -and $negative.failed_summary_not_passed -eq $true) '验证反例未通过。'
    [pscustomobject]@{ Status = 'passed'; Artifacts = $seen.Count; SuiteExecutions = $ctest.SuiteExecutions
        CaseExecutions = $ctest.CaseExecutions; UniqueTensors = 310; MatrixCasesPerRun = 88
        CompleteGpuModel = $false; Purpose = 'archive_revalidation' }
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
