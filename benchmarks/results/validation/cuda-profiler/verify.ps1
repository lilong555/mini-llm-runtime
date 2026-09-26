# 仅复核归档中的原始验证证据，不运行模型或 GPU 工具。
param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$evidence = Get-Content -Raw -LiteralPath (Join-Path $Directory 'evidence.json') | ConvertFrom-Json
if ($evidence.schema_version -ne 1 -or $evidence.scope -cne 'CUDA-VS-001 Step 9 tool validation' -or
    $evidence.status -cne 'passed' -or $evidence.gpu_serving -ne $false) {
    throw '工具验证归档范围或状态不符。'
}
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('cuda-profiler-checks-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
try {
    if ($evidence.source.state_file -cne 'source-state.json' -or
        $evidence.source.snapshot.path -cne 'source-snapshot.zip') { throw '源码路径无效。' }
    $snapshot = Join-Path $Directory 'source-snapshot.zip'
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -cne $evidence.source.snapshot.sha256) {
        throw '工具验证源码快照摘要不符。'
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($snapshot)
    try {
        foreach ($name in @('scripts/Benchmark-Common.ps1', 'scripts/Test-CtestEvidence.ps1')) {
            $entry = $archive.GetEntry($name)
            if ($null -eq $entry) { throw "源码快照缺少复核工具：$name" }
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path $temporary ([IO.Path]::GetFileName($name))))
        }
    } finally { $archive.Dispose() }
    . (Join-Path $temporary 'Benchmark-Common.ps1')
    Test-BenchmarkSourceArchive $Directory $evidence.source
    if ((Get-LowerSha256 (Join-Path $Directory 'source-state.json')) -cne $evidence.source.worktree_state_sha256) {
        throw '工具验证源码清单摘要不符。'
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($item in $evidence.artifacts) {
        if (-not $names.Add($item.path)) { throw '工具验证产物路径重复。' }
        $path = Resolve-BenchmarkArtifact $Directory $item.path
        if ((Get-LowerSha256 $path) -cne $item.sha256 -or (Get-Item -LiteralPath $path).Length -ne $item.size_bytes) {
            throw "工具验证产物摘要或长度不符：$($item.path)"
        }
    }
    foreach ($name in @('source-state.json', 'source-snapshot.zip', 'environment.json', 'verify.ps1',
        'own-ctest.xml', 'cpu-ctest.xml', 'core-ctest.xml', 'upstream-ctest.xml', 'ctest-audit.json',
        'model-cpu.json', 'http-cpu.json', 'model-cpu-command.json', 'http-cpu-command.json',
        'same-backend-regression.json', 'numerical-inheritance.json')) {
        if (-not $names.Contains($name)) { throw "工具验证缺少必需产物：$name" }
    }
    $ctest = & (Join-Path $temporary 'Test-CtestEvidence.ps1') -Directory $Directory
    if ($ctest.Status -cne 'passed' -or $ctest.Reports -ne 4 -or $ctest.SuiteExecutions -lt 62) {
        throw '工具验证没有通过四构建的完整 CTest。'
    }
    foreach ($stem in @('own', 'cpu', 'core', 'upstream')) {
        $xml = [Xml.XmlDocument]::new()
        $xml.XmlResolver = $null
        $xml.Load((Join-Path $Directory "$stem-ctest.xml"))
        foreach ($name in @('cuda-profiler-validation', 'cuda-bundle-validation',
            'cuda-benchmark-validation', 'cuda-profiler-process')) {
            if ($null -eq $xml.DocumentElement.SelectSingleNode("testcase[@name='$name']")) {
                throw "CTest 缺少工具套件：$stem/$name"
            }
        }
    }
    $model = Get-Content -Raw -LiteralPath (Join-Path $Directory 'model-cpu.json') | ConvertFrom-Json
    $http = Get-Content -Raw -LiteralPath (Join-Path $Directory 'http-cpu.json') | ConvertFrom-Json
    if ($model.status -cne 'passed' -or $model.passed -ne 13 -or $model.reference_gpu_layers -ne 0 -or
        $http.status -cne 'passed' -or $http.passed -ne 8 -or $http.server_after.gpu -ne $false -or
        $http.server_after.backend -cne 'minillm' -or $http.server_after.active_requests -ne 0) {
        throw 'CPU 模型或 HTTP 非回归检查未通过。'
    }
    foreach ($name in @('model-cpu', 'http-cpu')) {
        $command = Get-Content -Raw -LiteralPath (Join-Path $Directory "$name-command.json") | ConvertFrom-Json
        if ($command.exit_code -ne 0 -or $command.status -cne 'passed') { throw "回归进程失败：$name" }
    }
    $environment = Get-Content -Raw -LiteralPath (Join-Path $Directory 'environment.json') | ConvertFrom-Json
    $inheritance = Get-Content -Raw -LiteralPath (Join-Path $Directory 'numerical-inheritance.json') | ConvertFrom-Json
    if ($inheritance.status -cne 'passed' -or $inheritance.runtime_source_equal -ne $true -or
        $inheritance.validation_binary_sha256 -cne $environment.full_validation_binary_sha256 -or
        $inheritance.numeric_comparisons -ne 12528 -or $inheritance.rerun_full_numeric -ne $false) {
        throw '完整数值门禁的源码或编译继承不符。'
    }
    $regression = Get-Content -Raw -LiteralPath (Join-Path $Directory 'same-backend-regression.json') | ConvertFrom-Json
    if ($regression.status -cne 'passed' -or $regression.old_validator_accepted_inconsistent_aa -ne $true -or
        $regression.current_validator_rejected_inconsistent_aa -ne $true -or
        $regression.current_validator_rejected_cross_trial_difference -ne $true) {
        throw '同后端跨进程输出不一致的回归证据不完整。'
    }
    [pscustomobject]@{ Status='passed'; Purpose='archive_revalidation'; Artifacts=$names.Count
        SuiteExecutions=$ctest.SuiteExecutions; CaseExecutions=$ctest.CaseExecutions
        CpuModelChecks=13; CpuHttpChecks=8; FullNumericComparisonsInherited=12528; GpuServing=$false }
} finally { Remove-Item -LiteralPath $temporary -Recurse -Force }
