param(
    [string]$OutputDirectory = '',
    [string]$BinaryDirectory = '',
    [string]$Model = '',
    [string]$NumericalDirectory = '',
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
. (Join-Path $PSScriptRoot 'Cuda-Benchmark-Common.ps1')
$root = Split-Path -Parent $PSScriptRoot
$python = if (Get-Command python3 -ErrorAction SilentlyContinue) { 'python3' } else { 'python' }
$analyzer = Join-Path $PSScriptRoot 'analyze_cuda_benchmark.py'
$inputFile = Join-Path $root 'benchmarks/runtime-inputs/qwen3-cuda-v0.json'
if (-not $BinaryDirectory) {
    $platform = if ($env:OS -eq 'Windows_NT') { '' } else { 'wsl-' }
    $BinaryDirectory = Join-Path $root "build/${platform}own-cuda/bin"
}
if (-not $Model) { $Model = Join-Path $root 'models/Qwen3-0.6B-Q8_0.gguf' }
if (-not $NumericalDirectory) { $NumericalDirectory = Join-Path $root 'benchmarks/results/validation/cuda-micro' }
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$NumericalDirectory = (Resolve-Path -LiteralPath $NumericalDirectory).Path
$build = Split-Path -Parent $BinaryDirectory
while (-not (Test-Path -LiteralPath (Join-Path $build 'CMakeCache.txt'))) {
    $parent = Split-Path -Parent $build
    if (-not $parent -or $parent -eq $build) { throw '找不到 CMake cache。' }
    $build = $parent
}
$cache = Join-Path $build 'CMakeCache.txt'
$buildType = Read-CMakeValue $cache 'CMAKE_BUILD_TYPE'
if (-not $buildType) { $buildType = Split-Path -Leaf $BinaryDirectory }
if ((Read-CMakeValue $cache 'CMAKE_HOME_DIRECTORY') -cne $root -or
    (Read-CMakeValue $cache 'MINILLM_ENABLE_CUDA') -cne 'ON' -or
    (Read-CMakeValue $cache 'LLMSERVE_CUDA') -cne 'OFF' -or $buildType -cnotin @('Release', 'RelWithDebInfo')) {
    throw '正式对照要求本工作区的 Release/RelWithDebInfo、自有 CUDA=ON、上游 CUDA=OFF。'
}
$inputHash = Get-LowerSha256 $inputFile
$modelHash = Get-LowerSha256 $Model
if ($inputHash -cne 'f5a311a0d7c993640ba5b761844a39e70a5ae5015db3ce9dcd07c01b6ad2a6c6' -or
    $modelHash -cne '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031') {
    throw '固定性能输入或模型摘要不符。'
}
$provenance = Get-Content -Raw -LiteralPath (Join-Path $root 'models/manifest.json') | ConvertFrom-Json
if ($provenance.sha256 -cne $modelHash -or [long]$provenance.size_bytes -ne (Get-Item -LiteralPath $Model).Length) {
    throw '模型来源清单不符。'
}
& $python (Join-Path $PSScriptRoot 'analyze_cuda_validation.py') --directory (Join-Path $NumericalDirectory 'real-model')
if ($LASTEXITCODE -ne 0) { throw '必须先通过完整数值证据复核。' }
& cmake --build $build --config $buildType --target mini-cuda-runtime-bench minillm-cuda-model-tests --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'CUDA 模型基准构建失败。' }
$executable = Get-ProductExecutable $BinaryDirectory 'mini-cuda-runtime-bench'
$validationExecutable = Get-ProductExecutable $BinaryDirectory 'minillm-cuda-model-tests'
$numericalEnvironment = Get-Content -Raw (Join-Path $NumericalDirectory 'environment.json') | ConvertFrom-Json
$validationHash = Get-LowerSha256 $validationExecutable
if ($numericalEnvironment.full_validation_binary_sha256 -cne $validationHash) {
    throw '完整数值测试的编译产物已改变，必须在当前构建下重新建立数值验收，不能仅按源码继承。'
}
$compiler = Get-ChildItem -LiteralPath (Join-Path $build 'CMakeFiles') -Recurse -File -Filter 'CMakeCXXCompiler.cmake' |
    Select-Object -First 1
$cudaCompiler = Get-ChildItem -LiteralPath (Join-Path $build 'CMakeFiles') -Recurse -File -Filter 'CMakeCUDACompiler.cmake' |
    Select-Object -First 1
if ($null -eq $compiler -or $null -eq $cudaCompiler) { throw '缺少 host/CUDA 编译器元数据。' }
$dependency = Join-Path $root 'third_party/llama.cpp'
$revision = (& git -C $dependency rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $revision -cne '911f6cdc8ab8a530b2bee09ee61471a6f3178eeb') { throw 'llama.cpp 版本不符。' }
$gitSha = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw '不能识别源码版本。' }
$dirty = @(& git -C $root status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw '不能识别工作区状态。' }
$scope = @('CMakeLists.txt', '.gitattributes', 'cmake', 'apps', 'include', 'src', 'scripts', 'tests', '.github',
    'models/manifest.json', 'models/reference-manifest.json', 'benchmarks/runtime-inputs', 'docs/NEXT_SPEC.md')
$sourceFiles = @(Get-BenchmarkSourceState $root $scope)
$sourceJson = ConvertTo-Json -InputObject $sourceFiles -Depth 8 -Compress
$planText = & $python $analyzer --schedule
if ($LASTEXITCODE -ne 0) { throw '不能建立固定进程顺序。' }
$plan = ($planText -join "`n") | ConvertFrom-Json
$runId = '{0}-{1}-{2}' -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
    $gitSha.Substring(0, 12), ([guid]::NewGuid().ToString('N').Substring(0, 8))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "benchmarks/results/cuda-model-$runId" }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $OutputDirectory) -and @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count) {
    throw '基准输出目录必须为空；已有采样不会被覆盖。'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
Copy-Item -LiteralPath $inputFile -Destination (Join-Path $OutputDirectory 'input.json')
Copy-Item -LiteralPath $analyzer -Destination (Join-Path $OutputDirectory 'verify.py')
Copy-Item -LiteralPath (Join-Path $NumericalDirectory 'real-model/validation-summary.json') `
    -Destination (Join-Path $OutputDirectory 'validation-summary.json')
Copy-Item -LiteralPath (Join-Path $NumericalDirectory 'source-state.json') `
    -Destination (Join-Path $OutputDirectory 'numerical-source-state.json')
Copy-Item -LiteralPath (Join-Path $NumericalDirectory 'environment.json') `
    -Destination (Join-Path $OutputDirectory 'numerical-environment.json')
$stateFile = Join-Path $OutputDirectory 'source-state.json'
Write-BenchmarkJson $stateFile ([ordered]@{ scope = $scope; files = $sourceFiles })
$snapshot = Join-Path $OutputDirectory 'source-snapshot.zip'
Write-BenchmarkSourceSnapshot $root $sourceFiles $snapshot
$artifacts = @('verify.py', 'validation-summary.json', 'numerical-source-state.json', 'numerical-environment.json') | ForEach-Object {
    [ordered]@{ path = $_; sha256 = Get-LowerSha256 (Join-Path $OutputDirectory $_) }
}
$manifest = [ordered]@{
    schema_version = 1; benchmark = 'minillm-cuda-runtime'; protocol_id = 'qwen3-cuda-model-v0'; run_id = $runId
    created_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    source = [ordered]@{ git_sha = $gitSha; git_dirty = $dirty.Count -gt 0; scope = $scope; state_file = 'source-state.json'
        worktree_state_sha256 = Get-LowerSha256 $stateFile
        snapshot = [ordered]@{ path = 'source-snapshot.zip'; sha256 = Get-LowerSha256 $snapshot } }
    binary = [ordered]@{ path = $executable; sha256 = Get-LowerSha256 $executable; availability = 'rerun_dependency_not_bundled' }
    model = [ordered]@{ path = $Model; sha256 = $modelHash; size_bytes = (Get-Item -LiteralPath $Model).Length
        provenance = $provenance; availability = 'rerun_dependency_not_bundled' }
    input = [ordered]@{ path = 'input.json'; sha256 = $inputHash }
    dependencies = [ordered]@{ llama_commit = $revision }
    build = [ordered]@{ type = $buildType; generator = Read-CMakeValue $cache 'CMAKE_GENERATOR'
        own_cuda = Read-CMakeValue $cache 'MINILLM_ENABLE_CUDA'; upstream_cuda = Read-CMakeValue $cache 'LLMSERVE_CUDA'
        host_compiler = Read-CMakeValue $cache 'CMAKE_CXX_COMPILER'
        host_compiler_id = Read-CMakeSetValue $compiler.FullName 'CMAKE_CXX_COMPILER_ID'
        host_compiler_version = Read-CMakeSetValue $compiler.FullName 'CMAKE_CXX_COMPILER_VERSION'
        cuda_compiler = Read-CMakeValue $cache 'CMAKE_CUDA_COMPILER'
        cuda_compiler_version = Read-CMakeSetValue $cudaCompiler.FullName 'CMAKE_CUDA_COMPILER_VERSION'
        cuda_architectures = Read-CMakeValue $cache 'CMAKE_CUDA_ARCHITECTURES'
        cxx_flags = Read-CMakeValue $cache 'CMAKE_CXX_FLAGS'
        cuda_flags = Read-CMakeValue $cache 'CMAKE_CUDA_FLAGS'
        cxx_configuration_flags = Read-CMakeValue $cache "CMAKE_CXX_FLAGS_$($buildType.ToUpperInvariant())"
        cuda_configuration_flags = Read-CMakeValue $cache "CMAKE_CUDA_FLAGS_$($buildType.ToUpperInvariant())"
        linker_flags = Read-CMakeValue $cache 'CMAKE_EXE_LINKER_FLAGS' }
    environment = [ordered]@{ os = [Environment]::OSVersion.VersionString; machine = [Environment]::MachineName
        logical_processors = [Environment]::ProcessorCount; powershell = $PSVersionTable.PSVersion.ToString()
        frequency_control = 'uncontrolled'; affinity = 'inherited'; telemetry = 'process_boundaries_not_continuous'
        profiler = 'none' }
    statistics = $plan.statistics; reports = $plan.reports; artifacts = @($artifacts)
    numerical_evidence = [ordered]@{ repository_path = [IO.Path]::GetRelativePath($root, $NumericalDirectory).Replace('\', '/')
        scope = 'source_and_validation_binary_equivalent_full_corpus'; full_numeric_archive_included = $false
        validation_binary_sha256 = $validationHash }
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
Write-BenchmarkJson $manifestPath $manifest
$manifestHash = Get-LowerSha256 $manifestPath

function Assert-CudaBenchmarkInputs {
    if ((Get-LowerSha256 $executable) -cne $manifest.binary.sha256 -or
        (Get-LowerSha256 $validationExecutable) -cne $validationHash -or
        (Get-LowerSha256 $Model) -cne $modelHash -or (Get-LowerSha256 $inputFile) -cne $inputHash -or
        (Get-LowerSha256 (Join-Path $OutputDirectory 'input.json')) -cne $inputHash -or
        (Get-LowerSha256 $manifestPath) -cne $manifestHash) { throw '采集中的二进制、模型、输入或 manifest 发生变化。' }
    $current = ConvertTo-Json -InputObject @(Get-BenchmarkSourceState $root $scope) -Depth 8 -Compress
    if ($current -cne $sourceJson) { throw '采集中的源码发生变化。' }
    $status = @(& git -C $dependency status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $status.Count) { throw '依赖 checkout 不是干净状态。' }
    $currentRevision = & git -C $dependency rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $currentRevision -cne $revision) { throw '依赖版本发生变化。' }
}

$completed = [Collections.Generic.List[object]]::new()
$failed = $false
try {
    Assert-CudaBenchmarkInputs
    # 只核对数值源码继承，不构造 reference Runtime。
    $old = Get-Content -Raw (Join-Path $OutputDirectory 'numerical-source-state.json') | ConvertFrom-Json
    $productFiles = @($sourceFiles | Where-Object { $_.path -cmatch '^(include|src)/minillm/' })
    $oldProductFiles = @($old.files | Where-Object { $_.path -cmatch '^(include|src)/minillm/' })
    if ($productFiles.Count -ne $oldProductFiles.Count) { throw 'Runtime 源码集合发生变化，需要重新运行数值门禁。' }
    foreach ($item in $productFiles) {
        $previous = @($old.files | Where-Object path -CEQ $item.path)
        if ($previous.Count -ne 1 -or $previous[0].sha256 -cne $item.sha256) { throw "需要重新运行数值门禁：$($item.path)" }
    }
    Test-BenchmarkSourceArchive $OutputDirectory ([pscustomobject]$manifest.source)
    if ($PreflightOnly) {
        Write-BenchmarkJson (Join-Path $OutputDirectory 'preflight-environment.json') (Get-CudaBenchmarkEnvironment)
        Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
            status = 'preflight_only'; reports = @(); planned_reports = 70
            scope = '输入、源码、依赖、构建和环境检查；没有性能测量或统计结论' })
        Write-Host "预检通过：70 个独立进程；未执行性能采样。归档：$OutputDirectory"
        return
    }
    foreach ($slot in $plan.reports) {
        Assert-CudaBenchmarkInputs
        $path = Join-Path $OutputDirectory $slot.file
        $arguments = @('--model', $Model, '--input', (Join-Path $OutputDirectory 'input.json'), '--backend', $slot.backend,
            '--output', $path, '--manifest', $manifestPath, '--order', "$($slot.order)")
        $before = Get-CudaBenchmarkEnvironment
        $started = (Get-Date).ToUniversalTime().ToString('o')
        & $executable @arguments 1> "$path.stdout.txt" 2> "$path.stderr.txt"
        $exitCode = $LASTEXITCODE
        $after = Get-CudaBenchmarkEnvironment
        Write-BenchmarkJson "$path.process.json" ([ordered]@{ order = $slot.order; executable = $executable
            arguments = $arguments; exit_code = $exitCode; started_at_utc = $started
            finished_at_utc = (Get-Date).ToUniversalTime().ToString('o'); before = $before; after = $after })
        $record = [ordered]@{ file = $slot.file; exit_code = $exitCode; sha256 = $null; artifacts = @() }
        if (Test-Path -LiteralPath $path) { $record.sha256 = Get-LowerSha256 $path }
        foreach ($suffix in @('.stdout.txt', '.stderr.txt', '.process.json')) {
            $record.artifacts += [ordered]@{ path = $slot.file + $suffix; sha256 = Get-LowerSha256 ($path + $suffix) }
        }
        $completed.Add([pscustomobject]$record)
        if ($exitCode -ne 0) { $failed = $true }
        Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
            status = 'collecting'; reports = @($completed.ToArray()); planned_reports = 70 })
        Assert-CudaBenchmarkInputs
        Write-Host "CUDA 模型对照 $($completed.Count)/70：$($slot.file)，退出码 $exitCode"
    }
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = $(if ($failed) { 'failed' } else { 'passed' }); reports = @($completed.ToArray()); planned_reports = 70
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
    if ($failed) { throw '存在失败进程，全部已采集结果保留；不能发布通过摘要。' }
} catch {
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'failed'; reports = @($completed.ToArray()); planned_reports = 70; error = $_.Exception.Message
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
    throw
}
& (Join-Path $PSScriptRoot 'Analyze-CudaRuntime.ps1') -Directory $OutputDirectory
