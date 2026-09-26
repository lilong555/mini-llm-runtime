param(
    [string]$OutputDirectory = '',
    [string]$BinaryDirectory = '',
    [string]$Model = '',
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
. (Join-Path $PSScriptRoot 'Cuda-Benchmark-Common.ps1')
$root = Split-Path -Parent $PSScriptRoot
$python = if (Get-Command python3 -ErrorAction SilentlyContinue) { 'python3' } else { 'python' }
$inputFile = Join-Path $root 'benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json'
$analyzer = Join-Path $PSScriptRoot 'analyze_cuda_micro.py'
if (-not $BinaryDirectory) {
    $platform = if ($env:OS -eq 'Windows_NT') { '' } else { 'wsl-' }
    $BinaryDirectory = Join-Path $root "build/${platform}own-cuda/bin"
}
if (-not $Model) { $Model = Join-Path $root 'models/Qwen3-0.6B-Q8_0.gguf' }
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
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
    throw 'micro 采集要求本工作区的 Release/RelWithDebInfo、自有 CUDA=ON、上游 CUDA=OFF。'
}
$inputHash = Get-LowerSha256 $inputFile
$modelHash = Get-LowerSha256 $Model
if ($inputHash -cne '6aa2bc8af5de001ab3a8baedd305d0c77822a48b5baddc8f5708e79f496e706c' -or
    $modelHash -cne '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031') {
    throw '冻结的 micro 输入或模型摘要不符。'
}
$provenance = Get-Content -Raw (Join-Path $root 'models/manifest.json') | ConvertFrom-Json
if ($provenance.sha256 -cne $modelHash -or [long]$provenance.size_bytes -ne (Get-Item -LiteralPath $Model).Length) {
    throw '模型来源清单不符。'
}
& cmake --build $build --config $buildType --target mini-cuda-kernel-bench --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'CUDA micro 构建失败。' }
$executable = Get-ProductExecutable $BinaryDirectory 'mini-cuda-kernel-bench'
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
if ($LASTEXITCODE -ne 0) { throw '不能建立固定 micro 进程顺序。' }
$plan = ($planText -join "`n") | ConvertFrom-Json
$runId = '{0}-{1}-{2}' -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
    $gitSha.Substring(0, 12), ([guid]::NewGuid().ToString('N').Substring(0, 8))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "benchmarks/results/cuda-micro-$runId" }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $OutputDirectory) -and @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count) {
    throw 'micro 输出目录必须为空；已有采样不会被覆盖。'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
Copy-Item -LiteralPath $inputFile -Destination (Join-Path $OutputDirectory 'input.json')
Copy-Item -LiteralPath $analyzer -Destination (Join-Path $OutputDirectory 'verify.py')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'analyze_cuda_benchmark.py') -Destination $OutputDirectory
$stateFile = Join-Path $OutputDirectory 'source-state.json'
Write-BenchmarkJson $stateFile ([ordered]@{ scope = $scope; files = $sourceFiles })
$snapshot = Join-Path $OutputDirectory 'source-snapshot.zip'
Write-BenchmarkSourceSnapshot $root $sourceFiles $snapshot
$manifest = [ordered]@{
    schema_version = 1; benchmark = 'minillm-cuda-micro'; protocol_id = 'qwen3-cuda-micro-v0'; run_id = $runId
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
    statistics = $plan.statistics; reports = $plan.reports
    artifacts = @('verify.py', 'analyze_cuda_benchmark.py') | ForEach-Object {
        [ordered]@{ path = $_; sha256 = Get-LowerSha256 (Join-Path $OutputDirectory $_) } }
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
Write-BenchmarkJson $manifestPath $manifest
$manifestHash = Get-LowerSha256 $manifestPath

function Assert-CudaMicroInputs {
    if ((Get-LowerSha256 $executable) -cne $manifest.binary.sha256 -or (Get-LowerSha256 $Model) -cne $modelHash -or
        (Get-LowerSha256 $inputFile) -cne $inputHash -or
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
    Assert-CudaMicroInputs
    Test-BenchmarkSourceArchive $OutputDirectory ([pscustomobject]$manifest.source)
    if ($PreflightOnly) {
        Write-BenchmarkJson (Join-Path $OutputDirectory 'preflight-environment.json') (Get-CudaBenchmarkEnvironment)
        Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
            status = 'preflight_only'; reports = @(); planned_reports = 5; scope = '预检不包含性能采样或统计结论' })
        Write-Host "micro 预检通过：5 个独立进程，每个进程 375 个用例。归档：$OutputDirectory"
        return
    }
    foreach ($slot in $plan.reports) {
        Assert-CudaMicroInputs
        $path = Join-Path $OutputDirectory $slot.file
        $arguments = @('--model', $Model, '--input', (Join-Path $OutputDirectory 'input.json'),
            '--output', $path, '--trial', "$($slot.trial)", '--manifest', $manifestPath)
        $before = Get-CudaBenchmarkEnvironment
        $started = (Get-Date).ToUniversalTime().ToString('o')
        & $executable @arguments 1> "$path.stdout.txt" 2> "$path.stderr.txt"
        $exitCode = $LASTEXITCODE
        $after = Get-CudaBenchmarkEnvironment
        Write-BenchmarkJson "$path.process.json" ([ordered]@{ trial = $slot.trial; executable = $executable
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
            status = 'collecting'; reports = @($completed.ToArray()); planned_reports = 5 })
        Assert-CudaMicroInputs
        Write-Host "CUDA micro $($completed.Count)/5：$($slot.file)，退出码 $exitCode"
    }
    if ($failed) { throw '存在失败进程，全部样本保留；不能发布通过摘要。' }
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'passed'; reports = @($completed.ToArray()); planned_reports = 5
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
} catch {
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'failed'; reports = @($completed.ToArray()); planned_reports = 5; error = $_.Exception.Message
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
    throw
}
& (Join-Path $PSScriptRoot 'Analyze-CudaMicro.ps1') -Directory $OutputDirectory
