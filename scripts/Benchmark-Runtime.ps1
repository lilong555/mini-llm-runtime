param(
    [string]$InputFile = '',
    [string]$OutputDirectory = '',
    [string]$BinaryDirectory = '',
    [string]$Model = '',
    [string]$ModelManifest = '',
    [int[]]$Threads = @(1, 2, 4, 8, 16),
    [ValidateRange(1, 20)][int]$Trials = 3,
    [ValidateRange(1, 100)][int]$Repeats = 1,
    [ValidateRange(1, 100)][int]$Warmup = 1,
    [ValidateSet('auto', 'scalar')][string]$Kernel = 'auto'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$root = Split-Path -Parent $PSScriptRoot
if ($Threads.Count -eq 0 -or @($Threads | Sort-Object -Unique).Count -ne $Threads.Count -or
    @($Threads | Where-Object { $_ -lt 1 -or $_ -gt 256 }).Count -ne 0) {
    throw 'Threads must contain unique integers in [1, 256].'
}
if (-not $InputFile) { $InputFile = Join-Path $root 'benchmarks/runtime-inputs/qwen3-cpu.json' }
if (-not $BinaryDirectory) { $BinaryDirectory = Get-ProductDirectory $root 'mini' }
if (-not $Model) { $Model = Join-Path $root 'models/Qwen3-0.6B-Q8_0.gguf' }
if (-not $ModelManifest) { $ModelManifest = Join-Path $root 'models/manifest.json' }
$InputFile = (Resolve-Path -LiteralPath $InputFile).Path
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$ModelManifest = (Resolve-Path -LiteralPath $ModelManifest).Path
$inputSpec = Get-Content -Raw -LiteralPath $InputFile | ConvertFrom-Json
$provenance = Get-Content -Raw -LiteralPath $ModelManifest | ConvertFrom-Json
$modelHash = Get-LowerSha256 $Model
if ($provenance.file -cne (Split-Path -Leaf $Model) -or
    [long]$provenance.size_bytes -ne (Get-Item -LiteralPath $Model).Length -or
    $provenance.sha256.ToLowerInvariant() -cne $modelHash) {
    throw 'The model does not match its pinned provenance manifest.'
}
$build = Split-Path -Parent $BinaryDirectory
while (-not (Test-Path -LiteralPath (Join-Path $build 'CMakeCache.txt'))) {
    $parent = Split-Path -Parent $build
    if (-not $parent -or $parent -eq $build) { throw 'Cannot locate the product CMake cache.' }
    $build = $parent
}
$cache = Join-Path $build 'CMakeCache.txt'
if ((Read-CMakeValue $cache 'CMAKE_HOME_DIRECTORY') -ne $root) {
    throw 'The runtime benchmark must be built from this source directory.'
}
$buildType = Read-CMakeValue $cache 'CMAKE_BUILD_TYPE'
if (-not $buildType) { $buildType = Split-Path -Leaf $BinaryDirectory }
if ($buildType -notin @('RelWithDebInfo', 'Release')) {
    throw 'Runtime measurements require Release or RelWithDebInfo.'
}
& cmake --build $build --config $buildType --target mini-runtime-bench --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'The runtime benchmark build failed.' }
$executable = Get-ProductExecutable $BinaryDirectory 'mini-runtime-bench'
$compiler = Get-ChildItem -LiteralPath (Join-Path $build 'CMakeFiles') -Recurse -File `
    -Filter 'CMakeCXXCompiler.cmake' | Select-Object -First 1
if ($null -eq $compiler) { throw 'Cannot locate compiler metadata.' }
$gitSha = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the source revision.' }
$dirty = @(& git -C $root status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the source worktree.' }
$scope = @('CMakeLists.txt', '.gitattributes', 'cmake', 'apps', 'include', 'src', 'scripts',
    'tests', '.github', 'models/manifest.json', 'models/reference-manifest.json', 'benchmarks/runtime-inputs')
$sourceFiles = @(Get-BenchmarkSourceState $root $scope)
$sourceJson = ConvertTo-Json -InputObject $sourceFiles -Depth 8 -Compress
$dependency = Join-Path $root 'third_party/llama.cpp'
$revision = (& git -C $dependency rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the dependency revision.' }
$runId = '{0}-{1}-{2}' -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
    $gitSha.Substring(0, 12), ([guid]::NewGuid().ToString('N').Substring(0, 8))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "benchmarks/results/runtime-$runId" }
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $OutputDirectory) -and @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count) {
    throw 'The runtime benchmark output directory must be empty.'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$archivedInput = Join-Path $OutputDirectory 'input.json'
Copy-Item -LiteralPath $InputFile -Destination $archivedInput
$state = Join-Path $OutputDirectory 'source-state.json'
Write-BenchmarkJson $state ([ordered]@{ scope = $scope; files = $sourceFiles })
$snapshot = Join-Path $OutputDirectory 'source-snapshot.zip'
Write-BenchmarkSourceSnapshot $root $sourceFiles $snapshot
$reports = @()
for ($trial = 0; $trial -lt $Trials; ++$trial) {
    $threadOrder = @($Threads)
    if ($trial % 2 -ne 0) { [array]::Reverse($threadOrder) }
    foreach ($count in $threadOrder) {
        $modes = if ($trial % 2 -eq 0) { @('none', 'stages') } else { @('stages', 'none') }
        foreach ($mode in $modes) {
            $reports += [ordered]@{ file = "threads-$count-$mode-$trial.json"; threads = $count
                profiler = $mode; trial = $trial; order = $reports.Count }
        }
    }
}
$processor = $env:PROCESSOR_IDENTIFIER
if (-not $processor -and (Test-Path -LiteralPath '/proc/cpuinfo')) {
    $line = Get-Content -LiteralPath '/proc/cpuinfo' | Where-Object { $_ -match '^model name\s*:' } |
        Select-Object -First 1
    if ($line) { $processor = ($line -split ':', 2)[1].Trim() }
}
$manifest = [ordered]@{
    schema_version = 1; benchmark = 'minillm-runtime'; run_id = $runId
    created_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    source = [ordered]@{
        git_sha = $gitSha; git_dirty = $dirty.Count -gt 0; scope = $scope
        state_file = 'source-state.json'; worktree_state_sha256 = Get-LowerSha256 $state
        snapshot = [ordered]@{ path = 'source-snapshot.zip'; sha256 = Get-LowerSha256 $snapshot }
    }
    binary = [ordered]@{ path = $executable; sha256 = Get-LowerSha256 $executable }
    model = [ordered]@{ path = $Model; file = Split-Path -Leaf $Model; sha256 = $modelHash
        size_bytes = (Get-Item -LiteralPath $Model).Length; provenance = $provenance }
    input = [ordered]@{ path = 'input.json'; original_path = $InputFile; sha256 = Get-LowerSha256 $InputFile }
    dependencies = [ordered]@{ llama_commit = $revision }
    build = [ordered]@{
        type = $buildType; generator = Read-CMakeValue $cache 'CMAKE_GENERATOR'
        compiler = Read-CMakeValue $cache 'CMAKE_CXX_COMPILER'
        compiler_id = Read-CMakeSetValue $compiler.FullName 'CMAKE_CXX_COMPILER_ID'
        compiler_version = Read-CMakeSetValue $compiler.FullName 'CMAKE_CXX_COMPILER_VERSION'
        cxx_flags = Read-CMakeValue $cache 'CMAKE_CXX_FLAGS'
        configuration_flags = Read-CMakeValue $cache "CMAKE_CXX_FLAGS_$($buildType.ToUpperInvariant())"
        linker_flags = Read-CMakeValue $cache 'CMAKE_EXE_LINKER_FLAGS'
        configuration_linker_flags = Read-CMakeValue $cache "CMAKE_EXE_LINKER_FLAGS_$($buildType.ToUpperInvariant())"
        cuda_enabled = Read-CMakeValue $cache 'LLMSERVE_CUDA'
    }
    runtime = $inputSpec.runtime
    protocol = [ordered]@{ threads = @($Threads); kernel = $Kernel; trials = $Trials; repeats = $Repeats
        warmup = $Warmup; profiler_modes = @('none', 'stages'); order = 'alternating_processes_and_thread_order'
        setup = 'pinned_prefix_share_reset'; reference_model_resident = $false; seed = $null
        allowed_changes = @('threads', 'profiler'); sampling = 'greedy_argmax_separate'
        activation_dtype = 'F32'; kv_dtype = 'F16'
        logits_digest = 'sha256_seq_i32le_length_u32le_logits_f32le' }
    environment = [ordered]@{ os = [System.Environment]::OSVersion.VersionString
        machine = [System.Environment]::MachineName; processor = $processor
        logical_processors = [System.Environment]::ProcessorCount; powershell = $PSVersionTable.PSVersion.ToString()
        frequency_control = 'uncontrolled'; affinity = 'inherited'; concurrent_load = 'not_measured' }
    reports = $reports
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
Write-BenchmarkJson $manifestPath $manifest
$manifestHash = Get-LowerSha256 $manifestPath

function Assert-RuntimeInputs {
    foreach ($item in @($manifest.binary, $manifest.model)) {
        if ((Get-LowerSha256 $item.path) -cne $item.sha256) { throw "Runtime input changed: $($item.path)" }
    }
    foreach ($path in @($InputFile, $archivedInput)) {
        if ((Get-LowerSha256 $path) -cne $manifest.input.sha256) { throw 'Runtime input recipe changed.' }
    }
    if ((Get-LowerSha256 $manifestPath) -cne $manifestHash) { throw 'Runtime manifest changed.' }
    $current = ConvertTo-Json -InputObject @(Get-BenchmarkSourceState $root $scope) -Depth 8 -Compress
    if ($current -cne $sourceJson) { throw 'Runtime benchmark source changed during collection.' }
    $dependencyStatus = @(& git -C $dependency status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $dependencyStatus.Count) { throw 'The dependency checkout is dirty.' }
    $currentRevision = & git -C $dependency rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $currentRevision -cne $revision) { throw 'The dependency revision changed.' }
}

$completed = 0
try {
    foreach ($item in $reports) {
        Assert-RuntimeInputs
        $output = Join-Path $OutputDirectory $item.file
        $arguments = @('--model', $Model, '--input', $archivedInput, '--output', $output,
            '--manifest', $manifestPath, '--threads', "$($item.threads)", '--kernel', $Kernel,
            '--profiler', $item.profiler, '--trial', "$($item.trial)", '--warmup', "$Warmup", '--repeats', "$Repeats")
        & $executable @arguments 1> "$output.stdout.log" 2> "$output.stderr.log"
        if ($LASTEXITCODE -ne 0) { throw "Runtime measurement failed: $($item.file)" }
        Assert-RuntimeInputs
        ++$completed
        Write-Host "Runtime $completed/$($reports.Count): $($item.file)"
    }
    & (Join-Path $PSScriptRoot 'Analyze-Runtime.ps1') -Directory $OutputDirectory
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'passed'; reports = $completed; finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
} catch {
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'failed'; reports = $completed; error = $_.Exception.Message
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
    throw
}
