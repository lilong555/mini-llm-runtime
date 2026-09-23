param(
    [Parameter(Mandatory = $true)][string]$Trace,
    [ValidateSet('mini', 'llama')][string]$Backend = 'mini',
    [ValidateRange(1, 20)][int]$Trials = 3,
    [ValidateRange(0, 1)][int]$PolicyOrderOffset = 0,
    [ValidateRange(1024, 65515)][int]$Port = 8000,
    [string]$BinaryDirectory = '',
    [string]$OutputDirectory = '',
    [string]$Model = '',
    [string]$ModelManifest = '',
    [ValidateRange(1, 128)][int]$MaxActive = 8,
    [ValidateRange(1, 65536)][int]$QueueCapacity = 64,
    [ValidateRange(1, 65536)][int]$BatchTokens = 256,
    [ValidateRange(1, 65536)][int]$PrefillChunk = 32,
    [ValidateRange(0, 128)][int]$PrefixEntries = 4,
    [ValidateRange(0, 1048576)][int]$PrefixTokens = 2048,
    [ValidateSet(1, 2, 4, 8, 16, 32, 64, 128, 256)][int]$PageSize = 16,
    [ValidateRange(16, 1048576)][int]$Context = 8192,
    [ValidateRange(2, 1048576)][int]$MaxModelLen = 2048,
    [ValidateRange(1, 65536)][int]$EventBuffer = 128,
    [ValidateRange(1, 256)][int]$Threads = 8,
    [ValidateRange(0, 10000)][int]$GpuLayers = $(if ($Backend -eq 'mini') { 0 } else { 99 }),
    [ValidateSet('auto', 'scalar')][string]$Kernel = 'auto',
    [ValidateSet('off', 'batches', 'stages')][string]$Telemetry = 'off',
    [ValidateRange(1, 16384)][int]$TelemetryCapacity = 1024,
    [ValidateRange(0.000001, 10000)][double]$ArrivalScale = 1.0,
    [ValidateSet('success', 'queue_full', 'timeout', 'cancelled', 'backpressure')]
    [string[]]$AllowedRequestOutcomes = @('success'),
    [Nullable[long]]$TraceSeed = $null,
    [switch]$NoWarmup
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$root = Split-Path -Parent $PSScriptRoot

function Get-SourceState {
    return @(Get-BenchmarkSourceState $root $sourceScope)
}

function Assert-RunInputs {
    foreach ($item in @($manifest.binaries.server, $manifest.binaries.benchmark_client, $manifest.model)) {
        if ((Get-LowerSha256 $item.path) -cne $item.sha256) {
            throw "Benchmark input changed during the run: $($item.path)"
        }
    }
    if ((Get-LowerSha256 $archivedTrace) -cne $manifest.trace.sha256 -or
        (Get-LowerSha256 $Trace) -cne $manifest.trace.sha256) {
        throw 'The benchmark trace changed during the run.'
    }
    $currentState = ConvertTo-Json -InputObject @(Get-SourceState) -Depth 8 -Compress
    if ($currentState -cne $sourceStateJson) { throw 'Source files changed during the benchmark run.' }
    $dependencyState = @(& git -C $dependencyPath status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $dependencyState.Count -ne 0) { throw 'The dependency checkout is dirty.' }
    $dependencyRevision = & git -C $dependencyPath rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $dependencyRevision -cne $manifest.dependencies.llama_commit) {
        throw 'The dependency revision changed during the benchmark run.'
    }
}

if ($AllowedRequestOutcomes.Count -eq 0 -or $AllowedRequestOutcomes -notcontains 'success') {
    throw 'AllowedRequestOutcomes must include success so deterministic outputs have a comparison reference.'
}
if ($Backend -eq 'llama' -and $Kernel -ne 'auto') { throw 'The scalar kernel mode applies only to MiniLLM.' }
if ($Backend -eq 'mini' -and $GpuLayers -ne 0) { throw 'MiniLLM requires GpuLayers=0.' }
if ($Context % $PageSize -ne 0 -or $MaxModelLen -gt $Context -or $BatchTokens -lt $MaxActive -or
    $PrefillChunk -gt $BatchTokens -or $PrefixTokens -gt $Context -or
    ($PrefixEntries -gt 0 -and $PrefixTokens -lt $PageSize)) {
    throw 'The benchmark engine configuration violates server capacity constraints.'
}

$Trace = (Resolve-Path -LiteralPath $Trace).Path
if (-not $BinaryDirectory) {
    $BinaryDirectory = Get-ProductDirectory $root $Backend
}
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$serverExecutable = Get-ProductExecutable $BinaryDirectory 'llmserve'
$benchExecutable = Get-ProductExecutable $BinaryDirectory 'llmserve-bench'
if (-not $Model) { $Model = Join-Path $root 'models\Qwen3-0.6B-Q8_0.gguf' }
if (-not $ModelManifest) { $ModelManifest = Join-Path $root 'models\manifest.json' }
$Model = (Resolve-Path -LiteralPath $Model).Path
$ModelManifest = (Resolve-Path -LiteralPath $ModelManifest).Path

$modelProvenance = Get-Content -Raw -LiteralPath $ModelManifest | ConvertFrom-Json
$modelHash = Get-LowerSha256 $Model
$modelBytes = (Get-Item -LiteralPath $Model).Length
if ($modelProvenance.file -ne (Split-Path -Leaf $Model) -or
    [int64]$modelProvenance.size_bytes -ne $modelBytes -or
    $modelProvenance.sha256.ToLowerInvariant() -ne $modelHash) {
    throw 'The selected model does not match its pinned manifest.'
}

$traceRows = [System.Collections.Generic.List[object]]::new()
$traceIds = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($line in Get-Content -LiteralPath $Trace) {
    if ([string]::IsNullOrWhiteSpace($line)) { throw 'The trace contains an empty row.' }
    $row = $line | ConvertFrom-Json
    $id = [string]$row.request_id
    if ($id -cnotmatch '^[A-Za-z0-9_.:-]{1,128}$' -or -not $traceIds.Add($id)) {
        throw 'The trace contains a missing or duplicate request ID.'
    }
    $traceRows.Add($row)
}
if ($traceRows.Count -eq 0) { throw 'The trace is empty.' }

$gitSha = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the source Git revision.' }
$gitStatus = @(& git -C $root status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect the source worktree.' }
$sourceScope = @('CMakeLists.txt', '.gitattributes', 'cmake', 'apps', 'include', 'src',
    'scripts', 'tests', '.github', 'models/manifest.json', 'models/reference-manifest.json')
$sourceState = @(Get-SourceState)
$sourceStateJson = ConvertTo-Json -InputObject $sourceState -Depth 8 -Compress
$dependencyPath = Join-Path $root 'third_party/llama.cpp'
$dependencyRevision = & git -C $dependencyPath rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the dependency revision.' }

$runId = '{0}-{1}-{2}' -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
    $gitSha.Substring(0, [Math]::Min(12, $gitSha.Length)), ([guid]::NewGuid().ToString('N').Substring(0, 8))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "benchmarks\results\$runId" }
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $OutputDirectory) -and
    @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count -gt 0) {
    throw 'The benchmark output directory must be empty.'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if (Test-LoopbackPort $Port) {
    throw 'Benchmark port is occupied. Stop the project server or select another port.'
}

$archivedTrace = Join-Path $OutputDirectory 'trace.jsonl'
Copy-Item -LiteralPath $Trace -Destination $archivedTrace
$sourceStatePath = Join-Path $OutputDirectory 'source-state.json'
Write-BenchmarkJson $sourceStatePath ([ordered]@{ scope = $sourceScope; files = $sourceState })
$sourceArchivePath = Join-Path $OutputDirectory 'source-snapshot.zip'
Write-BenchmarkSourceSnapshot $root $sourceState $sourceArchivePath

$buildDirectory = Split-Path -Parent $BinaryDirectory
while (-not (Test-Path -LiteralPath (Join-Path $buildDirectory 'CMakeCache.txt'))) {
    $parent = Split-Path -Parent $buildDirectory
    if (-not $parent -or $parent -eq $buildDirectory) { throw 'Cannot locate the product CMake cache.' }
    $buildDirectory = $parent
}
$cache = Join-Path $buildDirectory 'CMakeCache.txt'
if ((Read-CMakeValue $cache 'CMAKE_HOME_DIRECTORY') -ne $root) {
    throw 'The benchmark binaries must be built from this source directory.'
}
$compilerMetadata = Get-ChildItem -LiteralPath (Join-Path $buildDirectory 'CMakeFiles') `
    -Filter 'CMakeCXXCompiler.cmake' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $compilerMetadata) { throw 'Cannot locate CMake compiler metadata.' }
$buildType = Read-CMakeValue $cache 'CMAKE_BUILD_TYPE'
if (-not $buildType) {
    $buildType = Split-Path -Leaf $BinaryDirectory
    if ($buildType -notin @('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel')) {
        throw 'For a multi-configuration build, BinaryDirectory must select the configuration directory.'
    }
}
& cmake --build $buildDirectory --config $buildType --target llmserve llmserve-bench --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'The benchmark product build failed.' }
$processor = $env:PROCESSOR_IDENTIFIER
if (-not $processor -and (Test-Path -LiteralPath '/proc/cpuinfo')) {
    $cpuLine = Get-Content -LiteralPath '/proc/cpuinfo' | Where-Object { $_ -match '^model name\s*:' } |
        Select-Object -First 1
    if ($cpuLine) { $processor = ($cpuLine -split ':', 2)[1].Trim() }
}
$gpu = @()
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuRows = @(& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits)
    if ($LASTEXITCODE -eq 0) { $gpu = @($gpuRows | ConvertFrom-Csv -Header name, driver_version, memory_mib) }
}
$reports = @()
for ($trial = 0; $trial -lt $Trials; ++$trial) {
    $policies = if (($trial + $PolicyOrderOffset) % 2 -eq 0) { @('mixed', 'prefill_first') } else { @('prefill_first', 'mixed') }
    foreach ($policy in $policies) {
        $reports += [ordered]@{ file = "$policy-$trial.json"; variant = $policy; trial = $trial; order = $reports.Count
            telemetry_file = if ($Telemetry -eq 'off') { $null } else { "$policy-$trial-telemetry.jsonl" } }
    }
}
$metricsBackend = if ($Backend -eq 'mini') { 'minillm' } else { 'llama.cpp' }
$manifest = [ordered]@{
    schema_version = 1
    benchmark = 'llmserve-policy-comparison'
    run_id = $runId
    created_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    source = [ordered]@{
        git_sha = $gitSha
        git_dirty = $gitStatus.Count -gt 0
        worktree_state_sha256 = Get-LowerSha256 $sourceStatePath
        state_file = 'source-state.json'
        scope = $sourceScope
        snapshot = [ordered]@{ path = 'source-snapshot.zip'; sha256 = Get-LowerSha256 $sourceArchivePath }
    }
    binaries = [ordered]@{
        server = [ordered]@{ path = $serverExecutable; sha256 = Get-LowerSha256 $serverExecutable }
        benchmark_client = [ordered]@{ path = $benchExecutable; sha256 = Get-LowerSha256 $benchExecutable }
    }
    build = [ordered]@{
        type = $buildType
        generator = Read-CMakeValue $cache 'CMAKE_GENERATOR'
        compiler = Read-CMakeValue $cache 'CMAKE_CXX_COMPILER'
        compiler_id = Read-CMakeSetValue $compilerMetadata.FullName 'CMAKE_CXX_COMPILER_ID'
        compiler_version = Read-CMakeSetValue $compilerMetadata.FullName 'CMAKE_CXX_COMPILER_VERSION'
        cxx_flags = Read-CMakeValue $cache 'CMAKE_CXX_FLAGS'
        configuration_flags = Read-CMakeValue $cache "CMAKE_CXX_FLAGS_$($buildType.ToUpperInvariant())"
        linker_flags = Read-CMakeValue $cache 'CMAKE_EXE_LINKER_FLAGS'
        configuration_linker_flags = Read-CMakeValue $cache "CMAKE_EXE_LINKER_FLAGS_$($buildType.ToUpperInvariant())"
        cuda_enabled = Read-CMakeValue $cache 'LLMSERVE_CUDA'
        cuda_architectures = Read-CMakeValue $cache 'CMAKE_CUDA_ARCHITECTURES'
    }
    dependencies = [ordered]@{
        llama_commit = $dependencyRevision
    }
    model = [ordered]@{
        file = Split-Path -Leaf $Model
        path = $Model
        size_bytes = $modelBytes
        sha256 = $modelHash
        provenance_manifest = $ModelManifest
        provenance = $modelProvenance
        weight_dtype = if ($modelProvenance.file -match '(Q[0-9]+_[0-9]+|F16|F32|BF16)') { $Matches[1] } else { 'unknown' }
    }
    trace = [ordered]@{
        path = 'trace.jsonl'
        original_path = $Trace
        size_bytes = (Get-Item -LiteralPath $Trace).Length
        sha256 = Get-LowerSha256 $Trace
        fnv1a64 = Get-TraceFnv1a64 $Trace
        request_count = $traceRows.Count
        seed = if ($null -eq $TraceSeed) { $null } else { [long]$TraceSeed }
    }
    engine = [ordered]@{
        backend = $Backend
        metrics_backend = $metricsBackend
        requested_kernel_mode = $Kernel
        metrics_kernel_mode = if ($Backend -eq 'mini') { $Kernel } else { 'upstream' }
        threads = $Threads
        gpu_layers = $GpuLayers
        context_tokens = $Context
        max_model_len = $MaxModelLen
        batch_tokens = $BatchTokens
        prefill_chunk = $PrefillChunk
        max_active = $MaxActive
        queue_capacity = $QueueCapacity
        block_size = $PageSize
        prefix_cache_entries = $PrefixEntries
        prefix_cache_tokens = $PrefixTokens
        event_buffer_size = $EventBuffer
        aging_ms = 250
        admission_reserve_ms = 2000
        telemetry_mode = $Telemetry
        telemetry_capacity = $TelemetryCapacity
    }
    comparison = [ordered]@{
        dimension = 'engine.policy'
        allowed_changes = @('engine.policy')
        variants = @('mixed', 'prefill_first')
        reference = [ordered]@{ variant = 'mixed'; trial = 0 }
    }
    protocol = [ordered]@{
        trials_per_variant = $Trials
        order = 'alternating'
        order_offset = $PolicyOrderOffset
        warmup = -not $NoWarmup.IsPresent
        warmup_request = if ($NoWarmup) { $null } else {
            [ordered]@{ prompt = 'Hello'; max_tokens = 8; ignore_eos = $true; cache_namespace = 'benchmark-warmup' }
        }
        allowed_request_outcomes = @($AllowedRequestOutcomes)
        activation_dtype = if ($Backend -eq 'mini') { 'F32' } else { 'upstream_native' }
        kv_dtype = 'F16'
        sampling = 'greedy'
        profiler_mode = if ($Telemetry -eq 'off') { 'none' } else { $Telemetry }
        arrival_scale = $ArrivalScale
    }
    environment = [ordered]@{
        os = [System.Environment]::OSVersion.VersionString
        machine = [System.Environment]::MachineName
        processor = $processor
        logical_processors = [System.Environment]::ProcessorCount
        gpu = $gpu
        cpu_frequency_temperature = $null
    }
    reports = $reports
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
Write-BenchmarkJson $manifestPath $manifest
$manifestHash = Get-LowerSha256 $manifestPath

foreach ($spec in $reports) {
    Assert-RunInputs
    $trial = $spec.trial
    $policy = $spec.variant
    $server = & (Join-Path $PSScriptRoot 'Start-LLMServe.ps1') -Backend $Backend -Policy $policy `
        -Port $Port -Executable $serverExecutable -Model $Model -MaxActive $MaxActive `
        -QueueCapacity $QueueCapacity -BatchTokens $BatchTokens -PrefillChunk $PrefillChunk `
        -PrefixEntries $PrefixEntries -PrefixTokens $PrefixTokens -PageSize $PageSize -Context $Context `
        -MaxModelLen $MaxModelLen -EventBuffer $EventBuffer -Threads $Threads -GpuLayers $GpuLayers `
        -Kernel $Kernel -Telemetry $Telemetry -TelemetryCapacity $TelemetryCapacity `
        -TelemetryOutput $(if ($spec.telemetry_file) { Join-Path $OutputDirectory $spec.telemetry_file } else { '' })
    try {
        $output = Join-Path $OutputDirectory "$policy-$trial.json"
        $arguments = @('--port', $server.Port, '--trace', $archivedTrace, '--output', $output,
            '--run-id', $runId, '--trial', $trial, '--variant', $policy,
            '--trace-sha256', $manifest.trace.sha256, '--manifest-sha256', $manifestHash,
            '--model-sha256', $modelHash, '--server-sha256', $manifest.binaries.server.sha256,
            '--client-sha256', $manifest.binaries.benchmark_client.sha256)
        $arguments += @('--arrival-scale', $ArrivalScale.ToString('R', [Globalization.CultureInfo]::InvariantCulture))
        if ($NoWarmup) { $arguments += '--no-warmup' }
        & $benchExecutable @arguments
        if ($LASTEXITCODE -notin @(0, 1) -or -not (Test-Path -LiteralPath $output -PathType Leaf)) {
            throw "Benchmark process failed without a complete report: $output"
        }
    } finally {
        & (Join-Path $PSScriptRoot 'Stop-LLMServe.ps1') -Port $server.Port
    }
    Assert-RunInputs
}

& (Join-Path $PSScriptRoot 'Analyze-Benchmarks.ps1') -Directory $OutputDirectory
if ($Telemetry -ne 'off') {
    $python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $python) { throw '在线观测验收需要 Python 3。' }
    & $python.Source (Join-Path $PSScriptRoot 'analyze_telemetry.py') $OutputDirectory `
        --output (Join-Path $OutputDirectory 'telemetry-summary.json')
    if ($LASTEXITCODE -ne 0) { throw '在线观测验收失败，原始报告已保留。' }
}
