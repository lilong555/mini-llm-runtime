# 完整模型外部诊断，不参与无 Profiler 的正式性能统计。
param(
    [string]$OutputDirectory = '',
    [string]$BaselineDirectory = '',
    [string]$BinaryDirectory = '',
    [string]$Model = '',
    [string]$NsightSystems = '',
    [string]$NsightCompute = '',
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
. (Join-Path $PSScriptRoot 'Cuda-Benchmark-Common.ps1')
$root = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python3 -ErrorAction Stop).Source
if (-not [Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([Runtime.InteropServices.OSPlatform]::Linux)) {
    throw '当前 Profiler 采集入口仅验收 Linux/WSL2；CPU 和 CUDA 产品构建不依赖该入口。'
}
if (-not $BaselineDirectory) { $BaselineDirectory = Join-Path $root 'benchmarks/results/cuda-model-baseline' }
if (-not $BinaryDirectory) { $BinaryDirectory = Join-Path $root 'build/wsl-own-cuda/bin' }
if (-not $Model) { $Model = Join-Path $root 'models/Qwen3-0.6B-Q8_0.gguf' }
if (-not $NsightSystems) {
    $localNsys = Join-Path $HOME '.local/bin/nsys'
    $NsightSystems = if (Test-Path -LiteralPath $localNsys) { $localNsys } else { (Get-Command nsys -ErrorAction Stop).Source }
}
if (-not $NsightCompute) { $NsightCompute = (Get-Command ncu -ErrorAction Stop).Source }
$BaselineDirectory = (Resolve-Path -LiteralPath $BaselineDirectory).Path
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
$NsightSystems = (Resolve-Path -LiteralPath $NsightSystems).Path
$NsightCompute = (Resolve-Path -LiteralPath $NsightCompute).Path
$executable = Get-ProductExecutable $BinaryDirectory 'mini-cuda-runtime-bench'
$analyzer = Join-Path $PSScriptRoot 'analyze_cuda_profiler.py'
$inputFile = Join-Path $root 'benchmarks/runtime-inputs/qwen3-cuda-v0.json'
& $python (Join-Path $PSScriptRoot 'analyze_cuda_benchmark.py') --directory $BaselineDirectory
if ($LASTEXITCODE -ne 0) { throw '必须先通过完整 70 进程模型基线复核。' }
$baselineManifestPath = Join-Path $BaselineDirectory 'manifest.json'
$baseline = Get-Content -Raw -LiteralPath $baselineManifestPath | ConvertFrom-Json
$baselineSummary = Get-Content -Raw -LiteralPath (Join-Path $BaselineDirectory 'summary.json') | ConvertFrom-Json
if ($baselineSummary.status -cnotin @('measured', 'measurement_inconclusive')) {
    throw '关联基线存在未解决的正确性问题。'
}
$binaryHash = Get-LowerSha256 $executable
$modelHash = Get-LowerSha256 $Model
$inputHash = Get-LowerSha256 $inputFile
if ($binaryHash -cne $baseline.binary.sha256 -or $modelHash -cne $baseline.model.sha256 -or
    $inputHash -cne $baseline.input.sha256) {
    throw 'Profiler 要求与关联基线相同的模型基准二进制、模型和冻结输入。'
}
$cache = Join-Path (Split-Path -Parent $BinaryDirectory) 'CMakeCache.txt'
if ((Read-CMakeValue $cache 'CMAKE_HOME_DIRECTORY') -cne $root -or
    (Read-CMakeValue $cache 'MINILLM_ENABLE_CUDA') -cne 'ON' -or
    (Read-CMakeValue $cache 'LLMSERVE_CUDA') -cne 'OFF') {
    throw 'Profiler 只接受本工作区的自有 CUDA=ON、上游 CUDA=OFF 构建。'
}
$dependency = Join-Path $root 'third_party/llama.cpp'
$revision = (& git -C $dependency rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $revision -cne '911f6cdc8ab8a530b2bee09ee61471a6f3178eeb') {
    throw 'Profiler 的 llama.cpp 依赖版本不符。'
}
$gitSha = (& git -C $root rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw '无法识别 Profiler 源码版本。' }
$dirty = @(& git -C $root status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw '无法识别 Profiler 工作区状态。' }
$scope = @('CMakeLists.txt', '.gitattributes', 'cmake', 'apps', 'include', 'src', 'scripts', 'tests', '.github',
    'models/manifest.json', 'models/reference-manifest.json', 'benchmarks/runtime-inputs', 'docs/NEXT_SPEC.md')
$sourceFiles = @(Get-BenchmarkSourceState $root $scope)
$sourceJson = ConvertTo-Json -InputObject $sourceFiles -Depth 8 -Compress
$oldState = Get-Content -Raw -LiteralPath (Join-Path $BaselineDirectory 'source-state.json') | ConvertFrom-Json
$productPattern = '^(include|src)/minillm/|^apps/(cuda_runtime_bench\.cpp|cuda_benchmark\.h|cuda_reports\.h|options\.h)$'
$product = @($sourceFiles | Where-Object { $_.path -cmatch $productPattern })
$oldProduct = @($oldState.files | Where-Object { $_.path -cmatch $productPattern })
if ($product.Count -eq 0 -or $product.Count -ne $oldProduct.Count) { throw '模型执行源码集合不同，不能继承基线。' }
foreach ($item in $product) {
    $old = @($oldProduct | Where-Object path -CEQ $item.path)
    if ($old.Count -ne 1 -or $old[0].sha256 -cne $item.sha256) { throw "模型执行源码已变化：$($item.path)" }
}
$planText = & $python $analyzer --schedule
if ($LASTEXITCODE -ne 0) { throw '无法建立 Profiler 诊断计划。' }
$plan = ($planText -join "`n") | ConvertFrom-Json
$runId = '{0}-{1}-{2}' -f (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'),
    $gitSha.Substring(0, 12), ([guid]::NewGuid().ToString('N').Substring(0, 8))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "benchmarks/results/cuda-profiler-$runId" }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $OutputDirectory) -and @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count) {
    throw 'Profiler 输出目录必须为空，原始采集不会被覆盖。'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$rawDirectory = Join-Path $root ".run/cuda-profiler-$runId"
if (Test-Path -LiteralPath $rawDirectory) { throw 'Profiler 原始采集目录已存在。' }
New-Item -ItemType Directory -Path $rawDirectory | Out-Null

function Invoke-ProfilerProcess([string]$Program, [string[]]$Arguments, [string]$Stdout, [string]$Stderr) {
    if ((Test-Path -LiteralPath $Stdout) -or (Test-Path -LiteralPath $Stderr)) { throw '不能覆盖进程原始输出。' }
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Program
    $info.WorkingDirectory = $root
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    # Profiler 可能保存子进程环境；只继承执行所需的非凭据字段，所有诊断进程使用相同白名单。
    $info.Environment.Clear()
    $allowed = @('PATH', 'HOME', 'USER', 'TMPDIR', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER', 'CUBLAS_WORKSPACE_CONFIG')
    $recordedEnvironment = [ordered]@{}
    foreach ($name in $allowed) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($null -ne $value) {
            $info.Environment[$name] = $value
            $recordedEnvironment[$name] = $value
        }
    }
    $info.Environment['LANG'] = 'C'
    $info.Environment['LC_ALL'] = 'C'
    $recordedEnvironment['LANG'] = 'C'
    $recordedEnvironment['LC_ALL'] = 'C'
    $started = (Get-Date).ToUniversalTime().ToString('o')
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    $outFile = [IO.File]::Create($Stdout)
    $errFile = [IO.File]::Create($Stderr)
    try {
        if (-not $process.Start()) { throw '无法启动 Profiler 子进程。' }
        $outTask = $process.StandardOutput.BaseStream.CopyToAsync($outFile)
        $errTask = $process.StandardError.BaseStream.CopyToAsync($errFile)
        $process.WaitForExit()
        $outTask.GetAwaiter().GetResult()
        $errTask.GetAwaiter().GetResult()
        $watch.Stop()
        return [ordered]@{ executable = $Program; arguments = $Arguments; exit_code = $process.ExitCode
            started_at_utc = $started; finished_at_utc = (Get-Date).ToUniversalTime().ToString('o')
            wrapper_elapsed_ns = [long]($watch.ElapsedTicks * 1000000000.0 / [Diagnostics.Stopwatch]::Frequency)
            environment_policy = 'allowlist_no_credentials'; environment = $recordedEnvironment }
    } finally {
        $outFile.Dispose()
        $errFile.Dispose()
        $process.Dispose()
    }
}

$nsysVersionProcess = Invoke-ProfilerProcess $NsightSystems @('--version') `
    (Join-Path $OutputDirectory 'nsys-version.stdout.txt') (Join-Path $OutputDirectory 'nsys-version.stderr.txt')
$ncuVersionProcess = Invoke-ProfilerProcess $NsightCompute @('--version') `
    (Join-Path $OutputDirectory 'ncu-version.stdout.txt') (Join-Path $OutputDirectory 'ncu-version.stderr.txt')
Write-BenchmarkJson (Join-Path $OutputDirectory 'nsys-version.process.json') $nsysVersionProcess
Write-BenchmarkJson (Join-Path $OutputDirectory 'ncu-version.process.json') $ncuVersionProcess
$nsysVersion = (Get-Content -Raw -LiteralPath (Join-Path $OutputDirectory 'nsys-version.stdout.txt')).Trim()
$ncuVersion = (Get-Content -Raw -LiteralPath (Join-Path $OutputDirectory 'ncu-version.stdout.txt')).Trim()
if ($nsysVersionProcess.exit_code -ne 0 -or $ncuVersionProcess.exit_code -ne 0 -or
    -not $nsysVersion.StartsWith('NVIDIA Nsight Systems version 2026.1.3') -or $ncuVersion -notmatch '2025\.1\.1') {
    throw '当前入口验收 Nsight Systems 2026.1.3 与 Nsight Compute 2025.1.1；工具原始输出已保留。'
}
Copy-Item -LiteralPath $inputFile -Destination (Join-Path $OutputDirectory 'input.json')
Copy-Item -LiteralPath $analyzer -Destination (Join-Path $OutputDirectory 'verify.py')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'analyze_cuda_benchmark.py') `
    -Destination (Join-Path $OutputDirectory 'analyze_cuda_benchmark.py')
foreach ($entry in @(
        @('manifest.json', 'baseline-manifest.json'),
        @('source-state.json', 'baseline-source-state.json'),
        @('summary.json', 'baseline-summary.json'),
        @('validation-summary.json', 'baseline-validation-summary.json'))) {
    Copy-Item -LiteralPath (Join-Path $BaselineDirectory $entry[0]) -Destination (Join-Path $OutputDirectory $entry[1])
}
$referenceSlot = $baseline.reports | Where-Object backend -CEQ 'cuda' | Select-Object -First 1
Copy-Item -LiteralPath (Join-Path $BaselineDirectory $referenceSlot.file) `
    -Destination (Join-Path $OutputDirectory 'baseline-reference.json')
$stateFile = Join-Path $OutputDirectory 'source-state.json'
Write-BenchmarkJson $stateFile ([ordered]@{ scope = $scope; files = $sourceFiles })
$snapshot = Join-Path $OutputDirectory 'source-snapshot.zip'
Write-BenchmarkSourceSnapshot $root $sourceFiles $snapshot
$manifest = [ordered]@{
    schema_version = 1; benchmark = 'minillm-cuda-profiler'; run_id = $runId
    created_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    source = [ordered]@{ git_sha = $gitSha; git_dirty = $dirty.Count -gt 0; scope = $scope; state_file = 'source-state.json'
        worktree_state_sha256 = Get-LowerSha256 $stateFile
        snapshot = [ordered]@{ path = 'source-snapshot.zip'; sha256 = Get-LowerSha256 $snapshot } }
    binary = [ordered]@{ path = $executable; sha256 = $binaryHash; availability = 'rerun_dependency_not_bundled' }
    model = [ordered]@{ path = $Model; sha256 = $modelHash; availability = 'rerun_dependency_not_bundled' }
    input = [ordered]@{ path = 'input.json'; execution_path = Join-Path $OutputDirectory 'input.json'; sha256 = $inputHash }
    baseline = [ordered]@{ run_id = $baseline.run_id; manifest_sha256 = Get-LowerSha256 $baselineManifestPath
        repository_path = [IO.Path]::GetRelativePath($root, $BaselineDirectory).Replace('\', '/')
        reference_report = $referenceSlot.file; full_baseline_included = $false }
    tools = [ordered]@{
        nsys = [ordered]@{ path = $NsightSystems; version = $nsysVersion; sha256 = Get-LowerSha256 $NsightSystems }
        ncu = [ordered]@{ path = $NsightCompute; version = $ncuVersion; sha256 = Get-LowerSha256 $NsightCompute } }
    environment = [ordered]@{ profiler_scope = 'external_diagnostic'; frequency_control = 'uncontrolled'
        affinity = 'inherited'; telemetry = 'process_boundaries_not_continuous'
        environment_policy = 'allowlist_no_credentials'; os = [Environment]::OSVersion.VersionString
        powershell = $PSVersionTable.PSVersion.ToString() }
    processes = $plan.processes; selector = $plan.selector
    ownership = [ordered]@{ model_and_kv = 'project'; matrix_kernels = 'NVIDIA cuBLAS'
        profiler = 'NVIDIA Nsight'; gpu_serving = $false; gpu_paged_attention = $false }
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
Write-BenchmarkJson $manifestPath $manifest
$manifestHash = Get-LowerSha256 $manifestPath

function Assert-ProfilerInputs {
    if ((Get-LowerSha256 $executable) -cne $binaryHash -or (Get-LowerSha256 $Model) -cne $modelHash -or
        (Get-LowerSha256 $inputFile) -cne $inputHash -or (Get-LowerSha256 $manifest.input.execution_path) -cne $inputHash -or
        (Get-LowerSha256 $manifestPath) -cne $manifestHash -or
        (Get-LowerSha256 $NsightSystems) -cne $manifest.tools.nsys.sha256 -or
        (Get-LowerSha256 $NsightCompute) -cne $manifest.tools.ncu.sha256) {
        throw 'Profiler 采集中模型、输入、二进制、工具或 manifest 发生变化。'
    }
    $current = ConvertTo-Json -InputObject @(Get-BenchmarkSourceState $root $scope) -Depth 8 -Compress
    if ($current -cne $sourceJson) { throw 'Profiler 采集中源码发生变化。' }
    $dependencyStatus = @(& git -C $dependency status --porcelain)
    if ($LASTEXITCODE -ne 0 -or $dependencyStatus.Count) { throw 'Profiler 依赖 checkout 不是干净状态。' }
    $currentRevision = & git -C $dependency rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $currentRevision -cne $revision) { throw 'Profiler 依赖版本发生变化。' }
}

$completed = [Collections.Generic.List[object]]::new()
try {
    Assert-ProfilerInputs
    Test-BenchmarkSourceArchive $OutputDirectory ([pscustomobject]$manifest.source)
    if ($PreflightOnly) {
        Write-BenchmarkJson (Join-Path $OutputDirectory 'preflight-environment.json') (Get-CudaBenchmarkEnvironment)
        Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
            status = 'preflight_only'; reports = @(); scope = '只核对基线、身份和工具，不执行模型或 Profiler' })
        Write-Host "Profiler 预检通过：$OutputDirectory"
        return
    }
    foreach ($slot in $plan.processes) {
        Assert-ProfilerInputs
        $reportPath = Join-Path $OutputDirectory $slot.report
        $targetArguments = @('--model', $Model, '--input', $manifest.input.execution_path, '--backend', 'cuda', '--output', $reportPath)
        $program = $executable
        $arguments = $targetArguments
        if ($slot.profiler -ceq 'nsys') {
            $program = $NsightSystems
            $arguments = @('profile') + @($plan.nsys_flags) + @("--output=$(Join-Path $rawDirectory 'nsys')", $executable) + $targetArguments
        } elseif ($slot.profiler -ceq 'ncu') {
            $program = $NsightCompute
            $arguments = @($plan.ncu_flags) + @('--export', (Join-Path $rawDirectory 'ncu'), $executable) + $targetArguments
        }
        $before = Get-CudaBenchmarkEnvironment
        $record = Invoke-ProfilerProcess $program $arguments "$reportPath.stdout.txt" "$reportPath.stderr.txt"
        $record.order = $slot.order
        $record.profiler = $slot.profiler
        $record.report_execution_path = $reportPath
        $record.before = $before
        $record.after = Get-CudaBenchmarkEnvironment
        Write-BenchmarkJson "$reportPath.process.json" $record
        $completed.Add([pscustomobject]@{ file = $slot.report; exit_code = $record.exit_code
            sha256 = $(if (Test-Path -LiteralPath $reportPath) { Get-LowerSha256 $reportPath } else { $null }) })
        Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
            status = 'collecting'; reports = @($completed.ToArray()); planned_reports = $plan.processes.Count })
        Assert-ProfilerInputs
        if ($record.exit_code -ne 0) { throw "Profiler 目标失败：$($slot.name)，原始输出保留。" }
        & $python (Join-Path $PSScriptRoot 'analyze_cuda_benchmark.py') --report $reportPath --input $manifest.input.execution_path
        if ($LASTEXITCODE -ne 0) { throw "Profiler 下的模型报告没有通过数据路径复核：$($slot.name)" }
        Write-Host "Profiler 诊断 $($completed.Count)/$($plan.processes.Count)：$($slot.name)"
    }
    $export = Invoke-ProfilerProcess $NsightCompute @('--import', (Join-Path $rawDirectory 'ncu.ncu-rep'),
        '--csv', '--page', 'raw', '--print-units', 'base') `
        (Join-Path $OutputDirectory 'ncu-metrics.csv') (Join-Path $OutputDirectory 'ncu-export.stderr.txt')
    Write-BenchmarkJson (Join-Path $OutputDirectory 'ncu-export.process.json') $export
    if ($export.exit_code -ne 0) { throw 'NCU 原始指标导出失败。' }
    $members = @('nsys.nsys-rep', 'nsys.sqlite', 'ncu.ncu-rep') | ForEach-Object {
        $path = Join-Path $rawDirectory $_
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) {
            throw "缺少完整 Profiler 原始文件：$_"
        }
        [ordered]@{ path = $_; sha256 = Get-LowerSha256 $path; size_bytes = (Get-Item -LiteralPath $path).Length }
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $rawZip = Join-Path $OutputDirectory 'profiler-raw.zip'
    $archive = [IO.Compression.ZipFile]::Open($rawZip, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($member in $members) {
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,
                (Join-Path $rawDirectory $member.path), $member.path, [IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally { $archive.Dispose() }
    Assert-ProfilerInputs
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'passed'; reports = @($completed.ToArray()); planned_reports = $plan.processes.Count
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
    $artifacts = @(Get-ChildItem -LiteralPath $OutputDirectory -File | Sort-Object Name | ForEach-Object {
        [ordered]@{ path = $_.Name; sha256 = Get-LowerSha256 $_.FullName; size_bytes = $_.Length }
    })
    Write-BenchmarkJson (Join-Path $OutputDirectory 'artifact-manifest.json') ([ordered]@{
        schema_version = 1; purpose = 'archive_revalidation'; manifest_sha256 = $manifestHash
        artifacts = $artifacts; raw_members = @($members); raw_availability = 'bundled'
        raw_archive = 'profiler-raw.zip'; external_tools_required_for_archive_revalidation = $false
        execution_dependencies_included = $false })
} catch {
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') ([ordered]@{
        status = 'failed'; reports = @($completed.ToArray()); error = $_.Exception.Message
        local_raw_directory = $rawDirectory; raw_availability = 'local_only_until_bundled'
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o') })
    throw
}
& $python $analyzer --directory $OutputDirectory --write
if ($LASTEXITCODE -ne 0) { throw 'Profiler 证据复核失败，原始文件和已有输出保持不变。' }
