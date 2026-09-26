param(
    [Parameter(Mandatory = $true)][string]$BaselineDirectory,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [string]$Nsys = "$HOME/.local/bin/nsys",
    [int]$Port = 8132
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
. (Join-Path $root 'scripts/Benchmark-Common.ps1')
$BaselineDirectory = (Resolve-Path $BaselineDirectory).Path
$base = Get-Content -Raw (Join-Path $BaselineDirectory 'manifest.json') | ConvertFrom-Json
if ($base.engine.backend -ne 'mini-cuda' -or $base.engine.telemetry_mode -ne 'off') {
    throw '时间线参照必须是无观测的 own-CUDA Serving 基线。'
}
foreach ($input in @($base.binaries.server, $base.binaries.benchmark_client, $base.model)) {
    if ((Get-LowerSha256 $input.path) -cne $input.sha256) { throw '时间线输入与正式基线不同。' }
}
$state = Get-Content -Raw (Join-Path $BaselineDirectory 'source-state.json') | ConvertFrom-Json
$current = @(Get-BenchmarkSourceState $root $base.source.scope)
if ((ConvertTo-Json -InputObject $current -Depth 8 -Compress) -cne
    (ConvertTo-Json -InputObject @($state.files) -Depth 8 -Compress)) { throw '时间线源码与正式基线不同。' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path $OutputDirectory) { throw '时间线输出目录必须尚不存在。' }
if (Test-LoopbackPort $Port) { throw '时间线端口已占用。' }
$Nsys = (Resolve-Path $Nsys).Path
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
Copy-Item (Join-Path $BaselineDirectory 'trace.jsonl') $OutputDirectory
Copy-Item (Join-Path $BaselineDirectory 'source-state.json') $OutputDirectory
Copy-Item (Join-Path $BaselineDirectory 'source-snapshot.zip') $OutputDirectory
$gpu = @(& nvidia-smi --id=0 --query-gpu=name,uuid,compute_cap --format=csv,noheader,nounits |
    ConvertFrom-Csv -Header name, uuid, compute_cap)[0]
if ($LASTEXITCODE -ne 0) { throw '无法查询时间线设备身份。' }
$version = (& $Nsys --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw '无法查询 NSys 版本。' }
$marker = Join-Path $OutputDirectory 'stop'
$serverArgs = [Collections.Generic.List[string]]::new()
$reference = Get-Content -Raw (Join-Path $BaselineDirectory 'mixed-0-process.json') | ConvertFrom-Json
foreach ($argument in $reference.server.Arguments) { $serverArgs.Add([string]$argument) }
$overrides = @{ '--port' = "$Port"; '--shutdown-file' = $marker; '--telemetry' = 'batches'; '--telemetry-capacity' = '2048' }
for ($i = 0; $i -lt $serverArgs.Count; ++$i) {
    if ($overrides.ContainsKey($serverArgs[$i])) { $serverArgs[$i + 1] = $overrides[$serverArgs[$i]] }
}
$serverArgs.Add('--telemetry-output')
$serverArgs.Add((Join-Path $OutputDirectory 'batches.jsonl'))
$nsysArgs = @('profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none', '--discard-environment=true',
    '--cuda-memory-usage=true', '--force-overwrite=false', '--export=sqlite',
    "--output=$(Join-Path $OutputDirectory 'nsys')", $base.binaries.server.path) + @($serverArgs.ToArray())
$engine = $base.engine
$engine.telemetry_mode = 'batches'
$engine.telemetry_capacity = 2048
$manifest = [ordered]@{
    schema_version = 1; spec_id = 'CUDA-SERVE-001'; benchmark = 'llmserve-cuda-profiler'
    source = $base.source; binaries = $base.binaries; model = $base.model; build = $base.build
    engine = $engine; trace = $base.trace
    device = [ordered]@{ name = $gpu.name.Trim(); uuid = ($gpu.uuid.Trim() -replace '^GPU-', '')
        compute_capability = @($gpu.compute_cap.Trim().Split('.') | ForEach-Object { [int]$_ }) }
    tool = [ordered]@{ path = $Nsys; sha256 = Get-LowerSha256 $Nsys; version = $version; arguments = $nsysArgs }
    report = @{ path = 'request.json' }; telemetry = @{ path = 'batches.jsonl' }; database = @{ path = 'nsys.sqlite' }
    formal_performance_baseline = $false
}
$manifestPath = Join-Path $OutputDirectory 'manifest.json'
Write-BenchmarkJson $manifestPath $manifest
$clientArgs = @('--port', "$Port", '--trace', (Join-Path $OutputDirectory 'trace.jsonl'),
    '--output', (Join-Path $OutputDirectory 'request.json'), '--arrival-scale', '1',
    '--run-id', 'CUDA-SERVE-001-nsys', '--trial', '0', '--variant', 'mixed',
    '--manifest-sha256', (Get-LowerSha256 $manifestPath), '--trace-sha256', $base.trace.sha256,
    '--model-sha256', $base.model.sha256, '--server-sha256', $base.binaries.server.sha256,
    '--client-sha256', $base.binaries.benchmark_client.sha256)
$quoted = @($nsysArgs | ForEach-Object {
    if ($_.Contains('"')) { throw '命令参数不能包含双引号。' }
    '"' + $_ + '"'
})
$started = (Get-Date).ToUniversalTime().ToString('o')
$process = Start-Process -FilePath $Nsys -ArgumentList $quoted -WorkingDirectory $root -PassThru `
    -RedirectStandardOutput (Join-Path $OutputDirectory 'nsys.stdout.log') `
    -RedirectStandardError (Join-Path $OutputDirectory 'nsys.stderr.log')
$clientExit = $null
$failure = $null
try {
    $ready = $false
    for ($i = 0; $i -lt 300; ++$i) {
        if ($process.HasExited) { throw '时间线服务在 ready 前退出。' }
        try { $ready = (Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 1).ready } catch {}
        if ($ready) { break }
        Start-Sleep -Milliseconds 200
    }
    if (-not $ready) { throw '时间线服务 ready 超时。' }
    & $base.binaries.benchmark_client.path @clientArgs 1> (Join-Path $OutputDirectory 'client.stdout.log') `
        2> (Join-Path $OutputDirectory 'client.stderr.log')
    $clientExit = $LASTEXITCODE
    if ($clientExit -ne 0) { throw '时间线请求失败，保留原始报告。' }
} catch {
    $failure = $_.Exception.Message
    throw
} finally {
    New-Item -ItemType File -Path $marker -Force | Out-Null
    if (-not $process.WaitForExit(180000)) {
        $process.Kill($true)
        $process.WaitForExit()
        $failure = '时间线停服或导出超时。'
    }
    Write-BenchmarkJson (Join-Path $OutputDirectory 'process.json') ([ordered]@{
        started_at_utc = $started; finished_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        executable = $Nsys; arguments = $nsysArgs; exit_code = $process.ExitCode
        client_arguments = $clientArgs; client_exit_code = $clientExit; error = $failure
    })
}
if ($failure -or $process.ExitCode -ne 0) { throw '时间线进程未正常完成。' }
& python3 (Join-Path $root 'scripts/analyze_cuda_profiler.py') --serving --directory $OutputDirectory --write
if ($LASTEXITCODE -ne 0) { throw '时间线分析未通过，原始报告保持不变。' }
