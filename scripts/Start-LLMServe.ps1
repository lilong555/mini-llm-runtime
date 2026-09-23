param(
    [ValidateSet('mini', 'llama')][string]$Backend = 'mini',
    [ValidateSet('mixed', 'prefill_first')][string]$Policy = 'mixed',
    [ValidateRange(1024, 65515)][int]$Port = 8000,
    [string]$Executable = '',
    [string]$Model = '',
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
    [string]$TelemetryOutput = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$root = Split-Path -Parent $PSScriptRoot
if (-not $Executable) {
    $Executable = Get-ProductExecutable (Get-ProductDirectory $root $Backend) 'llmserve'
}
$Executable = (Resolve-Path -LiteralPath $Executable).Path
if (-not $Model) { $Model = Join-Path $root 'models\Qwen3-0.6B-Q8_0.gguf' }
$Model = (Resolve-Path -LiteralPath $Model).Path
if (($Telemetry -eq 'off') -ne [string]::IsNullOrEmpty($TelemetryOutput)) {
    throw '启用观测时必须指定 TelemetryOutput；关闭观测时不得指定输出文件。'
}
if ($TelemetryOutput) { $TelemetryOutput = [IO.Path]::GetFullPath($TelemetryOutput) }
$first = $Port
while (Test-LoopbackPort $Port) {
    ++$Port
    if ($Port -gt $first + 20) { throw 'No unused port was found in the requested range.' }
}
if ($env:OS -eq 'Windows_NT' -and $env:CUDA_PATH -and
    (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin'))) {
    $env:Path = "$(Join-Path $env:CUDA_PATH 'bin');$env:Path"
}
$run = Join-Path $root '.run'
New-Item -ItemType Directory -Path $run -Force | Out-Null
$marker = Join-Path $run "server-$Port.stop"
if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
$argsList = @('--model', $Model, '--backend', $Backend, '--policy', $Policy, '--port', "$Port",
    '--max-active', "$MaxActive", '--queue-capacity', "$QueueCapacity", '--batch-tokens', "$BatchTokens",
    '--prefill-chunk', "$PrefillChunk", '--prefix-entries', "$PrefixEntries", '--prefix-tokens', "$PrefixTokens",
    '--threads', "$Threads", '--page-size', "$PageSize", '--context', "$Context",
    '--max-model-len', "$MaxModelLen", '--event-buffer', "$EventBuffer", '--gpu-layers', "$GpuLayers",
    '--kernel', $Kernel, '--shutdown-file', $marker)
$argsList += @('--telemetry', $Telemetry, '--telemetry-capacity', "$TelemetryCapacity")
if ($TelemetryOutput) { $argsList += @('--telemetry-output', $TelemetryOutput) }
$arguments = $argsList | ForEach-Object {
    if ($_.Contains('"')) { throw 'Argument cannot contain a double quote.' }
    '"' + $_ + '"'
}
$stdout = Join-Path $run "server-$Port.stdout.log"
$stderr = Join-Path $run "server-$Port.stderr.log"
$startOptions = @{
    FilePath = $Executable; ArgumentList = $arguments; WorkingDirectory = $root
    PassThru = $true; RedirectStandardOutput = $stdout; RedirectStandardError = $stderr
}
if ($env:OS -eq 'Windows_NT') { $startOptions.WindowStyle = 'Hidden' }
$process = Start-Process @startOptions
$record = [pscustomobject]@{
    Url = "http://127.0.0.1:$Port"
    Port = $Port
    ProcessId = $process.Id
    Executable = $Executable
    Model = $Model
    Backend = $Backend
    Policy = $Policy
    ShutdownFile = $marker
    StartedAt = $process.StartTime.ToUniversalTime().ToString('o')
    ErrorLog = $stderr
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "server-$Port.json") -Encoding utf8
try {
    for ($i = 0; $i -lt 300; ++$i) {
        if ($process.HasExited) {
            throw "Server exited with code $($process.ExitCode). See $stderr."
        }
        try {
            $health = Invoke-RestMethod -Uri ($record.Url + '/health') -TimeoutSec 1
            if ($health.ready) { return $record }
        } catch {}
        Start-Sleep -Milliseconds 200
    }
    throw "Server readiness timed out. See $stderr."
} catch {
    New-Item -ItemType File -Path $marker -Force | Out-Null
    if (-not $process.WaitForExit(60000)) {
        $process.Kill()
        $process.WaitForExit()
    }
    throw
}
