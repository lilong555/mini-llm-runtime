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
    [ValidateRange(1, 256)][int]$Threads = 8
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Executable) {
    $Executable = Join-Path $root $(if ($Backend -eq 'llama') { 'build\cuda\bin\llmserve.exe' } else { 'build\cpu\bin\llmserve.exe' })
    if (-not (Test-Path -LiteralPath $Executable)) { $Executable = Join-Path $root 'build\cuda\bin\llmserve.exe' }
}
$Executable = (Resolve-Path -LiteralPath $Executable).Path
if (-not $Model) { $Model = Join-Path $root 'models\Qwen3-0.6B-Q8_0.gguf' }
$Model = (Resolve-Path -LiteralPath $Model).Path
$first = $Port
while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    ++$Port
    if ($Port -gt $first + 20) { throw 'No unused port was found in the requested range.' }
}
if ($env:CUDA_PATH -and (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin'))) {
    $env:Path = "$(Join-Path $env:CUDA_PATH 'bin');$env:Path"
}
$run = Join-Path $root '.run'
New-Item -ItemType Directory -Path $run -Force | Out-Null
$marker = Join-Path $run "server-$Port.stop"
if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
$argsList = @('--model', $Model, '--backend', $Backend, '--policy', $Policy, '--port', "$Port",
    '--max-active', "$MaxActive", '--queue-capacity', "$QueueCapacity", '--batch-tokens', "$BatchTokens",
    '--prefill-chunk', "$PrefillChunk", '--prefix-entries', "$PrefixEntries", '--threads', "$Threads",
    '--shutdown-file', $marker)
$arguments = $argsList | ForEach-Object {
    if ($_.Contains('"')) { throw 'Argument cannot contain a double quote.' }
    '"' + $_ + '"'
}
$stdout = Join-Path $run "server-$Port.stdout.log"
$stderr = Join-Path $run "server-$Port.stderr.log"
$process = Start-Process -FilePath $Executable -ArgumentList $arguments -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$record = [pscustomobject]@{
    Url = "http://127.0.0.1:$Port"
    Port = $Port
    ProcessId = $process.Id
    Executable = $Executable
    Backend = $Backend
    ShutdownFile = $marker
    StartedAt = $process.StartTime.ToUniversalTime().ToString('o')
    ErrorLog = $stderr
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run "server-$Port.json") -Encoding utf8
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
New-Item -ItemType File -Path $marker -Force | Out-Null
throw "Server readiness timed out. Shutdown requested; see $stderr."
