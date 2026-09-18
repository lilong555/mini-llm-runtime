param([ValidateRange(1024, 65535)][int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$run = [IO.Path]::GetFullPath((Join-Path $root '.run'))
$record = Get-Content -Raw -LiteralPath (Join-Path $run "server-$Port.json") | ConvertFrom-Json
$marker = [IO.Path]::GetFullPath($record.ShutdownFile)
if (-not $marker.StartsWith($run + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Shutdown marker is outside the project runtime directory.'
}
$process = Get-Process -Id $record.ProcessId -ErrorAction SilentlyContinue
if (-not $process) { return }
$expectedStart = [DateTimeOffset]$record.StartedAt
if ($process.Path -ne $record.Executable -or
    $process.StartTime.ToUniversalTime().Ticks -ne $expectedStart.UtcDateTime.Ticks) {
    throw 'The saved process identity no longer matches this server.'
}
New-Item -ItemType File -Path $marker -Force | Out-Null
Wait-Process -Id $process.Id -Timeout 60
Write-Output "LLMServe on port $Port stopped."
