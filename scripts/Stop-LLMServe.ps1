param([ValidateRange(1024, 65535)][int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$run = [IO.Path]::GetFullPath((Join-Path $root '.run'))
$record = Get-Content -Raw -LiteralPath (Join-Path $run "server-$Port.json") | ConvertFrom-Json
$marker = [IO.Path]::GetFullPath($record.ShutdownFile)
$comparison = if ($env:OS -eq 'Windows_NT') { [StringComparison]::OrdinalIgnoreCase } else { [StringComparison]::Ordinal }
if (-not $marker.Equals((Join-Path $run "server-$Port.stop"), $comparison)) {
    throw 'Shutdown marker does not match the project server record.'
}
$process = Get-Process -Id $record.ProcessId -ErrorAction SilentlyContinue
if (-not $process) { return }
$expectedStart = [DateTimeOffset]$record.StartedAt
if (-not $process.Path.Equals([string]$record.Executable, $comparison) -or
    $process.StartTime.ToUniversalTime().Ticks -ne $expectedStart.UtcDateTime.Ticks) {
    throw 'The saved process identity no longer matches this server.'
}
New-Item -ItemType File -Path $marker -Force | Out-Null
Wait-Process -Id $process.Id -Timeout 60
Write-Output "LLMServe on port $Port stopped."
