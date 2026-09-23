param(
    [Parameter(Mandatory = $true)][string]$Trace,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [ValidateRange(1, 20)][int]$Trials = 3,
    [ValidateRange(0.000001, 10000)][double]$ArrivalScale = 1.0,
    [ValidateRange(1024, 65515)][int]$Port = 8081,
    [ValidateRange(1, 256)][int]$Threads = 8,
    [ValidateRange(0, 128)][int]$PrefixEntries = 4,
    [ValidateRange(0, 1048576)][int]$PrefixTokens = 2048,
    [ValidateRange(1, 16384)][int]$TelemetryCapacity = 1024,
    [string]$BinaryDirectory = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $python) { throw '在线观测验收需要 Python 3。' }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if ((Test-Path -LiteralPath $OutputDirectory) -and @(Get-ChildItem -LiteralPath $OutputDirectory -Force).Count -gt 0) {
    throw '输出目录必须为空。'
}
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$groups = @()
for ($trial = 0; $trial -lt $Trials; ++$trial) {
    $modes = if ($trial % 2 -eq 0) { @('off', 'batches', 'stages') } else { @('stages', 'batches', 'off') }
    foreach ($mode in $modes) {
        $groups += [ordered]@{ directory = "trial-$trial-$mode"; trial = $trial; mode = $mode; order = $groups.Count }
    }
}
$protocol = [ordered]@{
    schema_version = 1; benchmark = 'llmserve-telemetry-comparison'
    trials = $Trials; groups = $groups; allowed_changes = @('engine.policy', 'engine.telemetry_mode')
    limitations = @('CPU 频率、温度和背景负载未固定。', '开关差异包含系统噪声与 batch 组成变化。')
}
Write-BenchmarkJson (Join-Path $OutputDirectory 'experiment.json') $protocol
$completed = 0
try {
    foreach ($group in $groups) {
        & (Join-Path $PSScriptRoot 'Benchmark-Policies.ps1') -Trace $Trace -Trials 1 `
            -PolicyOrderOffset ($group.trial % 2) -ArrivalScale $ArrivalScale -Port $Port -Threads $Threads `
            -PrefixEntries $PrefixEntries -PrefixTokens $PrefixTokens -Telemetry $group.mode `
            -TelemetryCapacity $TelemetryCapacity -BinaryDirectory $BinaryDirectory `
            -OutputDirectory (Join-Path $OutputDirectory $group.directory)
        ++$completed
    }
    $directories = @($groups | ForEach-Object { Join-Path $OutputDirectory $_.directory })
    & $python.Source (Join-Path $PSScriptRoot 'analyze_telemetry.py') @directories `
        --output (Join-Path $OutputDirectory 'telemetry-summary.json')
    if ($LASTEXITCODE -ne 0) { throw '在线观测对照验收失败。' }
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') `
        ([ordered]@{ status = 'passed'; completed_groups = $completed; expected_groups = $groups.Count })
} catch {
    Write-BenchmarkJson (Join-Path $OutputDirectory 'collection-status.json') `
        ([ordered]@{ status = 'failed'; completed_groups = $completed; expected_groups = $groups.Count; error = $_.Exception.Message })
    throw
}
