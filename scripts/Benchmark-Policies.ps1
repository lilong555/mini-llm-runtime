param(
    [Parameter(Mandatory = $true)][string]$Trace,
    [ValidateSet('mini', 'llama')][string]$Backend = 'mini',
    [ValidateRange(1, 20)][int]$Trials = 3,
    [ValidateRange(1024, 65515)][int]$Port = 8000,
    [string]$BinaryDirectory = '',
    [string]$OutputDirectory = '',
    [ValidateRange(0, 128)][int]$PrefixEntries = 4
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$Trace = (Resolve-Path -LiteralPath $Trace).Path
if (-not $BinaryDirectory) {
    $BinaryDirectory = Join-Path $root $(if ($Backend -eq 'mini') { 'build\cpu\bin' } else { 'build\cuda\bin' })
}
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root "benchmarks\results\$Backend-scheduling" }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw 'Benchmark port is occupied. Stop the project server or select another port.'
}
for ($trial = 0; $trial -lt $Trials; ++$trial) {
    $policies = if ($trial % 2 -eq 0) { @('mixed', 'prefill_first') } else { @('prefill_first', 'mixed') }
    foreach ($policy in $policies) {
        $server = & (Join-Path $PSScriptRoot 'Start-LLMServe.ps1') -Backend $Backend -Policy $policy `
            -Port $Port -PrefixEntries $PrefixEntries -Executable (Join-Path $BinaryDirectory 'llmserve.exe')
        try {
            $output = Join-Path $OutputDirectory "$policy-$trial.json"
            & (Join-Path $BinaryDirectory 'llmserve-bench.exe') --port $server.Port --trace $Trace --output $output
            if ($LASTEXITCODE -ne 0) { throw "Benchmark request failed. Report: $output" }
        } finally {
            & (Join-Path $PSScriptRoot 'Stop-LLMServe.ps1') -Port $server.Port
        }
    }
}
