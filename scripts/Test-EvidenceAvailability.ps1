param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$result = Get-EvidenceAvailability $Directory
if ($Output) {
    # 报告只能写到新路径；检查不能覆盖归档中的原始文件或旧结论。
    $Output = [IO.Path]::GetFullPath($Output)
    if (Test-Path -LiteralPath $Output) { throw "检查输出已存在：$Output" }
    Write-BenchmarkJson $Output $result
}
$result | ConvertTo-Json -Depth 32
if ($result.status -cne 'AVAILABLE') { throw ('ARCHIVE_INCOMPLETE: ' + ($result.errors -join '; ')) }
