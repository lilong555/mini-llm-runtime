param([string]$Collector = (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts/Profile-CudaRuntime.ps1'))
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($PSVersionTable.PSVersion.Major -lt 7) { throw 'Profiler 进程检查需要 PowerShell 7。' }
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($Collector, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Profiler 采集脚本存在语法错误。' }
$function = $ast.Find({
    param($node)
    $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -ceq 'Invoke-ProfilerProcess'
}, $true)
if ($null -eq $function) { throw '找不到进程包装函数。' }
# 仅加载进程包装函数，不执行采集入口、模型预检或 NVIDIA 工具。
Invoke-Expression $function.Extent.Text
$root = Join-Path ([IO.Path]::GetTempPath()) ('cuda-process-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root | Out-Null
$privateName = 'MINILLM_PROFILER_TEST_PRIVATE'
$oldPrivate = [Environment]::GetEnvironmentVariable($privateName)
try {
    [Environment]::SetEnvironmentVariable($privateName, 'fixture_not_a_credential')
    $program = [Environment]::ProcessPath
    $stdout = Join-Path $root 'stdout.txt'
    $stderr = Join-Path $root 'stderr.txt'
    $child = '[Console]::Out.Write([string]::new([char]111,131072))
[Console]::Error.Write([string]::new([char]101,131072))
if ($env:MINILLM_PROFILER_TEST_PRIVATE) { exit 9 }
exit 7'
    $records = @(Invoke-ProfilerProcess $program @('-NoProfile','-Command',$child) $stdout $stderr)
    if ($records.Count -ne 1 -or $records[0] -isnot [Collections.IDictionary]) {
        throw "进程包装函数必须只返回一份记录，实际返回 $($records.Count) 个对象。"
    }
    $record = $records[0]
    if ($record.exit_code -ne 7 -or $record.wrapper_elapsed_ns -le 0 -or
        $record.environment_policy -cne 'allowlist_no_credentials' -or
        $record.environment.Contains($privateName) -or $record.environment.LANG -cne 'C' -or
        (Get-Content -Raw -LiteralPath $stdout) -cne [string]::new([char]111,131072) -or
        (Get-Content -Raw -LiteralPath $stderr) -cne [string]::new([char]101,131072)) {
        throw '进程返回码、输出流、环境白名单或时间记录不符。'
    }
    Write-Output '[PASS] process_result_streams_and_environment'
    $before = (Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash
    $rejected = $false
    try { Invoke-ProfilerProcess $program @('-NoProfile','-Command','exit 0') $stdout $stderr | Out-Null }
    catch {
        if ($_.Exception.Message -notlike '*不能覆盖*') { throw }
        $rejected = $true
    }
    if (-not $rejected -or (Get-FileHash -LiteralPath $stdout -Algorithm SHA256).Hash -cne $before) {
        throw '已有输出没有受到保护。'
    }
    Write-Output '[PASS] existing_process_output_is_preserved'
    Write-Output '2/2 tests passed'
} finally {
    [Environment]::SetEnvironmentVariable($privateName, $oldPrivate)
    Remove-Item -LiteralPath $root -Recurse -Force
}
