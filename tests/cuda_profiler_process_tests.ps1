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
    $repository = Split-Path -Parent $PSScriptRoot
    $runId = 'fixture'
    $GpuKvStudy = $false
    $PrecisionStudy = $false
    $ReferenceGpuLayers = 0
    $localPrefix = [IO.Path]::GetFullPath((Join-Path $root '.run')) + [IO.Path]::DirectorySeparatorChar
    $targets = @{
        'Benchmark-CudaRuntime.ps1' = @('OutputDirectory', 'NumericalDirectory')
        'Benchmark-CudaMicro.ps1' = @('OutputDirectory')
        'Profile-CudaRuntime.ps1' = @('OutputDirectory', 'rawDirectory', 'BaselineDirectory')
        'Benchmark-Runtime.ps1' = @('OutputDirectory')
        'Benchmark-Policies.ps1' = @('OutputDirectory')
        'Validate-Model.ps1' = @('Output')
    }
    $defaults = @{}
    foreach ($scriptName in $targets.Keys) {
        $path = if ($scriptName -ceq 'Profile-CudaRuntime.ps1') { $Collector }
            else { Join-Path $repository "scripts/$scriptName" }
        $tree = [Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
        if ($errors.Count) { throw "采集脚本存在语法错误：$scriptName" }
        $defaults[$scriptName] = @{}
        foreach ($variable in $targets[$scriptName]) {
            $assignments = @($tree.FindAll({
                param($node)
                $node -is [Management.Automation.Language.AssignmentStatementAst] -and
                $node.Left -is [Management.Automation.Language.VariableExpressionAst] -and
                $node.Left.VariablePath.UserPath -ceq $variable -and
                $node.Right -is [Management.Automation.Language.PipelineAst] -and
                $node.Right.PipelineElements[0] -is [Management.Automation.Language.CommandAst] -and
                $node.Right.PipelineElements[0].GetCommandName() -ceq 'Join-Path'
            }, $true))
            if ($assignments.Count -ne 1) { throw "默认产物路径定义不唯一：$scriptName/$variable" }
            foreach ($Backend in @('mini', 'mini-cuda', 'llama')) {
                $value = [IO.Path]::GetFullPath((Invoke-Expression $assignments[0].Right.Extent.Text))
                if (-not $value.StartsWith($localPrefix, [StringComparison]::Ordinal)) {
                    throw "默认产物应位于已忽略的 .run 目录：$scriptName/$variable/$Backend"
                }
            }
            $defaults[$scriptName][$variable] = $value
        }
    }
    $report = $defaults['Profile-CudaRuntime.ps1']['OutputDirectory']
    $raw = $defaults['Profile-CudaRuntime.ps1']['rawDirectory']
    $separator = [IO.Path]::DirectorySeparatorChar
    if ($report -ceq $raw -or $report.StartsWith($raw + $separator) -or $raw.StartsWith($report + $separator)) {
        throw 'Profiler 报告目录与原始采集目录应分别管理。'
    }
    Write-Output '[PASS] default_artifact_paths_are_local_and_distinct'
    Write-Output '3/3 tests passed'
} finally {
    [Environment]::SetEnvironmentVariable($privateName, $oldPrivate)
    Remove-Item -LiteralPath $root -Recurse -Force
}
