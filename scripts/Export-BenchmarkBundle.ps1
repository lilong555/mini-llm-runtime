param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$Output = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $Output) { throw "导出路径已存在：$Output" }
if (Test-Path -LiteralPath "$Output.sha256") { throw "摘要路径已存在：$Output.sha256" }
$availability = Assert-EvidenceAvailable $Directory
$manifest = Get-Content -Raw -LiteralPath (Join-Path $Directory 'manifest.json') | ConvertFrom-Json
$analyzer = if ($manifest.benchmark -ceq 'minillm-runtime') { 'Analyze-Runtime.ps1' } else { 'Analyze-Benchmarks.ps1' }
$hasTelemetry = $manifest.benchmark -ceq 'llmserve-policy-comparison' -and
    $manifest.engine.PSObject.Properties['telemetry_mode'] -and $manifest.engine.telemetry_mode -cne 'off'
function Invoke-BundleAnalysis([string]$Path, [string]$ToolDirectory = $PSScriptRoot) {
    if ($hasTelemetry) {
        $python = Get-Command python3, python -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $python) { throw '在线观测归档需要 Python 3 完成复验。' }
        # 在线分析器内部已调用策略分析器；每个副本只发布一次派生汇总。
        $result = & $python.Source (Join-Path $ToolDirectory 'analyze_telemetry.py') $Path `
            --output (Join-Path $Path 'telemetry-summary.json')
        if ($LASTEXITCODE -ne 0) { throw "在线观测归档验收失败：$($result -join ' ')" }
    } else { & (Join-Path $ToolDirectory $analyzer) -Directory $Path | Out-Null }
}
$temporary = Join-Path ([IO.Path]::GetTempPath()) ('benchmark-export-' + [guid]::NewGuid().ToString('N'))
$stage = Join-Path $temporary 'bundle'
$roundtrip = Join-Path $temporary 'roundtrip'
$pending = $Output + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
New-Item -ItemType Directory -Path $stage -Force | Out-Null
try {
    foreach ($name in @($availability.artifacts.locator | Sort-Object -Unique)) {
        $source = Resolve-BenchmarkArtifact $Directory $name
        $target = Resolve-BenchmarkArtifact $stage $name
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $target
        $expected = $availability.artifacts | Where-Object locator -CEQ $name | Select-Object -First 1
        if ((Get-LowerSha256 $target) -cne $expected.sha256) { throw "导出期间归档发生变化：$name" }
    }
    # 原 manifest 与绝对身份路径原样保存；仅在副本中运行严格分析器。
    Invoke-BundleAnalysis $stage
    $verification = Join-Path $stage 'verification'
    New-Item -ItemType Directory -Path $verification | Out-Null
    $verificationFiles = @('Benchmark-Common.ps1', 'Test-EvidenceAvailability.ps1', $analyzer)
    if ($hasTelemetry) { $verificationFiles += 'analyze_telemetry.py' }
    foreach ($name in $verificationFiles) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $verification $name)
    }
    $artifacts = @(Get-ChildItem -LiteralPath $stage -File -Recurse -Force | Sort-Object FullName | ForEach-Object {
        [ordered]@{ path = $_.FullName.Substring($stage.Length + 1).Replace('\', '/')
            sha256 = Get-LowerSha256 $_.FullName; size_bytes = $_.Length }
    })
    Write-BenchmarkJson (Join-Path $stage 'bundle-manifest.json') ([ordered]@{
        schema_version = 1; purpose = 'archive_revalidation'; run_id = $manifest.run_id
        manifest_sha256 = $availability.manifest_sha256; artifacts = $artifacts
        analyzer = $analyzer; analyzer_sha256 = Get-LowerSha256 (Join-Path $PSScriptRoot $analyzer)
        verification_entry = $(if ($hasTelemetry) { 'verification/analyze_telemetry.py' } else { "verification/$analyzer" })
        common_sha256 = Get-LowerSha256 (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
        telemetry_analyzer_sha256 = $(if ($hasTelemetry) { Get-LowerSha256 (Join-Path $PSScriptRoot 'analyze_telemetry.py') } else { $null })
        execution_dependencies_included = $false
    })
    Write-BenchmarkJson (Join-Path $stage 'evidence-availability.json') (Assert-EvidenceAvailable $stage)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory($stage, $pending)
    [IO.Compression.ZipFile]::ExtractToDirectory($pending, $roundtrip)
    $null = Assert-EvidenceAvailable $roundtrip
    Invoke-BundleAnalysis $roundtrip (Join-Path $roundtrip 'verification')
    [IO.File]::Move($pending, $Output)
    [IO.File]::WriteAllText("$Output.sha256", ((Get-LowerSha256 $Output) + "  " + [IO.Path]::GetFileName($Output) + "`n"),
        [Text.UTF8Encoding]::new($false))
    Write-Host "归档复验包已生成：$Output"
} finally {
    if (Test-Path -LiteralPath $pending) { Remove-Item -LiteralPath $pending }
    Remove-Item -LiteralPath $temporary -Recurse -Force
}
