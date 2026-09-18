param(
    [string]$BinaryDirectory = '',
    [ValidateRange(0, 10000)][int]$ReferenceGpuLayers = 0,
    [string]$Output = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $BinaryDirectory) { $BinaryDirectory = Join-Path $root 'build\cpu\bin' }
$BinaryDirectory = (Resolve-Path -LiteralPath $BinaryDirectory).Path
$model = & (Join-Path $PSScriptRoot 'Download-Model.ps1')
$manifest = Get-Content -Raw -LiteralPath (Join-Path $root 'models\reference-manifest.json') | ConvertFrom-Json
$reference = Join-Path $root ('models\' + $manifest.file)
if (-not (Test-Path -LiteralPath $reference)) {
    & (Join-Path $BinaryDirectory 'mini-llm.exe') --model $model --dequantize-ref $reference
    if ($LASTEXITCODE -ne 0) { throw 'Reference conversion failed.' }
}
if ((Get-Item -LiteralPath $reference).Length -ne $manifest.size_bytes -or
    (Get-FileHash -LiteralPath $reference -Algorithm SHA256).Hash -ne $manifest.sha256) {
    throw 'F32 reference does not match the pinned conversion manifest.'
}
if (-not $Output) {
    $Output = Join-Path $root "benchmarks\results\validation\model-reference-$ReferenceGpuLayers.json"
}
& (Join-Path $BinaryDirectory 'llmserve-model-tests.exe') --model $model --reference-model $reference `
    --gpu-layers $ReferenceGpuLayers --output $Output
if ($LASTEXITCODE -ne 0) { throw "Model validation failed: $Output" }
