param([switch]$UseWsl, [string]$Distribution = 'Ubuntu')

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$manifest = Get-Content -Raw -LiteralPath (Join-Path $root 'models\manifest.json') | ConvertFrom-Json
$model = Join-Path $root ('models\' + $manifest.file)
if (-not (Test-Path -LiteralPath $model)) {
    $partial = $model + '.part'
    if ($UseWsl) {
        $linuxPath = (& wsl.exe -d $Distribution --exec wslpath -a -u $partial).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the model download path in WSL.' }
        & wsl.exe -d $Distribution --exec curl --fail --location --retry 3 --output $linuxPath $manifest.url
        if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }
    } else {
        Invoke-WebRequest -Uri $manifest.url -OutFile $partial
    }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $manifest.sha256) {
        throw 'Downloaded model SHA-256 does not match the manifest.'
    }
    Move-Item -LiteralPath $partial -Destination $model
}
if ((Get-Item -LiteralPath $model).Length -ne $manifest.size_bytes -or
    (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash -ne $manifest.sha256) {
    throw 'Existing model does not match the pinned manifest. It was not modified.'
}
Write-Output $model
