function Get-LowerSha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-CMakeValue([string]$Cache, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Cache)) { return $null }
    $line = Get-Content -LiteralPath $Cache | Where-Object { $_ -match "^$([regex]::Escape($Name))(?::[^=]+)?=" } |
        Select-Object -First 1
    if ($null -eq $line) { return $null }
    return ($line -split '=', 2)[1]
}

function Read-CMakeSetValue([string]$Path, [string]$Name) {
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $null }
    $pattern = '^set\({0} "([^"]*)"\)' -f [regex]::Escape($Name)
    $line = Get-Content -LiteralPath $Path | Where-Object { $_ -match $pattern } | Select-Object -First 1
    if ($null -eq $line) { return $null }
    return [regex]::Match($line, $pattern).Groups[1].Value
}

function Get-BenchmarkSourceState([string]$Root, [string[]]$Scope) {
    $paths = @(& git -C $Root -c core.quotepath=false ls-files --cached --others --exclude-standard -- $Scope)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate benchmark sources.' }
    return @($paths | Sort-Object -CaseSensitive -Unique | ForEach-Object {
        $path = Join-Path $Root $_
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            [ordered]@{ path = $_; sha256 = Get-LowerSha256 $path; size_bytes = (Get-Item -LiteralPath $path -Force).Length }
        }
    })
}

function Write-BenchmarkSourceSnapshot([string]$Root, $Files, [string]$Path) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::Open($Path, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($item in $Files) {
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, (Join-Path $Root $item.path), $item.path) | Out-Null
        }
    } finally {
        $archive.Dispose()
    }
}

function Write-BenchmarkJson([string]$Path, $Value) {
    $json = ConvertTo-Json -InputObject $Value -Depth 32
    [System.IO.File]::WriteAllText($Path, $json + "`n", [System.Text.UTF8Encoding]::new($false))
}

function Get-TraceFnv1a64([string]$Path) {
    $hash = [System.Numerics.BigInteger]::Parse('14695981039346656037')
    $mask = [System.Numerics.BigInteger]::Parse('18446744073709551615')
    foreach ($byte in [System.IO.File]::ReadAllBytes($Path)) {
        $hash = (($hash -bxor $byte) * 1099511628211) -band $mask
    }
    return $hash.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Test-LoopbackPort([int]$Port) {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    return @($listeners | Where-Object Port -EQ $Port).Count -gt 0
}

function Get-ProductDirectory([string]$Root, [string]$Backend) {
    $platform = if ($env:OS -eq 'Windows_NT') { '' } else { 'wsl-' }
    $device = if ($Backend -eq 'mini') { 'cpu' } else { 'cuda' }
    return Join-Path $Root "build/$platform$device/bin"
}

function Get-ProductExecutable([string]$Directory, [string]$Name) {
    $suffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
    return (Resolve-Path -LiteralPath (Join-Path $Directory "$Name$suffix")).Path
}
