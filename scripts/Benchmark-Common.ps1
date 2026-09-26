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
    $Path = [IO.Path]::GetFullPath($Path)
    $temporary = $Path + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    try {
        [IO.File]::WriteAllText($temporary, $json + "`n", [Text.UTF8Encoding]::new($false))
        if ([IO.File]::Exists($Path)) { [IO.File]::Replace($temporary, $Path, [NullString]::Value) }
        else { [IO.File]::Move($temporary, $Path) }
    } finally {
        if ([IO.File]::Exists($temporary)) { [IO.File]::Delete($temporary) }
    }
}

function Publish-BenchmarkOutputs([string]$Directory, [System.Collections.IDictionary]$Outputs) {
    # 所有 JSON 序列化成功后才发布；发布失败恢复原文件，验收标记由调用方放在最后。
    $stage = Join-Path $Directory ('.analysis-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stage | Out-Null
    $published = [Collections.Generic.List[string]]::new()
    try {
        foreach ($name in $Outputs.Keys) { Write-BenchmarkJson (Join-Path $stage $name) $Outputs[$name] }
        foreach ($name in $Outputs.Keys) {
            $target = Join-Path $Directory $name
            if ([IO.File]::Exists($target)) {
                [IO.File]::Replace((Join-Path $stage $name), $target, (Join-Path $stage "$name.previous"))
            } else { [IO.File]::Move((Join-Path $stage $name), $target) }
            $published.Add($name)
        }
    } catch {
        foreach ($name in $published) {
            $backup = Join-Path $stage "$name.previous"
            $target = Join-Path $Directory $name
            if ([IO.File]::Exists($backup)) { [IO.File]::Replace($backup, $target, [NullString]::Value) }
            else { [IO.File]::Delete($target) }
        }
        throw
    } finally { Remove-Item -LiteralPath $stage -Recurse -Force }
}

function Resolve-BenchmarkArtifact([string]$Directory, [string]$Name) {
    if (-not $Name -or $Name -match '(^[/\\]|^[A-Za-z]:|(^|[/\\])\.\.?([/\\]|$)|:)' -or
        [IO.Path]::IsPathRooted($Name)) { throw "无效的归档相对路径：$Name" }
    $path = [IO.Path]::GetFullPath($Directory)
    foreach ($part in ($Name -split '[/\\]')) {
        if (-not $part) { throw "无效的归档相对路径：$Name" }
        $path = Join-Path $path $part
        if (Test-Path -LiteralPath $path) {
            $item = Get-Item -LiteralPath $path -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "归档不能依赖符号链接：$Name"
            }
        }
    }
    return $path
}

function Test-BenchmarkSourceArchive([string]$Directory, $Source) {
    $state = Get-Content -Raw -LiteralPath (Resolve-BenchmarkArtifact $Directory $Source.state_file) | ConvertFrom-Json
    if ($state.files.Count -eq 0) { throw 'Empty source snapshot.' }
    if ((ConvertTo-Json -Compress -InputObject $state.scope) -cne
        (ConvertTo-Json -Compress -InputObject $Source.scope)) { throw 'source scope differs.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead((Resolve-BenchmarkArtifact $Directory $Source.snapshot.path))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        if ($archive.Entries.Count -ne $state.files.Count) { throw 'source snapshot entries differ.' }
        $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($file in $state.files) {
            $null = Resolve-BenchmarkArtifact $Directory $file.path
            if (-not $seen.Add($file.path)) { throw 'Duplicate source snapshot path.' }
            $entry = $archive.GetEntry($file.path)
            if ($null -eq $entry -or $entry.Length -ne $file.size_bytes) { throw "source snapshot size differs: $($file.path)" }
            $stream = $entry.Open()
            try { $hash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
            finally { $stream.Dispose() }
            if ($hash -cne $file.sha256) { throw "source snapshot entry hash differs: $($file.path)" }
        }
    } finally { $sha.Dispose(); $archive.Dispose() }
}

function Get-EvidenceAvailability([string]$Directory, [string]$Manifest = '') {
    $Directory = (Resolve-Path -LiteralPath $Directory).Path
    if (-not $Manifest) { $Manifest = Join-Path $Directory 'manifest.json' }
    $items = [Collections.Generic.List[object]]::new()
    $errors = [Collections.Generic.List[string]]::new()
    $runId = $null
    $manifestHash = $null
    $add = {
        param([string]$Name, [string]$ExpectedHash, [string]$Label, [bool]$HashRequired = $false)
        $record = [ordered]@{ locator = $Name; mandatory = $true; exists = $false
            expected_sha256 = $ExpectedHash; sha256 = $null; size_bytes = $null; status = 'MISSING'; error = $null }
        try {
            $path = Resolve-BenchmarkArtifact $Directory $Name
            $record.exists = Test-Path -LiteralPath $path -PathType Leaf
            if (-not $record.exists) { throw "Missing report or artifact (does not exist): $Name" }
            $record.sha256 = Get-LowerSha256 $path
            $record.size_bytes = (Get-Item -LiteralPath $path -Force).Length
            $record.status = 'AVAILABLE'
            if (($HashRequired -or $ExpectedHash) -and
                ($ExpectedHash -cnotmatch '^[0-9a-f]{64}$' -or $record.sha256 -cne $ExpectedHash)) {
                $record.status = 'HASH_MISMATCH'
                throw "$Label hash differs: $Name"
            }
        } catch {
            if ($record.status -eq 'AVAILABLE') { $record.status = 'INVALID' }
            $record.error = $_.Exception.Message
            $errors.Add($record.error)
        }
        $items.Add([pscustomobject]$record)
    }
    try {
        $manifestPath = [IO.Path]::GetFullPath($Manifest)
        if ($manifestPath -cne [IO.Path]::GetFullPath((Join-Path $Directory 'manifest.json'))) {
            throw '可用性检查要求 manifest.json 位于归档根目录。'
        }
        & $add 'manifest.json' '' 'manifest'
        $data = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
        $runId = $data.run_id
        $manifestHash = Get-LowerSha256 $manifestPath
        if ($data.schema_version -ne 1 -or $data.benchmark -cnotin @('minillm-runtime', 'llmserve-policy-comparison')) {
            throw '不支持的实验 manifest 类型或版本。'
        }
        & $add $data.source.state_file $data.source.worktree_state_sha256 'source state / source.worktree_state_sha256' $true
        & $add $data.source.snapshot.path $data.source.snapshot.sha256 'source snapshot hash / source.snapshot.sha256' $true
        if ($data.benchmark -ceq 'minillm-runtime') { & $add $data.input.path $data.input.sha256 'input' $true }
        else { & $add $data.trace.path $data.trace.sha256 'trace' $true }
        foreach ($report in $data.reports) {
            & $add $report.file '' 'report'
            foreach ($field in @('process_file', 'server_stdout_file', 'server_stderr_file',
                'client_stdout_file', 'client_stderr_file', 'gpu_sample_file', 'gpu_sample_error_file')) {
                if ($report.PSObject.Properties[$field] -and $report.$field) {
                    & $add $report.$field '' $field
                }
            }
            if ($report.PSObject.Properties['telemetry_file'] -and $report.telemetry_file) {
                & $add $report.telemetry_file '' 'telemetry'
            } elseif ($data.benchmark -ceq 'llmserve-policy-comparison' -and
                $data.engine.PSObject.Properties['telemetry_mode'] -and $data.engine.telemetry_mode -cne 'off') {
                $errors.Add("缺少 telemetry_file：$($report.file)")
            }
        }
        $collectionPath = Join-Path $Directory 'collection-status.json'
        if (Test-Path -LiteralPath $collectionPath) {
            & $add 'collection-status.json' '' 'collection'
            $collection = Get-Content -Raw -LiteralPath $collectionPath | ConvertFrom-Json
            if ($collection.status -cne 'passed') { $errors.Add('采集状态不是 passed。') }
        }
        if ($errors.Count -eq 0) {
            try { Test-BenchmarkSourceArchive $Directory $data.source }
            catch { $errors.Add($_.Exception.Message) }
        }
        $bundlePath = Join-Path $Directory 'bundle-manifest.json'
        if (Test-Path -LiteralPath $bundlePath) {
            $bundle = Get-Content -Raw -LiteralPath $bundlePath | ConvertFrom-Json
            if ($bundle.schema_version -ne 1 -or $bundle.purpose -cne 'archive_revalidation' -or
                $bundle.manifest_sha256 -cne $manifestHash) { throw 'bundle manifest 身份不一致。' }
            $listed = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
            foreach ($artifact in $bundle.artifacts) {
                if (-not $listed.Add($artifact.path) -or $artifact.sha256 -cnotmatch '^[0-9a-f]{64}$') {
                    throw 'bundle manifest 包含重复路径或无效摘要。'
                }
                & $add $artifact.path $artifact.sha256 'bundle artifact' $true
            }
            foreach ($item in $items) {
                if (-not $listed.Contains($item.locator)) { throw "bundle manifest 缺少必需文件：$($item.locator)" }
            }
        }
    } catch { $errors.Add($_.Exception.Message) }
    return [pscustomobject][ordered]@{ schema_version = 1; purpose = 'archive_revalidation'; run_id = $runId
        manifest_sha256 = $manifestHash; status = $(if ($errors.Count) { 'ARCHIVE_INCOMPLETE' } else { 'AVAILABLE' })
        artifacts = @($items.ToArray()); errors = @($errors.ToArray()) }
}

function Assert-EvidenceAvailable([string]$Directory, [string]$Manifest = '') {
    $availability = Get-EvidenceAvailability $Directory $Manifest
    if ($availability.status -cne 'AVAILABLE') {
        throw ('ARCHIVE_INCOMPLETE: ' + ($availability.errors -join '; '))
    }
    return $availability
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
    $device = if ($Backend -eq 'mini') { 'cpu' } elseif ($Backend -eq 'mini-cuda') { 'own-cuda' } else { 'cuda' }
    return Join-Path $Root "build/$platform$device/bin"
}

function Get-ProductExecutable([string]$Directory, [string]$Name) {
    $suffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
    return (Resolve-Path -LiteralPath (Join-Path $Directory "$Name$suffix")).Path
}
