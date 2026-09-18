$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root 'third_party\llama.cpp'
$commit = '911f6cdc8ab8a530b2bee09ee61471a6f3178eeb'
if (-not (Test-Path -LiteralPath $target)) {
    & git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git $target
    if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed.' }
    & git -C $target checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw 'Cannot checkout the pinned llama.cpp commit.' }
}
$head = & git -C $target rev-parse HEAD
if ($head -ne $commit) { throw "Existing llama.cpp checkout must be at $commit. It was not modified." }
$dirty = & git -C $target status --porcelain
if ($dirty) { throw 'llama.cpp has local changes. A clean pinned checkout is required.' }
Write-Output "llama.cpp: $head"
