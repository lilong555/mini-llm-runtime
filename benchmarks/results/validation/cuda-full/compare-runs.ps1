param([string]$Directory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$previous = Join-Path $Directory 'diagnostics/fused-reference'
function Read-Json([string]$Root, [string]$Name) {
    Get-Content -Raw -LiteralPath (Join-Path $Root $Name) | ConvertFrom-Json
}
function Same-Bytes([string]$Left, [string]$Right) {
    (Get-FileHash -LiteralPath $Left -Algorithm SHA256).Hash -ceq (Get-FileHash -LiteralPath $Right -Algorithm SHA256).Hash
}
$oldState = Read-Json $previous 'source-state.json'
$newState = Read-Json $Directory 'source-state.json'
$oldProduct = @($oldState.files | Where-Object { $_.path -match '^(apps|src|include)/' })
$newProduct = @($newState.files | Where-Object { $_.path -match '^(apps|src|include)/' })
$sourceEqual = (ConvertTo-Json -Compress -Depth 8 $oldProduct) -ceq (ConvertTo-Json -Compress -Depth 8 $newProduct)
$counts = [ordered]@{}
$firstDifference = $null
function Compare-Score([string]$Category, [string]$Identity, $Old, $New) {
    if (-not $counts.Contains($Category)) { $counts[$Category] = [ordered]@{rows=0; bitwise_equal=0; token_equal=0} }
    $counter = $counts[$Category]
    ++$counter.rows
    $same = $Old.sha256 -ceq $New.sha256
    $counter.bitwise_equal += [int]$same
    $counter.token_equal += [int]($Old.token -eq $New.token)
    if (-not $same -and $null -eq $script:firstDifference) {
        $script:firstDifference = [ordered]@{category=$Category; identity=$Identity; before=$Old; after=$New}
    }
}
$oldReport = Read-Json $previous 'real-model/full-validation.json'
$newReport = Read-Json $Directory 'real-model/full-validation.json'
foreach ($entry in $newReport.canonical | Where-Object { $_.backend -in @('cpu','cuda_canonical') }) {
    $old = Read-Json $previous "real-model/$($entry.file)"
    $new = Read-Json $Directory "real-model/$($entry.file)"
    if ($old.samples.Count -ne $new.samples.Count) { throw '独立参照采样数量不同。' }
    for ($i=0; $i -lt $new.samples.Count; ++$i) {
        if ($old.samples[$i].position -ne $new.samples[$i].position) { throw '独立参照位置不同。' }
        Compare-Score $entry.backend "$($entry.id):$($new.samples[$i].position)" $old.samples[$i] $new.samples[$i]
    }
}
foreach ($entry in $newReport.teacher_forcing) {
    $old = @((Read-Json $previous "real-model/$($entry.file)").comparisons | Where-Object { $_.reference -ceq 'cpu' })
    $new = @((Read-Json $Directory "real-model/$($entry.file)").comparisons | Where-Object { $_.reference -ceq 'cpu' })
    if ($old.Count -ne $new.Count) { throw 'teacher-forcing 采样数量不同。' }
    for ($i=0; $i -lt $new.Count; ++$i) {
        if ($old[$i].position -ne $new[$i].position -or $old[$i].sequence -ne $new[$i].sequence) {
            throw 'teacher-forcing 采样身份不同。'
        }
        Compare-Score 'cuda_teacher' "$($entry.id):$i" `
            ([pscustomobject]@{sha256=$old[$i].actual_sha256; token=$old[$i].actual_argmax}) `
            ([pscustomobject]@{sha256=$new[$i].actual_sha256; token=$new[$i].actual_argmax})
    }
}
foreach ($entry in $newReport.generation) {
    $old = (Read-Json $previous "real-model/$($entry.file)").rows
    $new = (Read-Json $Directory "real-model/$($entry.file)").rows
    if ($old.Count -ne $new.Count) { throw '自然生成步数不同。' }
    for ($i=0; $i -lt $new.Count; ++$i) {
        Compare-Score 'cuda_generation' "$($entry.id):$i" $old[$i].cuda $new[$i].cuda
        Compare-Score 'cpu_generation' "$($entry.id):$i" $old[$i].references.cpu.score $new[$i].references.cpu.score
    }
}
[pscustomobject][ordered]@{schema_version=1; scope='同输入、同产品源码的参照模式对照；不是性能结果'
    product_sources_equal=$sourceEqual; product_source_files=$newProduct.Count
    dependency_commit_equal=(Read-Json $previous 'environment.json').llama_commit -ceq (Read-Json $Directory 'environment.json').llama_commit
    contract_bytes_equal=Same-Bytes (Join-Path $previous 'real-model/validation-contract.json') (Join-Path $Directory 'real-model/validation-contract.json')
    input_bytes_equal=Same-Bytes (Join-Path $previous 'real-model/input.json') (Join-Path $Directory 'real-model/input.json')
    scores=$counts; first_bitwise_difference=$firstDifference
    before_status=$oldReport.status; after_status=$newReport.status
    before_numeric_failures=$oldReport.totals.numeric_failures; after_numeric_failures=$newReport.totals.numeric_failures}
