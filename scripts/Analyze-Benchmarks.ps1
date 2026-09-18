param([Parameter(Mandatory = $true)][string]$Directory)

$ErrorActionPreference = 'Stop'
function Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$middle] }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2
}
$reports = @(Get-ChildItem -LiteralPath $Directory -Filter '*.json' |
    Where-Object Name -Match '^(mixed|prefill_first)-[0-9]+\.json$' |
    ForEach-Object { Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json })
if ($reports.Count -eq 0) { throw 'No policy replay reports found.' }
$baseline = $reports[0]
$expected = @{}
foreach ($request in $baseline.requests) {
    $expected[$request.id] = ConvertTo-Json -InputObject $request.token_ids -Compress
}
$mismatches = 0
foreach ($report in $reports) {
    if ($report.trace_fnv1a64 -ne $baseline.trace_fnv1a64 -or
        $report.server_before.backend -ne $baseline.server_before.backend -or
        $report.server_before.model -ne $baseline.server_before.model) {
        throw 'Reports do not use the same input trace, backend, and model.'
    }
    foreach ($request in $report.requests) {
        if (-not $expected.ContainsKey($request.id) -or
            (ConvertTo-Json -InputObject $request.token_ids -Compress) -ne $expected[$request.id]) {
            ++$mismatches
        }
    }
}
$policies = @{}
foreach ($group in ($reports | Group-Object { $_.server_before.policy })) {
    $rows = @($group.Group)
    $policies[$group.Name] = [ordered]@{
        trials = $rows.Count
        successful = ($rows.summary.successful | Measure-Object -Sum).Sum
        failed = ($rows.summary.failed | Measure-Object -Sum).Sum
        output_tokens_per_second_median = Median @($rows.summary.output_tokens_per_second)
        ttft_p95_ms_median = Median @($rows.summary.ttft_ms.p95)
        mean_tpot_p95_ms_median = Median @($rows.summary.mean_tpot_ms.p95)
        goodput_requests_per_second_median = Median @($rows.summary.goodput_requests_per_second)
    }
}
$summary = [ordered]@{
    backend = $baseline.server_before.backend
    model = $baseline.server_before.model
    trace_fnv1a64 = $baseline.trace_fnv1a64
    requests_per_trial = $baseline.summary.requests
    token_sequence_mismatches_against_first_trial = $mismatches
    policies = $policies
}
$output = Join-Path (Resolve-Path -LiteralPath $Directory).Path 'summary.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $output -Encoding utf8
$summary | ConvertTo-Json -Depth 8
