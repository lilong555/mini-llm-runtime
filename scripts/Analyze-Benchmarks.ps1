param(
    [Parameter(Mandatory = $true)][string]$Directory,
    [string]$Manifest = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$Directory = (Resolve-Path -LiteralPath $Directory).Path
if (-not $Manifest) { $Manifest = Join-Path $Directory 'manifest.json' }
$validationPath = Join-Path $Directory 'validation-summary.json'
$summaryPath = Join-Path $Directory 'summary.json'
$errors = [System.Collections.Generic.List[string]]::new()
$checks = [System.Collections.Generic.List[string]]::new()
$manifestHash = $null
$runId = $null

function Add-ValidationError([string]$Message) { $script:errors.Add($Message) }

function Require-Equal($Actual, $Expected, [string]$Label) {
    if ($null -eq $Actual -or $null -eq $Expected -or "$Actual" -cne "$Expected") {
        Add-ValidationError "$Label differs: expected '$Expected', got '$Actual'."
    }
}

function Require-Integer($Value, [long]$Minimum, [long]$Maximum, [string]$Label) {
    if (($Value -isnot [int] -and $Value -isnot [long] -and $Value -isnot [System.Numerics.BigInteger]) -or
        $Value -lt $Minimum -or $Value -gt $Maximum) {
        Add-ValidationError "$Label must be an integer in [$Minimum, $Maximum]."
    }
}

function Require-Number($Value, [string]$Label) {
    if (($Value -isnot [int] -and $Value -isnot [long] -and $Value -isnot [double] -and $Value -isnot [decimal]) -or
        [double]::IsNaN([double]$Value) -or [double]::IsInfinity([double]$Value) -or $Value -lt 0) {
        Add-ValidationError "$Label must be a finite, nonnegative number."
    }
}

function Require-Close($Actual, [double]$Expected, [string]$Label) {
    Require-Number $Actual $Label
    if ($null -eq $Actual -or [math]::Abs([double]$Actual - $Expected) -gt 1e-7 * [math]::Max(1, [math]::Abs($Expected))) {
        Add-ValidationError "$Label does not match the raw measurements."
    }
}

function Optional-Value($Object, [string]$Name, $Default) {
    if ($null -eq $Object.PSObject.Properties[$Name]) { return $Default }
    return $Object.$Name
}

function Resolve-Artifact([string]$Name) {
    if (-not $Name -or [System.IO.Path]::IsPathRooted($Name) -or $Name -match '(^|[/\\])\.\.([/\\]|$)') {
        throw "Invalid manifest artifact path: '$Name'."
    }
    return (Resolve-Path -LiteralPath (Join-Path (Split-Path -Parent $Manifest) $Name)).Path
}

function Check-Percentiles($Actual, [double[]]$Values, [string]$Label) {
    if ($Values.Count -eq 0) {
        if ($null -ne $Actual) { Add-ValidationError "$Label must be null without samples." }
        return
    }
    $sorted = @($Values | Sort-Object)
    foreach ($percentile in @(50, 95, 99)) {
        $rank = ($percentile / 100.0) * ($sorted.Count - 1)
        $lower = [int][math]::Floor($rank)
        $upper = [math]::Min($lower + 1, $sorted.Count - 1)
        $expected = $sorted[$lower] + ($sorted[$upper] - $sorted[$lower]) * ($rank - $lower)
        Require-Close $Actual."p$percentile" $expected "$Label.p$percentile"
    }
}

function Token-Key($Tokens) {
    return ConvertTo-Json -InputObject @($Tokens) -Compress
}

function Median([object[]]$Values) {
    $sorted = @($Values | Where-Object { $null -ne $_ } | Sort-Object)
    if ($sorted.Count -eq 0) { return $null }
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return $sorted[$middle] }
    return ($sorted[$middle - 1] + $sorted[$middle]) / 2
}

try {
    $null = Assert-EvidenceAvailable $Directory $Manifest
    $Manifest = (Resolve-Path -LiteralPath $Manifest).Path
    $manifestData = Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json
    $manifestHash = Get-LowerSha256 $Manifest
    $runId = $manifestData.run_id
    if ([string]::IsNullOrWhiteSpace([string]$runId)) { Add-ValidationError 'manifest.run_id is required.' }
    Require-Equal $manifestData.schema_version 1 'manifest.schema_version'
    Require-Equal $manifestData.benchmark 'llmserve-policy-comparison' 'manifest.benchmark'
    if ($manifestData.source.git_sha -cnotmatch '^[0-9a-f]{40}$' -or $manifestData.source.git_dirty -isnot [bool]) {
        Add-ValidationError 'Invalid source Git identity.'
    }
    Require-Equal (Get-LowerSha256 (Resolve-Artifact $manifestData.source.state_file)) `
        $manifestData.source.worktree_state_sha256 'source.worktree_state_sha256'
    Require-Equal (Get-LowerSha256 (Resolve-Artifact $manifestData.source.snapshot.path)) `
        $manifestData.source.snapshot.sha256 'source.snapshot.sha256'
    foreach ($binary in @('server', 'benchmark_client')) {
        if ([string]$manifestData.binaries.$binary.sha256 -notmatch '^[0-9a-f]{64}$') {
            Add-ValidationError "manifest.binaries.$binary.sha256 is invalid."
        }
    }
    if ([string]$manifestData.model.sha256 -notmatch '^[0-9a-f]{64}$') {
        Add-ValidationError 'manifest.model.sha256 is invalid.'
    }
    if ([string]$manifestData.trace.sha256 -notmatch '^[0-9a-f]{64}$') {
        Add-ValidationError 'manifest.trace.sha256 is invalid.'
    }
    foreach ($field in @('type', 'generator', 'compiler', 'compiler_id', 'compiler_version')) {
        if ([string]::IsNullOrWhiteSpace([string]$manifestData.build.$field)) {
            Add-ValidationError "manifest.build.$field is required."
        }
    }
    foreach ($field in @('cxx_flags', 'configuration_flags', 'linker_flags', 'configuration_linker_flags', 'cuda_enabled')) {
        if ($null -eq $manifestData.build.$field) { Add-ValidationError "manifest.build.$field is required." }
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifestData.environment.os) -or
        [string]::IsNullOrWhiteSpace([string]$manifestData.environment.processor)) {
        Add-ValidationError 'The OS and CPU identity are required.'
    }
    Require-Integer $manifestData.environment.logical_processors 1 65536 'environment.logical_processors'
    Require-Equal $manifestData.model.file $manifestData.model.provenance.file 'model.provenance.file'
    Require-Equal $manifestData.model.sha256 $manifestData.model.provenance.sha256 'model.provenance.sha256'
    Require-Equal $manifestData.model.size_bytes $manifestData.model.provenance.size_bytes 'model.provenance.size_bytes'
    if (@('Q8_0', 'F16', 'F32') -cnotcontains $manifestData.model.weight_dtype) {
        Add-ValidationError 'Unsupported or unknown model weight dtype.'
    }
    if (@('mini', 'llama') -cnotcontains $manifestData.engine.backend) { Add-ValidationError 'Unknown engine backend.' }
    $metricsBackend = if ($manifestData.engine.backend -ceq 'mini') { 'minillm' } else { 'llama.cpp' }
    $activationDtype = if ($manifestData.engine.backend -ceq 'mini') { 'F32' } else { 'upstream_native' }
    Require-Equal $manifestData.engine.metrics_backend $metricsBackend 'engine.metrics_backend'
    Require-Equal $manifestData.protocol.activation_dtype $activationDtype 'protocol.activation_dtype'
    Require-Equal $manifestData.protocol.sampling 'greedy' 'protocol.sampling'
    Require-Equal $manifestData.protocol.kv_dtype 'F16' 'protocol.kv_dtype'
    $telemetryMode = Optional-Value $manifestData.engine 'telemetry_mode' 'off'
    if ($telemetryMode -cnotin @('off', 'batches', 'stages')) { Add-ValidationError 'Invalid telemetry mode.' }
    $profilerMode = if ($telemetryMode -ceq 'off') { 'none' } else { $telemetryMode }
    Require-Equal $manifestData.protocol.profiler_mode $profilerMode 'protocol.profiler_mode'
    $arrivalScale = Optional-Value $manifestData.protocol 'arrival_scale' 1.0
    Require-Number $arrivalScale 'protocol.arrival_scale'
    if ($arrivalScale -lt 0.000001 -or $arrivalScale -gt 10000) { Add-ValidationError 'Invalid arrival scale.' }
    Require-Equal $manifestData.protocol.order 'alternating' 'protocol.order'
    $orderOffset = Optional-Value $manifestData.protocol 'order_offset' 0
    Require-Integer $orderOffset 0 1 'protocol.order_offset'
    if ($manifestData.protocol.warmup -isnot [bool]) { Add-ValidationError 'protocol.warmup must be boolean.' }
    if ($manifestData.protocol.warmup) {
        Require-Equal $manifestData.protocol.warmup_request.prompt 'Hello' 'protocol.warmup_request.prompt'
        Require-Equal $manifestData.protocol.warmup_request.max_tokens 8 'protocol.warmup_request.max_tokens'
        Require-Equal $manifestData.protocol.warmup_request.ignore_eos $true 'protocol.warmup_request.ignore_eos'
        Require-Equal $manifestData.protocol.warmup_request.cache_namespace 'benchmark-warmup' 'protocol.warmup_request.cache_namespace'
    } elseif ($null -ne $manifestData.protocol.warmup_request) {
        Add-ValidationError 'The warmup request must be null when warmup is disabled.'
    }
    if ($null -ne $manifestData.trace.seed) {
        Require-Integer $manifestData.trace.seed 0 ([long]::MaxValue) 'trace.seed'
    }
    Require-Equal $manifestData.comparison.dimension 'engine.policy' 'comparison.dimension'
    if (@($manifestData.comparison.allowed_changes).Count -ne 1 -or
        @($manifestData.comparison.allowed_changes)[0] -cne 'engine.policy') {
        Add-ValidationError 'Policy comparison must declare engine.policy as its only allowed change.'
    }

    $tracePath = Resolve-Artifact $manifestData.trace.path
    if (-not (Test-Path -LiteralPath $tracePath -PathType Leaf)) {
        Add-ValidationError "Trace file is unavailable: $tracePath"
        $traceRows = @()
    } else {
        Require-Equal (Get-LowerSha256 $tracePath) ([string]$manifestData.trace.sha256).ToLowerInvariant() 'trace.sha256'
        Require-Equal (Get-Item -LiteralPath $tracePath).Length $manifestData.trace.size_bytes 'trace.size_bytes'
        Require-Equal (Get-TraceFnv1a64 $tracePath) $manifestData.trace.fnv1a64 'trace.fnv1a64'
        $traceRows = @()
        foreach ($line in Get-Content -LiteralPath $tracePath) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                Add-ValidationError 'Trace contains an empty row.'
                continue
            }
            try { $traceRows += $line | ConvertFrom-Json } catch { Add-ValidationError "Trace row is invalid JSON: $($_.Exception.Message)" }
        }
    }
    Require-Equal $traceRows.Count $manifestData.trace.request_count 'trace.request_count'
    Require-Integer $traceRows.Count 1 256 'trace request count'
    $expectedRequests = [System.Collections.Generic.Dictionary[string,object]]::new([System.StringComparer]::Ordinal)
    $lastArrival = 0.0
    foreach ($row in $traceRows) {
        $id = [string]$row.request_id
        if ($id -cnotmatch '^[A-Za-z0-9_.:-]{1,128}$' -or $expectedRequests.ContainsKey($id)) {
            Add-ValidationError "Trace has a missing or duplicate request ID: '$id'."
        } else {
            $expectedRequests[$id] = $row
        }
        Require-Number $row.arrival_s "trace request $id arrival_s"
        if ($row.arrival_s -lt $lastArrival -or $row.arrival_s -gt 3600) {
            Add-ValidationError "Trace arrival is out of order or out of range: '$id'."
        }
        $lastArrival = $row.arrival_s
        if ($row.arrival_s * $arrivalScale -gt 3600) { Add-ValidationError 'Scaled trace duration exceeds 3600 seconds.' }
        Require-Integer (Optional-Value $row.request 'max_tokens' 64) 1 1048576 "trace request $id max_tokens"
        if ((Optional-Value $row.request 'ignore_eos' $false) -isnot [bool]) {
            Add-ValidationError "Trace request $id ignore_eos must be boolean."
        }
    }
    $checks.Add('trace_identity_and_request_set')

    $expectedReports = @{}
    $trials = $manifestData.protocol.trials_per_variant
    Require-Integer $trials 1 20 'protocol.trials_per_variant'
    $variants = @($manifestData.comparison.variants)
    if ($variants.Count -ne 2 -or $variants -cnotcontains 'mixed' -or $variants -cnotcontains 'prefill_first') {
        Add-ValidationError 'Policy comparison requires mixed and prefill_first.'
    }
    Require-Equal @($manifestData.reports).Count (2 * $trials) 'manifest report count'
    $order = 0
    foreach ($spec in @($manifestData.reports)) {
        $file = [string]$spec.file
        Require-Integer $spec.trial 0 ($trials - 1) "$file trial"
        Require-Equal $file "$($spec.variant)-$($spec.trial).json" "$file filename"
        Require-Equal $spec.order $order "$file order"
        $trial = [int][math]::Floor($order / 2)
        $expectedVariant = if (($trial + $order + $orderOffset) % 2 -eq 0) { 'mixed' } else { 'prefill_first' }
        Require-Equal $spec.trial $trial "$file alternating trial"
        Require-Equal $spec.variant $expectedVariant "$file alternating variant"
        ++$order
        if ($file -notmatch '^(mixed|prefill_first)-[0-9]+\.json$' -or $expectedReports.ContainsKey($file)) {
            Add-ValidationError "Manifest contains an invalid or duplicate report file: '$file'."
        } else {
            $expectedReports[$file] = $spec
        }
    }
    for ($trial = 0; $trial -lt $trials; ++$trial) {
        foreach ($variant in @('mixed', 'prefill_first')) {
            if (-not $expectedReports.ContainsKey("$variant-$trial.json")) {
                Add-ValidationError "Manifest is missing $variant-$trial.json."
            }
        }
    }
    $actualReportFiles = @(Get-ChildItem -LiteralPath $Directory -File -Filter '*.json' |
        Where-Object Name -Match '^(mixed|prefill_first)-[0-9]+\.json$')
    Require-Equal $actualReportFiles.Count $expectedReports.Count 'report count'
    foreach ($file in $actualReportFiles) {
        if (-not $expectedReports.ContainsKey($file.Name)) { Add-ValidationError "Unexpected report: $($file.Name)." }
    }
    foreach ($file in $expectedReports.Keys) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $file) -PathType Leaf)) {
            Add-ValidationError "Missing report: $file."
        }
    }

    $allowedOutcomes = @($manifestData.protocol.allowed_request_outcomes)
    if ($allowedOutcomes -cnotcontains 'success') { Add-ValidationError 'The manifest must allow successful requests.' }
    foreach ($outcome in $allowedOutcomes) {
        if (@('success', 'queue_full', 'timeout', 'cancelled', 'backpressure') -cnotcontains $outcome) {
            Add-ValidationError "Invalid allowed request outcome: '$outcome'."
        }
    }
    $modelName = [System.IO.Path]::GetFileNameWithoutExtension([string]$manifestData.model.file)
    $reportRecords = @()
    $baselineSnapshot = $null
    foreach ($file in $expectedReports.Keys | Sort-Object) {
        $path = Join-Path $Directory $file
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { continue }
        try { $report = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json } catch {
            Add-ValidationError "$file is invalid JSON: $($_.Exception.Message)"
            continue
        }
        $spec = $expectedReports[$file]
        Require-Equal $report.schema_version 1 "$file schema_version"
        Require-Equal $report.benchmark 'llmserve-open-loop' "$file benchmark"
        Require-Equal $report.clock 'steady_clock' "$file clock"
        Require-Equal $report.percentile_method 'linear' "$file percentile_method"
        Require-Equal $report.trace_fnv1a64 $manifestData.trace.fnv1a64 "$file trace_fnv1a64"
        Require-Integer $report.started_at_unix_ms 1 ([long]::MaxValue) "$file started_at_unix_ms"
        $identity = $report.run_identity
        Require-Equal $identity.run_id $manifestData.run_id "$file run_id"
        Require-Equal $identity.trial $spec.trial "$file trial"
        Require-Equal $identity.variant $spec.variant "$file variant"
        Require-Equal ([string]$identity.trace_sha256).ToLowerInvariant() ([string]$manifestData.trace.sha256).ToLowerInvariant() "$file trace_sha256"
        Require-Equal ([string]$identity.manifest_sha256).ToLowerInvariant() $manifestHash "$file manifest_sha256"
        Require-Equal ([string]$identity.model_sha256).ToLowerInvariant() ([string]$manifestData.model.sha256).ToLowerInvariant() "$file model_sha256"
        Require-Equal $identity.server_sha256 $manifestData.binaries.server.sha256 "$file server_sha256"
        Require-Equal $identity.client_sha256 $manifestData.binaries.benchmark_client.sha256 "$file client_sha256"
        if ($report.warmup -isnot [bool]) { Add-ValidationError "$file warmup must be boolean." }
        Require-Equal $report.warmup $manifestData.protocol.warmup "$file warmup"
        Require-Equal (Optional-Value $report 'arrival_scale' 1.0) $arrivalScale "$file arrival_scale"

        foreach ($snapshotName in @('server_before', 'server_after')) {
            $snapshot = $report.$snapshotName
            if ($snapshot.ready -isnot [bool] -or -not $snapshot.ready) {
                Add-ValidationError "$file $snapshotName is not ready."
            }
            if ($null -eq $baselineSnapshot) { $baselineSnapshot = $snapshot }
            foreach ($field in @('device', 'gpu')) {
                Require-Equal $snapshot.$field $baselineSnapshot.$field "$file $snapshotName.$field"
            }
            Require-Equal $snapshot.backend $manifestData.engine.metrics_backend "$file $snapshotName.backend"
            Require-Equal $snapshot.model $modelName "$file $snapshotName.model"
            Require-Equal $snapshot.policy $spec.variant "$file $snapshotName.policy"
            Require-Equal (Optional-Value $snapshot 'telemetry_mode' 'off') $telemetryMode "$file $snapshotName.telemetry_mode"
            Require-Equal (Optional-Value $snapshot 'telemetry_capacity' 1024) `
                (Optional-Value $manifestData.engine 'telemetry_capacity' 1024) "$file $snapshotName.telemetry_capacity"
            foreach ($field in @('threads', 'gpu_layers', 'context_tokens', 'max_model_len', 'batch_tokens',
                'prefill_chunk', 'max_active', 'queue_capacity', 'prefix_cache_entries',
                'prefix_cache_tokens', 'event_buffer_size', 'aging_ms', 'admission_reserve_ms')) {
                Require-Equal $snapshot.$field $manifestData.engine.$field "$file $snapshotName.$field"
            }
            Require-Equal $snapshot.kernel_mode $manifestData.engine.metrics_kernel_mode "$file $snapshotName.kernel_mode"
            Require-Equal $snapshot.kv_credits.block_size $manifestData.engine.block_size "$file $snapshotName.kv_credits.block_size"
            Require-Equal $snapshot.llama_commit $manifestData.dependencies.llama_commit "$file $snapshotName.llama_commit"
            foreach ($field in @('outstanding_requests', 'active_requests', 'waiting_requests')) {
                Require-Equal $snapshot.$field 0 "$file $snapshotName.$field"
            }
            Require-Equal $snapshot.kv_credits.active_unique_blocks 0 "$file $snapshotName.kv_active_unique_blocks"
        }
        $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
        $successCount = 0
        $failedCount = 0
        $goodCount = 0
        $outputTokens = 0
        $elapsed = 0.0
        $ttfts = @()
        $tpots = @()
        $e2es = @()
        $lags = @()
        $itls = @()
        foreach ($request in @($report.requests)) {
            $id = [string]$request.id
            if ([string]::IsNullOrWhiteSpace($id) -or -not $seen.Add($id)) {
                Add-ValidationError "$file has a missing or duplicate request ID: '$id'."
                continue
            }
            if (-not $expectedRequests.ContainsKey($id)) {
                Add-ValidationError "$file has an unexpected request ID: '$id'."
                continue
            }
            $traceRequest = $expectedRequests[$id]
            Require-Equal $request.class_name (Optional-Value $traceRequest 'class_name' 'default') "$file request $id class_name"
            if ($request.success -isnot [bool] -or $request.within_slo -isnot [bool]) {
                Add-ValidationError "$file request $id success and within_slo must be boolean."
            }
            foreach ($field in @('dispatch_lag_ms', 'e2e_ms', 'finished_s')) {
                Require-Number $request.$field "$file request $id $field"
            }
            Require-Close $request.finished_s ($traceRequest.arrival_s * $arrivalScale +
                ($request.dispatch_lag_ms + $request.e2e_ms) / 1000) "$file request $id finished_s"
            $elapsed = [math]::Max($elapsed, $request.finished_s)
            $lags += $request.dispatch_lag_ms
            if ($request.token_ids -isnot [array] -or $request.token_times_ms -isnot [array]) {
                Add-ValidationError "$file request $id token IDs and timestamps must be arrays."
            }
            foreach ($token in @($request.token_ids)) {
                Require-Integer $token 0 ([int]::MaxValue) "$file request $id token"
            }
            $previousTime = 0.0
            foreach ($time in @($request.token_times_ms)) {
                Require-Number $time "$file request $id token timestamp"
                if ($time -lt $previousTime -or $time -gt $request.e2e_ms) {
                    Add-ValidationError "$file request $id has out-of-order or out-of-range token timestamps."
                }
                $previousTime = $time
            }
            Require-Equal @($request.token_times_ms).Count @($request.token_ids).Count "$file request $id token timestamp count"
            $outcome = if ($request.success) { 'success' } else { [string]$request.error }
            if ([string]::IsNullOrWhiteSpace($outcome) -or $allowedOutcomes -cnotcontains $outcome) {
                Add-ValidationError "$file request $id has disallowed outcome '$outcome'."
            }
            if ($request.success) {
                ++$successCount
                Require-Equal $request.http_status 200 "$file request $id http_status"
                Require-Equal $request.sse_done_count 1 "$file request $id sse_done_count"
                Require-Equal $request.terminal_events 1 "$file request $id terminal_events"
                if (-not [string]::IsNullOrEmpty([string]$request.error)) {
                    Add-ValidationError "$file request $id is successful but has an error."
                }
                $tokenCount = @($request.token_ids).Count
                if ($tokenCount -eq 0) { Add-ValidationError "$file request $id has no output tokens." }
                $maximum = Optional-Value $traceRequest.request 'max_tokens' 64
                if ($tokenCount -gt $maximum) { Add-ValidationError "$file request $id exceeded max_tokens." }
                if ($request.finish_reason -ceq 'length') {
                    Require-Equal $tokenCount $maximum "$file request $id complete output length"
                } elseif ($request.finish_reason -cne 'stop' -or
                    (Optional-Value $traceRequest.request 'ignore_eos' $false)) {
                    Add-ValidationError "$file request $id has an invalid finish reason."
                }
                Require-Equal $request.usage.completion_tokens $tokenCount "$file request $id completion token count"
                Require-Integer $request.usage.prompt_tokens 1 1048576 "$file request $id prompt token count"
                Require-Equal $request.usage.total_tokens ($request.usage.prompt_tokens + $tokenCount) "$file request $id total token count"
                if ($traceRequest.request.prompt -is [array]) {
                    Require-Equal $request.usage.prompt_tokens $traceRequest.request.prompt.Count "$file request $id prompt token count"
                }
                Require-Close $request.ttft_ms $request.token_times_ms[0] "$file request $id ttft_ms"
                $tpot = 0.0
                if ($tokenCount -gt 1) {
                    $tpot = ($request.token_times_ms[-1] - $request.token_times_ms[0]) / ($tokenCount - 1)
                    Require-Close $request.mean_tpot_ms $tpot "$file request $id mean_tpot_ms"
                    $tpots += $tpot
                } elseif ($null -ne $request.mean_tpot_ms) {
                    Add-ValidationError "$file request $id single-token TPOT must be null."
                }
                $withinSlo = $request.ttft_ms -le (Optional-Value $traceRequest 'ttft_slo_ms' 1500) -and
                    $tpot -le (Optional-Value $traceRequest 'tpot_slo_ms' 100)
                Require-Equal $request.within_slo $withinSlo "$file request $id within_slo"
                if ($withinSlo) { ++$goodCount }
                $outputTokens += $tokenCount
                $ttfts += $request.ttft_ms
                $e2es += $request.e2e_ms
                for ($i = 1; $i -lt $tokenCount; ++$i) {
                    $itls += $request.token_times_ms[$i] - $request.token_times_ms[$i - 1]
                }
            } else {
                ++$failedCount
                if ($request.within_slo) { Add-ValidationError "$file request $id failed but is marked within SLO." }
                if ($request.http_status -eq 200) {
                    Require-Equal $request.sse_done_count 1 "$file failed request $id sse_done_count"
                    Require-Equal $request.terminal_events 1 "$file failed request $id terminal_events"
                } elseif ($request.http_status -notin @(408, 429, 499, 503)) {
                    Add-ValidationError "$file request $id has an invalid failure HTTP status."
                }
            }
        }
        foreach ($id in $expectedRequests.Keys) {
            if (-not $seen.Contains($id)) { Add-ValidationError "$file is missing request '$id'." }
        }
        Require-Equal $seen.Count $expectedRequests.Count "$file unique request count"
        Require-Equal $report.summary.requests $expectedRequests.Count "$file summary.requests"
        Require-Equal $report.summary.successful $successCount "$file summary.successful"
        Require-Equal $report.summary.failed $failedCount "$file summary.failed"
        Require-Equal $report.summary.slo_compliant $goodCount "$file summary.slo_compliant"
        Require-Equal $report.summary.successful_output_tokens $outputTokens "$file summary.successful_output_tokens"
        Require-Close $report.summary.elapsed_s $elapsed "$file summary.elapsed_s"
        if ($elapsed -le 0) { Add-ValidationError "$file has no positive elapsed time." } else {
            Require-Close $report.summary.output_tokens_per_second ($outputTokens / $elapsed) "$file output throughput"
            Require-Close $report.summary.goodput_requests_per_second ($goodCount / $elapsed) "$file goodput"
        }
        Check-Percentiles $report.summary.ttft_ms $ttfts "$file summary.ttft_ms"
        Check-Percentiles $report.summary.mean_tpot_ms $tpots "$file summary.mean_tpot_ms"
        Check-Percentiles $report.summary.inter_token_ms $itls "$file summary.inter_token_ms"
        Check-Percentiles $report.summary.e2e_ms $e2es "$file summary.e2e_ms"
        Check-Percentiles $report.summary.dispatch_lag_ms $lags "$file summary.dispatch_lag_ms"
        $reportRecords += [pscustomobject]@{ file = $file; spec = $spec; data = $report }
    }
    $checks.Add('report_identity_configuration_and_terminality')

    $referenceSpec = $manifestData.comparison.reference
    Require-Integer $referenceSpec.trial 0 ($trials - 1) 'comparison.reference.trial'
    $comparedOutputs = 0
    $referenceRequests = 0
    $referenceRecord = $reportRecords | Where-Object {
        $_.spec.variant -ceq $referenceSpec.variant -and "$($_.spec.trial)" -ceq "$($referenceSpec.trial)"
    } | Select-Object -First 1
    if ($null -eq $referenceRecord) {
        Add-ValidationError 'The declared output reference report is missing.'
    } else {
        $referenceTokens = [System.Collections.Generic.Dictionary[string,string]]::new([System.StringComparer]::Ordinal)
        foreach ($request in @($referenceRecord.data.requests)) {
            if ($request.success) { $referenceTokens[[string]$request.id] = Token-Key $request.token_ids }
        }
        $referenceRequests = $referenceTokens.Count
        foreach ($record in $reportRecords) {
            foreach ($request in @($record.data.requests)) {
                $id = [string]$request.id
                if ($request.success) {
                    if (-not $referenceTokens.ContainsKey($id)) {
                        Add-ValidationError "$($record.file) request $id has no successful declared output reference."
                    } elseif ((Token-Key $request.token_ids) -cne $referenceTokens[$id]) {
                        Add-ValidationError "$($record.file) request $id output tokens differ from the declared reference."
                    } else {
                        ++$comparedOutputs
                    }
                }
            }
        }
    }
    $checks.Add('deterministic_output_tokens')

    $validation = [ordered]@{
        schema_version = 1
        run_id = $manifestData.run_id
        valid = $errors.Count -eq 0
        checked_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        manifest_sha256 = $manifestHash
        checks = @($checks)
        reports_expected = $expectedReports.Count
        reports_checked = $reportRecords.Count
        requests_per_report = $expectedRequests.Count
        output_reference_successful_requests = $referenceRequests
        successfully_compared_requests = $comparedOutputs
        errors = @($errors)
    }
    if ($errors.Count -gt 0) {
        $preview = @($errors | Select-Object -First 10) -join [Environment]::NewLine
        throw "Benchmark validation failed with $($errors.Count) error(s):$([Environment]::NewLine)$preview"
    }

    $policies = [ordered]@{}
    foreach ($group in ($reportRecords.data | Group-Object { $_.server_before.policy })) {
        $rows = @($group.Group)
        $policies[$group.Name] = [ordered]@{
            trials = $rows.Count
            successful = ($rows.summary.successful | Measure-Object -Sum).Sum
            failed = ($rows.summary.failed | Measure-Object -Sum).Sum
            output_tokens_per_second_median = Median @($rows.summary.output_tokens_per_second)
            ttft_p95_ms_median = Median @($rows | ForEach-Object { if ($null -ne $_.summary.ttft_ms) { $_.summary.ttft_ms.p95 } })
            mean_tpot_p95_ms_median = Median @($rows | ForEach-Object { if ($null -ne $_.summary.mean_tpot_ms) { $_.summary.mean_tpot_ms.p95 } })
            goodput_requests_per_second_median = Median @($rows.summary.goodput_requests_per_second)
        }
    }
    $summary = [ordered]@{
        schema_version = 1
        run_id = $manifestData.run_id
        manifest_sha256 = $manifestHash
        validation = 'passed'
        backend = $manifestData.engine.metrics_backend
        model = $modelName
        trace_sha256 = $manifestData.trace.sha256
        requests_per_trial = $expectedRequests.Count
        policies = $policies
    }
    Publish-BenchmarkOutputs $Directory ([ordered]@{ 'summary.json' = $summary; 'validation-summary.json' = $validation })
    $failurePath = Join-Path $Directory 'analysis-failure.json'
    if (Test-Path -LiteralPath $failurePath) { Remove-Item -LiteralPath $failurePath }
    $summary | ConvertTo-Json -Depth 8
} catch {
    if ($errors.Count -eq 0) { $errors.Add($_.Exception.Message) }
    Write-BenchmarkJson (Join-Path $Directory 'analysis-failure.json') ([ordered]@{
        schema_version = 1; run_id = $runId; valid = $false
        checked_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        manifest_sha256 = $manifestHash; checks = @($checks); errors = @($errors)
    })
    throw
}
