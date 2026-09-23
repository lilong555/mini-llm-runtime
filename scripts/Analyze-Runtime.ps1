param([Parameter(Mandatory = $true)][string]$Directory)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'Benchmark-Common.ps1')
$Directory = (Resolve-Path -LiteralPath $Directory).Path
$outputs = @('validation-summary.json', 'runtime-prefill.json', 'runtime-decode.json',
    'runtime-mixed.json', 'forward-stages.json', 'profiler-overhead.json')
foreach ($name in $outputs) {
    $path = Join-Path $Directory $name
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
}

function Equal($Actual, $Expected, [string]$Label) {
    if ($null -eq $Actual -or $null -eq $Expected -or "$Actual" -cne "$Expected" -or
        (($Expected -is [int] -or $Expected -is [long]) -and $Actual -isnot [int] -and $Actual -isnot [long]) -or
        (($Expected -is [bool]) -and $Actual -isnot [bool])) {
        throw "$Label differs: expected '$Expected', got '$Actual'."
    }
}

function Integer($Value, [long]$Minimum, [long]$Maximum, [string]$Label) {
    if (($Value -isnot [int] -and $Value -isnot [long]) -or $Value -lt $Minimum -or $Value -gt $Maximum) {
        throw "$Label must be an integer in [$Minimum, $Maximum]."
    }
}

function Hash($Value, [string]$Label) {
    if ($Value -isnot [string] -or $Value -cnotmatch '^[0-9a-f]{64}$') { throw "$Label is not SHA-256." }
}

function Same-Json($Actual, $Expected, [string]$Label) {
    if ($null -eq $Expected) {
        if ($null -ne $Actual) { throw "$Label must be null." }
    } elseif ($Expected -is [System.Collections.IDictionary] -or $Expected -is [pscustomobject]) {
        $keys = if ($Expected -is [System.Collections.IDictionary]) { @($Expected.Keys) } else {
            @($Expected.PSObject.Properties.Name)
        }
        $actualKeys = if ($Actual -is [System.Collections.IDictionary]) { @($Actual.Keys) } else {
            @($Actual.PSObject.Properties.Name)
        }
        if (@(Compare-Object ($keys | Sort-Object -CaseSensitive) ($actualKeys | Sort-Object -CaseSensitive) -CaseSensitive).Count) {
            throw "$Label has different fields."
        }
        foreach ($key in $keys) { Same-Json $Actual.$key $Expected.$key "$Label.$key" }
    } elseif ($Expected -is [array]) {
        if ($Actual -isnot [array] -or $Actual.Count -ne $Expected.Count) { throw "$Label has different array length." }
        for ($i = 0; $i -lt $Expected.Count; ++$i) { Same-Json $Actual[$i] $Expected[$i] "$Label[$i]" }
    } else {
        if ($Expected -is [int] -or $Expected -is [long]) {
            Integer $Actual ([long]::MinValue) ([long]::MaxValue) $Label
        }
        Equal $Actual $Expected $Label
    }
}

function Artifact([string]$Name) {
    if (-not $Name -or [IO.Path]::IsPathRooted($Name) -or $Name -match '(^|[/\\])\.\.([/\\]|$)') {
        throw "Invalid artifact path: $Name"
    }
    return (Resolve-Path -LiteralPath (Join-Path $Directory $Name)).Path
}

function Median($Values) {
    $sorted = @($Values | Sort-Object)
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2
}

function Check-Snapshot($Source) {
    Equal (Get-LowerSha256 (Artifact $Source.state_file)) $Source.worktree_state_sha256 'source state hash'
    Equal (Get-LowerSha256 (Artifact $Source.snapshot.path)) $Source.snapshot.sha256 'source snapshot hash'
    $state = Get-Content -Raw -LiteralPath (Artifact $Source.state_file) | ConvertFrom-Json
    Same-Json $state.scope $Source.scope 'source scope'
    if ($state.files.Count -eq 0) { throw 'Empty source snapshot.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead((Artifact $Source.snapshot.path))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        Equal $zip.Entries.Count $state.files.Count 'source snapshot entries'
        $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($file in $state.files) {
            if (-not $seen.Add($file.path)) { throw 'Duplicate source snapshot path.' }
            $entry = $zip.GetEntry($file.path)
            if ($null -eq $entry) { throw "Missing source snapshot entry: $($file.path)" }
            Equal $entry.Length $file.size_bytes 'source snapshot size'
            $stream = $entry.Open()
            try { $digest = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
            finally { $stream.Dispose() }
            Equal $digest $file.sha256 'source snapshot entry hash'
        }
    } finally {
        $sha.Dispose()
        $zip.Dispose()
    }
}

function Stage-Contract([string]$Name, [int]$Layer, [long]$Count, [long]$Logits, $Dims) {
    $m = 0L; $n = 0L; $k = 0L; $tasks = 0L; $grain = 0L
    switch ($Name) {
        { $_ -in @('attention_norm', 'ffn_norm', 'qk_norm_rope_kv') } { $tasks = $Count; $grain = 1 }
        'attention' { $tasks = $Count * $Dims.heads; $grain = 1 }
        'swiglu' { $tasks = $Count * $Dims.feed_forward; $grain = 256 }
        'query_projection' { $n = $Dims.heads * $Dims.head_dim; $k = $Dims.embedding }
        { $_ -in @('key_projection', 'value_projection') } { $n = $Dims.kv_heads * $Dims.head_dim; $k = $Dims.embedding }
        'output_projection' { $n = $Dims.embedding; $k = $Dims.heads * $Dims.head_dim }
        { $_ -in @('gate_projection', 'up_projection') } { $n = $Dims.feed_forward; $k = $Dims.embedding }
        'down_projection' { $n = $Dims.embedding; $k = $Dims.feed_forward }
        'lm_head' { $n = $Dims.vocabulary; $k = $Dims.embedding }
    }
    if ($n) { $m = $Count; $tasks = $n; $grain = 16 }
    return [ordered]@{ stage = $Name; layer = $Layer; input_tokens = $Count; logits_tokens = $Logits
        matrix_m = $m; matrix_n = $n; matrix_k = $k; count = $tasks; grain = $grain }
}

function Check-Profile($Profile, $Sample, $Contracts, [long]$Threads, $Work) {
    Equal $Profile.completed $true 'profile completed'
    Equal $Profile.batch_id $Sample.batch_id 'profile batch_id'
    Equal $Profile.threads $Threads 'profile threads'
    Integer $Profile.wall_ns 1 $Sample.wall_ns 'profile wall_ns'
    Integer $Profile.unaccounted_ns 0 $Profile.wall_ns 'profile unaccounted_ns'
    $inputCount = $Work.prefill_tokens + $Work.decode_sequences
    $sequences = $Work.decode_sequences + [int]($Work.prefill_tokens -gt 0)
    Equal $Profile.input_tokens $inputCount 'profile input_tokens'
    Equal $Profile.logits_tokens $sequences 'profile logits_tokens'
    Equal $Profile.sequences $sequences 'profile sequences'
    Equal $Profile.context_before_sum ($Work.decode_sequences * $Work.kv_tokens) 'profile context_before_sum'
    Equal $Profile.context_before_max $Work.kv_tokens 'profile context_before_max'
    Equal $Profile.context_after_sum ($Work.decode_sequences * $Work.kv_tokens + $inputCount) 'profile context_after_sum'
    $maximum = [math]::Max($Work.prefill_tokens, $(if ($Work.decode_sequences) { $Work.kv_tokens + 1 } else { 0 }))
    Equal $Profile.context_after_max $maximum 'profile context_after_max'
    foreach ($field in @('kv_pages_before', 'kv_pages_after')) { Equal $Profile.$field $Sample.$field "profile $field" }
    Equal $Profile.stages.Count $Contracts.Count 'profile stage count'
    $sum = 0L
    for ($i = 0; $i -lt $Contracts.Count; ++$i) {
        $actual = $Profile.stages[$i]
        $expected = $Contracts[$i]
        foreach ($field in @('stage', 'layer', 'input_tokens', 'logits_tokens', 'matrix_m', 'matrix_n', 'matrix_k')) {
            Equal $actual.$field $expected.$field "stage $i $field"
        }
        Equal $actual.batch_id $Profile.batch_id 'stage batch_id'
        Equal $actual.threads $Threads 'stage threads'
        Integer $actual.wall_ns 0 $Profile.wall_ns 'stage wall_ns'
        $sum += $actual.wall_ns
        $p = $actual.parallel
        if (-not $expected.count) {
            if ($null -ne $p) { throw 'Unexpected parallel profile for serial stage.' }
            continue
        }
        Equal $p.completed $true 'parallel completed'
        Equal $p.threads $Threads 'parallel threads'
        Equal $p.count $expected.count 'parallel count'
        Equal $p.grain $expected.grain 'parallel grain'
        $chunks = [long][math]::Ceiling($expected.count / [double]$expected.grain)
        Equal $p.chunks $chunks 'parallel chunks'
        Integer $p.participating_threads 1 ([math]::Min($Threads, $chunks)) 'parallel participating_threads'
        Integer $p.wall_ns 0 $actual.wall_ns 'parallel wall_ns'
        foreach ($field in @('dispatch_ns', 'caller_work_ns', 'caller_wait_ns', 'worker_work_max_ns', 'worker_start_delay_max_ns')) {
            Integer $p.$field 0 $p.wall_ns "parallel $field"
        }
        Integer $p.worker_work_sum_ns 0 ($p.wall_ns * ($Threads - 1)) 'parallel worker_work_sum_ns'
        if ($p.dispatch_ns + $p.caller_work_ns + $p.caller_wait_ns -gt $p.wall_ns) {
            throw 'Overlapping caller parallel time.'
        }
    }
    Equal ($sum + $Profile.unaccounted_ns) $Profile.wall_ns 'forward time accounting'
}

$runId = $null
$manifestHash = $null
try {
    $manifestPath = Artifact 'manifest.json'
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    $manifestHash = Get-LowerSha256 $manifestPath
    $runId = $manifest.run_id
    Equal $manifest.schema_version 1 'manifest schema'
    Equal $manifest.benchmark 'minillm-runtime' 'manifest benchmark'
    if (-not $runId -or $manifest.source.git_sha -cnotmatch '^[0-9a-f]{40}$' -or
        $manifest.source.git_dirty -isnot [bool] -or $manifest.dependencies.llama_commit -cnotmatch '^[0-9a-f]{40}$') {
        throw 'Invalid runtime source identity.'
    }
    Check-Snapshot $manifest.source
    Hash $manifest.binary.sha256 'binary hash'
    Hash $manifest.model.sha256 'model hash'
    foreach ($field in @('file', 'sha256', 'size_bytes')) {
        Equal $manifest.model.$field $manifest.model.provenance.$field "model provenance $field"
    }
    foreach ($field in @('type', 'generator', 'compiler', 'compiler_id', 'compiler_version')) {
        if ([string]::IsNullOrWhiteSpace($manifest.build.$field)) { throw "Missing build $field." }
    }
    if ($manifest.build.type -notin @('Release', 'RelWithDebInfo')) { throw 'Invalid measurement build type.' }
    foreach ($field in @('cxx_flags', 'configuration_flags', 'linker_flags', 'configuration_linker_flags', 'cuda_enabled')) {
        if ($null -eq $manifest.build.$field) { throw "Missing build $field." }
    }
    if (-not $manifest.environment.os -or -not $manifest.environment.processor) { throw 'Missing CPU or OS identity.' }
    Integer $manifest.environment.logical_processors 1 65536 'logical processors'
    $inputPath = Artifact $manifest.input.path
    Equal (Get-LowerSha256 $inputPath) $manifest.input.sha256 'input hash'
    $inputSpec = Get-Content -Raw -LiteralPath $inputPath | ConvertFrom-Json
    Equal $inputSpec.schema_version 1 'input schema'
    Same-Json $manifest.runtime $inputSpec.runtime 'runtime config'
    $cfg = $inputSpec.runtime
    Integer $cfg.context_tokens 1 1048576 'context_tokens'
    Integer $cfg.page_tokens 1 256 'page_tokens'
    Integer $cfg.max_sequences 2 256 'max_sequences'
    Integer $cfg.batch_tokens 1 $cfg.context_tokens 'batch_tokens'
    Equal ($cfg.context_tokens % $cfg.page_tokens) 0 'page alignment'
    $protocol = $manifest.protocol
    Integer $protocol.trials 1 20 'trials'
    Integer $protocol.repeats 1 100 'repeats'
    Integer $protocol.warmup 1 100 'warmup'
    Same-Json $protocol.profiler_modes @('none', 'stages') 'profiler modes'
    Same-Json $protocol.allowed_changes @('threads', 'profiler') 'allowed changes'
    Equal $protocol.reference_model_resident $false 'reference model resident'
    Equal $protocol.order 'alternating_processes_and_thread_order' 'process order'
    Equal $protocol.setup 'pinned_prefix_share_reset' 'setup'
    Equal $protocol.sampling 'greedy_argmax_separate' 'sampling'
    Equal $protocol.activation_dtype 'F32' 'activation dtype'
    Equal $protocol.kv_dtype 'F16' 'KV dtype'
    Equal $protocol.logits_digest 'sha256_seq_i32le_length_u32le_logits_f32le' 'logits digest'
    if ($protocol.kernel -cnotin @('auto', 'scalar')) { throw 'Invalid kernel mode.' }
    $threadSet = [Collections.Generic.HashSet[int]]::new()
    foreach ($count in $protocol.threads) {
        Integer $count 1 256 'thread count'
        if (-not $threadSet.Add($count)) { throw 'Duplicate thread count.' }
    }
    if ($threadSet.Count -eq 0 -or $inputSpec.workloads.Count -eq 0 -or $inputSpec.token_ids.Count -eq 0) {
        throw 'Empty runtime experiment.'
    }
    $names = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($work in $inputSpec.workloads) {
        if ($work.name -cnotmatch '^[A-Za-z0-9_.-]{1,64}$' -or -not $names.Add($work.name)) {
            throw 'Duplicate or invalid workload name.'
        }
        Integer $work.prefill_tokens 0 $cfg.batch_tokens 'prefill_tokens'
        Integer $work.decode_sequences 0 ($cfg.max_sequences - 1) 'decode_sequences'
        Integer $work.kv_tokens 0 $cfg.context_tokens 'kv_tokens'
        if ($work.mode -cnotin @('prefill', 'decode', 'mixed') -or
            (($work.mode -cne 'decode') -ne ($work.prefill_tokens -gt 0)) -or
            (($work.mode -cne 'prefill') -ne ($work.decode_sequences -gt 0 -and $work.kv_tokens -gt 0)) -or
            ($work.mode -ceq 'prefill' -and ($work.decode_sequences -ne 0 -or $work.kv_tokens -ne 0)) -or
            $work.prefill_tokens + $work.decode_sequences -gt $cfg.batch_tokens -or
            $work.decode_sequences + [int]($work.prefill_tokens -gt 0) -ge $cfg.max_sequences) {
            throw 'Inconsistent workload recipe.'
        }
    }
    Equal $manifest.reports.Count ($threadSet.Count * 2 * $protocol.trials) 'manifest report count'
    $files = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $variants = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $signatures = [Collections.Generic.Dictionary[string,object]]::new([StringComparer]::Ordinal)
    $rows = [Collections.Generic.List[object]]::new()
    $stageRows = [Collections.Generic.List[object]]::new()
    $dimensions = $null
    $effectiveKernel = $null
    $reportHashes = @()
    $layerStages = @('attention_norm', 'query_projection', 'key_projection', 'value_projection',
        'qk_norm_rope_kv', 'attention', 'output_projection', 'attention_residual', 'ffn_norm',
        'gate_projection', 'up_projection', 'swiglu', 'down_projection', 'ffn_residual')
    foreach ($spec in $manifest.reports) {
        if (-not $files.Add($spec.file) -or -not $variants.Add("$($spec.threads):$($spec.profiler):$($spec.trial)")) {
            throw 'Duplicate runtime report.'
        }
        Integer $spec.trial 0 ($protocol.trials - 1) 'report trial'
        Equal $spec.order ($files.Count - 1) 'report order'
        if (-not $threadSet.Contains($spec.threads) -or $spec.profiler -cnotin @('none', 'stages')) {
            throw 'Undeclared runtime variant.'
        }
        $path = Artifact $spec.file
        $report = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
        $reportHashes += [ordered]@{ file = $spec.file; sha256 = Get-LowerSha256 $path }
        Equal $report.schema_version 1 'report schema'
        Equal $report.benchmark 'minillm-runtime' 'report benchmark'
        Equal $report.status 'passed' 'report status'
        Equal $report.trial $spec.trial 'report trial identity'
        Equal $report.run_identity.run_id $runId 'run_id'
        Equal $report.run_identity.manifest_sha256 $manifestHash 'manifest_sha256'
        Equal $report.run_identity.model_sha256 $manifest.model.sha256 'model_sha256'
        Equal $report.run_identity.binary_sha256 $manifest.binary.sha256 'binary_sha256'
        Equal $report.input_sha256 $manifest.input.sha256 'input_sha256'
        foreach ($field in @('context_tokens', 'page_tokens', 'max_sequences', 'batch_tokens')) {
            Equal $report.runtime.$field $cfg.$field "runtime $field"
        }
        Equal $report.runtime.threads $spec.threads 'runtime threads'
        Equal $report.runtime.kernel $protocol.kernel 'runtime kernel'
        if ($null -eq $effectiveKernel) { $effectiveKernel = $report.runtime.effective_kernel }
        Equal $report.runtime.effective_kernel $effectiveKernel 'effective kernel'
        if ($effectiveKernel -cnotin @('scalar', 'avx2-fma-f16c') -or
            ($protocol.kernel -ceq 'scalar' -and $effectiveKernel -cne 'scalar')) {
            throw 'Invalid effective kernel.'
        }
        Equal $report.protocol.profiler $spec.profiler 'profiler identity'
        Equal $report.protocol.clock 'steady_clock' 'clock'
        foreach ($field in @('warmup', 'repeats', 'setup', 'sampling', 'activation_dtype', 'kv_dtype', 'logits_digest')) {
            Equal $report.protocol.$field $protocol.$field "protocol $field"
        }
        if ($null -eq $dimensions) { $dimensions = $report.dimensions }
        Same-Json $report.dimensions $dimensions 'dimensions'
        foreach ($field in @('embedding', 'layers', 'heads', 'kv_heads', 'head_dim', 'feed_forward', 'vocabulary')) {
            Integer $dimensions.$field 1 1000000 "dimension $field"
        }
        Integer $dimensions.layers 1 256 'layers'
        foreach ($token in $inputSpec.token_ids) { Integer $token 0 ($dimensions.vocabulary - 1) 'input token ID' }
        Equal $report.workloads.Count $inputSpec.workloads.Count 'workload count'
        $batchId = 0L
        for ($w = 0; $w -lt $inputSpec.workloads.Count; ++$w) {
            $work = $inputSpec.workloads[$w]
            $actual = $report.workloads[$w]
            foreach ($field in @('name', 'mode', 'prefill_tokens', 'decode_sequences', 'kv_tokens')) {
                Equal $actual.$field $work.$field "workload $field"
            }
            $prefix = @()
            for ($i = 0; $i -lt $work.kv_tokens; ++$i) { $prefix += $inputSpec.token_ids[$i % $inputSpec.token_ids.Count] }
            Same-Json $actual.prefix_token_ids $prefix 'prefix token IDs'
            $tokens = @()
            for ($i = 0; $i -lt $work.prefill_tokens; ++$i) {
                $tokens += [ordered]@{ token = $inputSpec.token_ids[$i % $inputSpec.token_ids.Count]
                    position = $i; sequence = 1; logits = $i + 1 -eq $work.prefill_tokens; phase = 'prefill' }
            }
            $decodeStart = if ($work.prefill_tokens) { 2 } else { 1 }
            for ($i = 0; $i -lt $work.decode_sequences; ++$i) {
                $seq = $decodeStart + $i
                $tokens += [ordered]@{ token = $inputSpec.token_ids[($work.kv_tokens + $seq) % $inputSpec.token_ids.Count]
                    position = $work.kv_tokens; sequence = $seq; logits = $true; phase = 'decode' }
            }
            Same-Json $actual.input_tokens $tokens 'expanded input tokens'
            Equal $actual.setup_forward_calls ([math]::Ceiling($work.kv_tokens / [double]$cfg.batch_tokens)) 'setup calls'
            Integer $actual.setup_ns 0 ([long]::MaxValue) 'setup_ns'
            Equal $actual.used_pages_after_clear 0 'KV reclamation'
            Equal $actual.samples.Count $protocol.repeats 'sample count'
            $logits = $work.decode_sequences + [int]($work.prefill_tokens -gt 0)
            $contracts = @()
            foreach ($stage in @('kv_prepare', 'embedding', 'rope_prepare')) {
                $contracts += Stage-Contract $stage -1 $tokens.Count $logits $dimensions
            }
            for ($layer = 0; $layer -lt $dimensions.layers; ++$layer) {
                foreach ($stage in $layerStages) { $contracts += Stage-Contract $stage $layer $tokens.Count $logits $dimensions }
            }
            for ($i = 0; $i -lt $logits; ++$i) {
                foreach ($stage in @('final_norm', 'lm_head')) { $contracts += Stage-Contract $stage -1 1 1 $dimensions }
            }
            $batchId += $protocol.warmup
            for ($r = 0; $r -lt $protocol.repeats; ++$r) {
                $sample = $actual.samples[$r]
                Equal $sample.repeat $r 'sample repeat'
                Equal $sample.batch_id (++$batchId) 'sample batch_id'
                Integer $sample.wall_ns 1 ([long]::MaxValue) 'sample wall_ns'
                foreach ($field in @('sampling_ns', 'reset_ns', 'resident_bytes_before', 'resident_bytes_after')) {
                    Integer $sample.$field 0 ([long]::MaxValue) "sample $field"
                }
                $pagesBefore = [long][math]::Ceiling($work.kv_tokens / [double]$cfg.page_tokens)
                $pagesAfter = $pagesBefore + [long][math]::Ceiling($work.prefill_tokens / [double]$cfg.page_tokens) + $work.decode_sequences
                if ($pagesAfter -gt $cfg.context_tokens / $cfg.page_tokens) { throw 'Workload exceeds physical KV capacity.' }
                Equal $sample.kv_pages_before $pagesBefore 'KV pages before'
                Equal $sample.kv_pages_after $pagesAfter 'KV pages after'
                $pageBytes = 4L * $cfg.page_tokens * $dimensions.layers * $dimensions.kv_heads * $dimensions.head_dim
                Integer $sample.resident_bytes_before ($pagesBefore * $pageBytes) ($cfg.context_tokens / $cfg.page_tokens * $pageBytes) 'resident bytes before'
                Integer $sample.resident_bytes_after ([math]::Max($sample.resident_bytes_before, $pagesAfter * $pageBytes)) `
                    ($cfg.context_tokens / $cfg.page_tokens * $pageBytes) 'resident bytes after'
                Equal $sample.output.logits_vectors $logits 'output vector count'
                Hash $sample.output.logits_sha256 'logits SHA-256'
                Equal $sample.output.greedy_tokens.Count $logits 'greedy token count'
                for ($i = 0; $i -lt $logits; ++$i) {
                    Equal $sample.output.greedy_tokens[$i].sequence ($i + 1) 'greedy sequence'
                    Integer $sample.output.greedy_tokens[$i].token 0 ($dimensions.vocabulary - 1) 'greedy token'
                }
                if ($signatures.ContainsKey($work.name)) {
                    Same-Json $sample.output $signatures[$work.name] 'fixed-input output'
                } else { $signatures[$work.name] = $sample.output }
                if ($spec.profiler -ceq 'none') {
                    if ($null -ne $sample.profile) { throw 'Unexpected profile in uninstrumented measurement.' }
                } else {
                    Check-Profile $sample.profile $sample $contracts $spec.threads $work
                    foreach ($group in ($sample.profile.stages | Group-Object -Property stage)) {
                        $total = 0L; $wait = 0L; $caller = 0L; $workers = 0L; $parallel = 0L
                        foreach ($stage in $group.Group) {
                            $total += $stage.wall_ns
                            if ($null -ne $stage.parallel) {
                                $wait += $stage.parallel.caller_wait_ns
                                $caller += $stage.parallel.caller_work_ns
                                $workers += $stage.parallel.worker_work_sum_ns
                                $parallel += $stage.parallel.wall_ns
                            }
                        }
                        $stageRows.Add([pscustomobject]@{ workload = $work.name; threads = $spec.threads
                            stage = $group.Name; calls = $group.Count; wall_ns = $total
                            caller_wait_ns = $wait; caller_work_ns = $caller; worker_work_sum_ns = $workers
                            parallel_wall_ns = $parallel; forward_ns = $sample.profile.wall_ns })
                    }
                }
                $rows.Add([pscustomobject]@{ workload = $work.name; mode = $work.mode; threads = $spec.threads
                    profiler = $spec.profiler; trial = $spec.trial; repeat = $r; wall_ns = $sample.wall_ns
                    sampling_ns = $sample.sampling_ns; input_tokens = $tokens.Count; logits_tokens = $logits
                    forward_ns = $(if ($spec.profiler -ceq 'stages') { $sample.profile.wall_ns } else { $null })
                    unaccounted_ns = $(if ($spec.profiler -ceq 'stages') { $sample.profile.unaccounted_ns } else { $null }) })
            }
        }
    }
    $overhead = @()
    $modelRows = @()
    foreach ($group in ($rows | Group-Object -Property workload, threads)) {
        $off = @($group.Group | Where-Object profiler -CEQ 'none')
        $on = @($group.Group | Where-Object profiler -CEQ 'stages')
        $offMedian = Median $off.wall_ns
        $onMedian = Median $on.wall_ns
        $pairs = @()
        for ($trial = 0; $trial -lt $protocol.trials; ++$trial) {
            $a = Median @($off | Where-Object trial -EQ $trial | ForEach-Object wall_ns)
            $b = Median @($on | Where-Object trial -EQ $trial | ForEach-Object wall_ns)
            $pairs += [ordered]@{ trial = $trial; none_ns = $a; stages_ns = $b; ratio = $b / $a }
        }
        $overhead += [ordered]@{ workload = $off[0].workload; threads = $off[0].threads
            none_median_ns = $offMedian; stages_median_ns = $onMedian
            overhead_percent = 100 * ($onMedian / $offMedian - 1); paired_trials = $pairs
            profile_forward_median_ns = Median $on.forward_ns; unaccounted_median_ns = Median $on.unaccounted_ns }
        $modelRows += [ordered]@{ workload = $off[0].workload; mode = $off[0].mode; threads = $off[0].threads
            samples = $off.Count; wall_ns = @($off.wall_ns); median_ns = $offMedian
            min_ns = ($off.wall_ns | Measure-Object -Minimum).Minimum
            max_ns = ($off.wall_ns | Measure-Object -Maximum).Maximum
            sampling_median_ns = Median $off.sampling_ns; input_tokens = $off[0].input_tokens
            logits_tokens = $off[0].logits_tokens; input_tokens_per_second = $off[0].input_tokens * 1e9 / $offMedian }
    }
    $stages = @()
    foreach ($group in ($stageRows | Group-Object -Property workload, threads, stage)) {
        $values = $group.Group
        $stages += [ordered]@{ workload = $values[0].workload; threads = $values[0].threads
            stage = $values[0].stage; calls_per_forward = $values[0].calls; samples = $values.Count
            median_ns = Median $values.wall_ns; share_of_forward = (Median $values.wall_ns) / (Median $values.forward_ns)
            caller_wait_median_ns = Median $values.caller_wait_ns
            caller_work_median_ns = Median $values.caller_work_ns
            worker_work_sum_median_ns = Median $values.worker_work_sum_ns
            parallel_wall_median_ns = Median $values.parallel_wall_ns }
    }
    foreach ($mode in @('prefill', 'decode', 'mixed')) {
        Write-BenchmarkJson (Join-Path $Directory "runtime-$mode.json") ([ordered]@{
            schema_version = 1; run_id = $runId; manifest_sha256 = $manifestHash; profiler = 'none'
            results = @($modelRows | Where-Object mode -CEQ $mode) })
    }
    Write-BenchmarkJson (Join-Path $Directory 'profiler-overhead.json') ([ordered]@{
        schema_version = 1; run_id = $runId; manifest_sha256 = $manifestHash
        online_batch_perturbation = 'not_measured_offline_fixed_batches'; results = $overhead })
    Write-BenchmarkJson (Join-Path $Directory 'forward-stages.json') ([ordered]@{
        schema_version = 1; run_id = $runId; manifest_sha256 = $manifestHash
        worker_time_semantics = 'elapsed_consume_loop_including_dispatch_and_preemption_not_pure_cpu_work'
        caller_wait_semantics = 'elapsed_join_after_caller_consume_not_additive_with_worker_times'
        results = $stages })
    Write-BenchmarkJson (Join-Path $Directory 'validation-summary.json') ([ordered]@{
        status = 'passed'; run_id = $runId; manifest_sha256 = $manifestHash
        reports = $files.Count; measured_forwards = $rows.Count; profiled_forwards = @($rows | Where-Object profiler -CEQ 'stages').Count
        fixed_input_outputs_equal = $true; stage_accounting = 'exact'; raw_reports = $reportHashes })
    Write-Host "Runtime validation passed: $($files.Count) reports, $($rows.Count) measured forwards."
} catch {
    foreach ($name in $outputs) {
        $path = Join-Path $Directory $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path }
    }
    Write-BenchmarkJson (Join-Path $Directory 'validation-summary.json') ([ordered]@{
        status = 'failed'; run_id = $runId; manifest_sha256 = $manifestHash; error = $_.Exception.Message })
    throw
}
