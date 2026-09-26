param([Parameter(Mandatory = $true)][string]$Analyzer)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Analyzer = (Resolve-Path -LiteralPath $Analyzer).Path
. (Join-Path (Split-Path -Parent $Analyzer) 'Benchmark-Common.ps1')
$tempBase = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$testRoot = Join-Path $tempBase ("llmserve-benchmark-validation-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$passed = 0

function New-Snapshot([string]$Policy) {
    return [ordered]@{
        ready = $true; backend = 'minillm'; model = 'fixture-model'; policy = $Policy
        gpu = $false; device = 'CPU/fixture'; threads = 2; gpu_layers = 0; kernel_mode = 'auto'
        context_tokens = 128; max_model_len = 64; batch_tokens = 8; prefill_chunk = 4
        max_active = 2; queue_capacity = 4; prefix_cache_entries = 1
        prefix_cache_tokens = 16; event_buffer_size = 8; aging_ms = 250; admission_reserve_ms = 2000
        kv_credits = [ordered]@{ block_size = 4; active_unique_blocks = 0 }
        llama_commit = ('1' * 40)
        outstanding_requests = 0; active_requests = 0; waiting_requests = 0
    }
}

function New-Request([string]$Id, [string]$ClassName, [int[]]$Tokens, [double]$Arrival) {
    return [ordered]@{
        id = $Id; class_name = $ClassName; success = $true; within_slo = $true
        error = ''; http_status = 200; token_ids = $Tokens; token_times_ms = @(1.0, 2.0)
        finish_reason = 'length'; sse_done_count = 1; terminal_events = 1
        dispatch_lag_ms = 0.0; ttft_ms = 1.0; mean_tpot_ms = 1.0; e2e_ms = 3.0
        finished_s = $Arrival + 0.003
        usage = [ordered]@{ prompt_tokens = 2; completion_tokens = 2; total_tokens = 4 }
    }
}

function Quantiles([double[]]$Values) {
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $result = [ordered]@{}
    foreach ($percentile in @(50, 95, 99)) {
        $rank = ($percentile / 100.0) * ($sorted.Count - 1)
        $lower = [int][math]::Floor($rank)
        $upper = [math]::Min($lower + 1, $sorted.Count - 1)
        $result["p$percentile"] = $sorted[$lower] + ($sorted[$upper] - $sorted[$lower]) * ($rank - $lower)
    }
    return $result
}

function Update-Summary($Report) {
    $successful = @($Report.requests | Where-Object success)
    $good = @($successful | Where-Object within_slo).Count
    $elapsed = ($Report.requests.finished_s | Measure-Object -Maximum).Maximum
    $tokens = 0
    $intervals = @()
    foreach ($request in $successful) {
        $tokens += $request.token_ids.Count
        for ($i = 1; $i -lt $request.token_times_ms.Count; ++$i) {
            $intervals += $request.token_times_ms[$i] - $request.token_times_ms[$i - 1]
        }
    }
    $Report.summary = [ordered]@{
        requests = $Report.requests.Count; successful = $successful.Count
        failed = $Report.requests.Count - $successful.Count; slo_compliant = $good
        elapsed_s = $elapsed; successful_output_tokens = $tokens
        output_tokens_per_second = $tokens / $elapsed; goodput_requests_per_second = $good / $elapsed
        ttft_ms = Quantiles @($successful | ForEach-Object { $_.ttft_ms })
        mean_tpot_ms = Quantiles @($successful | Where-Object { $null -ne $_.mean_tpot_ms } | ForEach-Object { $_.mean_tpot_ms })
        inter_token_ms = Quantiles $intervals
        e2e_ms = Quantiles @($successful | ForEach-Object { $_.e2e_ms })
        dispatch_lag_ms = Quantiles @($Report.requests | ForEach-Object { $_.dispatch_lag_ms })
    }
}

function New-Report([string]$Policy, [string]$TraceHash, [string]$Fnv) {
    $report = [ordered]@{
        schema_version = 1; benchmark = 'llmserve-open-loop'; warmup = $true
        clock = 'steady_clock'; percentile_method = 'linear'; trace_fnv1a64 = $Fnv
        started_at_unix_ms = 1700000000000
        run_identity = [ordered]@{
            run_id = 'fixture-run'; trial = 0; variant = $Policy
            trace_sha256 = $TraceHash; manifest_sha256 = 'pending'; model_sha256 = ('a' * 64)
            server_sha256 = ('3' * 64); client_sha256 = ('4' * 64)
        }
        server_before = New-Snapshot $Policy
        server_after = New-Snapshot $Policy
        requests = @((New-Request 'r0' 'long' @(10, 11) 0), (New-Request 'r1' 'short' @(20, 21) 1))
        summary = $null
    }
    Update-Summary $report
    return $report
}

function Set-Failure($Request, [string]$Code = 'timeout', [int]$Status = 200) {
    $Request.success = $false
    $Request.within_slo = $false
    $Request.error = $Code
    $Request.http_status = $Status
    $Request.token_ids = @()
    $Request.token_times_ms = @()
    $Request.mean_tpot_ms = $null
    $Request.finish_reason = ''
    if ($Status -ne 200) { $Request.sse_done_count = 0; $Request.terminal_events = 0 }
}

function Set-Trace($Directory, $Manifest, $Mixed, $Prefill, $Rows) {
    $tracePath = Join-Path $Directory 'trace.jsonl'
    $lines = @($Rows | ForEach-Object { ConvertTo-Json -InputObject $_ -Depth 8 -Compress })
    [System.IO.File]::WriteAllText($tracePath, ($lines -join "`n") + "`n", [System.Text.UTF8Encoding]::new($false))
    $Manifest.trace.sha256 = Get-LowerSha256 $tracePath
    $Manifest.trace.fnv1a64 = Get-TraceFnv1a64 $tracePath
    $Manifest.trace.size_bytes = (Get-Item -LiteralPath $tracePath).Length
    $Manifest.trace.request_count = $Rows.Count
    foreach ($report in @($Mixed, $Prefill)) {
        $report.run_identity.trace_sha256 = $Manifest.trace.sha256
        $report.trace_fnv1a64 = $Manifest.trace.fnv1a64
    }
}

function New-Fixture([string]$Name, [scriptblock]$Mutate, [scriptblock]$AfterWrite) {
    $directory = Join-Path $testRoot $Name
    New-Item -ItemType Directory -Path $directory | Out-Null
    $trace = Join-Path $directory 'trace.jsonl'
    [System.IO.File]::WriteAllText($trace, (
        '{"request_id":"r0","class_name":"long","arrival_s":0,"request":{"prompt":[1,2],"max_tokens":2,"ignore_eos":true}}' + "`n" +
        '{"request_id":"r1","class_name":"short","arrival_s":1,"request":{"prompt":[3,4],"max_tokens":2,"ignore_eos":true}}' + "`n"))
    $traceHash = Get-LowerSha256 $trace
    $fnv = Get-TraceFnv1a64 $trace
    $sourceState = Join-Path $directory 'source-state.json'
    $sourceRoot = Join-Path $directory 'source'
    New-Item -ItemType Directory -Path $sourceRoot | Out-Null
    [IO.File]::WriteAllText((Join-Path $sourceRoot 'fixture.cpp'), "int main() { return 0; }`n")
    $sourceFiles = @([ordered]@{ path = 'fixture.cpp'; size_bytes = (Get-Item (Join-Path $sourceRoot 'fixture.cpp')).Length
        sha256 = Get-LowerSha256 (Join-Path $sourceRoot 'fixture.cpp') })
    Write-BenchmarkJson $sourceState ([ordered]@{ scope = @('fixture.cpp'); files = $sourceFiles })
    $snapshotPath = Join-Path $directory 'source-snapshot.zip'
    Write-BenchmarkSourceSnapshot $sourceRoot $sourceFiles $snapshotPath
    $manifest = [ordered]@{
        schema_version = 1; benchmark = 'llmserve-policy-comparison'; run_id = 'fixture-run'
        source = [ordered]@{
            git_sha = ('1' * 40); git_dirty = $false; worktree_state_sha256 = Get-LowerSha256 $sourceState
            state_file = 'source-state.json'; scope = @('fixture.cpp')
            snapshot = [ordered]@{ path = 'source-snapshot.zip'; sha256 = Get-LowerSha256 $snapshotPath }
        }
        build = [ordered]@{
            type = 'Release'; generator = 'Ninja'; compiler = 'c++'; compiler_id = 'fixture'; compiler_version = '1'
            cxx_flags = ''; configuration_flags = '-O2'; linker_flags = ''; configuration_linker_flags = ''; cuda_enabled = 'OFF'
        }
        environment = [ordered]@{ os = 'fixture'; processor = 'fixture'; logical_processors = 2 }
        binaries = [ordered]@{
            server = [ordered]@{ sha256 = ('3' * 64) }
            benchmark_client = [ordered]@{ sha256 = ('4' * 64) }
        }
        dependencies = [ordered]@{ llama_commit = ('1' * 40) }
        model = [ordered]@{
            file = 'fixture-model.gguf'; path = 'fixture-model.gguf'; sha256 = ('a' * 64); size_bytes = 64
            weight_dtype = 'Q8_0'
            provenance = [ordered]@{ file = 'fixture-model.gguf'; sha256 = ('a' * 64); size_bytes = 64 }
        }
        trace = [ordered]@{
            path = 'trace.jsonl'; size_bytes = (Get-Item -LiteralPath $trace).Length
            sha256 = $traceHash; fnv1a64 = $fnv; request_count = 2; seed = 0
        }
        engine = [ordered]@{
            backend = 'mini'; metrics_backend = 'minillm'; metrics_kernel_mode = 'auto'; threads = 2; gpu_layers = 0
            context_tokens = 128; max_model_len = 64; batch_tokens = 8; prefill_chunk = 4
            max_active = 2; queue_capacity = 4; block_size = 4; prefix_cache_entries = 1
            prefix_cache_tokens = 16; event_buffer_size = 8; aging_ms = 250; admission_reserve_ms = 2000
        }
        comparison = [ordered]@{
            dimension = 'engine.policy'; allowed_changes = @('engine.policy'); variants = @('mixed', 'prefill_first')
            reference = [ordered]@{ variant = 'mixed'; trial = 0 }
        }
        protocol = [ordered]@{
            warmup = $true; allowed_request_outcomes = @('success'); trials_per_variant = 1
            sampling = 'greedy'; activation_dtype = 'F32'; kv_dtype = 'F16'; profiler_mode = 'none'; order = 'alternating'
            warmup_request = [ordered]@{ prompt = 'Hello'; max_tokens = 8; ignore_eos = $true; cache_namespace = 'benchmark-warmup' }
        }
        reports = @(
            [ordered]@{ file = 'mixed-0.json'; variant = 'mixed'; trial = 0; order = 0 },
            [ordered]@{ file = 'prefill_first-0.json'; variant = 'prefill_first'; trial = 0; order = 1 }
        )
    }
    $mixed = New-Report 'mixed' $traceHash $fnv
    $prefill = New-Report 'prefill_first' $traceHash $fnv
    if ($null -ne $Mutate) { & $Mutate $manifest $mixed $prefill $directory }
    $manifestPath = Join-Path $directory 'manifest.json'
    Write-BenchmarkJson $manifestPath $manifest
    $manifestHash = Get-LowerSha256 $manifestPath
    foreach ($report in @($mixed, $prefill)) {
        if ($report.run_identity.manifest_sha256 -eq 'pending') { $report.run_identity.manifest_sha256 = $manifestHash }
    }
    Write-BenchmarkJson (Join-Path $directory 'mixed-0.json') $mixed
    Write-BenchmarkJson (Join-Path $directory 'prefill_first-0.json') $prefill
    if ($null -ne $AfterWrite) { & $AfterWrite $directory }
    return $directory
}

function Assert-Passes([string]$Name, [scriptblock]$Mutate = $null) {
    $directory = New-Fixture $Name $Mutate $null
    & $Analyzer -Directory $directory | Out-Null
    $validation = Get-Content -Raw -LiteralPath (Join-Path $directory 'validation-summary.json') | ConvertFrom-Json
    if (-not $validation.valid -or -not (Test-Path -LiteralPath (Join-Path $directory 'summary.json'))) {
        throw "$Name should pass validation."
    }
    ++$script:passed
    Write-Host "[PASS] $Name"
}

function Assert-Rejected([string]$Name, [scriptblock]$Mutate = $null, [string]$Pattern = '',
    [scriptblock]$AfterWrite = $null) {
    $directory = New-Fixture $Name $Mutate $AfterWrite
    $before = @{}
    foreach ($file in Get-ChildItem -LiteralPath $directory -File -Recurse -Force) {
        $before[$file.FullName] = Get-LowerSha256 $file.FullName
    }
    $rejected = $false
    try { & $Analyzer -Directory $directory | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw "$Name should be rejected." }
    $validation = Get-Content -Raw -LiteralPath (Join-Path $directory 'analysis-failure.json') | ConvertFrom-Json
    if ($validation.valid -or @($validation.errors).Count -eq 0) {
        throw "$Name 缺少失败诊断。"
    }
    foreach ($path in $before.Keys) {
        if (-not (Test-Path -LiteralPath $path) -or (Get-LowerSha256 $path) -cne $before[$path]) {
            throw "$Name 修改了原归档文件：$path"
        }
    }
    if ($Pattern -and ($validation.errors -join "`n") -notmatch $Pattern) {
        throw "$Name failed for an unexpected reason: $($validation.errors -join '; ')"
    }
    ++$script:passed
    Write-Host "[PASS] $Name"
}

function Set-CudaFixture($Manifest, $Mixed, $Prefill) {
    $Manifest.engine.backend = 'mini-cuda'
    $Manifest.engine.metrics_backend = 'minillm-cuda'
    $Manifest.engine.metrics_kernel_mode = 'cuda-f32'
    $Manifest.engine.prefix_cache_entries = 0
    $Manifest.engine.prefix_cache_tokens = 0
    $Manifest.build.own_cuda_enabled = 'ON'
    $Manifest.protocol.device_weight_dtype = 'F32'
    $Manifest.protocol.kv_layout = 'contiguous'
    $Manifest.protocol.single_stream = $true
    foreach ($report in @($Mixed, $Prefill)) {
        foreach ($snapshot in @($report.server_before, $report.server_after)) {
            $snapshot.backend = 'minillm-cuda'
            $snapshot.gpu = $true
            $snapshot.device = 'CUDA/fixture'
            $snapshot.kernel_mode = 'cuda-f32'
            $snapshot.prefix_cache_entries = 0
            $snapshot.prefix_cache_tokens = 0
            $snapshot.batches = 2
            $snapshot.capabilities = [ordered]@{ max_sequences = 2; max_batch_tokens = 8; max_model_len = 64
                prefix_copy = $false; runtime_stage_profile = $false; synchronous_execute = $true }
            $snapshot.resources = [ordered]@{ layout = 'contiguous'; live_kv_pages = $null
                resident_kv_payload_bytes = 4096; owned_device_bytes = 8192; capacity_tokens = 128
                live_tokens = 0; state_valid = $true; reusable = $true
                snapshot_boundary = 'model_thread_publish'; batch_id = 2 }
        }
    }
}

function Assert-CtestEvidence([string]$Name, [string]$Body, [bool]$ShouldPass,
    [string]$SuiteName = 'benchmark-validation', [int]$UncountedSuites = 0) {
    $directory = Join-Path $testRoot $Name
    New-Item -ItemType Directory -Path $directory | Out-Null
    $document = [System.Xml.XmlDocument]::new()
    $suite = $document.CreateElement('testsuite')
    $document.AppendChild($suite) | Out-Null
    foreach ($attribute in @{ tests = '1'; failures = '0'; skipped = '0'; disabled = '0' }.GetEnumerator()) {
        $suite.SetAttribute($attribute.Key, $attribute.Value)
    }
    $case = $document.CreateElement('testcase')
    $case.SetAttribute('name', $SuiteName)
    $case.SetAttribute('status', 'run')
    $suite.AppendChild($case) | Out-Null
    $output = $document.CreateElement('system-out')
    $output.InnerText = $Body
    $case.AppendChild($output) | Out-Null
    $document.Save((Join-Path $directory 'ctest.xml'))
    $accepted = $true
    $result = $null
    try {
        $result = & (Join-Path (Split-Path -Parent $Analyzer) 'Test-CtestEvidence.ps1') -Directory $directory
    } catch { $accepted = $false }
    if ($accepted -ne $ShouldPass -or ($accepted -and $result.SuitesWithoutCaseCounts -ne $UncountedSuites)) {
        throw "Unexpected CTest evidence result: $Name"
    }
    ++$script:passed
    Write-Host "[PASS] $Name"
}

try {
    $publishDirectory = Join-Path $testRoot 'atomic-publish'
    New-Item -ItemType Directory -Path $publishDirectory | Out-Null
    Write-BenchmarkJson (Join-Path $publishDirectory 'first.json') @{ value = 'original' }
    $originalHash = Get-LowerSha256 (Join-Path $publishDirectory 'first.json')
    New-Item -ItemType Directory -Path (Join-Path $publishDirectory 'second.json') | Out-Null
    $rejected = $false
    try {
        Publish-BenchmarkOutputs $publishDirectory ([ordered]@{
            'first.json' = @{ value = 'replacement' }; 'second.json' = @{ value = 'cannot_publish' } })
    } catch { $rejected = $true }
    if (-not $rejected -or (Get-LowerSha256 (Join-Path $publishDirectory 'first.json')) -cne $originalHash) {
        throw '发布失败没有恢复原有汇总。'
    }
    ++$passed
    Write-Host '[PASS] atomic-publication-rollback'
    foreach ($script in Get-ChildItem -LiteralPath (Split-Path -Parent $Analyzer) -Filter '*.ps1') {
        $tokens = $null
        $parseErrors = $null
        [System.Management.Automation.Language.Parser]::ParseFile($script.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null
        if ($parseErrors.Count) { throw "$($script.Name): $($parseErrors -join '; ')" }
    }
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        $port = $listener.LocalEndpoint.Port
        if (-not (Test-LoopbackPort $port)) { throw 'A listening port was not detected.' }
    } finally {
        $listener.Stop()
    }
    if (Test-LoopbackPort $port) { throw 'A stopped listener is still reported as active.' }
    ++$passed
    Write-Host '[PASS] native-port-detection'
    Assert-Passes 'normal'
    Assert-Passes 'own-cuda-contiguous' { param($m, $a, $b) Set-CudaFixture $m $a $b }
    Assert-Rejected 'own-cuda-fake-pages' {
        param($m, $a, $b)
        Set-CudaFixture $m $a $b
        $b.server_after.resources.live_kv_pages = 0
    } 'live_kv_pages must be null'
    Assert-Rejected 'own-cuda-poisoned-success' {
        param($m, $a, $b)
        Set-CudaFixture $m $a $b
        $b.server_after.resources.state_valid = $false
        $b.server_after.resources.reusable = $false
        $b.server_after.resources.live_tokens = $null
    } 'resources.state_valid|resources.reusable'
    Assert-Rejected 'own-cuda-prefix-capability' {
        param($m, $a, $b)
        Set-CudaFixture $m $a $b
        $b.server_after.capabilities.prefix_copy = $true
    } 'prefix_copy'
    $bundlePath = Join-Path $testRoot 'policy-bundle.zip'
    & (Join-Path (Split-Path -Parent $Analyzer) 'Export-BenchmarkBundle.ps1') `
        -Directory (Join-Path $testRoot 'normal') -Output $bundlePath *> $null
    $bundleDirectory = Join-Path $testRoot 'policy-relocated'
    [IO.Compression.ZipFile]::ExtractToDirectory($bundlePath, $bundleDirectory)
    & (Join-Path $bundleDirectory 'verification/Test-EvidenceAvailability.ps1') -Directory $bundleDirectory *> $null
    & (Join-Path $bundleDirectory 'verification/Analyze-Benchmarks.ps1') -Directory $bundleDirectory | Out-Null
    ++$passed
    Write-Host '[PASS] policy-bundle-relocation'
    Assert-Passes 'no-warmup' {
        param($manifest, $mixed, $prefill)
        $manifest.protocol.warmup = $false; $manifest.protocol.warmup_request = $null
        $mixed.warmup = $false; $prefill.warmup = $false
    }
    Assert-Passes 'single-token' {
        param($manifest, $mixed, $prefill, $directory)
        $rows = @(Get-Content -LiteralPath (Join-Path $directory 'trace.jsonl') | ConvertFrom-Json)
        foreach ($row in $rows) { $row.request.max_tokens = 1 }
        Set-Trace $directory $manifest $mixed $prefill $rows
        foreach ($report in @($mixed, $prefill)) {
            foreach ($request in $report.requests) {
                $request.token_ids = @($request.token_ids[0]); $request.token_times_ms = @(1.0)
                $request.mean_tpot_ms = $null; $request.usage.completion_tokens = 1; $request.usage.total_tokens = 3
            }
            Update-Summary $report
        }
    }
    Assert-Passes 'case-sensitive-ids' {
        param($manifest, $mixed, $prefill, $directory)
        $rows = @(Get-Content -LiteralPath (Join-Path $directory 'trace.jsonl') | ConvertFrom-Json)
        $rows[1].request_id = 'R0'
        Set-Trace $directory $manifest $mixed $prefill $rows
        $mixed.requests[1].id = 'R0'; $prefill.requests[1].id = 'R0'
    }
    Assert-Passes 'per-request-maximum-stall' {
        param($manifest, $mixed, $prefill)
        foreach ($report in @($mixed, $prefill)) {
            foreach ($request in $report.requests) { $request.max_itl_ms = 1.0 }
            $report.summary.request_max_itl_ms = Quantiles @(1.0, 1.0)
        }
    }
    Assert-Rejected 'forged-maximum-stall' {
        param($manifest, $mixed, $prefill)
        $prefill.requests[0].max_itl_ms = 999.0
    } 'max_itl_ms'
    Assert-Rejected 'forged-maximum-stall-distribution' {
        param($manifest, $mixed, $prefill)
        $prefill.summary.request_max_itl_ms = Quantiles @(999.0, 999.0)
    } 'request_max_itl_ms'
    Assert-Passes 'allowed-timeout' {
        param($manifest, $mixed, $prefill)
        $manifest.protocol.allowed_request_outcomes += 'timeout'
        Set-Failure $prefill.requests[1]
        Update-Summary $prefill
    }
    Assert-Passes 'allowed-rejection' {
        param($manifest, $mixed, $prefill)
        $manifest.protocol.allowed_request_outcomes += 'queue_full'
        Set-Failure $prefill.requests[1] 'queue_full' 429
        Update-Summary $prefill
    }
    Assert-Passes 'all-failed' {
        param($manifest, $mixed, $prefill)
        $manifest.protocol.allowed_request_outcomes += 'timeout'
        foreach ($report in @($mixed, $prefill)) {
            foreach ($request in $report.requests) { Set-Failure $request }
            Update-Summary $report
        }
    }
    Assert-Passes 'fractional-arrival-scale' {
        param($manifest, $mixed, $prefill)
        $manifest.protocol.arrival_scale = 0.5
        foreach ($report in @($mixed, $prefill)) {
            $report.arrival_scale = 0.5
            $report.requests[1].finished_s -= 0.5
            Update-Summary $report
        }
    }
    Assert-Passes 'reversed-policy-order' {
        param($manifest)
        $manifest.protocol.order_offset = 1
        $manifest.reports = @($manifest.reports[1], $manifest.reports[0])
        $manifest.reports[0].order = 0; $manifest.reports[1].order = 1
    }
    Assert-Rejected 'arrival-scale-mismatch' {
        param($manifest, $mixed, $prefill)
        $prefill.arrival_scale = 2.0
    } 'arrival_scale'
    Assert-Rejected 'invalid-arrival-scale' { param($manifest) $manifest.protocol.arrival_scale = 0 } 'arrival scale'
    Assert-Rejected 'telemetry-mode-mismatch' {
        param($manifest, $mixed, $prefill)
        $prefill.server_after.telemetry_mode = 'stages'
    } 'telemetry_mode'
    Assert-Rejected 'missing-request' {
        param($manifest, $mixed, $prefill)
        $prefill.requests = @($prefill.requests[0]); Update-Summary $prefill
    } 'missing request'
    Assert-Rejected 'duplicate-request' {
        param($manifest, $mixed, $prefill)
        $prefill.requests[1] = $prefill.requests[0]
    } 'duplicate request'
    Assert-Rejected 'unexpected-request' {
        param($manifest, $mixed, $prefill)
        $prefill.requests[1].id = 'unknown'
    } 'unexpected request'
    Assert-Rejected 'model-hash' { param($manifest, $mixed, $prefill) $prefill.run_identity.model_sha256 = ('b' * 64) } 'model_sha256'
    Assert-Rejected 'model-provenance' { param($manifest) $manifest.model.provenance.sha256 = ('b' * 64) } 'provenance.sha256'
    Assert-Rejected 'wrong-warmup' { param($manifest) $manifest.protocol.warmup_request.max_tokens = 1 } 'warmup_request.max_tokens'
    Assert-Rejected 'wrong-arithmetic' { param($manifest) $manifest.protocol.activation_dtype = 'F16' } 'activation_dtype'
    Assert-Rejected 'server-binary' { param($manifest, $mixed, $prefill) $prefill.run_identity.server_sha256 = ('b' * 64) } 'server_sha256'
    Assert-Rejected 'client-binary' { param($manifest, $mixed, $prefill) $prefill.run_identity.client_sha256 = ('b' * 64) } 'client_sha256'
    Assert-Rejected 'config-difference' { param($manifest, $mixed, $prefill) $prefill.server_before.batch_tokens = 16 } 'batch_tokens'
    Assert-Rejected 'scheduler-config' { param($manifest, $mixed, $prefill) $prefill.server_after.aging_ms = 100 } 'aging_ms'
    Assert-Rejected 'token-mismatch' { param($manifest, $mixed, $prefill) $prefill.requests[1].token_ids = @(20, 99) } 'output tokens differ'
    Assert-Rejected 'truncated-output' {
        param($manifest, $mixed, $prefill)
        $request = $prefill.requests[1]
        $request.token_ids = @(20); $request.token_times_ms = @(1.0); $request.mean_tpot_ms = $null
        $request.usage.completion_tokens = 1; $request.usage.total_tokens = 3
        Update-Summary $prefill
    } 'complete output length'
    Assert-Rejected 'invalid-success-type' { param($manifest, $mixed, $prefill) $prefill.requests[0].success = 'true' } 'must be boolean'
    Assert-Rejected 'negative-token' { param($manifest, $mixed, $prefill) $prefill.requests[0].token_ids = @(-1, 11) } 'must be an integer'
    Assert-Rejected 'token-timestamps' { param($manifest, $mixed, $prefill) $prefill.requests[0].token_times_ms = @(2.0, 1.0) } 'out-of-order'
    Assert-Rejected 'missing-done' { param($manifest, $mixed, $prefill) $prefill.requests[0].sse_done_count = 0 } 'sse_done_count'
    Assert-Rejected 'duplicate-terminal' { param($manifest, $mixed, $prefill) $prefill.requests[0].terminal_events = 2 } 'terminal_events'
    Assert-Rejected 'forged-throughput' { param($manifest, $mixed, $prefill) $prefill.summary.output_tokens_per_second = 1000.0 } 'output throughput'
    Assert-Rejected 'forged-latency' { param($manifest, $mixed, $prefill) $prefill.summary.ttft_ms.p95 = 999.0 } 'summary.ttft_ms.p95'
    Assert-Rejected 'undeclared-timeout' {
        param($manifest, $mixed, $prefill)
        Set-Failure $prefill.requests[1]; Update-Summary $prefill
    } 'disallowed outcome'
    Assert-Rejected 'missing-output-reference' {
        param($manifest, $mixed, $prefill)
        $manifest.protocol.allowed_request_outcomes += 'timeout'
        Set-Failure $mixed.requests[1]; Update-Summary $mixed
    } 'no successful declared output reference'
    Assert-Rejected 'trace-hash' { param($manifest, $mixed, $prefill) $prefill.run_identity.trace_sha256 = ('b' * 64) } 'trace_sha256'
    Assert-Rejected 'trace-fnv' { param($manifest, $mixed, $prefill) $prefill.trace_fnv1a64 = '0' } 'trace_fnv1a64'
    Assert-Rejected 'manifest-hash' { param($manifest, $mixed, $prefill) $prefill.run_identity.manifest_sha256 = ('b' * 64) } 'manifest_sha256'
    Assert-Rejected 'missing-variant' { param($manifest) $manifest.reports = @($manifest.reports[0]) } 'manifest report count'
    Assert-Rejected 'duplicate-report' { param($manifest) $manifest.reports[1] = $manifest.reports[0] } 'duplicate report'
    Assert-Rejected 'missing-trial' { param($manifest) $manifest.protocol.trials_per_variant = 2 } 'manifest report count'
    Assert-Rejected 'changed-comparison' { param($manifest) $manifest.comparison.allowed_changes += 'engine.threads' } 'only allowed change'
    Assert-Rejected 'live-kv' { param($manifest, $mixed, $prefill) $prefill.server_after.kv_credits.active_unique_blocks = 1 } 'kv_active_unique_blocks'
    Assert-Rejected 'missing-report-file' -Pattern 'Missing report' -AfterWrite {
        param($directory) Remove-Item -LiteralPath (Join-Path $directory 'prefill_first-0.json')
    }
    Assert-Rejected 'unexpected-report-file' -Pattern 'Unexpected report' -AfterWrite {
        param($directory) Copy-Item -LiteralPath (Join-Path $directory 'mixed-0.json') -Destination (Join-Path $directory 'mixed-1.json')
    }
    Assert-Rejected 'source-snapshot-hash' { param($manifest) $manifest.source.snapshot.sha256 = ('b' * 64) } 'source.snapshot.sha256'
    Assert-Rejected 'missing-source-snapshot' -Pattern 'ARCHIVE_INCOMPLETE' -AfterWrite {
        param($directory)
        & $Analyzer -Directory $directory | Out-Null
        Remove-Item -LiteralPath (Join-Path $directory 'source-snapshot.zip')
    }
    Assert-Rejected 'corrupt-source-state' -Pattern 'source.worktree_state_sha256' -AfterWrite {
        param($directory)
        & $Analyzer -Directory $directory | Out-Null
        [IO.File]::AppendAllText((Join-Path $directory 'source-state.json'), 'corrupt')
    }
    Assert-Rejected 'missing-manifest' -AfterWrite {
        param($directory) Remove-Item -LiteralPath (Join-Path $directory 'manifest.json')
    }
    Assert-Rejected 'stale-success' -AfterWrite {
        param($directory)
        & $Analyzer -Directory $directory | Out-Null
        [System.IO.File]::WriteAllText((Join-Path $directory 'manifest.json'), '{')
    }
    Assert-CtestEvidence 'ctest-standard' "[PASS] example`n1/1 tests passed" $true 'unit'
    Assert-CtestEvidence 'ctest-benchmark' "[PASS] example`n1/1 benchmark validation tests passed" $true
    Assert-CtestEvidence 'ctest-legacy-benchmark' 'benchmark validation fixtures passed' $true 'benchmark-validation' 1
    Assert-CtestEvidence 'ctest-legacy-wrong-suite' 'benchmark validation fixtures passed' $false 'unit'
    Assert-CtestEvidence 'ctest-truncated' 'benchmark validation fixtures' $false
    Assert-CtestEvidence 'ctest-failed' "[FAIL] example`n1/1 tests passed" $false
    Assert-CtestEvidence 'ctest-count-mismatch' '1/2 tests passed' $false
    $diagnosticDirectory = Join-Path $testRoot 'ctest-standard/diagnostics'
    New-Item -ItemType Directory -Path $diagnosticDirectory | Out-Null
    Copy-Item -LiteralPath (Join-Path $testRoot 'ctest-truncated/ctest.xml') -Destination $diagnosticDirectory
    $evidenceChecker = Join-Path (Split-Path -Parent $Analyzer) 'Test-CtestEvidence.ps1'
    $evidence = & $evidenceChecker -Directory (Join-Path $testRoot 'ctest-standard')
    if ($evidence.Reports -ne 1 -or @($evidence.DiagnosticReports).Count -ne 1) {
        throw 'Diagnostic XML was not classified separately.'
    }
    $rejected = $false
    try { & $evidenceChecker -Directory $diagnosticDirectory | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Truncated diagnostic XML was incorrectly accepted as complete evidence.' }
    ++$passed
    Write-Host '[PASS] ctest-diagnostic-classification'
    Write-Host "$passed/$passed benchmark validation tests passed"
} finally {
    $resolvedTestRoot = [System.IO.Path]::GetFullPath($testRoot)
    if (-not $resolvedTestRoot.StartsWith($tempBase, [System.StringComparison]::Ordinal) -or $resolvedTestRoot -eq $tempBase) {
        throw "Refusing to remove unexpected test directory: $resolvedTestRoot"
    }
    Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
}
