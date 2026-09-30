param(
    [Parameter(Mandatory = $true)][string]$Analyzer,
    [string]$Executable = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path (Split-Path -Parent $Analyzer) 'Benchmark-Common.ps1')
$temporary = Join-Path ([IO.Path]::GetTempPath()) ("runtime-fixtures-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporary | Out-Null
$passed = 0

function New-Fixture([string]$Name) {
    $directory = Join-Path $temporary $Name
    New-Item -ItemType Directory -Path (Join-Path $directory 'source/apps') -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $directory 'source/apps/fixture.cpp'), "int main() { return 0; }`n")
    $file = Join-Path $directory 'source/apps/fixture.cpp'
    $sourceFiles = @([ordered]@{ path = 'apps/fixture.cpp'; size_bytes = (Get-Item $file).Length; sha256 = Get-LowerSha256 $file })
    Write-BenchmarkJson (Join-Path $directory 'source-state.json') ([ordered]@{ scope = @('apps'); files = $sourceFiles })
    Write-BenchmarkSourceSnapshot (Join-Path $directory 'source') $sourceFiles (Join-Path $directory 'source-snapshot.zip')
    $cfg = [ordered]@{ context_tokens = 16; page_tokens = 2; max_sequences = 3; batch_tokens = 4 }
    $recipe = [ordered]@{ name = 'mixed'; mode = 'mixed'; prefill_tokens = 2; decode_sequences = 1; kv_tokens = 2 }
    $inputSpec = [ordered]@{ schema_version = 1; runtime = $cfg; token_ids = @(1, 2, 3); workloads = @($recipe) }
    Write-BenchmarkJson (Join-Path $directory 'input.json') $inputSpec
    $inputHash = Get-LowerSha256 (Join-Path $directory 'input.json')
    $manifest = [ordered]@{
        schema_version = 1; benchmark = 'minillm-runtime'; run_id = 'fixture'
        source = [ordered]@{ git_sha = ('1' * 40); git_dirty = $false; scope = @('apps')
            state_file = 'source-state.json'; worktree_state_sha256 = Get-LowerSha256 (Join-Path $directory 'source-state.json')
            snapshot = [ordered]@{ path = 'source-snapshot.zip'; sha256 = Get-LowerSha256 (Join-Path $directory 'source-snapshot.zip') } }
        dependencies = [ordered]@{ llama_commit = ('2' * 40) }
        binary = [ordered]@{ sha256 = ('a' * 64) }
        model = [ordered]@{ file = 'fixture.gguf'; sha256 = ('b' * 64); size_bytes = 42
            provenance = [ordered]@{ file = 'fixture.gguf'; sha256 = ('b' * 64); size_bytes = 42 } }
        input = [ordered]@{ path = 'input.json'; sha256 = $inputHash }
        runtime = $cfg
        build = [ordered]@{ type = 'RelWithDebInfo'; generator = 'Ninja'; compiler = '/usr/bin/c++'
            compiler_id = 'GNU'; compiler_version = '11.4'; cxx_flags = ''; configuration_flags = '-O2 -g -DNDEBUG'
            linker_flags = ''; configuration_linker_flags = ''; cuda_enabled = 'OFF' }
        environment = [ordered]@{ os = 'fixture'; processor = 'fixture'; logical_processors = 2 }
        protocol = [ordered]@{ threads = @(2); kernel = 'auto'; trials = 1; repeats = 1; warmup = 1
            profiler_modes = @('none', 'stages'); allowed_changes = @('threads', 'profiler')
            reference_model_resident = $false; order = 'alternating_processes_and_thread_order'
            setup = 'pinned_prefix_share_reset'; sampling = 'greedy_argmax_separate'; activation_dtype = 'F32'
            kv_dtype = 'F16'; logits_digest = 'sha256_seq_i32le_length_u32le_logits_f32le' }
        reports = @([ordered]@{ file = 'none.json'; threads = 2; profiler = 'none'; trial = 0; order = 0 },
                    [ordered]@{ file = 'stages.json'; threads = 2; profiler = 'stages'; trial = 0; order = 1 })
    }
    $stages = @()
    $descriptions = @(
        @('kv_prepare', -1, 3, 2, 0, 0, 0, 0, 0),
        @('embedding', -1, 3, 2, 0, 0, 0, 0, 0),
        @('rope_prepare', -1, 3, 2, 0, 0, 0, 0, 0),
        @('attention_norm', 0, 3, 2, 0, 0, 0, 3, 1),
        @('query_projection', 0, 3, 2, 3, 4, 4, 4, 16),
        @('key_projection', 0, 3, 2, 3, 2, 4, 2, 16),
        @('value_projection', 0, 3, 2, 3, 2, 4, 2, 16),
        @('qk_norm_rope_kv', 0, 3, 2, 0, 0, 0, 3, 1),
        @('attention', 0, 3, 2, 0, 0, 0, 6, 1),
        @('output_projection', 0, 3, 2, 3, 4, 4, 4, 16),
        @('attention_residual', 0, 3, 2, 0, 0, 0, 0, 0),
        @('ffn_norm', 0, 3, 2, 0, 0, 0, 3, 1),
        @('gate_projection', 0, 3, 2, 3, 8, 4, 8, 16),
        @('up_projection', 0, 3, 2, 3, 8, 4, 8, 16),
        @('swiglu', 0, 3, 2, 0, 0, 0, 24, 256),
        @('down_projection', 0, 3, 2, 3, 4, 8, 4, 16),
        @('ffn_residual', 0, 3, 2, 0, 0, 0, 0, 0),
        @('final_norm', -1, 1, 1, 0, 0, 0, 0, 0),
        @('lm_head', -1, 1, 1, 1, 32, 4, 32, 16),
        @('final_norm', -1, 1, 1, 0, 0, 0, 0, 0),
        @('lm_head', -1, 1, 1, 1, 32, 4, 32, 16)
    )
    foreach ($d in $descriptions) {
        $parallel = if ($d[7]) {
            [ordered]@{ completed = $true; threads = 2; count = $d[7]; grain = $d[8]
                chunks = [int][math]::Ceiling($d[7] / [double]$d[8]); participating_threads = 1
                wall_ns = 500; dispatch_ns = 10; caller_work_ns = 100; caller_wait_ns = 100
                worker_work_sum_ns = 200; worker_work_max_ns = 200; worker_start_delay_max_ns = 100 }
        } else { $null }
        $stages += [ordered]@{ batch_id = 2; stage = $d[0]; layer = $d[1]; input_tokens = $d[2]
            logits_tokens = $d[3]; matrix_m = $d[4]; matrix_n = $d[5]; matrix_k = $d[6]
            wall_ns = 1000; threads = 2; parallel = $parallel }
    }
    $profile = [ordered]@{ batch_id = 2; completed = $true; input_tokens = 3; logits_tokens = 2; sequences = 2
        context_before_sum = 2; context_before_max = 2; context_after_sum = 5; context_after_max = 3
        threads = 2; kv_pages_before = 1; kv_pages_after = 3; wall_ns = 21100; unaccounted_ns = 100; stages = $stages }
    $sample = [ordered]@{ repeat = 0; batch_id = 2; wall_ns = 22000; sampling_ns = 200; reset_ns = 100
        output = [ordered]@{ logits_vectors = 2; logits_sha256 = ('d' * 64)
            greedy_tokens = @([ordered]@{ sequence = 1; token = 4 }, [ordered]@{ sequence = 2; token = 5 }) }
        kv_pages_before = 1; kv_pages_after = 3; resident_bytes_before = 48; resident_bytes_after = 48
        profile = $null }
    $work = [ordered]@{ name = 'mixed'; mode = 'mixed'; prefill_tokens = 2; decode_sequences = 1; kv_tokens = 2
        prefix_token_ids = @(1, 2); setup_ns = 100; setup_forward_calls = 1; used_pages_after_clear = 0
        input_tokens = @(
            [ordered]@{ token = 1; position = 0; sequence = 1; logits = $false; phase = 'prefill' },
            [ordered]@{ token = 2; position = 1; sequence = 1; logits = $true; phase = 'prefill' },
            [ordered]@{ token = 2; position = 2; sequence = 2; logits = $true; phase = 'decode' })
        samples = @($sample) }
    $report = [ordered]@{ schema_version = 1; benchmark = 'minillm-runtime'; status = 'passed'; trial = 0
        input_sha256 = $inputHash; load_ns = 1
        run_identity = [ordered]@{ run_id = 'fixture'; manifest_sha256 = 'pending'
            model_sha256 = ('b' * 64); binary_sha256 = ('a' * 64) }
        runtime = [ordered]@{ context_tokens = 16; page_tokens = 2; max_sequences = 3; batch_tokens = 4
            threads = 2; kernel = 'auto'; effective_kernel = 'scalar' }
        protocol = [ordered]@{ profiler = 'none'; warmup = 1; repeats = 1; clock = 'steady_clock'
            setup = 'pinned_prefix_share_reset'; sampling = 'greedy_argmax_separate'; activation_dtype = 'F32'
            kv_dtype = 'F16'; logits_digest = 'sha256_seq_i32le_length_u32le_logits_f32le' }
        dimensions = [ordered]@{ embedding = 4; layers = 1; heads = 2; kv_heads = 1; head_dim = 2
            feed_forward = 8; vocabulary = 32 }; workloads = @($work) }
    $off = ConvertTo-Json -InputObject $report -Depth 32 | ConvertFrom-Json
    $on = ConvertTo-Json -InputObject $report -Depth 32 | ConvertFrom-Json
    $on.protocol.profiler = 'stages'
    $on.workloads[0].samples[0].profile = $profile
    return @{ directory = $directory; manifest = $manifest; off = $off; on = $on }
}

function Test-Fixture([string]$Name, [scriptblock]$Mutation, [string]$ErrorPattern = '', [scriptblock]$AfterWrite) {
    $fixture = New-Fixture $Name
    if ($null -ne $Mutation) { & $Mutation $fixture.manifest $fixture.off $fixture.on $fixture.directory }
    $manifestPath = Join-Path $fixture.directory 'manifest.json'
    Write-BenchmarkJson $manifestPath $fixture.manifest
    foreach ($report in @($fixture.off, $fixture.on)) {
        if ($report.run_identity.manifest_sha256 -ceq 'pending') {
            $report.run_identity.manifest_sha256 = Get-LowerSha256 $manifestPath
        }
    }
    Write-BenchmarkJson (Join-Path $fixture.directory 'none.json') $fixture.off
    Write-BenchmarkJson (Join-Path $fixture.directory 'stages.json') $fixture.on
    if ($AfterWrite) { & $AfterWrite $fixture.directory }
    $before = @{}
    foreach ($file in Get-ChildItem -LiteralPath $fixture.directory -File -Recurse -Force) {
        $before[$file.FullName] = Get-LowerSha256 $file.FullName
    }
    $caught = ''
    try { & $Analyzer -Directory $fixture.directory *> $null } catch { $caught = $_.Exception.Message }
    if ($ErrorPattern) {
        if (-not $caught -or $caught -notlike "*$ErrorPattern*") { throw "$Name expected '$ErrorPattern', got '$caught'." }
        $validation = Get-Content -Raw (Join-Path $fixture.directory 'analysis-failure.json') | ConvertFrom-Json
        if ($validation.status -cne 'failed') { throw "$Name left a passing validation." }
        foreach ($path in $before.Keys) {
            if (-not (Test-Path -LiteralPath $path) -or (Get-LowerSha256 $path) -cne $before[$path]) {
                throw "$Name 修改了原归档文件：$path"
            }
        }
    } elseif ($caught) { throw "$Name failed: $caught" }
    ++$script:passed
    Write-Host "[PASS] $Name"
    return $fixture.directory
}

try {
    $valid = Test-Fixture 'valid'
    $moved = Join-Path $temporary 'relocated'
    Copy-Item -LiteralPath $valid -Destination $moved -Recurse
    & $Analyzer -Directory $moved *> $null
    ++$passed
    Write-Host '[PASS] relocated'
    $null = Test-Fixture 'missing-source-snapshot' -ErrorPattern 'ARCHIVE_INCOMPLETE' -AfterWrite {
        param($dir)
        & $Analyzer -Directory $dir *> $null
        Remove-Item (Join-Path $dir 'source-snapshot.zip')
    }
    $null = Test-Fixture 'corrupt-source-state' -ErrorPattern 'source state' -AfterWrite {
        param($dir)
        & $Analyzer -Directory $dir *> $null
        [IO.File]::AppendAllText((Join-Path $dir 'source-state.json'), 'corrupt')
    }
    $bundlePath = Join-Path $temporary 'fixture-bundle.zip'
    $exporter = Join-Path (Split-Path -Parent $Analyzer) 'Export-BenchmarkBundle.ps1'
    & $exporter -Directory $valid -Output $bundlePath *> $null
    $bundleDir = Join-Path $temporary 'exported'
    [IO.Compression.ZipFile]::ExtractToDirectory($bundlePath, $bundleDir)
    & (Join-Path $bundleDir 'verification/Test-EvidenceAvailability.ps1') -Directory $bundleDir *> $null
    & (Join-Path $bundleDir 'verification/Analyze-Runtime.ps1') -Directory $bundleDir *> $null
    ++$passed
    Write-Host '[PASS] bundle-export-relocation'
    [IO.File]::AppendAllText((Join-Path $bundleDir 'none.json'), ' ')
    $rejected = $false
    try { $null = Assert-EvidenceAvailable $bundleDir } catch { $rejected = $true }
    if (-not $rejected) { throw '被篡改的 bundle 未被拒绝。' }
    ++$passed
    Write-Host '[PASS] bundle-tamper'
    $failedBundle = Join-Path $temporary 'incomplete.zip'
    $rejected = $false
    try { & $exporter -Directory (Join-Path $temporary 'missing-source-snapshot') -Output $failedBundle *> $null }
    catch { $rejected = $true }
    if (-not $rejected -or (Test-Path $failedBundle)) { throw '缺件归档不应产生导出包。' }
    ++$passed
    Write-Host '[PASS] bundle-missing-artifact'
    $null = Test-Fixture 'snapshot-entry-tamper' {
        param($m, $off, $on, $dir)
        $statePath = Join-Path $dir 'source-state.json'
        $state = Get-Content -Raw $statePath | ConvertFrom-Json
        $state.files[0].sha256 = ('f' * 64)
        Write-BenchmarkJson $statePath $state
        $m.source.worktree_state_sha256 = Get-LowerSha256 $statePath
    } 'source snapshot entry hash'
    $null = Test-Fixture 'snapshot-empty-hash' { param($m) $m.source.snapshot.sha256 = '' } 'source snapshot hash'
    $null = Test-Fixture 'artifact-traversal' { param($m) $m.source.snapshot.path = '../source.zip' } 'ARCHIVE_INCOMPLETE'
    $null = Test-Fixture 'missing-report' -ErrorPattern 'does not exist' -AfterWrite {
        param($dir) Remove-Item (Join-Path $dir 'stages.json')
    }
    $null = Test-Fixture 'duplicate-report' { param($m) $m.reports[1] = $m.reports[0] } 'Duplicate runtime report'
    $null = Test-Fixture 'missing-trial' { param($m) $m.protocol.trials = 2 } 'manifest report count'
    $null = Test-Fixture 'binary-mismatch' { param($m, $off, $on) $on.run_identity.binary_sha256 = ('f' * 64) } 'binary_sha256'
    $null = Test-Fixture 'model-mismatch' { param($m, $off, $on) $on.run_identity.model_sha256 = ('f' * 64) } 'model_sha256'
    $null = Test-Fixture 'provenance' { param($m) $m.model.provenance.sha256 = ('f' * 64) } 'model provenance'
    $null = Test-Fixture 'input-hash' { param($m, $off, $on) $on.input_sha256 = ('f' * 64) } 'input_sha256'
    $null = Test-Fixture 'manifest-hash' { param($m, $off, $on) $on.run_identity.manifest_sha256 = ('f' * 64) } 'manifest_sha256'
    $null = Test-Fixture 'snapshot-hash' { param($m) $m.source.snapshot.sha256 = ('f' * 64) } 'source snapshot hash'
    $null = Test-Fixture 'config' { param($m, $off, $on) $on.runtime.page_tokens = 4 } 'runtime page_tokens'
    $null = Test-Fixture 'config-type' { param($m, $off, $on) $on.runtime.page_tokens = '2' } 'runtime page_tokens'
    $null = Test-Fixture 'warmup' { param($m, $off, $on) $on.protocol.warmup = 2 } 'protocol warmup'
    $null = Test-Fixture 'missing-workload' { param($m, $off, $on) $on.workloads = @() } 'workload count'
    $null = Test-Fixture 'missing-sample' { param($m, $off, $on) $on.workloads[0].samples = @() } 'sample count'
    $null = Test-Fixture 'input-token' { param($m, $off, $on) $on.workloads[0].input_tokens[0].token = 2 } 'expanded input tokens'
    $null = Test-Fixture 'prefix-token' { param($m, $off, $on) $on.workloads[0].prefix_token_ids[0] = 2 } 'prefix token IDs'
    $null = Test-Fixture 'logits-hash' { param($m, $off, $on) $on.workloads[0].samples[0].output.logits_sha256 = ('e' * 64) } 'fixed-input output'
    $null = Test-Fixture 'greedy-token' { param($m, $off, $on) $on.workloads[0].samples[0].output.greedy_tokens[0].token = 9 } 'fixed-input output'
    $null = Test-Fixture 'negative-wall' { param($m, $off, $on) $on.workloads[0].samples[0].wall_ns = -1 } 'sample wall_ns'
    $null = Test-Fixture 'fractional-wall' { param($m, $off, $on) $on.workloads[0].samples[0].wall_ns = 1.5 } 'sample wall_ns'
    $null = Test-Fixture 'KV-leak' { param($m, $off, $on) $on.workloads[0].used_pages_after_clear = 1 } 'KV reclamation'
    $null = Test-Fixture 'KV-state' { param($m, $off, $on) $on.workloads[0].samples[0].kv_pages_after = 4 } 'KV pages after'
    $null = Test-Fixture 'KV-resident' { param($m, $off, $on) $on.workloads[0].samples[0].resident_bytes_after = 1 } 'resident bytes after'
    $null = Test-Fixture 'missing-stage' { param($m, $off, $on) $on.workloads[0].samples[0].profile.stages = @() } 'profile stage count'
    $null = Test-Fixture 'matrix-shape' { param($m, $off, $on) $on.workloads[0].samples[0].profile.stages[4].matrix_m = 1 } 'matrix_m'
    $null = Test-Fixture 'parallel-grain' { param($m, $off, $on) $on.workloads[0].samples[0].profile.stages[4].parallel.grain = 32 } 'parallel grain'
    $null = Test-Fixture 'worker-time' { param($m, $off, $on) $on.workloads[0].samples[0].profile.stages[4].parallel.worker_work_sum_ns = 9999 } 'worker_work_sum_ns'
    $null = Test-Fixture 'overlapping-time' { param($m, $off, $on) $on.workloads[0].samples[0].profile.unaccounted_ns = 101 } 'forward time accounting'
    $null = Test-Fixture 'wrong-completed-type' { param($m, $off, $on) $on.workloads[0].samples[0].profile.completed = 'true' } 'profile completed'
    $null = Test-Fixture 'profiling-off' { param($m, $off, $on) $off.workloads[0].samples[0].profile = $on.workloads[0].samples[0].profile } 'Unexpected profile'
    $null = Test-Fixture 'stale-summary' { param($m, $off, $on) $on.status = 'failed' } 'report status' {
        param($dir) [IO.File]::WriteAllText((Join-Path $dir 'profiler-overhead.json'), '{"status":"passed"}')
    }
    if ($Executable) {
        $fixture = New-Fixture 'invalid-cli'
        $inputPath = Join-Path $fixture.directory 'input.json'
        $data = Get-Content -Raw $inputPath | ConvertFrom-Json
        $data.token_ids[0] = 1.5
        Write-BenchmarkJson $inputPath $data
        $output = Join-Path $fixture.directory 'failed.json'
        & $Executable --model missing.gguf --input $inputPath --output $output *> $null
        if ($LASTEXITCODE -eq 0) { throw 'Invalid CLI input was accepted.' }
        $failure = Get-Content -Raw $output | ConvertFrom-Json
        if ($failure.error -cne 'invalid fixed runtime token ID') { throw "Unexpected input validation: $($failure.error)" }
        ++$passed
        Write-Host '[PASS] invalid-cli-token'
    }
    Write-Host "$passed/$passed tests passed"
} finally {
    Remove-Item -LiteralPath $temporary -Recurse -Force
}
