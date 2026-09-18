# Validation And Evidence

## Scope

The main implementation is C++20. The model is official Qwen3-0.6B Q8_0
at the revision and SHA-256 in `models/manifest.json`. Local auxiliary
experiments are excluded from source control and are not counted here.

The validated MiniLLM path uses CPU AVX2/FMA/F16C, FP32 arithmetic and
FP16 paged KV. The alternative llama.cpp backend provides CPU/CUDA
execution; it does not validate a custom GPU paged attention kernel.

## Environment

Local measurements were collected on 2026-09-18:

- Windows 11 x64, build 10.0.26200.
- AMD Ryzen 7 7745HX, 8 cores / 16 logical processors; 8 inference threads.
- MSVC 19.44.35225, CMake 3.31.6-msvc6, Ninja, Release `/O2 /Ob2 /DNDEBUG`.
- MiniLLM uses AVX2/FMA/F16C with scalar fallback. The local upstream CPU
  backend was configured for native AVX512.
- Upstream CUDA reference: RTX 4070 Laptop, 8188 MiB, driver 591.74,
  CUDA Toolkit 12.8.61, compute architecture 89.

`benchmarks/results/environment.json` records the environment and report
groups. These initial local reports predate the first source-control
checkpoint and do not embed an exact source commit. Future performance
comparisons must record the Git SHA and binary hash with the workload.

## Correctness

- `tests/core_tests.cpp`: 28 passing cases covering credit allocation, prefix indexing,
  UTF-8 fragments, scheduling, half conversion, SIMD accuracy, physical KV
  page aliases/COW, worker synchronization, cancellation, deadlines,
  backpressure, failure cleanup and bounded capacity. Randomized physical KV
  accounting, missing logits and duplicate finalization are included.
- `tests/gguf_tests.cpp`: four passing cases covering mapped rows, metadata
  types, truncated tensor data, invalid magic, missing files and unsupported
  tensor types. CTest evidence:
  `benchmarks/results/validation/windows-cpu-ctest.xml`.
- `benchmarks/results/validation/model-f32-cpu-reference.json`: 10 passing
  real-model checks against llama.cpp on CPU in the CUDA-enabled build.
- `benchmarks/results/validation/model-f32-cuda-reference.json`: the same
  10 checks against the CUDA reference.
- `benchmarks/results/validation/model-f32-cpu-only-reference.json`: all
  10 checks also pass in the independent CPU-only Release build.
- `benchmarks/results/validation/http-mini.json`: eight passing online
  checks from the CUDA-enabled binary running the MiniLLM CPU backend.
- `benchmarks/results/validation/http-mini-cpu-only.json`: all eight online
  checks pass in the CPU-only binary, including streaming, concurrency and
  both disconnect paths.
- `benchmarks/results/validation/cli-mini-cpu-only.json`: a real 16-token
  CLI generation with zero live physical KV pages after release.
- `benchmarks/results/validation/windows-cpu-imports.txt`: the CPU CLI and
  server import no CUDA or Python DLLs. CLI and HTTP checks also passed with
  CUDA/Python-related entries removed from the child process environment.
  Windows C++ redistributable and OpenMP runtime DLLs are still required.

Teacher-forced CPU-reference logits on three prompts have RMSE
0.000963-0.002472 in the CUDA-enabled build and 0.000946-0.003177 in the
CPU-only build. Both stay inside the same fixed thresholds; these are
separate runs, not bitwise reference reproducibility guarantees.
All checked greedy outputs match. Full prefill and
chunks of 1, 7 and 16 tokens produce identical MiniLLM logits on the
33-token test. A 17-token shared prefix with a COW tail also matches
fresh recomputation exactly. Releasing all sequences leaves zero live
physical KV pages.

These are focused regression tests, not a claim of complete model,
language, hardware or long-context coverage.

## Numerical Reference

The reference GGUF is made by dequantizing the exact Q8_0 weights to F32
with pinned llama.cpp. It does not restore the original unquantized model.
This isolates the model arithmetic from additional activation quantization
in upstream quantized matrix kernels.

Fixed acceptance bounds in `tests/model_tests.cpp`:

- Teacher-forced reference: RMSE < 0.05, max absolute error < 0.5,
  cosine similarity >= 0.9999, and matching argmax.
- Same-runtime chunking/COW: RMSE < 1e-6, max absolute error < 1e-5.
- Scalar/SIMD model path: RMSE < 0.02, max absolute error < 0.1,
  cosine similarity >= 0.99999.

Direct Q8 backend comparisons include different activation quantization
semantics and are not the matched-weight numerical oracle. Their
diagnostic reports are retained separately.

## SIMD Microbenchmark

Source: `apps/kernel_bench.cpp`.
Evidence: `benchmarks/results/simd-q8-dot.json`.

| Measurement | Value |
| --- | ---: |
| Weight matrix | 128 x 1024, Q8_0 |
| Input | F32 vector |
| Execution | Single thread, hot cache |
| Measured rounds | 7, alternating order |
| Repetitions per round | 300 |
| Scalar median | 661.53 ns/dot |
| AVX2 median | 154.13 ns/dot |
| Median ratio | 4.29x |
| Maximum absolute error | 0.00003815 |

This ratio applies only to this dot-product microbenchmark. It is not an
end-to-end model or serving throughput claim.

## Serving Protocol

`scripts/Benchmark-Policies.ps1` restarts the server before each trial,
warms it with the same request, and alternates the order of the two policies.
Each result records the server configuration, initial/final cache state,
exact input trace digest, token IDs and per-token receive timestamps.

- TTFT starts at the client's actual dispatch, excluding connection setup
  before dispatch only when a connection is reused; this benchmark creates
  a client per request, so connection time is included.
- TPOT is each request's mean interval between its first and last sampled
  token, not a percentile of individual token intervals.
- Inter-token percentiles are reported separately.
- Goodput counts successful requests meeting both their TTFT and mean-TPOT
  limits, divided by the full replay duration.
- Throughput counts output tokens from successful requests over that
  duration. Failed requests remain in the report.
- Arrival-to-dispatch lag is reported to expose load-generator delay.
- Small-sample P95/P99 values are descriptive, not production tail guarantees.

CPU trace: `benchmarks/traces/cpu-mixed-s0.jsonl`, 24 requests, 4 requests/s
Poisson arrivals, 128/16 prompt tokens, 16 generated tokens, seed 0.
The pressure level deliberately permits queueing. The two policies use
identical limits and the same raw trace.

## CPU Policy Results

Evidence: `benchmarks/results/mini-scheduling/summary.json` and all six raw
trial reports in the same directory.

| Median of three trials | Mixed | Prefill-first |
| --- | ---: | ---: |
| Output tokens/s | 18.40 | 18.59 |
| P95 TTFT, ms | 13105.77 | 12290.59 |
| P95 mean TPOT, ms | 398.89 | 352.85 |
| SLO goodput, requests/s | 0 | 0 |

All 144 requests completed successfully; output token sequences matched
across all six trials. Mixed batching did not improve this CPU workload.
The bottleneck is not yet established by profiling. See `ENG-008` in
`docs/ENGINEERING_LOG.md`. Lower-rate, distinct cold/warm-prefix and
long-context experiments are still needed.

## Remote CI

GitHub Actions run
[35352370060](https://github.com/lilong555/mini-llm-runtime/actions/runs/35352370060)
passed all five jobs for source commit
`07793ddd93410d04188092ade1c14b471078b1ea`:

- Windows and Ubuntu dependency-free core builds and unit tests.
- Windows and Ubuntu complete CPU product builds and both CTest suites.
- Ubuntu core tests under AddressSanitizer and UndefinedBehaviorSanitizer.

The remote run summary and JUnit reports are archived under
`benchmarks/results/ci/07793dd/`. Sanitizers cover the dependency-free core;
they do not cover the GGUF parser, HTTP transport or real-model execution.
Remote CI downloads the pinned C++ dependency, not model weights.

## Unverified Areas

Long-context stress, sustained overload fairness, failure injection into
real model execution, multi-tenant security, arbitrary Qwen3 variants,
F16-only model coverage, GPU custom kernels and large-sample SLO conclusions
require separate evidence. CUDA HTTP serving and model/HTTP sanitizer runs
also require separate evidence. Local GPU validation is not remote GPU CI.

## Reproduce

```powershell
.\scripts\Build-LLMServe.ps1
ctest --test-dir build/cpu --output-on-failure --test-output-size-passed 65536 `
    --output-junit ../../benchmarks/results/validation/windows-cpu-ctest.xml
.\scripts\Validate-Model.ps1 `
    -Output benchmarks/results/validation/model-f32-cpu-only-reference.json
$server = .\scripts\Start-LLMServe.ps1 -Backend mini
.\build\cpu\bin\llmserve-http-tests.exe --port $server.Port `
    --output benchmarks/results/validation/http-mini-cpu-only.json
.\scripts\Stop-LLMServe.ps1 -Port $server.Port
```

Run CTest from a developer shell or use the CTest executable beside CMake.
Model and HTTP tests are separate from CTest because they require downloaded
weights and a live service. Remote CI does not download model weights.
