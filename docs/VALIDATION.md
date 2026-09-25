# Validation And Evidence

## 当前 WSL 验收

[CUDA 完整模型与 CLI 验收](../benchmarks/results/validation/cuda-model/README.md) 记录四种构建共 31 次 CTest 套件、749 次用例执行，8 项 Runtime 检查、128 组 CPU/F32 logits 对照、六组 S=1/S=4 短金标准与三个 CLI 生成。最大 RMSE `0.005696512`、最大绝对误差 `0.024068833`，无 argmax 差异。设备与完整模型 memcheck、Runtime racecheck/synccheck 均通过；CPU 模型 13/13、HTTP 8/8。该组验证完整 GPU 模型路径，但未完成长语料全契约、正式性能或 GPU Serving 门禁。

[CUDA 连续 KV 与层验收](../benchmarks/results/validation/cuda-layer/README.md) 记录 7 项状态、FP16 RN-even、因果 GQA、NaN mask 与完整层测试，六组真实首层/末层对照，以及四种构建共 30 次 CTest 套件、741 次用例执行。CPU 模型 13/13、HTTP 8/8；设备和实模型层 memcheck、层 racecheck/synccheck 均通过。固定模型门槛与共享 Q/K/V 的算子门槛分别报告，FP16 边界导致的直接逐元素超差原件保留；不是完整 GPU 模型或性能验收。

[CUDA 基础算子验收](../benchmarks/results/validation/cuda-ops/README.md) 记录 11 项 gather、分组 RMSNorm、RoPE、逐元素与 finite/argmax 检查，覆盖实际宽度、151936 词表、padding、非法输入、原地操作、有限极值与同 stream 组合执行。四种构建共 29 次 CTest 套件、733 次用例执行；CPU 模型 13/13、HTTP 8/8。memcheck、racecheck、synccheck 均通过；不构成完整 GPU 模型或性能证据。

[CUDA 权重与存储验收](../benchmarks/results/validation/cuda-storage/README.md) 记录自有 CUDA 9/9、CPU 7/7、无 llama 核心 5/5、上游 CUDA 7/7 CTest，共 722 次用例执行；CPU 实模型 13/13、HTTP 8/8。固定 Q8_0 模型的 310 个唯一 tensor 全量回读通过，88 组真实形状 GEMM 满足预注册单元容差，S=4/Lmax=2048/B=128 的内存计划与自有分配一致。Compute Sanitizer 为 0 错误、0 泄漏；此项不证明完整 GPU 模型或性能。

[Host Model 验收](../benchmarks/results/validation/host-model/README.md) 记录 CPU 7/7、自有 CUDA 8/8、无 llama 核心 5/5、上游 CUDA 7/7 CTest；独立模型绑定/分词器实模型检查 8/8，CPU 和 CUDA 参照模型各 13/13，两个 HTTP 后端各 8/8。44 份固定输入报告的 792 次测量保留一致的完整 logits 摘要、greedy 和 KV 状态，36 对 profile 样本契约一致。12 个代表案例的中位退化均未超过预注册 A/A 阈值；正负波动和有限样本置信区间均保留，不作加速声明。

[CUDA 基础层验收](../benchmarks/results/validation/cuda-infra/README.md) 记录独立 CUDA、CPU、无 llama 核心和上游 CUDA 四组 CTest；设备资源及矩阵 11/11、Compute Sanitizer 0 错误/0 泄漏，CPU 模型 13/13 和 HTTP 8/8。自有 CUDA 模型数值与性能尚未验收。

2026-09-24 的 [M0 归档与验证契约验收](../benchmarks/results/validation/evidence-m0/README.md) 记录 CPU 产品 6/6、无 llama 依赖核心 5/5 CTest；CPU 实模型检查 13/13、HTTP 检查 8/8。反例覆盖缺 ZIP、源码与包内摘要篡改、路径迁移、失败时旧文件保留和发布回滚。三个短样例的 8-token 金标准及中文、英文、重复、特殊 token 的固定语料位于 `tests/data/qwen3_validation_cases.json`；GPU 数值与性能仍待真实 CUDA 模型实现后验收。

[M0 CPU 归档基线](../benchmarks/results/evidence-m0/README.md) 保留 6 个进程、36 次测量和可独立复验的包。`wsl-runtime-profile/context` 的原始 ZIP 已按历史 manifest 核对；其他历史归档不能仅凭旧 `passed` 推断依赖完整。本机 CPU 检查不替代 Windows、远程 CI 或 GPU 模型验收。

2026-09-23 的 [在线观测验证](../benchmarks/results/validation/wsl-batch-telemetry/README.md) 记录 CPU/CUDA 参照构建各 5/5 CTest 套件、ASan/UBSan 核心 4/4，共 463 次用例执行；CPU 1/8 线程及 CUDA 数值参照的模型检查各 13/13，CPU off/stages 和 CUDA stages 的 HTTP 各 8/8。SSE token 关联、缓冲耗尽、异常、阶段时间守恒、KV 回收与目录迁移均通过。

[在线 batch 基线](../benchmarks/results/wsl-batch-telemetry/README.md) 保留 42 个独立服务进程、594 个请求、9810 个输出 token，跨模式和策略的完整输出一致。原始负载三轮无观测的 mixed / prefill-first 中位吞吐为 18.41 / 18.66 token/s，P95 请求平均 TPOT 为 395.28 / 333.42 ms，P99 ITL 为 1224.77 / 2480.67 ms；各轮分位数取中位数，不混作合并分位数。报告保留阶段观测约 +1.82% / +5.82% 的全程耗时差异，以及低到达率的 batch 组成变化。该差异包含系统噪声及在线扰动，不是精确插桩成本或计算优化收益。

2026-09-22 的 [Runtime profiler 验证](../benchmarks/results/validation/wsl-runtime-profile/README.md) 记录 CPU/CUDA 参照构建各 4/4 CTest 套件、ASan/UBSan 核心构建 3/3，共 358 次用例执行。CPU 1/8 线程和 CUDA 数值参照的三次模型检查各 12/12，两个 HTTP 后端各 8/8。profile 开关的逐字节 logits、续写、共享 KV、异常恢复与存储复用均有检查；源码、模型和二进制身份固定在该目录的 `evidence.json`。

[Runtime 阶段基线](../benchmarks/results/wsl-runtime-profile/README.md) 包含 1/2/4/8/16 线程扩展和独立的 8 线程长上下文组，36 份报告、558 次测量全部通过输出与计时验收。8 线程 `prefill-128` 的矩阵投影占 profile forward 中位数约 96.16%，LM head 约 0.54%；16 线程的 prefill 更快，但单序列 decode 慢于 8 线程。长上下文组的 attention 从约 2.00 ms 增至 15.28 ms。原始开销、离群值和实验限制均保留；这些不是 Serving 或自研 CUDA 加速结论。

此前的 [WSL 确定性混合批验证](../benchmarks/results/validation/wsl-deterministic/README.md) 保留 CPU/CUDA 各 3/3 套件、ASan/UBSan 核心构建 2/2 套件，以及 CPU 1、2、8 线程和 CUDA 参照共六次 10/10 模型检查。8 线程 CPU 连续三次通过，六份报告均形成确定的真实混合批；CPU/CUDA HTTP 各 8/8 通过。51 项基准与归档 fixture 检查不完整报告、身份差异、合法失败及诊断分类。

[WSL 策略基线](../benchmarks/results/wsl-policy-validation/README.md) 保留六次独立服务回放、144 个成功请求、完整 token 对照、manifest 和源码快照。mixed / prefill-first 的中位吞吐为 18.98 / 19.30 token/s，P95 请求平均 TPOT 为 381.77 / 321.92 ms，goodput 均为 0；单次 ITL 的 P99 则为 1203.67 / 2436.45 ms。该结果不是优化收益证明，不与以下 Windows 历史基线直接计算加速比。

[原生回放终态验证](../benchmarks/results/wsl-protocol-validation/README.md) 覆盖真实 HTTP 429 与 SSE 超时；失败请求保留，未计入成功或 goodput。完整协议见 [策略回放与验收](BENCHMARKS.md)。旧截断 XML 作为诊断保存，默认归档检查将 `diagnostics/` 单独列出，不把它算作完整通过证据；旧套件没有提供的用例数量不会被推测补齐。

以下章节对应各自标明日期的历史环境与证据，不代替上述当前 WSL 验收。

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

## CPU KV Cache Comparison

Evidence: `benchmarks/results/kv-cache-cpu/summary.json`, the 29 raw reports
in that directory, and the four byte-stable traces under `benchmarks/traces/kv-cpu-*.jsonl`.
Measurements were collected on 2026-09-21 from source commit
`600a1b93cdc95aa11dcbcb56d52d73ad6d5ae8ea` and pinned llama.cpp commit
`911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`.

Both backends used the same Qwen3-0.6B Q8_0 file, CPU execution, eight
threads, F16 K and V, an 8192-token context capacity, one active sequence,
256-token prefill chunks, no prefix cache, and 33 forced output tokens.
MiniLLM used its AVX2/FMA/F16C kernels and 16-token physical pages. The
upstream CPU backend used its native AVX512 build and FlashAttention. Each
cell below is the median of three complete single-request runs; all raw
rounds are retained.

| Prompt tokens | Mini TPOT, ms | llama.cpp TPOT, ms | Mini / llama.cpp |
| ---: | ---: | ---: | ---: |
| 16 | 32.65 | 23.16 | 1.41x |
| 256 | 34.32 | 25.52 | 1.34x |
| 1024 | 39.09 | 30.01 | 1.30x |
| 1536 | 49.26 | 33.84 | 1.46x |

From 16 to 1536 cached prompt tokens, MiniLLM TPOT increased by 16.62 ms
and llama.cpp TPOT increased by 10.68 ms. The measured context-growth cost
was therefore 1.56x higher for MiniLLM, or 10.93 versus 7.03 microseconds
per additional cached token per decoded token. This is the closest
end-to-end estimate of the KV/attention-path gap in this experiment; it is
not a pure allocator measurement.

For this model both F16 caches carry exactly 112 KiB per token. MiniLLM's
16-token page is 1.75 MiB and its full 8192-token payload capacity is
896 MiB. MiniLLM lazily materializes pages and retains freed backing
storage; llama.cpp preallocates the configured KV tensors. Whole-process
Private Bytes were 65.0 MiB for MiniLLM immediately after startup and
237.9 MiB after the longest workload, versus 1153.0 MiB and 1157.4 MiB for
llama.cpp. These process counters include non-KV allocations and are only
supporting evidence for the different allocation policies, not direct KV
tensor sizes.

`mini-kv-cache-bench` isolates layout while keeping the QK, softmax, PV,
F16 conversion and AVX2/F16C math identical. For one resident-memory layer
with the model's real 16 heads, 8 KV heads and 128-wide heads, PagedKV took
1.20x, 1.26x, 1.35x and 1.32x the time of a per-layer contiguous F16 layout
at 16, 256, 1024 and 1536 tokens. This demonstrates a measurable cost from
per-token access and page discontinuities, but it is not an actual
llama.cpp kernel benchmark. The full runtime difference also includes
AVX512 versus AVX2, graph execution, FlashAttention, softmax, batching and
other kernel differences.

An exploratory five-run MiniLLM test changed the page size from 16 to 256
tokens at a 1536-token prompt. TPOT drifted monotonically from 49.78 to
36.22 ms, so its 38.59 ms median is retained but not used as comparative
evidence. A future page-size experiment must interleave configurations and
control frequency and thermal state.

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
`scripts/Test-CtestEvidence.ps1` checks the archived local/remote XML reports
and rejects failed, missing or truncated suite output.

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
