# Mini LLM Runtime: C++ + SIMD + GGUF

一个从底层推理到在线服务的 C++20 项目。**MiniLLM** 独立执行 Qwen3 前向计算，**LLMServe** 在其上实现迭代级调度与流式服务；llama.cpp 提供格式解析、tokenizer 和可切换的 CPU/CUDA 参照后端。

```text
                  Mini LLM Runtime
GGUF -> Memory Mapping -> Tensor Views -> SIMD / Scalar Kernels
                                             |
                         Qwen3: RMSNorm / RoPE / GQA / SwiGLU
                                             |
                              FP16 Paged KV + Page Tables
                                             |
                           LLM Serving
HTTP / SSE -> Bounded Queue -> Priority + Aging -> BatchPlan
                                             |
                              Continuous Batching
                              Chunked Prefill + Decode
                                             |
                         ModelRunner: MiniLLM | llama.cpp
```

MiniLLM 不调用 `llama_decode()` 执行模型。它使用自有矩阵计算、attention 和物理 KV 页；llama.cpp 后端的 CUDA 算子与 GPU KV 存储属于上游能力。

## 能力边界

| 层次 | 已实现 |
| --- | --- |
| GGUF | 只读文件映射、TensorView、形状与文件范围检查；F32/F16/Q8_0 权重 |
| Host model | 独立的 immutable Qwen3 绑定与 vocab-only tokenizer；不创建执行线程或 KV |
| 自有 CUDA 基础 | 常驻 FP32 权重、显存预算、cuBLAS GEMM、自有 gather/RMSNorm/RoPE/SwiGLU/greedy 算子 |
| CPU SIMD | Q8_0 × F32、F16 × F32、F32 dot、FP16 V 到 F32 的加权累加；AVX2/FMA/F16C 运行时检测、非对齐尾部处理及 scalar fallback |
| 模型执行 | Dense Qwen3、GQA、Q/K RMSNorm、NeoX RoPE、SwiGLU、FP32 accumulation、贪心采样 |
| 物理 KV | FP16 页存储、free list、序列页表、引用计数、完整页共享、部分尾页 copy-on-write |
| Batching | 单模型执行线程；每轮重新组批；同一次前向混合 prefill/decode |
| 调度 | token budget、chunked prefill、优先级 aging、等待保护、保守容量预留 |
| Prefix cache | token Trie、命名空间、完整块复用、LRU 淘汰；全命中时重算最后一块 |
| 服务 | C++ HTTP/SSE、取消、超时、断连回收、慢消费者背压、严格参数校验 |
| 实验 | scalar/SIMD 微基准、模型 logits 对照、在线负载生成与回放、TTFT/TPOT/goodput |
| 在线观测 | 默认关闭的有界 batch 记录、SSE token 关联、Runtime 阶段汇总与跨模式验收 |

支持范围：单机、单模型、纯文本、`temperature=0`、`n=1`。首个验证模型为 Qwen3-0.6B Q8_0。MiniLLM 模型在 CPU 执行，完整 CUDA 模型执行通过 llama.cpp 后端提供。自有 CUDA 提供常驻权重、存储与真实形状矩阵验证，尚无完整 GPU forward，见 [CUDA 运行基础](docs/CUDA_RUNTIME.md)。

不支持：chat-template 自动套用、随机采样、任意 GGUF 模型架构、Q4/MoE、多 GPU、抢占重算、PD 分离、自研 CUDA PagedAttention。自有 CPU paged attention 与上游 GPU attention 必须分别评价。

## 快速运行

### WSL2 原生开发（默认）

主工作区：Ubuntu `/home/li/code/mini-llm-runtime`。源码、依赖、模型与构建产物位于 Linux 文件系统。

```bash
cd /home/li/code/mini-llm-runtime
bash scripts/dev.sh dependencies
bash scripts/dev.sh model
bash scripts/dev.sh build
bash scripts/dev.sh test
bash scripts/dev.sh serve --port 8000
```

完整的模型验证、HTTP 检查、编辑器入口和 CUDA 条件见 [WSL2 开发指南](docs/WSL_DEVELOPMENT.md)。

### Windows / PowerShell

需要 Visual Studio 2022 C++ 工具链、CMake >= 3.24 和 Ninja。运行不依赖 Python。

```powershell
.\scripts\Fetch-Dependencies.ps1
.\scripts\Download-Model.ps1
.\scripts\Build-LLMServe.ps1

.\build\cpu\bin\mini-llm.exe --model models\Qwen3-0.6B-Q8_0.gguf `
    --prompt "The capital of France is" --tokens 16

.\scripts\Start-LLMServe.ps1 -Backend mini
```

服务默认监听 `http://127.0.0.1:8000`。启动脚本会避开占用端口并返回实际地址、PID 和日志路径。服务仅绑定 loopback，没有认证，不应直接暴露到公网。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
curl.exe -N http://127.0.0.1:8000/v1/completions `
    -H "Content-Type: application/json" --data-binary '@examples/completion.json'

.\scripts\Stop-LLMServe.ps1 -Port 8000
```

模型下载由 manifest 固定版本与 SHA-256。存在可用 WSL 网络时，可使用 `Download-Model.ps1 -UseWsl` 下载；WSL 不是 C++ 运行时的依赖。

### CUDA 参照后端

```powershell
.\scripts\Build-LLMServe.ps1 -Cuda -CudaArchitectures 89
.\scripts\Start-LLMServe.ps1 -Backend llama -Port 8001
```

本机验证环境：RTX 4070 Laptop GPU，8188 MiB 显存，驱动 591.74，CUDA Toolkit 12.8，MSVC 19.44。`89` 对应 Ada；其他 GPU 应选择适合的架构。CUDA DLL 所在的 `%CUDA_PATH%\bin` 需在 `PATH` 中。

### Linux

```bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout --detach 911f6cdc8ab8a530b2bee09ee61471a6f3178eeb
cmake -S . -B build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j 8
ctest --test-dir build/cpu --output-on-failure
build/cpu/bin/llmserve --model models/Qwen3-0.6B-Q8_0.gguf --backend mini
```

没有模型或外部依赖时，可独立构建算法与资源管理测试：

```bash
cmake -S . -B build/unit -DLLMSERVE_WITH_LLAMA=OFF
cmake --build build/unit
ctest --test-dir build/unit --output-on-failure
```

## 接口

| 接口 | 内容 |
| --- | --- |
| `GET /health` | readiness |
| `GET /v1/models` | 实际加载的模型 ID |
| `POST /tokenize` | `{"text":"..."}` -> token IDs |
| `POST /v1/completions` | 文本或 token ID prompt；完整 JSON 或 SSE |
| `DELETE /v1/requests/{id}` | 取消在途请求；ID 可由 `X-Request-ID` 指定 |
| `GET /metrics` | 批次、生成、缓存、容量预留与请求状态统计 |

Completions 是受限的兼容接口，不是完整 API 实现。支持 `prompt`、`model`、`max_tokens`、`temperature=0`、`stream`、`priority=0..3`、`timeout_ms`、`ignore_eos`、`cache_namespace`、`n=1`；其他参数明确拒绝。

SSE 每个采样 token 带 `token_id`，文本经过 UTF-8 增量缓冲；末事件带 `usage` 和 `timings`，随后恰好一个 `[DONE]`。EOS 计入采样 token 数，不输出 EOS 文本。断连与超时在一次模型前向完成后的边界生效，不中断在途 kernel。

`cache_namespace` 只用于单租户实验隔离，不构成身份认证。终态请求不保留在查询注册表中。

## 验证与测量

```powershell
ctest --test-dir build\cpu --output-on-failure
.\scripts\Validate-Model.ps1 -Output benchmarks\results\validation\model-f32-cpu-only-reference.json
.\build\cpu\bin\llmserve-http-tests.exe --port 8000 `
    --output benchmarks\results\validation\http-mini.json
.\build\cpu\bin\mini-kernel-bench.exe --output benchmarks\results\simd-q8-dot.json
.\build\cpu\bin\mini-kv-cache-bench.exe `
    --output benchmarks\results\kv-cache-cpu\layout-isolation.json
```

模型验证使用由同一份 Q8_0 权重解量化得到的 F32 GGUF，避免把上游额外的激活量化误差混入参考。参照文件约 2.39 GB，只用于验证，生成命令、来源和哈希固定在 `scripts/Validate-Model.ps1` 与 `models/reference-manifest.json`。

负载生成与策略对照：

```powershell
.\build\cpu\bin\llmserve-bench.exe --make-trace --port 8000 `
    --trace benchmarks\traces\cpu-mixed-s0.jsonl --requests 24 --rate 4 `
    --long-tokens 128 --short-tokens 16 --max-tokens 16 --seed 0

.\scripts\Stop-LLMServe.ps1 -Port 8000
.\scripts\Benchmark-Policies.ps1 -Backend mini -Trace benchmarks\traces\cpu-mixed-s0.jsonl
```

WSL 原生入口为 `bash scripts/dev.sh benchmark -Trace benchmarks/traces/cpu-mixed-s0.jsonl`。策略对照的实验身份、源码快照、合法失败终态和严格验收规则见 [策略回放与验收](docs/BENCHMARKS.md)。

`bash scripts/dev.sh runtime-benchmark` 直接测量 CPU Runtime 的固定 prefill、decode 和 mixed 输入，交替采集无计时与分阶段计时的独立进程，核对完整 logits 摘要及 KV 状态。接口、矩阵形状、线程池等待时间和开销边界见 [Runtime 计时与模型基准](docs/RUNTIME_PROFILING.md)。

`scripts/Benchmark-Telemetry.ps1` 交替运行 `off / batches / stages` 与两种调度策略，关联每个 SSE token 的 batch、Engine 发布间隔及客户端 ITL。数据结构、到达率缩放、有界采集与验收见 [在线 batch 与 token 时间线](docs/BATCH_TELEMETRY.md)。

对照只改变 `mixed` / `prefill_first` 策略，每次重启服务、执行相同 warmup、交替运行顺序。报告保留逐请求 token ID、token 到达时间、失败、调度延迟和服务端配置；失败请求不会从总请求数中删除。

结果与限制见 [验证记录](docs/VALIDATION.md)。SIMD 内核的微基准加速不能当作模型或 Serving 的端到端加速。

CPU KV 对照使用同一模型、相同 F16 K/V、8 线程和固定 token trace，分别记录 MiniLLM 与 llama.cpp 的长上下文 TPOT。隔离基准在相同 AVX2/F16C attention 数学下只切换物理分页与按层连续布局，不把该结果表述为 llama.cpp 内核性能。原始报告及汇总位于 `benchmarks/results/kv-cache-cpu/`。

## KV 容量

```text
FP16 KV bytes/token = 2 * 28 layers * 8 KV heads * 128 head_dim * 2 bytes
                    = 112 KiB
16-token physical page = 1.75 MiB
8192-token KV capacity = 896 MiB
```

MiniLLM 的物理页在追加 token 时按需分配；释放后进入 free list，已分配的底层缓冲保留供复用。Serving 的 `BlockPool` 是独立的容量信用管理，按请求的 prompt + 输出上限保守预留，不是 GPU 地址分配器，也不是已经完成的增量准入优化。

## 代码导航

```text
include/minillm/      Mini Runtime 公共接口
src/minillm/          SIMD、GGUF mmap、Qwen3 forward、物理 paged KV
include/llmserve/     Serving 公共接口
src/engine.cpp       请求生命周期、准入、缓存和单执行线程
src/scheduler.cpp    mixed / prefill-first 调度
src/prefix_index.cpp Token Trie 和 LRU
src/mini_runner.cpp  Mini Runtime 适配器
src/llama_runner.cpp llama.cpp CPU/CUDA 适配器
src/http_server.cpp  HTTP/SSE
apps/                CLI、服务入口和 C++ 基准工具
tests/               单元、模型与在线验证
benchmarks/          固定输入及实测报告
```

模型所有权与分词接口见 [Host Model](docs/HOST_MODEL.md)；依赖与贡献边界见 [THIRD_PARTY.md](THIRD_PARTY.md)；设计问题见 [PROBLEM_CHECKLIST.md](docs/PROBLEM_CHECKLIST.md)；论文与固定源码入口见 [REFERENCES.md](docs/REFERENCES.md)。

## 版本与问题管理

本仓库公开发布；`main` 为主分支，功能改动使用独立分支与有意义的提交，已发布标签不覆盖。具体约定见 [VERSION_CONTROL.md](docs/VERSION_CONTROL.md)。

遇到的问题按编号记录在 [ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md)，包含现象、原因、解决方法、验证证据和仍未解决的事项。模型权重、第三方 checkout、构建产物、运行状态与本地 Python 辅助实验不上传。

## 项目计划

当前路线见 [PROJECT_PLAN_V2](docs/PROJECT_PLAN_V2.md)，实施规范见 [NEXT_SPEC](docs/NEXT_SPEC.md)，验收状态见 [执行状态](docs/EXECUTION_STATUS.md)。[原项目计划](docs/PROJECT_PLAN.md) 保留为历史参考。

1. V2-M0：实验依赖可用性检查、完整证据包导出与固定数值验证契约。
2. V2-M1：独立自有 CUDA 构建、常驻有效权重、连续 GPU KV、完整 Qwen3 模型与真实 token CLI。
3. V2-M2：接入现有 Serving，验证 HTTP/SSE 与请求生命周期。
4. V2-M4/M5：在连续 GPU 基线和压力证据上推进分页 attention、公平性与准入策略。

CPU 保持独立产品与数值参照；V2-M3 的两项有界研究按证据启动。CUDA 完整模型、GPU Serving 与 GPU PagedAttention 分别验收。
