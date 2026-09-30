# Mini LLM Runtime

C++20 实现的 CPU/CUDA Qwen3 推理 Runtime 与单 GPU 在线 Serving 研究原型。
从 GGUF 权重到生成 token，连接物理 KV、动态组批、HTTP/SSE 和故障回收；
目标是把模型计算、内存容量和服务完成点放在同一个可解释系统里。

## 自有实现

- **CPU Runtime**：文件映射与张量视图、SIMD/scalar dispatch、线程池、完整 Qwen3 forward。
- **KV 内存系统**：CPU 物理分页、prefix 共享与 COW；GPU 连续槽位和独占页池，信用与物理分配分开。
- **Own CUDA Serving**：常驻权重、自有 attention/greedy、动态 mixed batching、HTTP/SSE、poisoned fail-stop 和有界停服排空。
- **分层实验**：算子、模型和服务分别验证；保留数值失败、性能退化和不确定结果，不用微基准代替端到端收益。

GGUF 元数据解析、tokenizer 和可选参照 backend 复用 llama.cpp；
CUDA GEMM 使用 cuBLAS，归约使用 CUB。自有 CPU/CUDA forward 不调用
`llama_decode()`，也不通过 CLI 子进程执行模型。[依赖与归属](THIRD_PARTY.md)

## 架构与范围

```text
HTTP / SSE -> Engine: admission, credits, lifecycle -> schedule_batch
                                                    |
                       MiniRunner / MiniCudaRunner / LlamaRunner
                            |             |               |
                       CPU Runtime    CUDA Runtime    upstream reference
                            |             |
                      FP16 paged KV   FP16 contiguous KV (default)
                                                    |
                                  checked samples -> RequestHandle -> SSE
```

默认快速入口为 CPU `mini`，GPU 展示入口为 `mini-cuda`。
Own CUDA 限定 Qwen3-0.6B、单 GPU、S≤4、L≤2048、B≤128、greedy、单 stream、
同步 execute；source Q8_0 在初始化时转为 F32 device weights，不是 native Q8 CUDA。
`llama` 是独立的上游参照路径。

GPU paged 为 **B：容量／研究 opt-in**，不改变 contiguous 默认。
FP16 matrix 候选未通过原数值门槛，禁止用于 Serving。
不支持随机采样、任意模型架构、GPU prefix sharing、Graph/async、多 GPU 或生产级 SLO。

## 结果与取舍

以下为不同阶段的冻结采集，不是当前 main 重跑；不能混为一个统一加速比。

| 实验 | 结果 | 限制与证据 |
| --- | --- | --- |
| CUDA Serving，mixed-length，mixed 对 prefill_first | 吞吐中位数约 +7.09% | 特定 trace 的调度对照；[原始研究](benchmarks/results/cuda-serving-001/README.md) |
| GPU KV，同容量 8192 tokens，paged 对 contiguous | 配对吞吐中位数 -15.73% | 未通过 10% 护栏，不节省该组显存；[分页研究](docs/GPU_KV_STUDY.md) |
| GPU KV，同 288 MiB KV 子预算 | 配对吞吐中位数 +4.31% | 异长容量更灵活，但 TPOT/ITL 更高，不是低延迟改进；[证据包](benchmarks/results/gpu-kv-001/README.md) |

完整 CPU micro、CUDA model、Serving 与 FP16 负结果见[性能](docs/PERFORMANCE.md)。
主要功能和性能实验已冻结，剩余范围仅为文档、交付和必要修复。

## 快速开始

Linux／WSL2 需要 Git、CMake ≥3.24、C++20 编译器、Ninja、Python 3、
PowerShell 7 和 curl。Python/PowerShell 用于脚本及完整测试，不是 C++ Runtime
的执行依赖。Own CUDA 另需 CUDA Toolkit ≥12.8、兼容驱动和 GPU。
WSL 使用 Windows GPU 驱动，不在 WSL 内安装 Linux 显示驱动。

### CPU

在自行选择的 Linux 工作目录中执行：

```bash
git clone https://github.com/lilong555/mini-llm-runtime.git
cd mini-llm-runtime
bash scripts/dev.sh dependencies
bash scripts/dev.sh model
bash scripts/dev.sh build
bash scripts/dev.sh test
build/wsl-cpu/bin/mini-llm --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --tokens 8
bash scripts/dev.sh serve --port 8000
```

### Own CUDA

承接已准备的依赖和模型。`89` 是已验证的 RTX 4070 Laptop 架构设置，
也是脚本当前默认值，不适用于所有 GPU；其它设备须选对应架构。

```bash
CUDA_ARCHITECTURES=89 bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda generate --prompt "The capital of France is" --tokens 8
bash scripts/dev.sh own-cuda serve --port 8001
```

`serve` 前台运行，Ctrl+C 停止。另开终端访问对应端口，CPU 用 8000，CUDA 用 8001：

```bash
PORT=8001
curl --fail --silent "http://127.0.0.1:$PORT/health"
curl --fail -N "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"The capital of France is","max_tokens":8,"temperature":0,"stream":true}'
curl --fail --silent "http://127.0.0.1:$PORT/metrics"
```

只监听 loopback，没有认证，不应直接暴露到公网。
安装与编译耗时取决于网络、机器和缓存，不承诺固定完成时间。
参照 F32 模型生成、Windows 命令和进阶验证见[开发指南](docs/WSL_DEVELOPMENT.md)；
公共 HTTP 合同和 GPU 参数见[Serving 说明](docs/CUDA_SERVING.md)。

## 深入阅读

本页是系统总览，其余四个主入口：
[架构与所有权](docs/ARCHITECTURE.md)、
[性能与证据边界](docs/PERFORMANCE.md)、
[GPU KV 取舍](docs/GPU_KV_STUDY.md)、
[验证](docs/VALIDATION.md)。

代码入口：[CPU Runtime](src/minillm/runtime.cpp)、
[CUDA Runtime](src/minillm/cuda/runtime.cpp)、
[MiniCudaRunner](src/mini_cuda_runner.cpp)、
[Engine](src/engine.cpp)、[Scheduler](src/scheduler.cpp)、
[HTTP/SSE](src/http_server.cpp)。
问题与失败见[工程记录](docs/ENGINEERING_LOG.md)，发布规则见[版本约定](docs/VERSION_CONTROL.md)。
