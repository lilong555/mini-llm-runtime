# WSL2 原生开发

建议在 WSL 的 Linux 文件系统内选择工作目录，源码、依赖、模型和构建产物
使用同一套 Linux 工具链。不需要作者的目录布局，不复用 Windows CMake 缓存。

从 PowerShell 进入：

```powershell
wsl -d Ubuntu
```

进入 WSL 后，在自行选择的工作目录中 clone 仓库。编辑器使用 WSL 远程模式；
终端、Git、CMake、编译器、调试器和服务均在 Ubuntu 中运行。

## 工具链

前置工具为 Git、CMake ≥3.24、C++20 编译器、Ninja、Python 3、PowerShell 7、
curl；完整测试需要 Python/PowerShell，不能通过关闭测试工具要求来替代验收。
C++ Runtime 运行时不依赖 Python/PowerShell。GPU 另需 CUDA Toolkit ≥12.8
及兼容的驱动、GPU 和目标架构。`CUDA_ARCHITECTURES` 默认 89 是已验证
RTX 4070 Laptop 的设置，不是通用默认。

历史验证环境为 Ubuntu 22.04、WSL2 `5.15.167.4-microsoft-standard-WSL2`、Ryzen 7 7745HX、RTX 4070 Laptop GPU。工具与验证证据见 [环境验收](../benchmarks/results/wsl-environment/README.md)；新机器需要实际验证，不自动继承此表。

| 用途 | 当前工具 |
| --- | --- |
| 默认 C++20 编译器 | GCC 11.4.0 |
| 构建 | CMake 4.4.0、Ninja 1.13.0 |
| 编辑器与辅助编译器 | clangd、clang-tidy、clang-format、Clang 19.1.7 |
| 脚本 | Python 3.10.12、PowerShell 7.6.6 |
| CPU 采集 | perf 5.15.209；进程级用户态事件可用 |
| CUDA 编译 | CUDA Toolkit 12.8、nvcc 12.8.93；host compiler 为 GCC 11.4.0 |
| GPU 时间线 | Nsight Systems 2026.1.3.425 |
| GPU 内存检查 | CUDA 12.8 的 Compute Sanitizer |
| GPU 硬件计数器 | Nsight Compute 2025.1.1；单 kernel 基础指标采集可用 |

CUDA 驱动由 Windows 提供，当前为 `591.74`；WSL 中只使用 Linux Toolkit，不安装 Linux NVIDIA 显示驱动。单 GPU 工作流不需要 NCCL。

`nsys` 的用户级入口为 `~/.local/bin/nsys`，安装目录为 `~/.local/opt/nsight-systems-2026.1.3`；登录终端优先使用该入口。安装包来源、版本和 SHA-256 固定在 [环境元数据](../benchmarks/results/wsl-environment/environment.json)。`nvcc`、`ncu`、`compute-sanitizer` 和 `cuda-gdb` 使用 `/usr/local/cuda-12.8/bin`。

## CPU 工作流

以下入口使用完整测试配置。模型和依赖由受版本控制的 manifest 固定。

```bash
git clone https://github.com/lilong555/mini-llm-runtime.git
cd mini-llm-runtime
bash scripts/dev.sh dependencies
bash scripts/dev.sh model
bash scripts/dev.sh build
bash scripts/dev.sh test
bash scripts/dev.sh validate
bash scripts/dev.sh check-http 8015
```

构建目录为 `build/wsl-cpu`，配置为 `RelWithDebInfo`，默认 4 个编译任务，可用 `JOBS=8 bash scripts/dev.sh build` 调整。模型文件依据 `models/manifest.json` 和 `models/reference-manifest.json` 校验大小及 SHA-256；已有不匹配文件导致失败，不会自动覆盖。

根目录 `.clangd` 指向 `build/wsl-cpu/compile_commands.json`。先完成 CPU 构建，再打开 C++ 文件；补全和诊断使用实际的 C++20、头文件路径及 SIMD 编译参数。

```bash
clangd --check=src/minillm/kernels.cpp
clang-tidy -p build/wsl-cpu src/minillm/kernels.cpp
```

`build` 在缺少前置工具时给出诊断，不自动安装系统软件。
`test` 保留完整的成功用例输出，并在没有注册测试时返回失败；
实际注册项可用 `ctest --test-dir build/wsl-cpu -N` 查看。
PowerShell 的基准验收与服务启停均可在 Linux 原生执行。

完整模型验证独立于 CTest。模型套件通过测试侧屏障构造真实 prefill/decode 混合批，保留数值、生成、前缀复用和 KV 回收检查。CPU 1、2、8 线程及 CUDA 参照的完整报告见 `benchmarks/results/validation/wsl-deterministic/`；原有时序问题及失败证据见 `ENG-017`。

## 服务与 HTTP 验证

```bash
bash scripts/dev.sh serve --port 8000
```

在第二个 WSL 终端运行：

```bash
curl --fail http://127.0.0.1:8000/health
bash scripts/dev.sh http-test 8000
build/wsl-cpu/bin/llmserve-bench --port 8000 \
  --trace benchmarks/traces/cpu-mixed-s0.jsonl --output .run/wsl-replay.json
```

服务在前台运行，使用 Ctrl+C 退出；独立自动化进程可传入 `--shutdown-file .run/server.stop`，创建该文件请求停止。只监听 loopback，Windows 可通过 WSL localhost 转发访问。测试报告位于 `.run/wsl-model.json`、`.run/wsl-http.json` 和 `build/wsl-cpu/test-results.xml`。

`check-http` 使用指定端口启动临时服务，完成检查后通过停止标记回收自己的服务进程。运行前应选择空闲端口。Linux 的原始 replay 不等同于完整策略对照验收。

## 策略对照

```bash
bash scripts/dev.sh benchmark \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl \
  -Trials 3 -TraceSeed 0 -Port 8031 \
  -OutputDirectory benchmarks/results/policy-run
```

该入口使用原生 PowerShell，核对源码和二进制身份，按轮次重启服务、交替策略并严格验收。输出目录必须为空；manifest、原始 trace、源码快照、全部请求结果和验收报告保存在同一目录。协议、压力实验终态和统计口径见 [策略回放与验收](BENCHMARKS.md)。

## 自有 CUDA Runtime

```bash
CUDA_ARCHITECTURES=89 bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda generate --prompt "The capital of France is" --tokens 8

# 短数值检查需要 matched-weight F32 reference，不仅是 Q8_0 模型。
python3 scripts/models.py --reference --converter build/wsl-cpu/bin/mini-llm
OUT="$PWD/.run/finalization-$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%dT%H%M%SZ)-$$"
test ! -e "$OUT" || exit 1
mkdir -p "$OUT"
bash scripts/dev.sh own-cuda model-check "$OUT/model"
bash scripts/dev.sh own-cuda serving-check "$OUT/serving.json"
bash scripts/dev.sh own-cuda check-http 8015 "$OUT/http.json"
```

`own-cuda` 使用独立的 `build/wsl-own-cuda`，设置 `MINILLM_ENABLE_CUDA=ON`、`LLMSERVE_CUDA=OFF`，提供常驻 FP32 有效权重、连续 FP16 KV、完整 Qwen3 forward、greedy CLI 和 `serve`。GPU HTTP 与实模型检查使用 `own-cuda check-http`、`own-cuda serving-check`，配置与边界见 [CUDA Serving](CUDA_SERVING.md)。`storage-*`、`model-*` 使用固定模型，模型数值验证还需要 matched-weight F32 参照；报告路径必须尚不存在。构建与接口契约见 [CUDA Runtime](CUDA_RUNTIME.md)，全量语料见 [CUDA 数值验证](CUDA_NUMERICS.md)。CPU 可执行文件不链接该 CUDA target。

## CUDA 参照后端

RTX 4070 Laptop GPU 使用架构 `89`。Windows CUDA/MSVC 产物不能用于 Linux 构建。`cuda` 前缀选择独立的 `build/wsl-cuda` 目录、llama.cpp 后端和 `--gpu-layers 99`：

```bash
bash scripts/dev.sh cuda build
bash scripts/dev.sh cuda test
bash scripts/dev.sh cuda validate
bash scripts/dev.sh cuda check-http 8016
bash scripts/dev.sh cuda serve --port 8001
```

其他架构通过 `CUDA_ARCHITECTURES=数字 bash scripts/dev.sh cuda build` 指定。CUDA 的报告为 `.run/wsl-cuda-model.json`、`.run/wsl-cuda-http.json` 和 `build/wsl-cuda/test-results.xml`，不覆盖 CPU 报告。`cuda validate` 中 MiniLLM 仍在 CPU 执行，只将 F32 数值参照放在 GPU。

上游 CUDA 参照、自有 CUDA 连续 KV Runtime 和下述向量冒烟检查是不同执行路径，均不能作为自有 GPU PagedAttention 的验收证据。

## 性能采集

### CPU

```bash
mkdir -p .run
perf stat -e task-clock:u,cycles:u,instructions:u -- \
  build/wsl-cpu/bin/mini-llm --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --tokens 8 --threads 8

perf record -e cpu-clock:u -F 199 --call-graph dwarf -o .run/perf.data -- \
  build/wsl-cpu/bin/mini-llm --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --tokens 8 --threads 8
perf report -i .run/perf.data --stdio --no-children --call-graph none
```

当前 `perf_event_paranoid=2`，只验证进程级用户态事件与调用栈，不假定系统级或内核态采集可用。不支持的事件应记为不可用，不能填零；不为运行上述命令放宽全局采集权限。短命令用于工具验收，不是性能基线。

### GPU

```bash
bash scripts/dev.sh cuda smoke
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 .run/cuda-smoke

profile=$(mktemp -d "$PWD/.run/profile.XXXXXX")
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --output="$profile/cuda" .run/cuda-smoke
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$profile/cuda.nsys-rep"
```

`smoke` 使用单个数字架构，默认 `89`；它编译 `scripts/cuda_smoke.cu`，执行 64 次 kernel，校验 1048593 个结果。验收要求 Nsight 报告确实包含 64 次 GPU kernel 和 3 次内存传输，不能只检查进程退出码或 CUDA API 表。当前 Unified Memory 跟踪有设备限制，不宣称已验证该能力。

GPU 硬件计数器以 Windows 主机授权为前提：在 NVIDIA 控制面板启用开发者设置，并在“管理 GPU 性能计数器”中授权访问。WSL 的 root 权限不能替代主机授权。当前环境已通过普通 WSL 用户的基础指标采集验证，复核命令：

```bash
ncu --target-processes all --launch-count 1 --set basic .run/cuda-smoke
```

报告应包含实际 kernel 耗时、SM/DRAM 周期和占用率，不只包含启动参数。该命令只采集一个 kernel，profiler 的多次 replay 不作为正式性能测量。验证结果见 [ncu-validation.json](../benchmarks/results/wsl-environment/ncu-validation.json)；出现 `ERR_NVGPUCTRPERM` 时仍应检查主机授权，见 `ENG-023`。

`.ncu-rep`、`.nsys-rep`、`.sqlite` 和 `perf.data` 保留在忽略的 `.run` 目录；入库的是小型摘要、诊断和摘要清单，不提交大型采集文件。

## 工作区与版本管理

Linux 副本保留原分支、提交历史和未提交内容。后续开发以 Linux 副本为准，避免两侧交替修改同一分支。构建产物、服务状态、依赖 checkout 和模型权重不提交；保留模型来源与小型验证报告。Windows 副本保留用于恢复，不自动删除。

## Windows 原生与参照后端

Windows 原生构建需要 Visual Studio 2022 C++ 工具链、CMake ≥3.24、
Python 3 和 PowerShell 7；选择 Ninja 时还需可用的 Ninja。
以下路径从仓库根目录开始，与 WSL 构建目录分开：

```powershell
.\scripts\Fetch-Dependencies.ps1
.\scripts\Download-Model.ps1
.\scripts\Build-LLMServe.ps1
.\build\cpu\bin\mini-llm.exe --model models\Qwen3-0.6B-Q8_0.gguf `
    --prompt "The capital of France is" --tokens 8
.\scripts\Start-LLMServe.ps1 -Backend mini
.\scripts\Stop-LLMServe.ps1 -Port 8000
```

启动脚本返回实际端口、PID 和日志；不要假定占用端口仍为 8000。
CUDA 参照后端的 `-Cuda` 控制上游 ggml，不是 own-CUDA：

```powershell
.\scripts\Build-LLMServe.ps1 -Cuda -CudaArchitectures 89
.\scripts\Start-LLMServe.ps1 -Backend llama -Port 8001
```

`89` 只对应适用的 Ada GPU；CUDA DLL 的 `%CUDA_PATH%\bin` 需要在 `PATH` 中。
自有 CUDA 的 Windows 选项与精度／布局边界见 [CUDA Serving](CUDA_SERVING.md)。
