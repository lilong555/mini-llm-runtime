# 策略回放与验收

`llmserve-bench` 是 C++ HTTP/SSE 回放客户端。`scripts/Benchmark-Policies.ps1` 负责构建检查、实验身份、独立服务进程和策略轮次；`scripts/Analyze-Benchmarks.ps1` 负责离线验收与统计汇总。当前比较维度只允许 `engine.policy`，固定其他模型、算术、二进制和服务配置。

## 运行

WSL2 / Linux 需要已配置的 CPU 构建、固定模型和原生 PowerShell：

```bash
bash scripts/dev.sh build
bash scripts/dev.sh benchmark \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl \
  -Trials 3 -TraceSeed 0 -Port 8031 \
  -OutputDirectory benchmarks/results/policy-run

pwsh -NoProfile -File scripts/Analyze-Benchmarks.ps1 \
  -Directory benchmarks/results/policy-run
```

Windows 使用相同的 PowerShell 入口：

```powershell
.\scripts\Build-LLMServe.ps1
.\scripts\Benchmark-Policies.ps1 -Backend mini `
    -Trace benchmarks/traces/cpu-mixed-s0.jsonl -Trials 3 -TraceSeed 0
```

`BinaryDirectory` 默认选择当前平台的 CPU 或 CUDA 产品目录。WSL 的 `bash scripts/dev.sh cuda benchmark ...` 选择 llama.cpp CUDA 参照后端；它不是自研 CUDA Runtime。模型和来源可通过 `Model`、`ModelManifest` 指定，两者的文件名、字节数和 SHA-256 必须相符。

输出目录必须为空；未指定时使用包含时间、源码提交和随机后缀的独立目录。采集前对当前 CMake 工程增量构建 `llmserve` 和 `llmserve-bench`，失败时不使用旧二进制继续测量。Windows 增量编译仍需要完整的 MSVC 开发环境。

## 实验身份

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 源码、二进制、编译器与配置、依赖、模型来源、trace、完整 EngineConfig、算术、环境、轮次和比较规则 |
| `source-state.json` | `source.scope` 所列构建输入的路径、大小和 SHA-256 |
| `source-snapshot.zip` | 对应的源码与构建、测试、运行脚本快照，不含模型、依赖 checkout 或编译产物 |
| `trace.jsonl` | 输入 trace 的原始字节副本 |
| `mixed-N.json`、`prefill_first-N.json` | 每轮服务前后快照、完整请求、token、终态和时间戳 |
| `validation-summary.json` | 验收状态、检查项、实际比较的成功请求数和错误 |
| `summary.json` | 仅在验收通过时生成的策略统计 |

`source.worktree_state_sha256` 绑定 `source-state.json`；`source.snapshot.sha256` 绑定源码快照。`git_dirty=true` 不替代源码快照。该范围覆盖构建和运行输入，不包含文档、实验结果和本地服务状态。

每轮原始报告绑定 manifest、服务端、客户端、模型和 trace 的 SHA-256。客户端还对实际读取的 trace 原始字节计算 FNV-1a，保留 CRLF 和文件末尾换行的差异。采集脚本在每轮前后核对源码、二进制、模型、trace 和依赖状态。

归档内的 trace 和源码路径是相对路径，整个目录移动后仍可验收。离线验收不要求原模型和可执行文件仍位于采集机器的绝对路径；它核对归档及采集时绑定的身份，不构成可信执行证明。

## 运行协议

- 每轮启动并回收独立服务，按 `mixed / prefill_first`、`prefill_first / mixed` 交替顺序执行。
- 默认预热为 `Hello`、8 个输出 token、`ignore_eos=true`，使用独立的 `benchmark-warmup` namespace。`NoWarmup` 明确关闭此步骤。
- 服务重启不代表操作系统文件缓存冷启动；默认协议也不是独立的冷、热前缀对照实验。
- 已知 trace seed 通过 `TraceSeed` 提供；未知值记录为 `null`，不从文件名推测。
- 采集不常驻数值参照模型，观测默认关闭。`Telemetry` 可选择 `batches` 或 `stages`；同一个策略组固定该选项。CPU 频率与温度未采集时记录为 `null`，不能解释为零或恒定。
- `ArrivalScale` 默认 1，以固定比例缩放原始到达时间并写入 manifest。`PolicyOrderOffset` 默认 0；值为 1 时从 `prefill_first` 开始交替，用于外层观测模式实验。
- MiniLLM 使用 F32 激活和 F16 KV；llama.cpp 的激活算术记录为 `upstream_native`，不能把其量化内核与 MiniLLM 当作相同算术。

## 验收规则

验收使用归档 trace 的完整请求集合，并按大小写区分请求 ID。缺失、重复、额外请求，缺失轮次、错误策略、未声明的配置变化，以及模型、二进制或 trace 身份差异均被拒绝。

成功请求必须具有单个 SSE `[DONE]` 和单个终态，usage、token 数、`max_tokens` 和结束原因相符。`ignore_eos=true` 时必须生成完整输出。成功请求的 token 序列与显式声明的 `mixed-0` 参照逐请求一致。

`AllowedRequestOutcomes` 默认只有 `success`。压力实验可以显式加入 `queue_full`、`timeout`、`cancelled`、`backpressure`；客户端连接错误或不完整响应不能作为合法终态。所有请求仍须保留，失败请求不能计入 goodput。其他轮次中的成功请求必须在声明的参照轮次中有成功输出，否则验收失败。全部失败的合法压力记录可以通过结构验收，但 `successfully_compared_requests=0`，不代表模型输出验证通过。

吞吐、goodput、延迟分位数及完整回放时长均根据原始请求重新核对。TTFT 从客户端实际发送时刻开始；TPOT 是每请求平均 token 间隔，ITL 单独统计。单输出 token 的 TPOT 为 `null`，不作为零延迟样本。吞吐和 goodput 的分母包含整个回放时长，失败请求不从请求集合中删除。

验收失败时保存 `valid=false` 的报告并移除旧 `summary.json`；原始报告不修改。已有历史报告缺少 manifest 时不能直接进入该验收链路，也不能通过补写猜测的摘要将其视为当前基线。

## 覆盖范围

CTest 的 `benchmark-validation` 套件覆盖正常报告、合法失败、单 token、大小写 ID、身份与配置差异、输出截断、统计不一致、轮次缺失、旧汇总残留和 CTest 归档格式。

本协议用于 Serving 策略对照。Runtime 的模型级计时、独立进程对照及 manifest 见 [Runtime 计时与模型基准](RUNTIME_PROFILING.md)。dot、KV 布局的统一 manifest，以及跨源码/二进制的优化前后比较仍属于独立工作。微基准结果和上游 CUDA 执行不能作为自研端到端加速证据。

启用在线观测时，每份报告另有 `*-telemetry.jsonl`，通过 Python 3 验收完整的 batch/token 关联。跨观测模式比较使用 [在线观测入口](BATCH_TELEMETRY.md)，不能在同一策略组中隐式改变观测模式。
