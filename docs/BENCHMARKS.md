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

`BinaryDirectory` 默认按 `mini`、`mini-cuda`、`llama` 选择当前平台的 CPU、自有 CUDA 或上游 CUDA 产品目录。WSL 的 `bash scripts/dev.sh own-cuda benchmark ...` 使用自有 Runtime；`cuda benchmark ...` 使用 llama.cpp CUDA 参照。模型和来源可通过 `Model`、`ModelManifest` 指定，两者的文件名、字节数和 SHA-256 必须相符。

own-CUDA 的默认配置为 S=4、Lmax=2048、credits=8192、B=128、chunk=32、prefix=0，
采用 source Q8_0 / device F32 / activation F32 / KV F16、单 stream 同步执行。
`Device` 和 `DeviceBudgetBytes` 仅适用于此后端。连续 KV 不提供页数，ready/live/resident
按 [CUDA Serving](CUDA_SERVING.md) 解释。完整 raw 默认放在 `.run/`，按
[产物政策](ARTIFACT_POLICY.md) 发布一个 canonical bundle，Git 只保留固定输入、小索引和摘要。

输出目录必须为空；未指定时使用包含时间、源码提交和随机后缀的独立目录。采集前对当前 CMake 工程增量构建 `llmserve` 和 `llmserve-bench`，失败时不使用旧二进制继续测量。Windows 增量编译仍需要完整的 MSVC 开发环境。

## 实验身份

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 源码、二进制、编译器与配置、依赖、模型来源、trace、完整 EngineConfig、算术、环境、轮次和比较规则 |
| `source-state.json` | `source.scope` 所列构建输入的路径、大小和 SHA-256 |
| `source-snapshot.zip` | 对应的源码与构建、测试、运行脚本快照，不含模型、依赖 checkout 或编译产物 |
| `trace.jsonl` | 输入 trace 的原始字节副本 |
| `mixed-N.json`、`prefill_first-N.json` | 每轮服务前后快照、完整请求、token、终态和时间戳 |
| `*-process.json`、`*-server.*.log`、`*-client.*.log` | 每轮命令、进程身份、客户端退出码、停服状态与原始日志；未直接取得的服务端退出码为 null |
| `*-gpu.csv`、`*-gpu.stderr.log` | 可选低频 `nvidia-smi` 整设备采样与诊断 |
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
- `GpuSamplePeriodMs` 默认 0；启用时至少 1000 ms，使用 `nvidia-smi` 采集所选整张设备，而非本进程。窗口从服务 ready 后至停服，包含预热与客户端调度等待；不将该窗口的采样均值称为稳态 SM 利用率。
- MiniLLM 使用 F32 激活和 F16 KV；llama.cpp 的激活算术记录为 `upstream_native`，不能把其量化内核与 MiniLLM 当作相同算术。

## 验收规则

验收使用归档 trace 的完整请求集合，并按大小写区分请求 ID。缺失、重复、额外请求，缺失轮次、错误策略、未声明的配置变化，以及模型、二进制或 trace 身份差异均被拒绝。

成功请求必须具有单个 SSE `[DONE]` 和单个终态，usage、token 数、`max_tokens` 和结束原因相符。`ignore_eos=true` 时必须生成完整输出。成功请求的 token 序列与显式声明的 `mixed-0` 参照逐请求一致。

`AllowedRequestOutcomes` 默认只有 `success`。压力实验可以显式加入 `queue_full`、`timeout`、`cancelled`、`backpressure`；客户端连接错误或不完整响应不能作为合法终态。所有请求仍须保留，失败请求不能计入 goodput。其他轮次中的成功请求必须在声明的参照轮次中有成功输出，否则验收失败。全部失败的合法压力记录可以通过结构验收，但 `successfully_compared_requests=0`，不代表模型输出验证通过。

吞吐、goodput、延迟分位数及完整回放时长均根据原始请求重新核对。TTFT 从客户端实际发送时刻开始；TPOT 是每请求平均 token 间隔，ITL 单独统计。`max_itl_ms` 是每请求最大间隔，`request_max_itl_ms` 汇总其分布；历史报告可缺省这两项。单输出 token 的 TPOT 和最大 ITL 为 `null`，不作为零延迟样本。吞吐和 goodput 的分母包含整个回放时长，失败请求不从请求集合中删除。

## 固定 Trace

`llmserve-bench --make-trace` 通过现有 `/tokenize` 取得词元，再按指定长度重复/截断，
不需要执行模型生成。`--workload default` 保留每四个请求一长三短；
`mixed-length` 按 short/medium/long 循环，`burst-reuse` 每四个请求同时到达且短长交错。
指数到达使用 `--rate`、`--seed`，突发间隔使用 `--burst-gap-s`。
SLO 写入每行，冻结后以原始字节和 SHA-256 为准，不依赖跨标准库重现随机分布。

`CUDA-SERVE-001` 的固定输入与预算见
[Serving 协议](../benchmarks/results/cuda-serving-001/protocol.json)：
两个 24 请求 trace、每请求 32 个输出、两策略各三轮；SLO 是实验目标，不是产品承诺。

验收失败时写入 `analysis-failure.json`，返回非零状态；原始文件与既有 `validation-summary.json`、统计汇总保持原字节。旧的成功记录只代表当时的验收，不能代替本次退出状态。成功时先完成所有验证和 JSON 序列化，再逐文件原子发布汇总，最后发布验收标记；发布异常恢复原文件。本接口不支持并发写同一归档目录。

## 归档可用性与导出

```bash
pwsh -NoProfile -File scripts/Test-EvidenceAvailability.ps1 \
  -Directory benchmarks/results/evidence-m0/baseline

pwsh -NoProfile -File scripts/Export-BenchmarkBundle.ps1 \
  -Directory benchmarks/results/evidence-m0/baseline \
  -Output /tmp/runtime-evidence.zip
```

可用性检查只读输入，输出每项必需文件的 `locator / mandatory / exists / sha256 / status`。可选 `-Output` 必须指向新文件。缺失 ZIP、摘要不符、源码状态与 ZIP 内容不一致或未完成采集均返回 `ARCHIVE_INCOMPLETE`；文件齐全的 `AVAILABLE` 与严格数值、统计验收通过是不同结论。

导出接受 Runtime 与单组 Serving 策略归档。它复制必需输入到临时目录，执行相应严格分析器，生成含逐文件 SHA-256 的 `bundle-manifest.json` 和可用性报告；ZIP 再解压到独立目录复验后发布，并提供 `.sha256`。包内 `verification/` 保存本次验收脚本，与原测量源码快照分别记录身份；`verification_entry` 指定复验入口。在线观测组包含 manifest 声明的 JSONL，并通过 `analyze_telemetry.py` 验收。跨组 `experiment.json` 的整体比较需要原在线观测入口单独处理。

包的用途是 `archive_revalidation`：脱离采集机器的原绝对路径，重算归档的身份、结果与统计。模型权重、实测二进制和工具链不在包内，`execution_dependencies_included=false`；重新执行模型仍需取得 manifest 固定的模型与依赖、构建相应源码并建立新的实验身份。归档复验不等于原二进制重跑。

历史归档和导出的包均在临时副本中复验。包中的文件哈希绑定导出时的字节，不能把重分析后改写的派生文件当作原包。缺少历史源码时不得依据当前源码重造同名 ZIP 或改写历史摘要；本机保留的原始 ZIP 也必须核对 manifest 和逐源码摘要后才能纳入交付。

## 覆盖范围

CTest 的 `benchmark-validation` 套件覆盖正常报告、合法失败、单 token、大小写 ID、身份与配置差异、输出截断、统计不一致、轮次缺失、旧汇总残留和 CTest 归档格式。

本协议用于 Serving 策略对照。Runtime 的模型级计时、独立进程对照及 manifest 见 [Runtime 计时与模型基准](RUNTIME_PROFILING.md)。dot、KV 布局的统一 manifest，以及跨源码/二进制的优化前后比较仍属于独立工作。微基准结果和上游 CUDA 执行不能作为自研端到端加速证据。

自有 CPU8/CPU16/CUDA 的固定模型 workload、独立 KV 重建、A/A 与配对统计使用单独的 [CUDA 模型性能对照](CUDA_BENCHMARKS.md)，不适用本页的策略回放协议或旧 CPU Runtime 的共享前缀重置。

启用在线观测时，每份报告另有 `*-telemetry.jsonl`，通过 Python 3 验收完整的 batch/token 关联。跨观测模式比较使用 [在线观测入口](BATCH_TELEMETRY.md)，不能在同一策略组中隐式改变观测模式。
