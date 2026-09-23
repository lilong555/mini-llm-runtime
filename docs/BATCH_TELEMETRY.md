# 在线 batch 与 token 时间线

LLMServe 提供默认关闭的在线观测：`off` 保持普通响应，`batches` 记录调度轮次与输出关联，`stages` 同时采集 MiniLLM Runtime 阶段。所有观测在模型线程写入启动时预分配的有界内存，`Engine::stop()` 完成后序列化为 JSONL。请求处理和 Runtime 热循环不写 JSON 或文件。

## 采集

WSL2 需要原生 PowerShell、Python 3 和已构建的 CPU 产品。下面的入口交替观测模式与策略顺序，每份报告使用独立服务进程：

```bash
bash scripts/dev.sh build
pwsh -NoProfile -File scripts/Benchmark-Telemetry.ps1 \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl -Trials 3 \
  -OutputDirectory .run/online-profile
```

每轮包含 `off / batches / stages`，每种模式各运行 `mixed / prefill_first`。奇数轮反转模式及策略顺序。每个子目录具有完整的 [Serving manifest](BENCHMARKS.md)、源码快照、trace、二进制和模型摘要；根目录的 `experiment.json` 记录预定采集顺序，`collection-status.json` 记录完整或失败终态。实验期间源码和二进制必须保持相同。

单组采集可直接使用已有策略入口：

```bash
bash scripts/dev.sh benchmark \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl -Trials 3 \
  -Telemetry stages -TelemetryCapacity 1024 -ArrivalScale 6 \
  -OutputDirectory .run/online-low-load
```

`ArrivalScale` 乘以每个原始 `arrival_s`，保留 trace 原始字节和摘要；放大值降低到达密度，不改变 prompt 或输出上限。缩放后的到达时间不能超过 3600 秒。生成新 trace 的 `llmserve-bench --make-trace --rate` 支持小数到达率。

直接启动服务时指定文件：

```bash
bash scripts/dev.sh serve --port 8081 --telemetry stages \
  --telemetry-capacity 1024 --telemetry-output .run/batch-telemetry.jsonl \
  --shutdown-file .run/online.stop
```

父目录须存在，输出文件不得已存在，启动前须移除上次运行的停止标记。创建停止标记可请求正常停服。强制结束进程或写文件失败时，产物不具有完整 footer，不能用于性能验收。

## 数据与计时

每份 JSONL 由 header、按顺序排列的 batch、footer 组成。header 包含模式、后端、策略、容量、实际预分配载荷字节和时钟口径；footer 包含记录数、丢弃数、Engine 错误及停服后的物理资源。`storage_bytes` 不包括 allocator 元数据或 Runtime 自身的 profile 存储。

| 字段 | 口径 |
| --- | --- |
| `batch_id` | 单个服务实例内的调度轮次，包含预热 |
| `start_ns`、`finish_ns` | 相对 Engine 启动观测时刻的 steady clock 纳秒 |
| `admission_ns` | 接收队列转移、过期回收、等待排序、准入及其缓存操作 |
| `scheduler_ns` | 构造调度输入、选择 batch，包含该边界内的观测记录准备 |
| `prepare_ns` | 构造 token batch、记录 slice 和前置资源快照 |
| `runner_ns` | 完整 `ModelRunner::execute` 或 `execute_profiled`，包含 forward、采样及适配器工作 |
| `finish_ns - runner_start_ns - runner_ns` | runner 返回后的缓存、token 发布、终态清理及本轮局部对象管理 |
| `waiting_requests`、`active_requests` | 准入结束后的模型线程视图；不包含随后到达的 incoming 请求 |
| `context_before/after_sum/max` | 本轮实际参与序列的逻辑 KV 长度；不是池内全部缓存的长度 |
| `reserved_unique_blocks` | Serving 容量信用，按输出上限预留 |
| `resources_before/after` | 执行前后 MiniLLM 的活跃物理 KV 页及保留存储的 KV payload 字节 |

`resources_after` 在缓存发布和本轮终态回收之前采集；`resources_final` 在全部活动与缓存引用释放后采集。live pages 为零不代表已分配存储已经归还系统。llama.cpp 的物理快照为 `null`；当前没有发布共享引用、COW 字节或进程 RSS，不能从容量信用推算这些值。

`context_after` 根据调度输入计算；只有成功完成的 batch 才代表已经提交的上下文。异常记录的 `completed=false`，预期长度不能用作部分失败后的真实模型状态。

每条 slice 包含 request ID、请求创建顺序、sequence、prefill/decode、输入数、KV 起始位置、logits 数、输出序号、采样 token 及事件入队时间。同一 sequence 可用于不同请求，关联键使用请求创建顺序，不能只使用 sequence 或可复用的请求 ID。

成功执行满足：

```text
start_ns + admission_ns + scheduler_ns + prepare_ns == runner_start_ns
runner_start_ns + runner_ns <= finish_ns
sum(runner.stages.wall_ns) + runner.unaccounted_ns == runner.forward_ns
runner.forward_ns + runner.sampling_ns <= runner_ns
```

MiniLLM 的阶段按固定枚举在各层求和，保留调用次数、矩阵 M/N/K 和线程池经过时间。LM head 当前每个 logits 单独执行 `M=1`。共享矩阵阶段只记录一次，不分别完整归入 prefill 与 decode。阶段细节和重叠线程时间的解释见 [Runtime 计时](RUNTIME_PROFILING.md)。llama.cpp 没有接入这份 Runtime profile，`runner` 为 `null`，Engine 仍能记录整体执行时间。

## SSE 与客户端关联

启用观测时，每个带 `token_id` 的 SSE 事件具有 `telemetry`：`batch_id`、`request_order`、从零开始的 `token_index`、`engine_elapsed_ns`。后者是事件入队尝试前的时间，不是客户端收到字节的时间。批次记录的 `emitted` 表示入队是否成功；取消、超时或背压可能使已有 sample 没有成功发布。

回放报告为每个客户端 token 保存对应的 `token_telemetry`。关闭观测时这些元素为 `null`，SSE 不增加观测字段。Engine 与客户端时钟原点不同，不相减其绝对值。分析器比较同一请求中相邻 token 的客户端间隔与 Engine 发布间隔；两者之差还包含线程调度、事件队列、socket 和客户端缓冲，不能直接解释为网络耗时。

## 验收与分析

`Benchmark-Policies.ps1 -Telemetry ...` 自动运行在线验收。目录迁移后可重新验收并跨模式比较：

```bash
python3 scripts/analyze_telemetry.py \
  .run/online-profile/trial-0-off \
  .run/online-profile/trial-0-batches \
  .run/online-profile/trial-0-stages \
  --output .run/online-profile/rechecked.json
```

先执行 Serving 的请求、配置、终态、统计及源码身份验收，再检查所有 batch 的顺序、计时守恒、slice 集合、上下文推进、阶段、形状和 SSE 逐 token 对应关系。预热 batch 保留在原始 JSONL，通过 `server_before.batches` 从测量汇总中排除。跨模式比较固定源码、二进制、模型、trace、到达缩放、其他配置及环境，检查完整输出 token 一致。

`telemetry-summary.json` 保存原始文件 SHA-256、batch 组成直方图、scheduler/admission/runner 经过时间占比、阶段时间和、客户端及 Engine 的每请求最大 ITL。模式对照保留进程数、延迟及吞吐范围、中位数差异、batch 数和 mixed batch 数。异常值与不利结果不剔除。

缓冲达到容量后，服务继续处理请求，记录 `dropped`，后续轮次不再采集阶段；分析器拒绝将这类报告作为完整证据。默认容量 1024，允许 1–16384，采集规模超过容量时需要新目录及更大容量重新运行。`Engine::telemetry()` 仅允许在 `stop()` 后读取。

失败分析会写入 `status=failed`，不会保留旧成功汇总。本时间线分析要求成功请求集合；取消、背压与超时由核心和 HTTP 回归覆盖，协议压力报告仍可使用 Serving 验收器。

## 适用边界

固定容量与模式的启动分配不计入请求阶段，观测字段复制、时钟、SSE 序列化和额外 profile 都可能改变在线 batch 组成。独立进程 off/on 差异包含系统噪声与该扰动，不能当作纯插桩耗时，也不是相对旧二进制的零开销证明。

目前没有 sequence create/share/clear 事件和完整 token 输入流，JSONL 不能用于离线 Runtime replay。attention 仍合并页访问、QK、softmax、PV；worker 字段不是操作系统调度时间线。进一步判断 `ENG-008`、`ENG-018` 的根因，需要相应受控实验，不能仅凭占比证明因果或将已有页布局微基准等同于端到端收益。
