# WSL Runtime 阶段基线

2026-09-22，WSL2、Ryzen 7 7745HX（8 核、16 逻辑 CPU）、GCC 11.4、`RelWithDebInfo`，Qwen3-0.6B Q8_0、F32 激活、F16 KV、AVX2/FMA/F16C。数据来自直接调用 MiniLLM CPU Runtime 的独立进程，不包含 HTTP 或调度器。

## 证据与协议

| 数据组 | 输入 | 线程数 | 独立进程 | 测量 forward | Profile forward |
| --- | --- | --- | ---: | ---: | ---: |
| [scaling](scaling/manifest.json) | `qwen3-scaling.json`，五种输入 | 1、2、4、8、16 | 30 | 450 | 225 |
| [context](context/manifest.json) | `qwen3-cpu.json`，六种输入 | 8 | 6 | 108 | 54 |

每个线程/模式三轮独立进程，每种输入预热一次、测量三次，因此每个配置有九个样本，但不是九个独立进程。off/on 和线程顺序交替。两组使用相同源码快照、模型和基准二进制，各自的 manifest 固定输入和比较规则；不合并两组样本计算加速比或开销。

36 份原始报告、558 次测量均通过验收：完整 logits SHA-256 与贪心 token 在各组的线程、模式和轮次间一致，KV 状态合法，逐样本满足阶段时间加未归类时间等于 profile forward 时间。源码快照、原始输入、逐层矩阵形状、线程池统计及每个样本均在子目录保留。

每次测量从同一只读 KV 前缀恢复活动序列，不随重复次数增长上下文。模型加载、前缀构建、恢复、argmax 和摘要计算单独记录或排除在 forward 外。协议及计时定义见 [Runtime 计时与模型基准](../../../docs/RUNTIME_PROFILING.md)；数值、HTTP、内存检查及目录迁移证据见 [正确性验证](../validation/wsl-runtime-profile/README.md)。

## 线程扩展

下表为 `scaling` 的无 profiler forward 中位数，单位 ms：

| 线程 | Prefill 16 | Prefill 128 | Decode 16 | Decode 256 | Mixed 16+2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1235.34 | 9630.31 | 101.17 | 108.45 | 1451.12 |
| 2 | 634.07 | 4947.61 | 66.57 | 71.91 | 750.08 |
| 4 | 346.32 | 2697.25 | 46.52 | 49.35 | 408.91 |
| 8 | 214.15 | 1590.54 | 45.48 | 47.15 | 256.48 |
| 16 | 167.03 | 1128.19 | 58.45 | 58.28 | 205.31 |

16 线程包含 SMT，在这些固定 prefill 和 mixed 输入上更快，但两个单序列 decode 输入均慢于 8 线程。不能把同一个线程数视为所有模型阶段的最优配置，也不能据此推导在线调度策略的优劣。

异常样本未剔除：4 线程 `decode-256` 的最大值为 133.12 ms，中位数为 49.35 ms；1 线程 `prefill-128` 最大值为 11302.79 ms，中位数为 9630.31 ms。完整范围及逐样本值见各组 `runtime-*.json`，不只保留最快结果。

## 阶段归因

`scaling` 的 8 线程 `prefill-128` 中，七类矩阵投影合计中位数为 1499.28 ms，占 profile forward 中位数的 96.16%；LM head 为 8.40 ms，占 0.54%。因此可以排除 LM head 是该固定 prefill 输入的主要直接耗时阶段。投影时间还包含分配、数据访问和线程池，不能仅凭此占比判定为算术计算瓶颈。

当前每个 logits token 单独执行一次 `M=1` 的 LM head。`mixed-16-2` 的三次 LM head 合计中位数为 23.76 ms，占 9.16%；其普通投影采用完整 `M=18`，不存在把共享阶段同时完整计入 prefill 和 decode 的重复归因。[投影分组数据](stage-groups.json) 按每个 forward 先求和，再取中位数。

`decode-16` 的 attention 合计中位数在 1、8、16 线程下分别为 0.486、1.977、3.487 ms。8 线程时，worker 消费循环的经过时间总和为 0.653 ms，调用线程等待为 0.633 ms，parallel wall 为 1.957 ms。这支持继续检查小任务发布、唤醒和尾部等待；这些统计不是纯 CPU 工作时间，不能直接据此认定全部退化由锁或线程池造成。

## 长上下文

下表只使用 `context` 组。forward 列来自无 profiler 进程，阶段列来自独立的 profiler 进程，不能跨列相加，单位 ms：

| 初始 KV 长度 | Forward 中位数 | Attention 中位数 | LM head 中位数 | Attention 占 profile forward |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 42.98 | 2.00 | 6.80 | 4.56% |
| 256 | 44.85 | 3.27 | 6.51 | 7.30% |
| 1536 | 58.47 | 15.28 | 6.43 | 26.00% |

attention 的经过时间随上下文显著增长，LM head 没有同样的增长。该阶段仍合并 QK、softmax、PV 和 KV accessor；数据不能证明分页布局或 allocator 导致了全部增长，更不能把旧 llama.cpp 服务端差距全部归于该项。`ENG-018` 保持待解决。

## 开销与限制

`scaling` 的 off/on 中位数相对差异为 -3.11% 至 +4.82%；`context` 为约 -1.13% 至 +1.90%。完整配对轮次见各组 `profiler-overhead.json`。频率、温度和后台负载没有固定，负值不是 profiler 加速，较小差异也不能作为插桩成本的精确上界。

两组 8 线程 profile 的未归类时间中位数均小于 0.2 ms。各阶段中位数未必相加等于总中位数；严格等式在每个原始 forward 上验收。worker 经过时间互相重叠，也与调用线程工作/等待重叠，不能累计成 forward wall time。

这是同一二进制的 off/on 对照，没有测量“禁用插桩”相对旧源码的成本。在线 batch 组成的扰动、worker 调度时间线、硬件计数器及客户侧 ITL 关联尚未验收，属于 `PLAN-003` 及后续实验。本基线不构成 Serving 加速或自研 CUDA 实现的证据。

## 复现

新采集目录必须为空：

```bash
bash scripts/dev.sh build
bash scripts/dev.sh runtime-benchmark \
  -InputFile benchmarks/runtime-inputs/qwen3-scaling.json \
  -Trials 3 -Repeats 3 -Warmup 1 -OutputDirectory .run/runtime-scaling
bash scripts/dev.sh runtime-benchmark \
  -Threads 8 -Trials 3 -Repeats 3 -Warmup 1 -OutputDirectory .run/runtime-context

pwsh -NoProfile -File scripts/Analyze-Runtime.ps1 \
  -Directory benchmarks/results/wsl-runtime-profile/scaling
pwsh -NoProfile -File scripts/Analyze-Runtime.ps1 \
  -Directory benchmarks/results/wsl-runtime-profile/context
```
