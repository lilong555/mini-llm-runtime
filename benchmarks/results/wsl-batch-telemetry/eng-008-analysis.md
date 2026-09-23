# ENG-008：平均 TPOT 与单次停顿的在线归因

同一 CPU 二进制、模型、原始 trace、8 线程、相同预热和容量条件下，mixed 的请求平均 TPOT 较高，而单次 ITL 的尾部较低。调度器选 batch 的计算耗时很小，输出停顿主要与实际模型执行批次重叠。

## 未启用观测的服务指标

原始 trace 每进程 24 请求、384 个输出 token；每策略三轮。下表取三份报告各自指标的中位数，单位为 ms，吞吐除外：

| 策略 | 输出 token/s | P95 TTFT | P95 请求平均 TPOT | P99 单次 ITL |
| --- | ---: | ---: | ---: | ---: |
| mixed | 18.41 | 12708.12 | 395.28 | 1224.77 |
| prefill-first | 18.66 | 12207.44 | 333.42 | 2480.67 |

两者 goodput 均为 0。mixed 三轮吞吐为 18.41、18.39、18.81 token/s；prefill-first 为 18.00、19.11、18.66 token/s，保留了其中一轮 prefill-first 较慢的结果。没有把有限轮次差异推广为 CPU mixed 必然有效或无效。

原始请求数据位于 `mixed/trial-{0,1,2}-off/`，逐报告摘要、SHA-256 与跨模式比较见 [mixed/telemetry-summary.json](mixed/telemetry-summary.json)。源 trace 字节不变，历史 Windows 结果不参与本组加速比计算。

## 停顿对应的 batch

阶段模式第 0 轮，prefill-first 请求 `s0-r1` 的输出序号 0、1 分别来自 batch 10、17。Engine 发布间隔为 **3673.993311 ms**，客户端间隔为 **3673.860555 ms**。两者原点不同，只比较间隔：

| 期间 batch | Prefill token | Decode token | Runner ms |
| ---: | ---: | ---: | ---: |
| 11 | 32 | 0 | 387.63 |
| 12 | 64 | 0 | 780.81 |
| 13 | 80 | 0 | 1007.24 |
| 14 | 32 | 0 | 412.62 |
| 15 | 32 | 0 | 401.63 |
| 16 | 32 | 0 | 418.52 |
| 17 | 0 | 8 | 265.10 |

该 Engine 间隔与 runner 时间区间的交集之和为 **3673.558444 ms**，与 scheduler 区间的交集仅 **0.012814 ms**。对应 [原始 JSONL](mixed/trial-0-stages/prefill_first-0-telemetry.jsonl) 和 [客户端报告](mixed/trial-0-stages/prefill_first-0.json) 保留精确 token 与时间。

同轮 mixed 的最大客户端停顿为 **1463.775715 ms**，发生在 `s0-r9` 输出序号 0→1、batch 28→29，期间 runner 为 **1463.754074 ms**，涉及 96 个 prefill token。它没有跨越多批纯 prefill，但同批模型计算仍带来明显停顿。其他两轮各自最长停顿见 [hypothesis-table.json](hypothesis-table.json)，没有只保留最好的一轮。

阶段模式下，按每个进程汇总再取三轮中位数，七类矩阵投影占迭代时间之和约 **78.95% / 77.83%**，LM head 占 **14.71% / 15.26%**，对应 mixed / prefill-first。七类为 Q/K/V、output、gate/up/down projection；LM head 单独统计。这些 wall time 包含访存、线程池和操作系统调度，不能直接认定为纯算术成本。共享阶段没有重复分摊到每个请求或 prefill/decode。

## 低到达率控制

`arrival_s × 6` 保留同一原始 trace，只降低到达密度。每策略/模式一个独立进程，不能作为稳定策略优势证明：

| 无观测策略 | 输出 token/s | P95 TTFT ms | P95 请求平均 TPOT ms | P99 ITL ms | Goodput 请求/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed | 10.39 | 1790.26 | 148.89 | 419.07 | 0.379 |
| prefill-first | 10.37 | 1650.47 | 166.46 | 907.02 | 0.351 |

请求平均 TPOT 的相对方向与原始密集到达不同，说明需要结合到达过程研究。低负载阶段模式还出现批数组成变化；完整原始指标见 [low-load/telemetry-summary.json](low-load/telemetry-summary.json)。

## 假设结论

- 在这些输入中排除“scheduler 自身计算是主要耗时”：全部观测报告该部分占比小于 0.004%；最长停顿的区间交集也支持这一结论。它不排除策略通过执行顺序影响等待。
- 支持继续验证“prefill 模型执行与 decode 停顿有关”：纯 prefill 的连续执行和大型 mixed batch 均与停顿重叠。
- 尚未证明批量 LM head、多 token 权重解码或线程池 cutoff 的收益。后续需要实际矩阵形状对照、对应数值门槛及独立模型/HTTP 优化前后实验。

这不是 KV 容量压力或长期公平性实验，也没有 worker 调度时间线。`ENG-008` 的性能策略问题仍保持开放；在线记录提供了可核验的执行归因。
