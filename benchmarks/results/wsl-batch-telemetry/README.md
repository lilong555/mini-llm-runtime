# WSL 在线 batch 观测与归因

2026-09-23，WSL2、Ryzen 7 7745HX、GCC 11.4、`RelWithDebInfo`，MiniLLM CPU 8 线程，Qwen3-0.6B Q8_0、F32 激活、F16 KV。MiniLLM 使用自有 forward；本组不运行 llama.cpp forward 或自研 CUDA。

42 个独立服务进程、594 个请求、9810 个输出 token 全部通过 Serving 及在线关联验收。同一输入和配置在 `off / batches / stages`、两种策略和轮次间输出一致；有观测的 28 份 JSONL 均有完整 footer、无丢记录，batch、阶段时间、上下文及 SSE token 关联合法。

[evidence.json](evidence.json) 绑定各组验收汇总、假设表及正确性证据。21 份 manifest 的源码状态摘要、服务端、客户端和模型摘要分别一致；源码状态摘要为 `4330c019937be02fc09ca3e34435df5bc7d2ae6d186e0ad760d7a92919e29acd`。该身份包含未提交的构建、运行和测试输入，不能只用基线 Git 提交号代替。

| 组 | 输入 | 到达缩放 | 每策略/模式独立进程 | 全部进程 | 全部请求 |
| --- | --- | ---: | ---: | ---: | ---: |
| [mixed](mixed/experiment.json) | 原始 `cpu-mixed-s0.jsonl` | 1 | 3 | 18 | 432 |
| [low-load](low-load/experiment.json) | 相同原始 trace | 6 | 1 | 6 | 144 |
| [context-16](context-16/experiment.json) | `kv-cpu-16.jsonl` | 1 | 1 | 6 | 6 |
| [context-256](context-256/experiment.json) | `kv-cpu-256.jsonl` | 1 | 1 | 6 | 6 |
| [context-1536](context-1536/experiment.json) | `kv-cpu-1536.jsonl` | 1 | 1 | 6 | 6 |

模式顺序和策略顺序逐轮反转。长上下文组关闭 prefix cache，其余使用 4 条、2048 token 缓存；全部采用 `context=8192`、`max_model_len=2048`、`max_active=8`、`batch_tokens=256`、`prefill_chunk=32`、16-token 页。各子目录保留独立 manifest、源码快照、原始 trace、模型/二进制身份、逐请求报告与 JSONL；源码和二进制在全部采集期间保持相同。

## 结果

原始负载中，未启用观测的 mixed / prefill-first 三轮中位吞吐为 **18.41 / 18.66 token/s**，P95 请求平均 TPOT 为 **395.28 / 333.42 ms**，P99 单次 ITL 为 **1224.77 / 2480.67 ms**，goodput 均为 0。这些是每轮指标的中位数，不是合并所有请求后重新计算的分位数。

所有已测有观测进程中，scheduler 经过时间占迭代时间之和的 **0.00053%–0.00365%**，可排除 scheduler 自身计算是这些输入下的主要耗时。实际执行顺序仍然影响停顿：一段 prefill-first 的 3.674 秒 token 间隔包含六批 prefill 和随后一批 decode，runner 占约 3.674 秒，scheduler 仅约 0.013 ms。完整分析见 [ENG-008](eng-008-analysis.md)。

长上下文阶段证据将 attention 的贡献与 LM head 区分开：mixed 的初始 16/256/1536 KV 组，32 次 decode 的 attention 中位数为 **2.04 / 3.90 / 15.36 ms**，LM head 为 **7.41 / 7.56 / 6.96 ms**。这不能证明分页 accessor 是全部差距的来源；见 [ENG-018](eng-018-analysis.md) 和 [逐项假设与原始证据映射](hypothesis-table.json)。

## 观测成本与扰动

原始 mixed trace 的全程耗时中位数相对关闭观测的变化：

| 策略 | batches | stages | 关闭观测的批数 | stages 批数 |
| --- | ---: | ---: | --- | --- |
| mixed | +0.36% | +1.82% | 55、55、55 | 55、55、55 |
| prefill-first | +2.40% | +5.82% | 61、61、61 | 61、61、61 |

低到达率的单轮观测出现组批差异：mixed 的 off/stages 为 271/267 批，混合批为 20/21；prefill-first 为 271/240 批。不能将在线开关差异当作固定的插桩成本，也不能只凭总批数相同证明所有 batch 组成相同。

单轮诊断组还存在负差异：`context-16` mixed 的 batches 全程耗时相对 off 低 6.65%，而 `context-256` mixed 的 stages 高 6.70%。这些值全部保留，不能解释为 profiler 加速或精确开销上界。CPU 频率、温度和背景负载没有固定，各组 `telemetry-summary.json` 保存范围及原始指标。低负载和上下文组每配置只有一个独立进程，不构成稳定优劣结论。

## 复现

输出目录必须为空：

```bash
pwsh -NoProfile -File scripts/Benchmark-Telemetry.ps1 \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl -Trials 3 \
  -OutputDirectory .run/online-mixed
pwsh -NoProfile -File scripts/Benchmark-Telemetry.ps1 \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl -Trials 1 -ArrivalScale 6 \
  -OutputDirectory .run/online-low-load
pwsh -NoProfile -File scripts/Benchmark-Telemetry.ps1 \
  -Trace benchmarks/traces/kv-cpu-1536.jsonl -Trials 1 \
  -PrefixEntries 0 -PrefixTokens 0 -OutputDirectory .run/online-context-1536
```

16/256 组使用对应 `kv-cpu-16.jsonl`、`kv-cpu-256.jsonl`，其余参数与 1536 组一致。协议、导出文件、离线重新验收和时间口径见 [在线观测文档](../../../docs/BATCH_TELEMETRY.md)，模型、HTTP、sanitizer 及目录迁移证据见 [正确性验证](../validation/wsl-batch-telemetry/README.md)。

这些结果没有切换计算内核，不是性能优化收益证明。sequence 生命周期重放、attention 内部分离、worker 调度时间线和同源码页布局对照仍待完成。
