# ENG-018：长上下文 decode 的在线阶段归因

同一 MiniLLM CPU 产品、8 线程、16-token 页、关闭 prefix cache，使用初始 16、256、1536 token 的单请求 trace，每个请求生成 33 token。每种策略/观测模式各一个独立进程。这是限定输入诊断，没有同源码的分页/连续布局切换。

## 测量范围

首个输出由 prefill 产生，之后 32 次纯 decode 的 KV 起始长度分别为 16–47、256–287、1536–1567。这里的上下文随生成增长，不是离线 Runtime 基准的固定 KV 长度重复测量。预热 batch 不进入阶段汇总；长 prompt 的 prefill 与 decode 分开分析。

下表为阶段模式每个进程 32 次纯 decode 的中位数，单位 ms：

| 初始 KV | 策略 | Runner | Attention | LM head |
| ---: | --- | ---: | ---: | ---: |
| 16 | mixed | 47.80 | 2.04 | 7.41 |
| 16 | prefill-first | 43.55 | 1.95 | 6.69 |
| 256 | mixed | 49.64 | 3.90 | 7.56 |
| 256 | prefill-first | 47.64 | 3.62 | 7.31 |
| 1536 | mixed | 58.95 | 15.36 | 6.96 |
| 1536 | prefill-first | 60.70 | 16.20 | 7.25 |

这是单序列负载，两种策略没有并发请求竞争；它们的时间差不能被解释为调度策略收益。表中阶段中位数也不能直接相加为 runner 中位数。逐进程来源、实际上下文范围和计算方法保留在 [hypothesis-table.json](hypothesis-table.json)。

独立关闭观测进程的请求平均 TPOT，mixed 为 **48.36 / 46.98 / 61.64 ms**，prefill-first 为 **45.12 / 47.22 / 59.09 ms**。16→256 的 mixed 数字没有单调增长，说明这一有限样本也存在系统噪声；不删掉不符合预期的结果。

原始数据与严格验收分别见 [context-16](context-16/telemetry-summary.json)、[context-256](context-256/telemetry-summary.json)、[context-1536](context-1536/telemetry-summary.json)。每份 JSONL 记录执行前后真实 KV live pages 与 resident payload；停服后 live pages 为零，resident 不要求为零。

## 结论与边界

这些记录显示长上下文下 attention 的时间增长，LM head 没有相同趋势，支持把 attention 内部作为下一项测量重点。该阶段仍包含 QK、softmax、PV 和逐 token 的 KV accessor，当前证据不能判断其中各项的独立贡献。

[历史同算术布局隔离](../kv-cache-cpu/summary.json) 具有独立的微基准证据，但平台、工具链和实验协议与本组不同，没有直接合并计算端到端加速比。旧 MiniLLM/llama.cpp 差距还包含 ISA、FlashAttention 与其他内核差异；本组也没有运行 llama.cpp 服务对照。

`ENG-018` 保持待解决。后续验收需要分离 QK/softmax/PV 和页访问，按交替顺序对照页布局或页大小，并完成长上下文数值及模型/HTTP 检查；本次没有提出经过验证的 allocator 优化收益。
