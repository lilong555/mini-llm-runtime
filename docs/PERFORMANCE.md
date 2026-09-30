# 性能与取舍

所有数字绑定各自的源码、二进制、模型、输入和机器状态。
微基准、完整模型和 online Serving 的时钟与工作负载不同，不互相替代。

## GPU KV 结论

**B：容量／研究模式，默认 contiguous，paged 显式 opt-in。**

冻结输入为 `qwen3-gpu-kv-v1.json`，P16、F32 矩阵、F16 KV、
单 stream、同步执行。一次局部修订只将页大小除余变为编译期常量；
之后停止 kernel 优化。

| 模型 workload | P16 确认的配对延迟退化中位数 | 冻结护栏 |
| --- | ---: | --- |
| prefill-128 | 7.26% | 通过 |
| long-prefill-1536-chunk128 | 32.61% | 未通过 |
| decode-prefix1536 | 71.21% | 未通过 |
| decode-batch4-prefix256 | 7.36% | 通过 |

每项三对独立 trial，主时钟包含 forward 到 token 的完成点；
护栏为退化中位数≤10%，且最多一轮>10%。不以短用例抵消长用例。

| Serving 比较 | 连续 token/s | 分页 token/s | 配对吞吐变化中位数 |
| --- | ---: | ---: | ---: |
| 同容量 8192 tokens | 115.67 | 97.48 | -15.73% |
| 同 KV 子预算 288 MiB | 33.64 | 35.15 | +4.31% |

每组两臂各三进程，共 288 成功请求、9216 输出 token，无失败，输出一致。
同容量失败于 10% 吞吐护栏；同预算改善了限定异长负载的排队和吞吐，
但请求最大 ITL P95 从约 21.29 ms 增至 85.49 ms。
不能将同预算吞吐收益说成低延迟收益、四倍吞吐或普适优势。

同预算比较中连续方案为 S1/cap2048，paged 为 S4/cap2560；
paged 实际分配更多 KV 字节，但两者均在 288 MiB 子预算内。
同容量比较不节省显存。未锁频；跨 cohort 的连续基线也有漂移，
不能把前后差值全部解释成局部修订的因果收益。

## 其它证据

- CPU SIMD 微基准不等于端到端模型或 Serving 加速。
- 原 CUDA Serving 的 mixed 调度收益仅适用于对应旧 trace 和采集身份，
  不替代本次分页比较。
- FP16 矩阵候选曾减少 owned device memory，但未通过原模型数值门槛，
  以 `blocked_correctness` 收束，不开放 FP16 Serving。
- NSys 表明长 decode 的 paged PV 是主要设备时间来源；
  源码调用次数不等于 DRAM transaction 数，静态指令也不独立证明动态瓶颈。

逐轮指标、TTFT/TPOT/ITL、SLO、身份、局限与产物状态见
[GPU KV 研究](GPU_KV_STUDY.md)。正式性能预算 30/30 已用完，不增加 trial。
