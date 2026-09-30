# 性能与取舍

六项结果来自各自冻结的采集身份，不是当前 main 的重新测量。
CPU micro、完整模型和在线服务的时钟不同，不能合并成统一加速比；
正结果、负结果和测量不确定性都保留。

| 层次与 workload | Baseline / candidate | Metric 与结果 | 限制与身份入口 |
| --- | --- | --- | --- |
| CPU micro：128 行、K=1024 的热缓存 Q8_0×F32 dot | 单线程 scalar / auto SIMD | 中位 661.53 / 154.13 ns/dot，约 4.29× | 不是模型或 Serving 加速；[原始记录](../benchmarks/results/simd-q8-dot.json) |
| CUDA model：12 类固定 prefill/decode/mixed，分别对 CPU8/CPU16 | CPU8 或 CPU16 / own CUDA | 无 profiler host forward-to-token；24 项中 14 项 `faster`，10 项不确定 | A/A 噪声超限不能称为获胜或无退化；[clean 0754722 采集](../benchmarks/results/cuda-model-baseline/README.md) |
| CUDA Serving：mixed-length trace | prefill_first / mixed，同模型同二进制 | 122.03 / 130.68 token/s，中位数之比约 +7.09% | burst-reuse 吞吐不确定，不是普适调度收益；[b1ced89 + snapshot](../benchmarks/results/cuda-serving-001/analysis.md) |
| GPU KV：同容量 8192 tokens | contiguous S4 / paged S4 | 吞吐配对中位退化 15.73% | KV 都是 896 MiB，paged 多 2 KiB table；未通过护栏；[3987492 + snapshot](GPU_KV_STUDY.md) |
| GPU KV：同 288 MiB KV 子预算，异长请求 | contiguous S1/cap2048 / paged S4/cap2560 | 吞吐配对中位提升 4.31% | TPOT/ITL 更高，不是低延迟收益或四倍吞吐；[冻结证据](../benchmarks/results/gpu-kv-001/README.md) |
| FP16 matrix：固定模型数值合同 | F32 / F16 matrix + F32 accumulation | owned device bytes 3,449,229,312 / 2,258,046,976；cosine 0.999885866 < 0.9999 | `blocked_correctness`，不能因显存下降放宽数值阈值，不开放 Serving；[精度研究](PRECISION_STUDY.md) |

## 阅读口径

CUDA model 的主时钟包含 metadata、finite/argmax 和完成同步；
初始化、prefix setup 单列。不同配置的 measured repetitions 不是独立 trial。
Serving 使用完整 trace 时间计算吞吐和 goodput，SLO 不达标请求不从分母中删除。

“中位数之比”与“每对 trial 比值的中位数”是不同统计量。
GPU KV 数字采用后者，不直接拿表中两个吞吐中位数相除重算。
各研究均未锁频；跨 cohort 的连续基线有漂移，
不能把前后差值全部解释为某个局部修改的因果收益。

## GPU KV 的停止决定

最终为 **B：容量／研究模式**，默认 contiguous，paged 显式 opt-in。
一次 P16 编译期地址修订后，长 prefill 和长 decode 的模型延迟仍分别退化
32.61% 和 71.21%，未通过冻结护栏；不再优化第二次。

同预算 Serving 的请求最大 ITL P95 从约 21.29 ms 增至 85.49 ms。
两组共 288 请求成功、9216 输出 token、输出一致，
但正确性通过不能替代模型或吞吐护栏。
没有统一预注册的尾延迟百分比门槛，不宣称通过不存在的 tail gate。

分页提供的是相对当前等长静态槽的容量灵活性，不是对所有连续 allocator 的优越性。
原始 micro 的显著退化、两组模型 cohort、唯一 NSys 与全部 Serving raw
均保留在[单一公开包](../benchmarks/results/gpu-kv-001/evidence.json)。
30 个正式性能进程已结束；交付 smoke 不产生新性能结论。
