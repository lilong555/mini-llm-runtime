# GPU KV 收尾合同

- 决策标识：`GPU-KV-CLOSEOUT-001`。
- 功能已完成，新增 Major Feature 冻结；不扩展分页正确性矩阵，保留既有回归。
- 当前默认 `contiguous`，唯一模型确认仍超限，A 不成立；最终 B/C 待 Serving。
- 冻结输入：`benchmarks/runtime-inputs/qwen3-gpu-kv-v1.json`，
  SHA-256 `77b44ce8578e73e05889c21e4aa167b5cff5f858110fc4f62981cc49e002bb5e`。
- 本合同优先于历史计划中未执行的建议，不产生新功能路线。

## 模型护栏

每个 workload 分别计算三对 trial 的 `paged/contiguous-1`；
中位数不超过 10%，且最多一轮超过 10%。不合并不同 workload、
revision 或 profiler 数据。初始六进程均完成，四项均超限，详见
[研究记录](GPU_KV_STUDY.md)。

## 唯一局部修订

允许的一次修订为 P16 编译期地址专用化：只在 `PagedKvAccess::row`
中用既有 `kv_page_tokens` 常量计算逻辑块、页内偏移及物理行。
Host 仍拒绝其它页大小；所有 bounds checks、错误状态、表加载、
slab 布局、数学与累加顺序不变。不增加 warp 广播或 PV 分段设计。

依据为初始模型护栏失败，以及同一二进制的 NSys 与 PTX：
长 decode 的三个 measured forward 中 PV 为 20.04～20.92 ms，
QK 约 1.06 ms；PV 内层仍加载运行时 `page_tokens` 并执行除余。
这支持有限修订假设，不预先证明收益或全部根因。

修订后既有正确性、模型、HTTP 与 memcheck 均通过；独立的六进程模型确认
已完成，长 prefill/decode 仍超限。停止优化，完成 Serving 并决定 B/C。
禁止第二次优化、fusion、Graph、异步、精度和调度变更。

## 预算

| 项目 | 上限 |
| --- | ---: |
| 已完成 micro | 6 |
| 已完成初始 model | 6 |
| 唯一修订后的 model 确认 | 6 |
| Serving | 12 |
| 正式性能进程合计 | 30 |
| 新 NSys | 1，已用于模型诊断 |
| 新 NCU | 1，尚未使用，非必需 |

额外六进程仅属于已授权的修订确认，不能并入原始三对 trial。
原始不利数据保留；不重新采 micro，不追加 trial 追求通过。

## Serving 与结束条件

只测试一个最终 paged 版本，固定 mixed policy。同容量为两臂
S4/L2048/cap8192，旧 mixed trace，顺序 C/P、P/C、C/P；
同 288 MiB KV 预算为 contiguous S1/cap2048 与 paged S4/cap2560，
capacity trace，顺序 P/C、C/P、P/C。共 12 进程。

同容量吞吐配对退化中位数不超过 10%；保留原 SLO、TTFT、
mean TPOT、每请求最大 ITL 及失败，不追加不存在的统一尾延迟百分比门槛。
tail slack 只在同一 ready 快照计算，未分派池容量不算尾页碎片。

- A：容量价值、模型及 Serving 护栏通过，无未解释正确性或尾部风险。
- B：容量能力成立，但开销或证据只支持显式研究入口。
- C：对预期用途不值得采用，保留负结果及研究记录。

最终只交付一个 canonical bundle、小摘要及证据索引，收敛
README、ARCHITECTURE、PERFORMANCE、GPU_KV_STUDY、VALIDATION
入口，然后结束主要功能开发。发布整合仍需用户审阅，不自动合并 main。
