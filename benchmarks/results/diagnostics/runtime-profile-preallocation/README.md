# 未完成的 Runtime 采集

`collection-status.json` 为失败状态，完成报告数为 0。目录保留采集前的源码快照、manifest 和输入；没有完整的模型性能样本，不能用于吞吐、阶段占比或 profiler 开销结论。

首个单线程进程在完成前被主动终止，用于处理 `ENG-027` 的 profile 预分配失败状态问题。原始诊断为 `Runtime measurement failed: threads-1-none-0.json`，不是模型数值失败或算子超时证据。
