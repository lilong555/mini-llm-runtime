# 自有 CUDA 真实形状基线

本目录包含 RTX 4070 Laptop、WSL2、CUDA 12.8 上的 375 项微基准。5 个独立进程共保留 9375 个原始样本，其中 5625 个测量样本对应 180000 次 API 调用。数值、跨进程输出一致性、显式传输和项目设备分配检查通过；这是绝对时间基线，不是优化加速报告。

## 结果

每个样本连续调用同一 API 32 次；每个进程先取三次测量中位数，再汇总五个独立 trial。下表为区间均摊值，单位为微秒/调用：

| 用例 | host | device interval |
| --- | ---: | ---: |
| Q projection，M=1 | 38.127 | 37.369 |
| Q projection，M=128 | 73.603 | 72.729 |
| LM head，M=1 | 2654.861 | 2653.655 |
| LM head，M=128 | 3524.716 | 3523.424 |
| attention，单序列、context=1536 | 114.025 | 113.216 |
| attention，mixed M=18、context=1537 | 378.662 | 377.623 |

全部形状与区间见 [analysis.md](analysis.md)，原始 trial 和逐样本验证见 `cuda-micro-t0.json` 至 `cuda-micro-t4.json`。最大绝对误差为 `1.552663471e-05`，最大容差占比为 `0.000797035`；矩阵与 attention 为固定抽样对照，其他算子全元素对照，所有输出检查 finite。

## 波动与边界

`matrix-Q-m1` 的五轮 device 区间均摊值为 `37.664/10.240/37.369/10.656/37.792` 微秒，与正序/逆序分组一致。`rope-query-m1-p1535` 出现 `16.960` 微秒的长样本，其余四轮为 `6.275` 至 `6.937` 微秒。375 个用例中有 48 个相对 MAD 超过 10%；全部样本保留，没有用最快轮次代表整体。

这些区间包含 host 提交空隙，不是纯 kernel 时间。未锁频、未固定 affinity，环境仅在进程边界查询；当前证据不能判定上述差异来自热状态、库执行选择还是提交间隙，见 `ENG-048`。这里的 MAD 不是模型 A/A 协议的噪声带，也不能证明没有退化。

初始化、输入重置、状态/输出下载和 FP64 验证均在计时外。RoPE 每个 sample 前恢复原输入，随后连续原地旋转 32 次。attention 使用初始化的 synthetic KV，而非真实 prompt 的模型 KV。

项目设备预分配为 `3449229312` 字节，计时区间没有项目设备分配、释放或显式 H2D/D2H。这些计数不覆盖 NVIDIA 库内部资源；逻辑 FLOP 数不能作为硬件 DRAM 带宽。模型性能、HTTP TTFT、GPU Serving 与自有 PagedAttention 均不由本报告验收。

## 身份与复核

源码基点为 `dcb07b7c179d9f9710b606fbf11b5ce33e9b40e9`，采集时为 dirty worktree；实际内容以 `source-state.json` 和 `source-snapshot.zip` 为准。二进制 SHA-256 为 `d07516a97c1c74b9ff4db2359bb29c2e3c262c7bbe98ae5f67b9ff39ad551c3f`。构建为 GNU 11.4.0、RelWithDebInfo、自有 CUDA=ON、上游 CUDA=OFF、架构 89。

模型、二进制和依赖 checkout 不装入归档。复制本目录后，Python 3.10+ 可独立复核：

```bash
python3 verify.py --directory .
```

`availability.json` 登记 27 个必需原始产物。归档复核不需要原绝对路径下的模型或可执行文件，也不是原二进制在其他机器重跑的证明。运行与数值规则见 [微基准协议](../../../docs/CUDA_MICROBENCHMARKS.md)，完整模型数值、CTest 和 CPU HTTP 证据见 [验证归档](../validation/cuda-micro/README.md)。

`revalidation.json` 记录独立目录复核，以及缺失源码/复核工具、缺少 trial、数值点篡改、设备身份不一致和计时区间传输篡改的六项拒绝结果；原始样本与已有汇总均保持不变。
