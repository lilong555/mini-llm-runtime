# 自有 CUDA 模型性能基线

本目录包含 RTX 4070 Laptop、WSL2、CUDA 12.8 上的 CPU8、CPU16 与自有 CUDA 完整模型对照。30 个 A/A 进程和 40 个异构配对进程全部完成，共 40950 次 forward、2520 次 measured repetition；每个配置以五个独立 trial 统计，不将进程内重复视为独立 trial。

## 结果

| 检查 | 结果 |
| --- | --- |
| 独立进程 | 70/70，退出码均为 0 |
| 输入、KV、权重与内存契约 | 通过 |
| 同后端全部进程 token 一致性 | 通过 |
| CPU/CUDA 输出 token | 全部一致 |
| 稳态项目设备分配、释放 | 0 |
| 稳态权重、层间 hidden 传输 | 0 |
| greedy 全词表下载 | 0 |
| 性能统计 | `measurement_inconclusive` |

完整 24 项结果见 [analysis.md](analysis.md) 和 `summary.json`。14 项按冻结规则判为 `faster`，10 项因 A/A 噪声超过 10% 保留为 `measurement_inconclusive`，没有 `slower` 项。CPU8 对照的不确定项有 3 项，CPU16 有 7 项；噪声最高约 23.06%。不能将这些结果合并为统一加速承诺，或对不确定项宣称没有退化。

主指标为无 Profiler 的 host forward 到 token 延迟，包含 metadata copy、finite/argmax 和完成同步。前缀每次独立重建，setup、clear、初始化和报告编码不计入该主指标。全部 warmup、重复、原始时间及进程边界环境保留；未锁频、未固定 affinity，边界遥测不是连续功耗或频率曲线。

## 执行身份

- 源码：`0754722511b98a8d834167e3befb2c3307e81fa2`，采集源码为 clean worktree。
- Run ID：`20260926T025352Z-0754722511b9-f23c5edd`。
- 模型基准二进制 SHA-256：`d7eb569942f6ffa6621f346c351e16d531dad70041afc76c0cc25dbac0f53180`。
- 数值测试二进制 SHA-256：`fb9dc71c05a44f19eec764ada02d9a9a6918c89327dd65228ce268f23b213f34`。
- 模型：固定 Q8_0 checkpoint；GPU 常驻 FP32 有效权重、FP16 连续 KV、cuBLAS pedantic FP32。
- 配置：S=4、Lmax=2048、B=128、单项目 stream；CPU 为 auto SIMD、8/16 线程、16-token paged KV。
- GPU 项目预分配：`3449229312` 字节；不包含 NVIDIA 工具或库内部资源。

完整数值证据来自 [CUDA 微基准与数值验收](../validation/cuda-micro/README.md)，12528 次比较通过。源码集合、摘要及数值测试二进制相同是继承前提；本次性能采集不重复常驻 F32 reference。

## 独立复核

需要 Python 3.10+，无需原采集绝对路径、模型权重、编译产物、CUDA 或 Nsight：

```bash
python3 -B revalidate.py --directory .
```

`manifest.json`、原始报告、`source-state.json`、`source-snapshot.zip` 和 `verify.py` 固定采集身份。`revalidate.py` 是包含全部同后端一致性门禁的当前复核入口；两份工具的摘要分别记录于 `revalidation.json`，不改写采集身份。

`availability.json` 登记 289 个必需原始产物。独立目录复核通过，295 个既有文件保持不变；缺源码 ZIP、进程不完整、稳态传输、设备身份和权重别名五项反例均被拒绝。同后端 A/A 与跨 trial 的真实报告反例见 [工具回归证据](../validation/cuda-profiler/same-backend-regression/same-backend-regression.json)。

本目录只验收模型层，不是 HTTP TTFT、GPU Serving、自有 PagedAttention 或微基准加速结论。Profiler 和完整交付包有独立门禁。
