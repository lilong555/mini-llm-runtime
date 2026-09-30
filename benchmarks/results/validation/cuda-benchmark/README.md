# CUDA 模型基准工具验收

范围为 `CUDA-VS-001 / Step 8` 的模型 workload、采集协议与统计工具。**不是正式性能基线**，没有 A/A 加速结论、microbenchmark、Profiler 或 GPU Serving 完成声明。

## 验收结果

| 项目 | 结果 |
| --- | --- |
| 自有 CUDA / CPU / 独立核心 / 上游 CUDA CTest | 16/16、10/10、7/7、10/10 |
| 当前 CTest 用例执行 | 875 次，全部通过 |
| CPU 实模型 / HTTP | 13/13、8/8 |
| CUDA 默认短模型 | 128 组 logits、六组短金标准通过 |
| 实际基准进程 | CPU8、CPU16、CUDA，以及带 manifest 的 CUDA 进程 |
| 完整 workload | 每进程 12 项，每项 2 次 warmup、3 次测量 |
| 实际 forward / 输入 / 输出 | 2,340 次 / 120,500 个 token / 1,600 个 token |
| 跨后端输出 | 全部一致 |
| 稳态项目设备分配与释放 | 0 |
| 稳态权重与中间激活传输 | 0 |
| greedy 全词表下载 | 0 |
| 项目 GPU 预分配 | 3,449,229,312 字节，与计划一致 |
| 独立目录复验 | 100 个产物通过，五项缺件或语义反例被拒绝 |

13 项 Python 检查覆盖完整进程计划、统计单位、噪声边界、负结果、输入重放、独立 KV、copy/allocation、权重/内存、生成历史、完整合成归档与发布回滚；自有 CUDA 构建另有真实 CLI 拒绝检查，共 14 项。五项 C++ 检查直接覆盖 workload 生成和执行语义。

三项真实预检反例分别拒绝不完整基线、非空输出目录与不同的数值验收二进制；原文件不变。CPU resident KV 的高水位与活跃页数分开核对，GPU 不使用 CPU 页数语义。

## 原始证据

- `real-processes/`：CPU8、CPU16、CUDA 的完整单进程报告。
- `preflight/`：固定 70 进程计划、初始源码快照、环境和实际 manifest 绑定检查。采集状态为 `preflight_only`。
- `final-preflight/`：完整数值测试二进制身份检查、最终验证工具源码及预检。采集状态仍为 `preflight_only`。
- `single-process-times.json`、[计时记录](analysis.md)：所有用例的原始测量与单进程中位值，保留较慢结果，不计算加速比。
- `*-ctest.xml`、`model-cpu.json`、`http-cpu.json`、`short-model/`：当前回归结果。
- `diagnostics/`：构建、Python 兼容性、CLI fixture 的原始诊断及先前 CTest；不计入当前 CTest 总数。
- `evidence.json`、`revalidation.json`：身份、逐文件摘要和独立目录复核结果。

三个初始功能进程的源码与环境在预检阶段归档；基准 executable 在这些检查和预检前后保持同一 SHA-256：`b49eeaba5732509a91d8cbe743ffa0e4ed986ed7fdb47b28cd8859f514b6dae0`。这不构成冻结的独立 trial cohort。

完整模型数值测试二进制为 `4b31b3d60cf3b5d79fcbec054a234ad273eee67a03d661a7a40bde2ea08c518f`，与 [全量数值验收](../cuda-full/README.md) 相同。本组没有重复运行完整长语料，也未重复执行 sanitizer。当前未运行 Windows 或远程 CI。

## 复核

```bash
pwsh -NoProfile -File benchmarks/results/validation/cuda-benchmark/verify.ps1
```

整个目录迁移后可直接运行其中的 `verify.ps1`，不需要原绝对路径中的模型、二进制或依赖 checkout。归档复核不等于二进制重跑。完整采集和分析入口见 [模型性能对照](../../../../docs/CUDA_BENCHMARKS.md)；正式 70 进程 A/A/异构基线、microbenchmark 和 Step 9 仍待验收。
