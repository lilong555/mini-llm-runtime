# CUDA 权重与存储验收

范围为 `CUDA-VS-001 / Step 4`，环境为 WSL2、RTX 4070 Laptop、CUDA 12.8、cuBLAS 12.8.5、GCC 11.4。自有 CUDA 开启，上游 ggml CUDA 关闭。这里提供常驻有效权重、预分配 workspace、KV 容量预留和真实矩阵正确性证据，不包含完整 GPU forward、token 生成、GPU Serving 或性能结论。

## 验收结果

| 配置 | CTest 套件 | 用例执行 |
| --- | --- | ---: |
| 自有 CUDA ON、上游 CUDA OFF | 9/9 | 199 |
| CPU | 7/7 | 178 |
| 独立核心，无 llama | 5/5 | 167 |
| 上游 CUDA ON、自有 CUDA OFF | 7/7 | 178 |

共 28 次套件、722 次用例执行，含重复执行。设备基础单测 11 项，存储单测 10 项；实模型命令另含 1 项整组权重与矩阵检查。CPU 模型 13/13、HTTP 8/8，HTTP 临时服务已退出。现有模型词表警告仍保留，参见 `ENG-011`。

- 固定 Q8_0 模型的 SHA-256 与数值契约一致，实际包含 197 个 Q8_0 和 113 个 F32 tensor。全部 310 个唯一 tensor 转为 FP32 后逐字节回读，与 host `decode_row` 相同；输出权重与 embedding 共用设备地址。
- 权重上传总量为 2,384,199,680 字节，468 个分块，最大分块和 host staging 上限均为 8 MiB。`weight-plan.json` 含 311 条记录，其中一条为 tied 别名，保留各 tensor 的有效权重 SHA-256。
- S=4、Lmax=2048、B=128 的实际项目分配为 3,448,180,736 字节，等于计划。权重、workspace、KV 和 cuBLAS workspace 各有一个 owner；KV 预留为 896 MiB，不代表逻辑长度、append 或 attention 已实现。
- Q/K/V、attention output、gate/up/down、LM head 八类真实 N/K 各检查 11 组。M=1/2/4/8/16/18/32/64/128 使用稀疏四点输入，M=1/2 另有稠密输入；每组全部输出对照 CPU FP64，容差固定为 `atol=2e-4, rtol=2e-4`。
- 每组重复入队两次 GEMM，共 176 次。矩阵阶段项目设备 allocation/free 调用为零；不统计 CUDA/cuBLAS 内部分配，不能推广为尚未实现的完整 forward 稳态结果。
- Linux 链接器测试在四个项目分配点逐个返回 `cudaErrorMemoryAllocation`，检查部分 owner 的清理与恢复；F32/F16 的 NaN、正负 Inf 均在初始化时拒绝。正常和受控失败路径的 Compute Sanitizer 为 0 错误、0 泄漏。
- 已有目录重跑在开始前拒绝，原文件摘要不变。错误 checkpoint 产生失败摘要；进程中断前未完成的报告为 `incomplete`，不能作为通过依据。

## 文件与复核

`real/` 与 `memcheck/` 分别为普通执行和内存检查的独立进程原始报告。两者的权重、矩阵及分配摘要字节相同，free memory 单独保留各自实测值。`*-ctest.xml`、构建和验证日志、`build-boundaries.json`、`negative-checks.json` 固定测试与构建边界；CPU 产品没有 CUDA 动态依赖，`mini-llm`、`mini-runtime-bench`、`llmserve` 的二进制摘要与 Host Model 验收一致。存储测试没有 CPU Runtime 或上游模型 forward 定义。

`evidence.json` 记录基点 SHA、dirty 状态、逐文件源码清单、完整源码 ZIP、模型与二进制身份及各产物摘要。不能将工作区构建归为基点的 clean build。归档不含模型权重、依赖 checkout、二进制或工具链；重跑需按模型 manifest 和依赖 pin 获取它们。

`revalidation.json` 记录独立目录通过、缺少源码 ZIP 和篡改矩阵报告均被拒绝、原件摘要不变。辅助汇总的已复现 PowerShell 类型问题保留在 `diagnostics/collection-error.txt`，见 `ENG-037`；不改变原始验证结论。

只需 PowerShell 即可核对归档，验证工具从包内源码快照取得，不依赖原采集绝对路径：

```bash
pwsh -NoProfile -File benchmarks/results/validation/cuda-storage/verify.ps1
```

实机重跑：

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda storage-check
bash scripts/dev.sh own-cuda storage-memcheck
bash scripts/dev.sh own-cuda memcheck
bash scripts/dev.sh validate
bash scripts/dev.sh check-http 8067
```

单元矩阵正确性和显存容量不是模型性能证据。尚无自有 GPU 模型时延、GPU HTTP、S=1 完整模型、Profiler 或 Windows 实测；后续基础算子、KV 状态、attention、完整模型和性能各有独立门禁。
