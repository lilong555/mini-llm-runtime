# CUDA 基础层验收

范围为 `CUDA-VS-001 / Step 2`，环境为 WSL2、RTX 4070 Laptop、CUDA 12.8、GCC 11.4。`evidence.json` 固定实际工作区源码快照、模型摘要和各项原始证据；`build-boundaries.json` 记录编译参数、构建选项、二进制摘要及动态依赖。源码为 `e2fcb6d` 上的未提交实现，不能标成基点的 clean build。

## 结果

| 配置 | CTest 套件 | 用例执行 |
| --- | --- | --- |
| 自有 CUDA ON、上游 CUDA OFF | 7/7 | 183 |
| CPU，两种 CUDA 均 OFF | 6/6 | 172 |
| 独立核心，无 llama | 5/5 | 167 |
| 上游 CUDA ON、自有 CUDA OFF | 6/6 | 172 |

四组共有 24 次套件执行、694 次用例执行，包含重复执行。设备资源和矩阵单测为 11/11；Compute Sanitizer 的 `memcheck --leak-check full` 报告 0 错误、0 字节泄漏。CPU Qwen3 实模型检查 13/13、HTTP 检查 8/8。

CPU 构建及自有 CUDA 构建内的 CPU 产品均无 CUDA 动态依赖。`minillm-cuda-unit-tests` 链接 `libcudart.so.12`、`libcublas.so.12`，使用 CUDA C++20 且未启用 fast math。自有 CUDA 开启、llama 关闭的配置按设计拒绝，原始诊断位于 `rejected-no-llama.txt`。

## 复核

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda memcheck
bash scripts/dev.sh build
bash scripts/dev.sh test
bash scripts/dev.sh validate
bash scripts/dev.sh check-http 8057
bash scripts/dev.sh cuda build
bash scripts/dev.sh cuda test
pwsh -NoProfile -File scripts/Test-CtestEvidence.ps1 \
  -Directory benchmarks/results/validation/cuda-infra
```

`source-state.json` 的逐文件摘要与 `source-snapshot.zip` 条目一致。原始构建、CTest 控制台、模型和 HTTP 输出以 `.txt.gz` 保存；完整 CTest XML、模型和 HTTP JSON 可直接读取。模型、二进制、依赖 checkout 和工具链不随证据提交，重跑需依据清单另行取得。

本记录没有 CUDA 完整模型、GPU Serving、GPU PagedAttention 或性能加速结论。Windows 和远程 CI 未在本轮执行。
