# CUDA 基础算子验收

范围为 `CUDA-VS-001 / Step 5`。单 stream 基础算子由本项目实现，block 归约复用 CUDA 12.8 随附的 CUB 2.7.0，矩阵继续使用 cuBLAS FP32 PEDANTIC。没有完整 GPU transformer layer、模型、Serving 或性能结论。

## 正确性

| 配置 | CTest 套件 | 用例执行 |
| --- | --- | ---: |
| 自有 CUDA ON、上游 CUDA OFF | 10/10 | 210 |
| CPU | 7/7 | 178 |
| 独立核心，无 llama | 5/5 | 167 |
| 上游 CUDA ON、自有 CUDA OFF | 7/7 | 178 |

共 29 次套件、733 次用例执行，含重复执行。CPU 模型 13/13、HTTP 8/8；HTTP 临时服务已退出。311 条权重记录中的 310 个唯一 tensor 全量回读一致，88 组真实形状矩阵回归通过。

11 项算子用例覆盖：

- gather 的 37/1024 宽度、重复和首尾索引、stride、NaN padding，以及先屏蔽读取的非法设备索引。
- RMSNorm 的 1/17/31/32/33/128/1024/3072 宽度、分组 head、原地与独立输出、部分重叠拒绝；有限输入幅值从 0 到 `FLT_MAX`，epsilon 从最小正 subnormal 到 `FLT_MAX`。
- NeoX RoPE 的 128 head_dim、1/8/16 heads、0/1/15/16/17/1535/2047 位置及非法位置屏蔽。
- residual/SwiGLU 的非整 warp 尾部、不同 stride、正负极值 gate。
- argmax 的 1/31/32/33/257/151936 宽度、跨 lane 的相等最大值、NaN/Inf、最小 ID 选择及错误行 `-1`。
- descriptor 的设备、容量、形状、地址对齐和别名检查；预检失败不写入设备结果。
- 同 stream 的 gather → norm → GEMM → Q/K norm → RoPE → residual → SwiGLU → argmax 组合；无中间 host 读回，项目设备分配/释放调用不增加。

数值检查保持 `atol=2e-4, rtol=2e-4`，使用 CPU FP64 对照；组合参考在算子边界转回 FP32。memcheck 为 0 错误、0 泄漏，racecheck 为 0 hazards，synccheck 为 0 错误。极值处理范围见 `ENG-038`。

## 证据与命令

`source-state.json`、`source-snapshot.zip`、`evidence.json` 固定实际工作区源码、基点与二进制身份。`real-storage/` 保留本轮权重与矩阵原始报告；CTest、CPU 回归和三种 sanitizer 输出分别保存。模型、依赖 checkout、二进制和工具链不在归档内。

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda memcheck
compute-sanitizer --tool racecheck --error-exitcode 1 \
  build/wsl-own-cuda/bin/minillm-cuda-ops-tests
compute-sanitizer --tool synccheck --error-exitcode 1 \
  build/wsl-own-cuda/bin/minillm-cuda-ops-tests
pwsh -NoProfile -File benchmarks/results/validation/cuda-ops/verify.ps1
```

归档复核不运行 GPU，也不等于模型重跑。源/设备权重分别为 Q8_0/F32；算子通过不能证明连续 KV 状态、causal attention、完整 28 层或生成 token 已完成。
