# CUDA 连续 KV 与层执行验收

范围为 `CUDA-VS-001 / Step 6`。提供连续 FP16 KV、逐 query 因果 GQA、QK → softmax → PV 和完整 transformer 层。单层通过不代表完整 28 层、真实生成 token、GPU Serving 或性能已验收。

## 正确性

| 配置 | CTest 套件 | 用例执行 |
| --- | --- | ---: |
| 自有 CUDA ON、上游 CUDA OFF | 11/11 | 218 |
| CPU | 7/7 | 178 |
| 独立核心，无 llama | 5/5 | 167 |
| 上游 CUDA ON、自有 CUDA OFF | 7/7 | 178 |

共 30 次套件、741 次用例执行，含重复执行。CPU 模型 13/13、HTTP 8/8；临时服务已退出。310 个唯一 tensor 全量回读和 88 组真实矩阵检查通过。目标 S=4、Lmax=2048、B=128 的实际项目分配为 3,449,229,312 字节，包含 1 MiB 初始化 RoPE 表。

7 项层级用例覆盖：

- 预检、准备、执行、提交、clear 和终止性 poisoned；固定 seed 的 1000 次随机交错 append/clear。
- 非连续 position、非法 token/sequence、batch/context 上界；预检失败不提交逻辑长度。
- FP16 RN-even 的 subnormal、正负零、中点与极值；跨 slot/layer、stride、guard 和非法转换。
- 设备写入后的受控 nonfinite 故障，逻辑长度不提交，poisoned 不能 clear 或继续使用。
- GQA、长度 1/17/33/1536、非整 warp head_dim、真实 16/8/128 头形状。
- 未使用 KV 与 padding 的 NaN 污染；每个 query 只读 `position+1`，不会读取同批未来 token。
- 完整层的交错序列、追加、clear 后重用，以及固定 workspace 的分配/释放计数。

设备单测和真实层 memcheck 均为 0 错误、0 泄漏；层测试 racecheck 为 0 hazards，synccheck 为 0 错误。

## 真实权重

首层和末层各执行 M=2、混合 M=18、交错追加 M=4，共六组。使用同一 Q8_0 checkpoint 的 FP32 有效权重、FP32 QK/PV、FP16 KV 和 FP64 softmax 分母；矩阵由 cuBLAS FP32 PEDANTIC 执行。

独立 CPU FP64 整层对照使用固定模型门槛：RMSE < 0.05、max absolute < 0.5、cosine >= 0.9999。最大观察误差为 `0.12060546875`，最大 RMSE 为 `0.0033648982414092882`，最小 cosine 约 `0.999999999874`。这是受控层输入，不是第 27 层由前 27 层实际传播得到的 hidden。

基础算子和受控 fixture 保持 `atol=2e-4, rtol=2e-4`。真实 Q/K/V 本身满足该逐元素门槛；额外以实际 Q/K/V 为共享输入，在 CPU FP64 中计算 attention 和 FFN，同样满足该门槛。两个 CPU oracle 分别持有独立 KV 历史，不用相互替换的结果伪装独立整层对照。

`diagnostics/` 保留独立整层误套算子逐元素门槛时的失败、源码和二进制身份。末层两次 V 舍入跨界将约 `1.14e-5` 的连续差异放大为 `0.001953125`；三组末层的直接逐元素比值均保留在 `max_unit_tolerance_ratio`，不宣称它们满足算子门槛。参见 `ENG-039`。

## 复核

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda layer-check
bash scripts/dev.sh own-cuda layer-memcheck
pwsh -NoProfile -File benchmarks/results/validation/cuda-layer/verify.ps1
```

`evidence.json` 固定实际源码、二进制、工具环境及产物摘要。归档复核使用源码 ZIP 内的工具，不依赖原工作区；缺 ZIP、篡改和已有目录覆盖的反例单独记录。模型、二进制、依赖 checkout 与工具链不在归档中。归档复核不执行模型，不等于在另一台 GPU 重跑。
