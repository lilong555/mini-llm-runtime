# CUDA 数值验证

自有 CUDA Runtime 的数值入口使用固定 Qwen3-0.6B Q8_0、同有效权重的 F32 llama 参照和自有 CPU Runtime。模型 SHA-256、输入语料、计算精度与容差由 [验证契约](../tests/data/qwen3_validation_cases.json) 固定；完整执行与资源接口见 [CUDA Runtime](CUDA_RUNTIME.md)。

## 执行

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda model-check .run/cuda-short
bash scripts/dev.sh own-cuda model-full-check .run/cuda-full
python3 scripts/analyze_cuda_validation.py --directory .run/cuda-full
```

模型文件位于 `models/`，通过 `scripts/models.py` 与固定 manifest 获取。每个输出目录必须尚不存在。实模型检查不属于默认 CTest；无模型的输入生成、比较器和报告反例属于 CTest。

CPU、F32 llama 与 GPU 顺序加载。CPU 参照使用 8 线程、auto SIMD、FP16 KV；全量验证的 F32 llama 在 CPU 上执行，使用 FP16 KV、非融合 attention 和 FP32 QK/PV 累加。GPU 使用 FP32 有效权重、FP32 activation、FP16 KV、FP32 QK/PV 累加与 FP64 softmax 分母，不使用 fast math。

参照端的 F32 指有效权重，不能据此推断所有中间运算的精度。固定上游版本的 CPU FlashAttention 在单 query、FP16 KV 路径使用 FP16 PV 累加缓冲；长重复语料续写的两处 cosine 超限及融合/非融合固定输入重放见 `ENG-042`。全量入口显式关闭参照端 FlashAttention，保留相同 checkpoint、输入、KV dtype 和容差。默认短模式仍使用原有融合参照。

## 覆盖

| 范围 | 输入与检查 |
| --- | --- |
| 固定输入 | 中文、英文、重复 token、特殊 token，长度 16/33/128/256/1536 |
| 批次 | 整批最多 1/16/33/128 行，S=1/2/4，共 240 个组合 |
| 独立序列 | 按 position、sequence 交错；第 s 个序列使用 `(corpus_index+s)%4` 的语料，各自维护 KV |
| 参照 | 每个 corpus/length 独立执行 S=1、chunk=128 的 CPU、F32 llama、GPU 路径 |
| 采样 | 过滤契约中的 position；3920 个 GPU 采样行，11760 次 teacher-forcing 比较 |
| 自然生成 | 四类语料，prompt 长度 16/128/1536，各生成 32 个 token，共 12 组；最后 KV 长度为 prompt+31 |
| 稳定金标准 | 三个英文短样例，在 S=1/S=4 下严格匹配已有 8-token 序列 |
| 计时开关 | S=4、chunk=33、length=1536，40 个采样行 logits 位级一致 |
| 容量与复用 | slot 3 追加到 2048；拒绝 position=2048，不改变状态；clear 后与相同槽位初始结果位级一致 |

chunk 是整个 batch 的容量，不是每个序列的容量。1536-token 输入分批执行，不能视为一次 B=1536。特殊 token 作为固定输入，不触发提前结束；自然生成也固定执行 32 个输出位置，不因 EOG 提前结束。

## 判定

- 所有 logits 必须有限；RMSE `<0.05`、最大绝对误差 `<0.5`、cosine `>=0.9999`。
- 参照 top1-top2 margin `>2*max_absolute` 时要求 argmax 相同；否则记录为 near-tie，保留真实 argmax，不宣称严格 greedy 等价。
- teacher-forcing 始终使用相同输入，允许跨 chunk、batch 和 backend 比较。
- 自然生成仅比较输入历史仍相同的位置，包含首次输出分歧那一步。此后保留每一步 token、margin、logits SHA-256，不比较已经漂移的 logits。
- 数值超限不会删除已完成用例，可以继续收集其他组合；执行异常或 poisoned 状态终止该实例，报告不能标为完成。
- CPU 原有同实现的 chunk/COW 容差不变。GPU 没有 prefix alias/COW，本入口不构造虚假的 GPU 共享测试。

## 报告

`input.json` 保存展开后的语料、组合描述及包含 batch 边界的输入摘要。`canonical/` 保存三类独立参照的采样分数和摘要，`teacher/` 保存每个组合的逐行指标，`generation/` 保存三条生成路径及共同输入比较。

`full-validation.json` 包含覆盖数、首处失败、首处分歧、计时开关、容量边界和各配置的 diagnostics。`validation-summary.json` 只有执行完成后才发布完成状态；初始 `incomplete`、异常 `failed` 都不能作为通过证据。`weight-plan.json` 与 `memory-plan.json` 记录唯一权重、别名和设备分配。

`analyze_cuda_validation.py` 只读原始报告，独立枚举完整输入集合、重算输入摘要、检查参照关联、冻结门槛、生成历史和分类传输计数，并向标准输出给出复核结果。它不重新执行模型或从 SHA-256 恢复 logits。源码、二进制、工具链和逐文件摘要由外层证据包记录，归档复核不等于在另一台机器重新运行模型。

## 参照诊断

```bash
build/wsl-own-cuda/bin/minillm-cuda-reference-diagnostic \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --reference-model models/Qwen3-0.6B-Q8_0-dequant-F32.gguf \
  --contract tests/data/qwen3_validation_cases.json \
  --generation-report benchmarks/results/validation/cuda-full/diagnostics/fused-reference/real-model/generation/cuda-repeated-l1536-g32.json \
  --output .run/cuda-reference-check
```

该入口固定重放同一段输入，记录 CPU、上游融合/非融合 attention 和 GPU 的全部 32 步分数、摘要及对照指标。它检查融合参照的已知失败是否复现，以及非融合参照是否满足相同门槛；通过只代表这一诊断成立，不代替全量验证。

## 验收边界

当前编译产物的完整数值证据位于 [微基准与数值验收](../benchmarks/results/validation/cuda-micro/README.md)。模型性能采集同时核对 Runtime 源文件集合、摘要与 `minillm-cuda-model-tests` 的二进制 SHA-256，不能仅凭旧报告的通过状态继承数值门禁。

本入口只验收数值与数据路径。CPU8/16 性能、A/A 噪声、矩阵和算子 microbenchmark、完整模型 Profiler 属于独立门禁；不提供 GPU Serving、GPU paging 或自有 PagedAttention 的验收结论。
