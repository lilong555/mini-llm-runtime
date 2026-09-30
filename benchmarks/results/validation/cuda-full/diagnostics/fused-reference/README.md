# CPU 融合 attention 参照诊断

本目录的完整数值门禁状态为 **失败**，不是验收通过的模型基线。源码、模型与二进制身份、全部输入、240 个组合和 12 组续写结果均保留。

## 结果

- 11760 次 teacher-forcing 比较通过。
- 768 次自然生成共同输入比较中，两次 F32 llama 参照 cosine 低于 `0.9999`；RMSE、最大绝对误差均未超限。
- 两次失败均位于 `repeated-l1536-g32`：step 6、position 1541 的 cosine 为 `0.999858573856455`；step 19、position 1554 为 `0.9998883358146589`。
- 所有生成 token 相同；GPU 对自有 CPU 参照无数值失败。token 一致不替代 logits 门槛。
- 六组短金标准、2048-token 容量边界、slot 复用与计时开关位级检查通过。

## 算术边界

参照模型的 F32 指权重 dtype。固定上游版本的 `ggml_compute_forward_flash_attn_ext_f16_one_chunk` 在 FP16 KV 路径将 Q 转换为 FP16，并使用 `VKQ16` 与 `ggml_vec_mad_f16` 累加 PV。长上下文单 query 的 split-KV 分支使用这一实现；prefill 的 tiled 分支使用 FP32 PV 缓冲。graph 的 `GGML_PREC_F32` 标记不等于这一 CPU 分支实际使用 FP32 PV 累加。

[固定输入重放](reference-diagnostic/attention-reference.json) 使用相同的 1536-token prompt 和 32-token 续写输入，比较自有 CPU、上游融合 attention、上游非融合 attention 和自有 CUDA。原 CPU、融合参照和 GPU 的 96 个分数及摘要完全复现；仅切换上游 attention 模式，非融合参照的 32 步全部通过，最小 cosine 为 `0.999995715032576`。

重放是参照精度诊断，不是完整语料通过证明。两种上游路径均使用相同 F32 checkpoint、FP16 KV 和 8 线程，未修改上游源码或数值门槛；自有 CPU/CUDA 计算不变。

## 复核

```bash
pwsh -NoProfile -File benchmarks/results/validation/cuda-full/diagnostics/fused-reference/verify.ps1
```

脚本校验源码 ZIP、逐文件摘要、原始数值失败及重放结果，返回 `Status=completed_numeric_failure`、`NumericalGatePassed=False`。验证工具来自本目录的诊断源码快照，不需要模型权重、二进制或原始采集绝对路径。

`source-snapshot.zip` 对应完整融合参照采集；`reference-diagnostic-source/` 对应固定输入重放及其复核工具。`report-fixtures-initial.txt` 保留 `ENG-041` 的 fixture 参数错误，`report-fixtures.txt` 为该问题的 12/12 验证。`core-ctest.xml` 只属于此诊断采集阶段，不能代替主目录的完整构建回归。
