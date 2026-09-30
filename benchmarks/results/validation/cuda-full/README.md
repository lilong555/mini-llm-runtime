# CUDA 全量数值验收

范围为 `CUDA-VS-001` Step 8 的数值与数据路径门禁。正式性能、A/A、microbenchmark、完整模型 Profiler 和 GPU Serving 不在本次结论内。

## 验收结果

| 检查 | 结果 |
| --- | --- |
| 自有 CUDA / CPU / 独立核心 / 上游 CUDA CTest | 14/14、8/8、6/6、8/8 |
| CTest 总量 | 36 次套件、807 次用例执行；不是 807 个不同用例 |
| CPU 模型 / HTTP | 13/13、8/8；HTTP 后端为 CPU |
| 默认短模式 | 128 次 logits 比较、六组 8-token 金标准通过 |
| 全量固定输入 | 四类语料 × 五个长度 × 四个 chunk × 三个序列数，共 240 个组合 |
| 全量比较 | 3920 个 GPU 采样行，11760 次 teacher-forcing 比较 |
| 自然生成 | 四类语料 × prompt16/128/1536，各 32-token，共 12 组、768 次共同输入比较 |
| 数值与 greedy | 12528 次比较全部通过，无非有限值、argmax 差异或 near-tie |
| 计时开关 | S=4、chunk=33、length=1536，40 个采样行位级一致 |
| 容量与复用 | slot 3 长度 2048；拒绝越界追加且状态不变；clear 后位级复用通过 |
| 独立归档复验 | 783 个产物通过；12 项缺件、篡改及语义反例被拒绝，原件未改变 |

长度为 16/33/128/256/1536，chunk 为整批 1/16/33/128 行，S=1/2/4。每个 sequence 使用轮换后的独立语料与 KV；不采用 prefix alias 或共享重置。

## 数值范围

| 参照 | 比较数 | 最大 RMSE | 最大绝对误差 | 最小 cosine |
| --- | ---: | ---: | ---: | ---: |
| 自有 Q8_0 CPU、8 线程、FP16 KV | 4304 | 0.005444215 | 0.027509690 | 0.999999171 |
| 同有效权重 F32 llama、8 线程、FP16 KV、非融合 attention | 4304 | 0.019126342 | 0.072307349 | 0.999986580 |
| 独立 GPU S=1、chunk=128 | 3920 | 0.002325767 | 0.011589528 | 0.999999777 |

原门槛为 RMSE `<0.05`、最大绝对误差 `<0.5`、cosine `>=0.9999`；argmax 规则按固定 margin 公式执行。最大 RMSE 与最小 cosine 位于 `special-l1536-g32` 的 step 4、position 1539。这些结论只适用于本模型、平台、语料和配置，不表示跨后端 logits 位级等价。

## 参照精度

[融合参照诊断](diagnostics/fused-reference/README.md) 保留一份完整的失败采集：11760 次固定输入比较通过，重复 token 长上下文续写有两处 cosine 超限。固定上游版本的 CPU 单 query 融合 attention 使用 FP16 PV 累加缓冲，与自有 CPU/CUDA 的 FP32 累加不同。

固定输入重放完整复现原 CPU、融合参照、GPU 的 96 个分数及摘要；非融合参照的 32 步满足相同门槛。全量采集使用明确的非融合参照配置，不修改 checkpoint、FP16 KV、输入或门槛。

`reference-mode-comparison.json` 核对两次独立采集：65 个产品源文件、依赖 commit、验证契约和输入字节相同；同一 GPU 路径的 4444 个采样行与 CPU 路径的 524 个采样行摘要分别一致。参照配置的差异没有混入产品计算变化，原失败没有被删除或标为通过。

## 资源与边界

S=4、Lmax=2048、B=128 的项目设备分配为 3,449,229,312 字节，与计划一致。三个配置的稳态权重和 RoPE 上传不增加，中间激活没有 host 传输，项目设备 allocation/free 为零。metadata、token、status 和显式 debug logits 的累计字节由复核器按全部输入重新核算；短金标准的 greedy 路径不下载全词表。

运行时及设备源码与前一模型 gate 相同。sanitizer 证据见 [完整模型与 CLI 验收](../cuda-model/README.md)；本目录没有把该记录当作全部长语料已重跑 sanitizer。未执行 Windows 或远程 CI，没有性能加速结论。

## 使用与复核

```bash
bash scripts/dev.sh own-cuda model-full-check .run/cuda-full
python3 scripts/analyze_cuda_validation.py --directory .run/cuda-full
pwsh -NoProfile -File benchmarks/results/validation/cuda-full/verify.ps1
```

实模型运行需要固定权重和新输出目录；归档复核只需要 Python 3、PowerShell 及本目录文件，不依赖权重、二进制或原采集绝对路径。`source-snapshot.zip` 对应完整模型采集，`verification-source/` 对应回归与复核工具；失败采集及其诊断另有独立源码身份。

`real-model/` 保存逐组合、逐位置和逐生成步的原始指标与摘要。`numerical-audit.json` 为独立数值检查，`revalidation.json` 记录迁移与反例；源码及逐文件身份由 `evidence.json` 固定。协议和接口见 [CUDA 数值验证](../../../../docs/CUDA_NUMERICS.md)，后续门禁见 [执行状态](../../../../docs/EXECUTION_STATUS.md)。
