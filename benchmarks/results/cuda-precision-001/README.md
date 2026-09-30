# FP16 矩阵边界研究证据

规范为 `CUDA-PREC-001`，结论为 **`blocked_correctness`**，候选不具备产品资格。
默认自有 CUDA 模型和 Serving 保持 F32；F16 模型路径仅用于研究。

[研究报告](../../../docs/PRECISION_STUDY.md) · [摘要](summary.json) ·
[完整证据索引](evidence.json)

## 结果

| 层次 | 范围与结论 |
| --- | --- |
| 矩阵微基准 | 16 shape、6 个独立进程，960 个样本全部保留，其中正式测量 576 个 |
| Owned 显存 | F32 3,449,229,312 bytes，F16 2,258,046,976 bytes，减少 34.53% |
| 模型数值 | 48 配置、840 行和 12 组短 golden 通过；长续写的一行 cosine 未达门槛 |
| 模型性能 | 未执行；在数值停止线停止，不用微基准代替模型收益 |
| Serving 对照 | 未执行；候选模式在入口被拒绝，保留原 F32 HTTP 回归 |
| 新 Profiler | 未采集；Tensor Core 使用为 `unverified` |

`generation-repeated` 的 prompt 长度为 1536，step=19 的 cosine 为
`0.9998858663580449 < 0.9999`。RMSE 和 max absolute 达标、argmax 一致
均不豁免此门槛。2 次失败计数来自同一行的自然续写和固定轨迹检查，不是两个独立样本。
完整失败向量与独立 `math.fsum` 重算结果均保留。显存收益不能使它成为 memory-only 成功。

## 原始包

唯一 canonical bundle 为
`https://github.com/lilong555/mini-llm-runtime/releases/download/cuda-prec-001-20260928/cuda-prec-001-20260928.zip`。
SHA-256、大小和文件数见 `evidence.json`。原始数值/日志/source snapshots 不进入 Git，
包内不含模型权重、依赖 checkout、构建产物或本地服务状态。

| 包内目录 | 内容 |
| --- | --- |
| `contract/` | 合同、原 F32 回归及初始 HTTP 失败记录 |
| `matrix-boundary/` | F16 存储、转换、矩阵 oracle、memcheck 与 F32 回归 |
| `micro-preflight/` | 正式采集前的固定输入和进程计划 |
| `micro-validation/` | 微基准入口回归、正式采集控制台、目录迁移复核 |
| `micro/` | 六进程完整 raw、配对摘要和既有离线复核器 |
| `model-validation/` | 48 配置、续写失败向量、身份、CTest、HTTP、memcheck |

不同阶段有不同 source/binary 身份。微基准来自 clean `03492ca`；
模型数值来自 `a8a56de` 加必要 dirty snapshot，不是 Release 标签的 clean build 测量。
Release 标签绑定可构建实现 `103070a`，其自身 CI run `36426140443` 五任务通过；
CI 不替代本机 GPU 数值检查，也不使失败候选成为产品。

## 离线复核

下载并在新目录解包。从解包目录执行，不需要模型、GPU 或原始绝对路径：

```bash
sha256sum -c SHA256SUMS
python3 micro/verify.py --directory micro
```

微基准复核器检查输入、源码快照、六进程计划、所有样本及数值/资源合同。
`SHA256SUMS` 检查包内文件完整性，不替代模型正确性。模型数值程序的原始退出码为 1，
`model-validation/precision/validation-summary.json` 中的 `passed=false` 是应保留的结果。

下面只从原始失败向量重算 cosine，不运行 GPU：

```bash
python3 - <<'PY'
import json
import math
from pathlib import Path

path = Path("model-validation/precision/first-numeric-failure-logits.json")
data = json.loads(path.read_text())
a, b = data["f16_logits"], data["f32_logits"]
assert len(a) == len(b) == 151936
assert all(math.isfinite(x) for x in a + b)
cosine = math.fsum(x*y for x, y in zip(a, b)) / math.sqrt(
    math.fsum(x*x for x in a) * math.fsum(x*x for x in b))
assert cosine < 0.9999
print(cosine)
PY
```

## 上板复现

按仓库模型和依赖 metadata 准备环境，输出目录必须尚不存在：

```bash
bash scripts/dev.sh own-cuda build
build/wsl-own-cuda/bin/minillm-cuda-model-tests \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --contract tests/data/qwen3_validation_cases.json \
  --precision-study benchmarks/runtime-inputs/qwen3-precision-v1.json \
  --output .run/precision-reproduction
```

原微基准采集入口为 `scripts/Benchmark-CudaMicro.ps1 -PrecisionStudy`。
项目研究不再追加微基准 trial，也不启动被数值门禁阻止的性能采集。
重跑依赖原模型、CUDA/cuBLAS 环境与硬件，不将跨环境复现解释为新的配对性能结论。
