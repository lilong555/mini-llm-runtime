# 历史运行与验证证据

本目录提供 CPU 阶段基线、在线 batch 观测和早期 CPU/CUDA 验证的完整归档入口。
仓库保留结果摘要、协议、固定输入、代表性样本和全部历史 CTest XML。
各实验的采集身份、原始结果、源码快照及失败诊断由包内原有清单固定。

| 组件 | 摘要入口 | 完整包内目录 |
| --- | --- | --- |
| Runtime 阶段基线 | [线程扩展与上下文](../wsl-runtime-profile/README.md) | `benchmarks/results/wsl-runtime-profile/` |
| 在线 batch 观测 | [Serving 与阶段归因](../wsl-batch-telemetry/README.md) | `benchmarks/results/wsl-batch-telemetry/` |
| CPU/CUDA 历史验证 | [验证索引](../../../docs/VALIDATION.md) | `benchmarks/results/validation/` |

Runtime 包含 36 个独立进程、558 次测量；在线观测包含 42 个服务进程、594 个请求和
28 份 JSONL。CUDA 数值原件保留完整通过结果及融合参照的原始失败。
`validation/cuda-micro/` 和 `validation/cuda-profiler/` 使用
[CUDA-VS-001 完整包](../cuda-vs-001/README.md)。

## 获取

- [完整 ZIP](https://github.com/lilong555/mini-llm-runtime/releases/download/historical-evidence-20261024/historical-evidence-20261024.zip)：18428252 字节，1682 个文件。
- SHA-256：`3fb7e3aea8de34be470aff1f8abf92c3d29488bb1dadd5962a246ff7f3f2d5b5`。
- [独立摘要](historical-evidence-20261024.zip.sha256)与[公开索引](evidence.json)。

从仓库根目录执行，使用新的解包目录 `.run/historical-evidence`。依赖为 curl、
unzip、Python 3 和 PowerShell 7。

```bash
mkdir -p .run
cd .run
curl --fail --location --output historical-evidence-20261024.zip \
  https://github.com/lilong555/mini-llm-runtime/releases/download/historical-evidence-20261024/historical-evidence-20261024.zip
sha256sum -c ../benchmarks/results/historical-evidence/historical-evidence-20261024.zip.sha256
unzip -n historical-evidence-20261024.zip -d historical-evidence
cd ..
```

包内 `archive-manifest.json` 固定逐文件 SHA-256 和字节数。22 份历史源码快照与其原始
manifest 摘要一致。归档标签绑定 `155dd980`，各实验的实际采集源码由包内元数据标识。

## 离线复核

以下命令读取独立解包副本，Runtime 和 telemetry 分析器会更新该副本的派生汇总。

```bash
pwsh -NoProfile -File .run/historical-evidence/scripts/Analyze-Runtime.ps1 \
  -Directory .run/historical-evidence/benchmarks/results/wsl-runtime-profile/scaling
pwsh -NoProfile -File .run/historical-evidence/scripts/Analyze-Runtime.ps1 \
  -Directory .run/historical-evidence/benchmarks/results/wsl-runtime-profile/context
python3 .run/historical-evidence/scripts/analyze_telemetry.py \
  .run/historical-evidence/benchmarks/results/wsl-batch-telemetry/context-16/trial-0-off \
  .run/historical-evidence/benchmarks/results/wsl-batch-telemetry/context-16/trial-0-batches \
  .run/historical-evidence/benchmarks/results/wsl-batch-telemetry/context-16/trial-0-stages \
  --output .run/telemetry-rechecked.json
pwsh -NoProfile -File .run/historical-evidence/benchmarks/results/validation/cuda-full/verify.ps1
pwsh -NoProfile -File .run/historical-evidence/benchmarks/results/validation/cuda-benchmark/verify.ps1
```

其余在线组按各自的 `experiment.json` 传入全部 trial 目录。
`cuda-storage`、`cuda-ops`、`cuda-layer` 和 `cuda-model` 也分别提供 `verify.ps1`。
公开下载件的 14 项既有离线检查通过，包含两组 Runtime、五组在线观测、
六个 CUDA 归档及历史 CTest 证据。每项实验的原始结论与失败分类保持原样。
