# M0 证据归档基线

本目录提供可独立复验的 CPU Runtime 基线及一组历史在线观测包。用途为 `archive_revalidation`；包含输入、原始报告、源码快照、派生汇总与验收脚本，不包含模型权重、实测二进制或工具链。

## CPU Runtime

- 模型：固定 SHA-256 的 Qwen3-0.6B Q8_0；8 线程、自动 SIMD、F32 激活、F16 分页 KV。
- 配置：context=2048、page=16、max_sequences=4、batch=128。
- 协议：3 个独立 trial；每 case 1 次预热、1 次正式测量；`none / stages` 交替进程顺序。
- 覆盖：6 个进程、6 种工作负载、36 次测量；18 次带阶段计时。
- 初始化与 prefix setup 单列；`sampling_ns` 单列，下面的 forward 时间不包含采样。
- CPU 原协议使用 `pinned_prefix_share_reset`；它不作为后续 CPU/CUDA 公平对照的连续 KV setup 协议。

| 工作负载 | 无 profiler 中位 ms | 有 profiler 中位 ms | 中位差异 |
| --- | ---: | ---: | ---: |
| prefill-16 | 202.601 | 216.402 | +6.81% |
| prefill-128 | 1499.844 | 1495.692 | -0.28% |
| decode-16 | 40.422 | 40.469 | +0.11% |
| decode-256 | 44.947 | 41.201 | -8.33% |
| decode-1536 | 54.911 | 55.771 | +1.57% |
| mixed-16-2 | 241.199 | 248.422 | +2.99% |

全部原始样本、每轮配对与开销保留在 `baseline/`。没有锁频、温度或背景负载控制，也未执行 A/A；正负差异包含噪声，不能用于宣称代码加速或无退化。`prefill-16` 的观测组较慢，`decode-256` 的观测组较快，均不能解释为纯插桩成本。

采集 run_id：`20260924T153032Z-68ac27591320-7e661c79`。基点为 `68ac275913207975a88e2090c6617467e351301c`，`git_dirty=true`，实际构建输入绑定 `baseline/source-state.json` 和原始 ZIP；不归属于随后提交的 clean build。模型、二进制、输入和编译配置以 `baseline/manifest.json` 为准。

## 可交付包

- `baseline-bundle.zip`：上述 CPU Runtime 的完整归档复验包。
- `telemetry-bundle.zip`：历史 `wsl-batch-telemetry/context-256/trial-0-batches` 的两份服务报告及 JSONL；原 source-state、manifest、模型/二进制身份保持原测量归属，不作为新性能样本。
- 每个 ZIP 有同名 `.sha256`，包内 `bundle-manifest.json` 逐文件固定 SHA-256，`verification_entry` 指定包内复验入口。
- `source-snapshot.zip` 保存测量时源码，`verification/` 保存封包时验收脚本，两者身份分开记录。

在仓库根目录，选择一个新临时目录解压；保留 ZIP 原件：

```bash
work=$(mktemp -d)
python3 -m zipfile -e benchmarks/results/evidence-m0/baseline-bundle.zip "$work"
pwsh -NoProfile -File "$work/verification/Test-EvidenceAvailability.ps1" -Directory "$work"
pwsh -NoProfile -File "$work/verification/Analyze-Runtime.ps1" -Directory "$work"
```

观测包使用相同的 availability 检查，随后运行包内 `verification/analyze_telemetry.py`，传入解压目录与 `--output`。每次复验使用新副本；分析会更新派生结果，更新后不再对应原包的派生文件摘要。

独立目录检查及文件访问跟踪见 [验证记录](../validation/evidence-m0/README.md)。此处无自有 GPU 模型、GPU Serving 或性能加速结论。
