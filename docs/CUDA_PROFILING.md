# 自有 CUDA 完整模型 Profiler

`Profile-CudaRuntime.ps1` 使用与正式模型基线相同的 `mini-cuda-runtime-bench`、模型、输入和 Runtime 源码，执行完整 Qwen3 模型。它不替代无 Profiler 的 [模型性能基线](CUDA_BENCHMARKS.md)，不提供 GPU Serving 或 PagedAttention。

## 运行

采集入口面向 Linux/WSL2，需要已有完整 70 进程模型基线、Python 3.10+、PowerShell 7、Nsight Systems 2026.1.3 和 Nsight Compute 2025.1.1：

```bash
bash scripts/dev.sh own-cuda runtime-profile \
  -PreflightOnly -OutputDirectory .run/cuda-profiler-preflight

bash scripts/dev.sh own-cuda runtime-profile \
  -OutputDirectory benchmarks/results/cuda-profiler-new-run
```

输出目录必须为空。`BaselineDirectory`、`BinaryDirectory`、`Model`、`NsightSystems`、`NsightCompute` 可显式指定；不能更换模型或基准二进制身份。默认优先使用 `$HOME/.local/bin/nsys`，不将 CUDA 附带的旧版 Nsight Systems 当成已验证工具。NCU 硬件计数器权限由主机配置决定，WSL root 不替代 Windows 主机授权。

五个独立进程固定为 `off-before → nsys → off-middle → ncu → off-after`。每个进程均执行全部 12 项 workload、585 次 forward，保持 S=4、Lmax=2048、B=128；没有缩短模型或共享 prefix。每种 Profiler 只有一个独立进程，相邻无 Profiler 进程只提供诊断参照，不计算显著性。

## 时间线与指标

NSys 使用 CUDA trace，关闭 CPU sampling 和 context-switch 采集。复核器检查 585 个完整 forward 的项目 kernel 顺序、每次 28 层、单项目 stream、逐调用显式传输及真实设备身份，保存 CUDA Runtime API、kernel 时间、设备区间并集与间隙。Runtime API 观察不覆盖 Driver API；项目分配归属仍由 Runtime 计数验证。

本机采用 legacy software-instrumented trace，硬件 tracing 模式不可用；Unified Memory 跟踪也未验收。原始诊断保留在 `nsys-summary.json`，这些限制不改称已通过的硬件追踪能力。

NCU 固定选择 `chunked-prefill-1536` 的 iteration 2、最后一个 128-token chunk、第 27 层 `pv_kernel`。对应 forward index=55、匹配 kernel 的 `launch-skip=1567`、`launch-count=1`。采集使用 basic set、kernel replay，clock/cache control 均为 none，不提前终止模型进程。

NCU 自动名称简化关闭，CSV 使用原始 mangled symbol，与 NSys 的 `mangledName` 精确相等；设备、launch 形状和硬件指标单位继续核对。时间、SM throughput、DRAM throughput、occupancy 与 replay passes 分别保存，不将逻辑 bytes/time 当成硬件带宽。

所有诊断进程使用相同的环境白名单，NSys 使用 `--discard-environment=true`。原始采集不保存完整继承环境；模型、二进制和工具摘要在各进程前后核对。

## 归档

[完整模型 Profiler](../benchmarks/results/cuda-model-profiler/README.md) 包含五份模型报告、原始命令和工具版本、源码快照、NCU CSV，以及 `profiler-raw.zip`：

- `nsys.nsys-rep`：原始 Nsight Systems 报告。
- `nsys.sqlite`：完整时间线的离线导出。
- `ncu.ncu-rep`：选定 kernel 的原始 Nsight Compute 报告。

`artifact-manifest.json` 记录包和成员的 SHA-256、尺寸及可获取位置。无需 CUDA、Nsight、模型或原绝对路径即可复核：

```bash
python3 -B /path/to/profiler/verify.py --directory /path/to/profiler
```

完整交付包将模型基线、微基准、数值、Profiler 和工具回归放在各自目录，保留独立来源。导出和包内复核还需要 PowerShell 7：

```bash
python3 -B scripts/cuda_evidence_bundle.py \
  --model benchmarks/results/cuda-model-baseline \
  --micro benchmarks/results/cuda-micro-baseline \
  --numerical benchmarks/results/validation/cuda-micro \
  --profiler benchmarks/results/cuda-model-profiler \
  --validation benchmarks/results/validation/cuda-profiler \
  --export /existing/directory/cuda-vs-001.zip

python3 -B /path/to/extracted/verify.py --directory /path/to/extracted
```

归档不含模型权重、编译产物或依赖 checkout；离线复核不是同一二进制重跑或可信执行证明。

## 解释边界

正式模型基线的 24 项比较中，14 项为 `faster`、10 项为 `measurement_inconclusive`。Profiler 不改变该结论。当前 NSys 诊断相对差异约为 -14.46% 至 +9.31%，NCU 约为 +9.96% 至 +150.98%；只有一组独立诊断，负差异不代表 Profiler 带来加速。

NSys 的全表还包含 warmup 和 prefix setup，kernel 累积时间不等于 HTTP 请求分布。选定 NCU kernel 的回放指标不能外推所有形状。`ENG-048` 的微基准顺序相关差异和 `ENG-051` 的模型 A/A 噪声仍未完成受控归因；这里不修改门槛，也不筛除慢样本。
