# 自有 CUDA 完整模型 Profiler

本目录为完整 Qwen3 模型的外部诊断，关联 [70 进程无 Profiler 基线](../cuda-model-baseline/README.md)。五个进程依次为无 Profiler、NSys、无 Profiler、NCU、无 Profiler，均执行 585 次 forward；输出 token 与关联基线一致。

## 结果

- NSys：585 次完整 forward、每次 28 层、368610 次 kernel，单项目 stream；测量部分为 258 次 forward。
- 每个 forward 的项目 kernel 顺序和显式传输满足契约；观察区间未出现 CUDA Runtime allocation/free API，Driver API 不在此项计数范围。
- NCU：`chunked-prefill-1536`、iteration 2、最后一个 128-token chunk、第 27 层 `pv_kernel`，forward index=55。
- NSys 与 NCU 的原始符号精确相等；grid=`2048×1×1`、block=`256×1×1`，8 次 kernel replay。
- NCU 设备时间 `971808 ns`，SM throughput `77.42%`、DRAM throughput `7.51%`、occupancy `70.96%`；这些只属于该选定 kernel。

完整表格见 [analysis.md](analysis.md)、`nsys-summary.json`、`ncu-selected-kernel.json` 与 `profiler-overhead.json`。NSys 表格还包含 warmup 和 prefix setup，不等于 HTTP 请求分布。

单组开关诊断中，NSys 相对延迟约 -14.46% 至 +9.31%，NCU 约 +9.96% 至 +150.98%。没有显著性区间，也不据负差异宣称 Profiler 加速。NCU 回放和插桩时间不进入正式模型性能基线。

## 原始证据

Run ID 为 `20260926T080547Z-0754722511b9-2218194b`，源码基点为 `0754722511b98a8d834167e3befb2c3307e81fa2`，采集时为 dirty worktree；实际代码由源码状态与 ZIP 固定。模型基准二进制与关联基线相同，SHA-256 为 `d7eb569942f6ffa6621f346c351e16d531dad70041afc76c0cc25dbac0f53180`。

`artifact-manifest.json` 登记 42 个必需原始产物。`profiler-raw.zip` 为 24217543 字节，实际包含 `nsys.nsys-rep`、`nsys.sqlite`、`ncu.ncu-rep`，各成员及 ZIP 均有 SHA-256，不依赖作者 `.run` 路径获取。工具为 NSys `2026.1.3.425`、NCU `2025.1.1.0`，完整命令、版本输出、进程环境白名单和日志均保留。

Python 3.10+ 可在独立目录复核，无需模型、二进制、CUDA、Nsight 或原路径：

```bash
python3 -B verify.py --directory .
```

48 个既有文件在迁移复核中保持不变；缺少原始 ZIP、NCU 单位篡改、选定层不符三项反例被拒绝，见 `revalidation.json`。

## 限制

本机 NSys 使用 legacy software-instrumented CUDA trace，硬件 tracing 模式不可用；报告保留 `Unified Memory cannot be traced` 诊断。项目使用显式设备分配与复制，不以此报告验收 Unified Memory 跟踪。NCU 的硬件计数器与 NSys 模式分别说明，见 `ENG-022`。

未锁频，环境只在进程边界采集。单个选定 kernel 不能代表全部形状，设备间隙不能单独确定 host、驱动或调度根因。`ENG-048` 的微基准顺序差异和 `ENG-051` 的模型 A/A 噪声继续保留。

项目拥有模型执行、KV 和生命周期；矩阵 kernel 归 NVIDIA cuBLAS。本目录不验收 GPU Serving、自有 PagedAttention 或端到端 HTTP 性能。
