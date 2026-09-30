# 自有 CUDA Serving 基线

规范为 `CUDA-SERVE-001`，冻结输入和实验预算见 [protocol.json](protocol.json)。
数据路径为原 HTTP/SSE → Engine → MiniCudaRunner → CudaRuntime；
上游 `GGML_CUDA=OFF`，source Q8_0、device weights F32、activation F32、KV F16，
单 stream、同步 execute、无 prefix cache。

[分析与限制](analysis.md) · [逐轮摘要](summary.json) · [验证摘要](validation.json) ·
[完整证据包索引](evidence.json)

12 个正式进程的 288 请求全部成功，9216 个输出 token 跨轮次一致；
SLO 未达标请求与突发负载的吞吐不确定项保留。一次 NSys 包含 288 次完整 forward，
不扩展 M1 或自动启动 M3-2。

## 采集

从仓库根目录运行，模型与依赖按仓库 metadata 准备，输出目录必须为空。
每条 trace 有 24 个请求，各输出 32 token，`ignore_eos=true`。
正式采集共 12 个服务进程；两个策略按 trial 反转顺序，telemetry 关闭，
GPU 每 1000 ms 采样一次。冻结输入已有 token ID，不需要重新生成。

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda benchmark \
  -Trace benchmarks/traces/cuda-serving-mixed-s20260926.jsonl \
  -Trials 3 -TraceSeed 20260926 -PolicyOrderOffset 0 -ArrivalScale 1 \
  -GpuSamplePeriodMs 1000 -Port 8131 \
  -OutputDirectory .run/cuda-serving-001/mixed-length

bash scripts/dev.sh own-cuda benchmark \
  -Trace benchmarks/traces/cuda-serving-burst-s20260927.jsonl \
  -Trials 3 -TraceSeed 20260927 -PolicyOrderOffset 1 -ArrivalScale 1 \
  -GpuSamplePeriodMs 1000 -Port 8131 \
  -OutputDirectory .run/cuda-serving-001/burst-reuse

pwsh -NoProfile -File benchmarks/results/cuda-serving-001/capture.ps1 \
  -BaselineDirectory .run/cuda-serving-001/mixed-length \
  -OutputDirectory .run/cuda-serving-001/profiler
```

`capture.ps1` 只复现本实验的一次 `mixed-length/mixed` NSys 采集，复用正式基线的
源代码身份、二进制、模型、trace 和服务参数，仅开启 batches 观测。
它不执行 NCU，也不将 Profiler 下的延迟计入正式基线。

## 复核

```bash
pwsh -NoProfile -File scripts/Analyze-Benchmarks.ps1 \
  -Directory .run/cuda-serving-001/mixed-length
pwsh -NoProfile -File scripts/Analyze-Benchmarks.ps1 \
  -Directory .run/cuda-serving-001/burst-reuse
python3 scripts/analyze_cuda_profiler.py --serving \
  --directory .run/cuda-serving-001/profiler --write
```

客户端 TTFT 从发送时刻开始，mean TPOT 是每请求平均 token 间隔，ITL 和每请求最大停顿
分别报告。吞吐和 goodput 使用完整 trace 时长；固定 SLO 为 TTFT≤1000 ms 且
mean TPOT≤100 ms，失败保留在分母，不因结果调整阈值。

初始化与 weight decode/upload 单列；upload 已包含在 storage initialization 中，
不能重复相加。`nvidia-smi` 是整设备采样，窗口包含 ready 后预热、请求和停服，
不是本进程的独占利用率。host runner、CUDA API 与设备 timeline 重叠，不相加。
两条 trace 的策略差异分开解释；24 个请求的尾分位数不代表生产级 P99 保证。
