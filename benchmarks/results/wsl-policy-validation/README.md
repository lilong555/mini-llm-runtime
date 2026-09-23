# WSL 策略基线

2026-09-22，MiniLLM CPU、Qwen3-0.6B Q8_0、F32 激活、F16 KV、8 线程、GCC 11.4.0 `RelWithDebInfo`。输入为 `cpu-mixed-s0.jsonl` 的原始字节：24 个请求、128/16-token prompt、16 个输出 token、seed 0。每种策略三轮，独立服务、相同预热、交替顺序。

全部 144 个请求成功，输出 token 与声明的参照一致。严格验收结果见 [validation-summary.json](validation-summary.json)，完整身份见 [manifest.json](manifest.json)。工作区有未提交内容，其构建输入保存在 `source-snapshot.zip`，逐文件摘要为 `source-state.json`；没有用单独的 `git_dirty=true` 替代源码证据。

| 三轮统计的中位数 | 混合调度 | 预填充优先 |
| --- | ---: | ---: |
| 输出吞吐，token/s | 18.98 | 19.30 |
| P95 TTFT，ms | 12367.61 | 11667.64 |
| P95 请求平均 TPOT，ms | 381.77 | 321.92 |
| P99 单次 ITL，ms | 1203.67 | 2436.45 |
| SLO goodput，请求/s | 0 | 0 |

混合调度没有体现整体吞吐或请求平均延迟优势，但单次 token 停顿的 P99 更低。请求平均 TPOT 与单次 ITL 不能互相替代。六轮吞吐分别为 mixed 的 19.80、18.48、18.98 token/s，以及 prefill-first 的 19.30、19.59、19.04 token/s；波动和不利轮次均保留。

该结果用于可信基线，不是优化前后对照，也不能与 Windows 历史结果直接计算加速比。没有采集 CPU 频率、温度或阶段 profiler；`ENG-008` 的归因工作仍未完成。完整轮次和原始输出见本目录六份策略 JSON 与 `collection.txt`。

复现命令：

```bash
bash scripts/dev.sh benchmark \
  -Trace benchmarks/traces/cpu-mixed-s0.jsonl \
  -Trials 3 -TraceSeed 0 -Port 8031 \
  -OutputDirectory benchmarks/results/policy-run
```

输出目录必须为空。当前归档可直接用 `pwsh -NoProfile -File scripts/Analyze-Benchmarks.ps1 -Directory benchmarks/results/wsl-policy-validation` 重新验收；目录整体复制后的离线验证证据见 [benchmark-relocation.json](../validation/wsl-deterministic/benchmark-relocation.json)。
