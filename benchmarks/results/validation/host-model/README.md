# Host Model 验收

范围为 `CUDA-VS-001 / Step 3`。当前系统提供独立的 `Qwen3Model` 与 `Tokenizer`，CPU Runtime 的计算区段、SIMD、线程池、物理 KV 和 Serving 保持已有实现；`source-comparison.json` 固定计算区段的字节摘要。环境为 WSL2、Ryzen 7 7745HX、GCC 11.4、RelWithDebInfo，依赖和模型身份见 `checks.json`、模型 manifest 与 `environment.json`。

## 正确性与构建

| 配置 | CTest 套件 | 用例执行 |
| --- | --- | --- |
| CPU | 7/7 | 178 |
| 自有 CUDA ON、上游 CUDA OFF | 8/8 | 189 |
| 独立核心，无 llama | 5/5 | 167 |
| 上游 CUDA ON、自有 CUDA OFF | 7/7 | 178 |

共 27 次套件、712 次用例执行，含重复执行。独立 host-model 实模型检查为 8/8；CPU 与上游 CUDA 数值参照的模型检查各 13/13，MiniLLM CPU 与 llama.cpp CUDA HTTP 各 8/8。`host-real.txt` 包含全部 28 层绑定、7 组固定分词输入和 151936 个 token 的 piece/EOG 检查。现有词表警告仍存在，参见 `ENG-011`，本验证没有修复源模型元数据。

独立 host-model 测试二进制没有 `minillm::Runtime`、`ParallelExecutor` 或 `PagedKV` 定义。CPU 产品没有 CUDA 动态依赖。编译参数、各构建二进制摘要与动态依赖见 `build-boundaries.json`。

## 固定输入对照

- 输入为 `input.json`：prefill-16/128、decode-16/256/1536、mixed-16-2；8 和 16 线程分别比较。
- 20 份 A/A、20 份前后配对、4 份独立 profile 报告，共 44 个进程、792 次测量。每进程每案例 warmup=2、repeats=3；setup 和 KV reset 不计入 forward 时间。
- 全部测量的 logits SHA-256、greedy token、输入、KV 页数及常驻字节一致；两版的 36 对 profile 样本具有相同阶段、形状、token 数、上下文与资源契约，计时和 worker 参与数允许变化。
- 每种线程数有 5 个独立配对 trial，执行顺序交替；先取进程内中位数，再分析配对相对差异。所有样本、异常波动及置信区间均保留。

下表的差异为 `(新二进制 / 原二进制 - 1) * 100%`，正数表示更慢。噪声阈值为预先固定的 `max(5%, abs(median(d_AA)) + 2 * MAD(d_AA))`，上限为 10%。各案例的配对中位退化均未超过自己的阈值。95% 区间为 10000 次配对 trial bootstrap，seed=20260926；仅 5 对 trial，区间稳定性有限。噪声带内或区间跨零的差异标记为 `inconclusive`，与退化门禁是否通过分开记录；不据此宣称加速。

| 线程 | 案例 | A/A 噪声阈值 | 前后配对中位差异 | 95% 区间 |
| --- | --- | --- | --- | --- |
| 8 | prefill-16 | 5.000% | +1.018% | [-2.681%, +11.296%] |
| 8 | prefill-128 | 5.000% | +1.606% | [-0.350%, +5.682%] |
| 8 | decode-16 | 9.901% | +0.739% | [-18.982%, +2.685%] |
| 8 | decode-256 | 6.325% | -0.029% | [-2.698%, +2.184%] |
| 8 | decode-1536 | 5.781% | -2.238% | [-21.701%, +3.150%] |
| 8 | mixed-16-2 | 9.996% | +0.301% | [+0.025%, +3.713%] |
| 16 | prefill-16 | 5.000% | -6.054% | [-12.336%, -4.193%] |
| 16 | prefill-128 | 5.000% | -1.840% | [-6.274%, +7.549%] |
| 16 | decode-16 | 5.000% | -0.094% | [-4.430%, +7.315%] |
| 16 | decode-256 | 5.000% | +1.640% | [-30.448%, +4.086%] |
| 16 | decode-1536 | 8.683% | +1.046% | [-2.994%, +15.171%] |
| 16 | mixed-16-2 | 6.693% | +5.799% | [-5.452%, +10.304%] |

## 证据身份与复核

`protocol.json` 在采集前固定输入、配置、重复数和判定方法。`*-collection.json` 记录实际直接 CLI 调用、顺序、时间、退出码、二进制与报告 SHA-256。`comparison.json` 保留各 trial 中位数、原始相对差异、噪声带和统计结论。

原二进制对应 `ad8b450` 的源码范围，源码位于 `before/`；新二进制来自该提交上的工作区，源码位于 `after/`，不能标成 `ad8b450` 的 clean build。两个 ZIP 均随归档提供并与各自的逐文件清单一致。首轮 A/A 曾被用户中断，已完成的一份报告保留，未完成进程没有测量报告；继续采集使用相同二进制、输入与协议，记录见 `source-comparison.json`。

```bash
bash scripts/dev.sh build
bash scripts/dev.sh test
build/wsl-cpu/bin/minillm-host-model-tests \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --contract tests/data/qwen3_validation_cases.json
bash scripts/dev.sh validate
bash scripts/dev.sh check-http 8057
pwsh -NoProfile -File scripts/Test-CtestEvidence.ps1 \
  -Directory benchmarks/results/validation/host-model
```

归档不含模型、二进制、依赖 checkout 或工具链；模型重跑需按清单取得它们。这里是 CPU 提取的行为与性能门禁，CUDA 完整模型、GPU Serving、GPU PagedAttention、Windows 与远程 CI 不在本轮验收范围内。
