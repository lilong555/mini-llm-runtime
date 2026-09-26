# 自有 CUDA 模型性能对照

`mini-cuda-runtime-bench` 在同一可执行文件中选择自有 CPU8、CPU16 或 CUDA Runtime，每个进程只构造一个后端。固定协议为 `benchmarks/runtime-inputs/qwen3-cuda-v0.json`，模型为固定 Q8_0 checkpoint；CUDA 使用常驻 FP32 有效权重、FP16 连续 KV 和 cuBLAS pedantic FP32，CPU 使用 auto SIMD 与 FP16 paged KV。

当前提供模型层的采集、严格复核和配对统计工具。单进程检查、合成报告 fixture 和数值验收均不能替代完整 A/A 性能基线；microbenchmark、Profiler 和 GPU Serving 仍有各自的阶段门禁。

## 运行入口

需要自有 CUDA 构建、固定模型、Python 3.10+、PowerShell，以及已通过复核的完整数值归档：

```bash
bash scripts/dev.sh own-cuda build

bash scripts/dev.sh own-cuda runtime-benchmark \
  -PreflightOnly -OutputDirectory .run/cuda-benchmark-preflight

bash scripts/dev.sh own-cuda runtime-benchmark \
  -OutputDirectory benchmarks/results/cuda-model-baseline

pwsh -NoProfile -File scripts/Analyze-CudaRuntime.ps1 \
  -Directory benchmarks/results/cuda-model-baseline
```

输出目录必须为空。`PreflightOnly` 检查模型、冻结输入、源码继承、依赖、构建和可获取环境信息，只生成预检证据，不执行性能采样；正式采集使用另一个新目录。`BinaryDirectory`、`Model`、`NumericalDirectory` 可指定路径，但不能改变模型摘要、工作区、构建开关或冻结参数。

完整采集固定启动 70 个独立进程：30 个 A/A 进程，以及 CPU8/CUDA、CPU16/CUDA 各 20 个对照进程。每个进程包含 585 次 forward，其中包括不计入主指标的前缀重建。长上下文重建可能占据大部分采集墙钟时间，不能省略、共享或归入其他 trial。

单进程功能检查不产生配对统计结论：

```bash
mkdir -p .run
build/wsl-own-cuda/bin/mini-cuda-runtime-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --input benchmarks/runtime-inputs/qwen3-cuda-v0.json \
  --backend cuda --output .run/cuda-one-process.json

python3 scripts/analyze_cuda_benchmark.py \
  --report .run/cuda-one-process.json \
  --input benchmarks/runtime-inputs/qwen3-cuda-v0.json
```

`--backend` 也接受 `cpu8`、`cpu16`。报告必须为新文件，父目录须已存在；失败保留错误及已完成的 workload。正式进程由采集器传入 `--manifest/--order`，可执行文件核对自身、模型、输入和进程位置的 SHA-256/身份。

## Workload 与时钟

12 个用例包含 prefill 16/128、128-token chunk 的总 prefill 256/1536、固定 KV=16/256/1536 decode、独立序列 batch 2/4、16 prefill+2 decode 的 mixed，以及 prompt 16/128 后的 32-token 自然生成。

每项先运行 2 次 warmup，再运行 3 次 measured repetition。每次都清空 sequence 0–3，独立重建所需 prefix；CPU 不调用 `share_prefix`。固定 token 按 recipe 循环，独立序列可具有相同内容，但不共享物理 KV。mixed 使用新 sequence 0 和已独立预填充的 sequence 1/2，M=18、logits rows=3。自然生成忽略 EOS，第一项输出来自 prefill，随后 31 次 decode 使用实际前一项输出，KV 分别增长至 47/159。

主指标 `host_forward_to_token_ns` 从 host descriptors 就绪开始，到 token 可用为止。CPU 包含 logits finite 检查和 deterministic argmax；CUDA 包含必要 metadata H2D、token/status D2H、argmax 和 checked stream completion。报告编码、输入摘要和资源查询位于计时区间外。多次 forward 的 workload 以各次主计时之和作为指标，不包含 clear、prefix setup 或 JSON 开销，不是整个进程墙钟时间。

每次 forward 保存输入摘要、输入/输出行数、前后 context 和 token；每轮保存 clear 后、setup 后、测量后的状态。初始化单列；CUDA 的 weight decode/upload 包含在 storage initialization 内，不能重复相加。正式模型基准关闭 events 和 profiler，`device_elapsed_ms=null`。

## 配对统计

- A/A 与异构对照分开采集，每种配置有 5 个独立 trial。A/A 每个 trial 为两个独立进程，A/B 顺序逐轮反转。
- CPU8/CUDA 使用 A/B/B/A，下一 trial 使用 B/A/A/B；CPU16 为独立对照组。每个进程先取 3 次 measured repetition 的中位数，再对同一 trial、同一后端的两个进程中位数取中位数。
- 差异为 `100*(B/A-1)`，B 为 CUDA；负值表示 CUDA 延迟更低。统计单位始终是 5 个独立 trial，不把进程内重复扩充为更多 trial。
- 各 backend/case 的 A/A 噪声为 `max(5, abs(median(d_AA)) + 2*MAD(d_AA))`。异构比较取 CPU、CUDA 两个噪声带的较大值，超过 10% 时为 `measurement_inconclusive`。
- 95% 区间使用全部 `5^5=3125` 个成对 trial 重采样的 percentile bootstrap。仅有 5 个 trial，区间稳定性有限。差异落入噪声带或区间跨零时为 `inconclusive`；有效负结果保留为 `slower`。
- 跨后端 token 不同会保留逐 trial 序列，并标为 `correctness_followup_required`，不发布加速结论。相同后端重复输出不一致直接拒绝验收。

温度、时钟、功耗、显存与背景负载在进程边界采集，可用性及原始输出一并保留。未锁频，未固定 affinity，也不是连续硬件遥测。

## 证据与复核

归档包含 manifest、源码状态/ZIP、冻结输入、完整数值摘要及对应 Runtime 源码状态、70 份进程报告、stdout/stderr、原始命令和环境。分析器核对完整顺序、全部 repetition、输入重放、生成历史、KV 状态、权重形状/别名、内存计划与传输/分配计数，再发布：

`availability.json`、`weight-plan.json`、`memory-plan.json`、`copy-allocation-summary.json`、`summary.json`、`analysis.md`。

发布前完成全部验证与序列化；异常时保留原文件，发布故障回滚。不要并发写入同一目录。CPU resident KV 是实际已分配页的高水位，clear 后可仍然非零；GPU 的连续预留不是 CPU 活跃页数。项目 allocation/copy 计数不包含 NVIDIA 库内部资源，不等于硬件 DRAM 流量。

整个目录可迁移后复核，无需原绝对路径下的模型或二进制：

```bash
python3 /path/to/bundle/verify.py --directory /path/to/bundle
```

模型权重、二进制、依赖 checkout 和完整数值归档不重复装入该模型性能包。数值摘要只在 `include/minillm/`、`src/minillm/` 文件集合和摘要一致，且重新构建的 `minillm-cuda-model-tests` 与完整数值验收的二进制 SHA-256 一致时继承；编译身份不同须在当前构建下重新进行数值验收。完整数值原始证据见 [CUDA 全量数值验收](../benchmarks/results/validation/cuda-full/README.md)。归档复核不是原二进制重跑，也不是可信执行证明。Step 8 的 microbenchmark 与 Step 9 的完整 Profiler/交付门禁不由本报告替代。
