# 自有 CUDA 真实形状微基准

`mini-cuda-kernel-bench` 使用固定 Qwen3-0.6B Q8_0 的实际 metadata 与常驻 FP32 有效权重，直接调用自有 CUDA 算子和 cuBLAS 矩阵接口。进程只构造 `Qwen3Model` 与 `CudaStorage`，不构造完整 CPU Runtime 或上游模型参照。模型级 CPU8/16 对照使用独立的 [模型基准](CUDA_BENCHMARKS.md)。

## 运行

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda micro-benchmark \
  -PreflightOnly -OutputDirectory .run/cuda-micro-preflight
bash scripts/dev.sh own-cuda micro-benchmark \
  -OutputDirectory benchmarks/results/cuda-micro-baseline

pwsh -NoProfile -File scripts/Analyze-CudaMicro.ps1 \
  -Directory benchmarks/results/cuda-micro-baseline
```

采集目录必须为空。`PreflightOnly` 不采样；正式采集使用另一个新目录。模型、设备容量、输入、算术和重复次数由 `benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json` 固定，文件 SHA-256 为 `6aa2bc8af5de001ab3a8baedd305d0c77822a48b5baddc8f5708e79f496e706c`。不能通过删减慢用例、降低容量或减少重复次数形成正式结果。

单进程功能检查：

```bash
build/wsl-own-cuda/bin/mini-cuda-kernel-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --input benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json \
  --output .run/cuda-micro-single.json
python3 scripts/analyze_cuda_micro.py --report .run/cuda-micro-single.json \
  --input benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json
```

该入口不形成独立 trial 统计。报告必须为新文件；失败保存已完成用例与错误，不能将部分报告验收为通过。

## 形状与输入

每个独立进程包含 375 项，奇数 trial 反转完整用例顺序：

| 类别 | 数量 | 覆盖 |
| --- | ---: | --- |
| 矩阵 | 72 | 第 0 层 Q/K/V/output、gate/up/down，以及 LM head；M=1/2/4/8/16/18/32/64/128 |
| RMSNorm | 27 | hidden、query、key 的真实宽度与分组 |
| RoPE | 108 | query/key，位置 0/1/15/16/17/1535 |
| softmax | 84 | 同一 causal attention softmax 内核的独立入口，FP64 分母 |
| attention | 84 | GQA、独立序列、因果 prefill、16 prefill+2 decode |

N/K 从 metadata 读取，并与实际权重清单核对。attention 的 context 为 1/17/33/256/257/1536/1537/2048，decode 包含 1/2/4 个序列，prefill 只使用不超过 context 的 M。mixed 的两个独立 decode prefix 为 256 或 1536，计入当前 query 后的长度为 257 或 1537。

密集输入按冻结 modular recipe 生成，每个 canonical 用例使用独立 seed。KV 的 layer 0 一次性初始化全部 4 个 slot、2048 个位置，使用与产品相同的 FP16 store；它不是由真实 prompt 生成的模型 KV。softmax 的未使用 scores 填 NaN，概率 scratch 的远端尾部保留 sentinel。

## 时钟与正确性

每项 2 次 warmup、3 次 measured sample；每个 sample 连续调用同一 API 32 次。统计先取每个进程的三次测量中位数，再以五个独立 trial 为单位汇总；不将 API 调用或进程内重复当成独立实验。

CUDA events 包围 API 调用序列，包含 host 提交空隙。`host_enqueue_to_completion_ns` 包含 event 提交及 checked stream completion。两种时钟均不包含初始化、输入重置、状态/输出下载或 FP64 验证。表中的每次调用时间是 32 次调用的区间均摊值，不是单个 kernel 的纯执行时间。

RoPE 是原地算子，每个 sample 前恢复原输入，sample 内连续旋转 32 次；CPU FP64 参照使用相同 FP32 系数表执行同样的递推。它不能直接当成一次模型 RoPE 的孤立延迟。

全部输出检查 finite。矩阵检查首/中/末行及八个均匀列；attention 检查首/中/末行、head 0/7/8/15、维度 0/1/63/127。其他算子全元素对照，固定 `atol=2e-4, rtol=2e-4`。抽样值、全量对照摘要、误差指标与输出摘要均保留；全部 sample 和 trial 的输出应一致。离线分析检查记录与协议，不在无模型的归档中重新计算全部数学结果。

## 归档与边界

归档含源码状态/ZIP、冻结输入、实际权重与内存计划、五份完整进程报告、命令、stdout/stderr 和进程边界环境。源码、模型、二进制与依赖在每个进程前后核对。显式传输和项目设备分配分别记录准备、测量、验证与析构，不覆盖 NVIDIA 库内部资源。

模型权重、二进制和依赖 checkout 不装入归档。目录迁移后可直接复核，Python 3.10+ 即可：

```bash
python3 /path/to/bundle/verify.py --directory /path/to/bundle
```

五个 trial 的 95% 区间使用全部 `5^5=3125` 次 bootstrap 重采样，有限样本的区间稳定性有限。未锁频、未固定 affinity，环境查询不是连续遥测。矩阵 FLOP 数只是逻辑运算量；本工具不提供硬件 DRAM 带宽、CPU/GPU 模型加速、HTTP TTFT、GPU Serving 或自有 PagedAttention 结论。
