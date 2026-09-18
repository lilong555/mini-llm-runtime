# MiniLLM / LLMServe 参考资料

阅读顺序：C++ 模型执行与 GGUF → CPU kernel 与数值契约 → KV 页表 → 迭代级服务调度 → 可复现实验。

论文中的性能数字和开源项目的 benchmark 不属于本项目的实测结果。

## 1. 固定 C++ 参考

llama.cpp 提交：`911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`。

| 入口 | 需要理解的问题 | 本项目对应实现 |
| --- | --- | --- |
| [llama.h](https://github.com/ggml-org/llama.cpp/blob/911f6cdc8ab8a530b2bee09ee61471a6f3178eeb/include/llama.h) | batch token、position、sequence、logits 与内存别名的执行契约 | `ModelRunner`、`LlamaRunner` |
| [Qwen3 graph](https://github.com/ggml-org/llama.cpp/blob/911f6cdc8ab8a530b2bee09ee61471a6f3178eeb/src/models/qwen3.cpp) | Q/K norm、显式 head_dim、GQA、RoPE、SwiGLU、输出权重共享 | `src/minillm/runtime.cpp` |
| [GGUF API](https://github.com/ggml-org/llama.cpp/blob/911f6cdc8ab8a530b2bee09ee61471a6f3178eeb/ggml/include/gguf.h) | 元数据类型、tensor offset、alignment、合法文件边界 | `GgufModel` 与 mmap 适配器 |
| [Quantization definitions](https://github.com/ggml-org/llama.cpp/blob/911f6cdc8ab8a530b2bee09ee61471a6f3178eeb/ggml/src/ggml-quants.c) | Q8_0 block layout、scale、解量化、激活量化与数值差异 | `kernels.cpp`、`kernels_avx2.cpp`、F32 权重参照 |
| [Batched example](https://github.com/ggml-org/llama.cpp/blob/911f6cdc8ab8a530b2bee09ee61471a6f3178eeb/examples/batched/batched.cpp) | 多序列 batch、logits index、共享 prompt 的含义 | Serving 的 batch 构造与参照后端 |

MiniLLM 复用成熟 GGUF parser 和 tokenizer，不自行实现不完整的 BPE 或字符串版二进制解析器。自有前向、SIMD 和物理 CPU KV 路径不依赖 llama.cpp 的模型图执行。

### 数值对照的关键约束

Q8_0 权重相同不代表所有矩阵路径使用相同算术：某些后端还量化激活，矩阵形状也可能改变所选 kernel。模型级严格对照应固定有效权重与数值路径。

本项目的 F32 参照是对固定 Q8_0 GGUF 解量化，保留其量化后的权重值，而不是另行下载一个不同精度、不同 revision 的模型。直接 Q8 后端差异保留为诊断，不通过放宽阈值混入正确性结论。

## 2. 基础模型论文

| 论文 | 对应问题 |
| --- | --- |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | RMSNorm 的归一化轴、epsilon 和权重乘法 |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | RoPE 位置、频率与旋转布局；分块不能重置 position |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | SwiGLU 的 gate/up/down 数据流 |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | query heads 与 KV heads 不同，分组映射和 KV 容量不能套用 MHA 公式 |

## 3. Serving 论文

| 论文与一手来源 | 阅读重点 | 项目验收问题 |
| --- | --- | --- |
| [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | Iteration-level scheduling、selective batching | 请求能否逐轮入批和移除，而不是等待整个 batch 结束？ |
| [Efficient Memory Management for Large Language Model Serving with PagedAttention, SOSP 2023](https://arxiv.org/abs/2309.06180) | 逻辑块到物理块映射、共享、COW、内存浪费 | Attention 是否真正读页表？是否处理共享尾块和回收？ |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | RadixAttention、LRU、引用保护、缓存感知调度 | 命中收益如何与容量、公平性及租户边界协调？ |
| [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310) | Chunked prefill、stall-free batching、token budget | Prefill/decode 是否在同一次前向？chunk 对 TTFT/TPOT 有何代价？ |
| [Fast Distributed Inference Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | 未知输出长度、抢占、MLFQ、迁移成本 | 为什么不能将真实调度直接当作已知长度的背包问题？ |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving, OSDI 2024](https://arxiv.org/abs/2401.09670) | TTFT/TPOT 双约束、goodput、阶段干扰 | 满足 SLO 的有效吞吐与 token/s 是否得出不同结论？ |
| [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005) | 动态 KV 布局、attention kernel 接口、负载均衡 | 自有 CPU 页表未来怎样接入成熟 GPU attention，而不混淆能力归属？ |

### 不应直接类比的地方

- Continuous batching 不等于多 HTTP 线程，也不等于并发调用多次完整 `generate()`。
- `free list + page table` 不等于 PagedAttention；计算路径必须正确使用页表。MiniLLM 的 CPU 路径已做到，GPU 自有路径未实现。
- Prefix Trie 不等于完整 RadixAttention；本项目没有实现其全部压缩树、调度和运行时功能。
- Mixed batching 在 CPU 上可能受计算成本与矩阵复用制约，不保证比 prefill-first 更快。
- 单机共享内存不构成 DistServe 式的多 GPU PD 分离。
- SIMD 内核加速需要同时报告精度、微基准条件和模型/服务端实际影响。

## 4. 其他开源入口

- [vLLM](https://github.com/vllm-project/vllm)：资源预留、computed-token 进度、抢占与 prefix cache 的成熟工程参考，不作为本项目构建依赖。
- [SGLang](https://github.com/sgl-project/sglang)：Radix cache 与缓存感知调度参考。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：GPU paged/ragged attention 的后续后端候选。
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：可作为独立的 Python 对照，不参与本仓库的 C++ 构建与验收。

## 5. 固定模型

官方 GGUF：[Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a)。

元数据实测：28 层、hidden size 1024、16 query heads、8 KV heads、head_dim 128、FFN 3072、训练上下文 40960。**不能用 1024 / 16 替换显式 head_dim 128。**

页容量公式与默认限制见 README；完整下载和参照转换信息见两个 model manifest。
