# 在线观测正确性验证

2026-09-23，WSL2 原生 CPU/CUDA 参照产品与 ASan/UBSan 核心构建。MiniLLM 始终在 CPU 上执行；CUDA 仅用于 llama.cpp 数值参照及 HTTP 后端。

[evidence.json](evidence.json) 固定采集源码快照、构建参数、各验证二进制、两份实际模型文件摘要和本目录原始报告 SHA-256；对应源码 manifest 位于在线基准的 `mixed/trial-0-off/`。完整归档检查为 14 次套件、463 次用例执行，没有数量缺失。

| 范围 | 结果 | 原始证据 |
| --- | --- | --- |
| CPU `RelWithDebInfo` CTest | 5/5 套件 | [cpu-ctest.xml](cpu-ctest.xml) |
| CUDA 参照构建 CTest | 5/5 套件 | [cuda-ctest.xml](cuda-ctest.xml) |
| ASan/UBSan 核心 CTest | 4/4 套件 | [asan-ctest.xml](asan-ctest.xml) |
| CPU 数值参照，1 线程 | 13/13 | [cpu-model-1.json](cpu-model-1.json) |
| CPU 数值参照，8 线程 | 13/13 | [cpu-model-8.json](cpu-model-8.json) |
| CUDA 数值参照，8 线程 | 13/13 | [cuda-model-8.json](cuda-model-8.json) |
| MiniLLM HTTP，关闭观测 | 8/8 | [http-cpu-off.json](http-cpu-off.json) |
| MiniLLM HTTP，阶段观测 | 8/8 | [http-cpu-stages.json](http-cpu-stages.json) |
| llama.cpp CUDA HTTP，阶段模式 | 8/8 | [http-cuda-stages.json](http-cuda-stages.json) |
| 在线归档目录迁移 | 两份报告重新验收通过，除目录字段外汇总相同 | [relocation.json](relocation.json) |

CPU/CUDA 各包含 35 项核心、27 项在线观测验收、56 项 Serving 基准与归档、34 项 Runtime 基准、4 项 GGUF 用例。ASan/UBSan 不依赖 llama，Runtime 套件为 33 项，无 GGUF。合计 463 次用例执行，不是 463 个不同用例。

## 覆盖

- 受控注入真实混合批，覆盖缓存命中、序列复用、输出顺序和资源回收。三份模型报告均验证 32 个生成 token、阶段时间守恒及停服后的零活跃 KV 页。
- 关闭观测不分配 capture 存储；`batches / stages` 缓冲耗尽后仍保持生成结果，并显式统计丢弃。运行中读取 capture 被拒绝，后端异常保留未完成 batch。
- 验收反例覆盖缺失 footer、丢失或重复 batch、非法类型、计时不守恒、上下文错误、错误 SSE token、缺失 Runtime 阶段及错误矩阵形状。
- 到达时间小数缩放与逆序策略采集有独立 fixture；原有 Runtime profile 开关的完整 logits 一致性、KV/COW、异常恢复和生成对照仍通过。
- HTTP 覆盖流式/非流式输出等价、严格输入检查、并发、取消、断连与超时；流式 token 的 batch ID、请求顺序和输出序号合法。

[CPU HTTP 原始观测](http-cpu-stages.jsonl) 的 28 个 batch 均具有 Runtime 阶段，停服后活跃 KV 页为 0、保留载荷为 14680064 bytes。[CUDA HTTP 原始观测](http-cuda-stages.jsonl) 的 29 个 batch 不伪造 Runtime 阶段和物理页信息，这两项均为 `null`。两份采集均无丢弃，Engine 错误为空。

## 边界

ASan/UBSan 仅覆盖独立核心，不包含真实模型、GGUF 或 HTTP 的 sanitizer 验收。Windows、远程 CI、完整 F16/F32 被测模型和自研 CUDA 均不在本批范围。正确性验证独立于性能采集，不能将这些执行耗时作为性能基线。
