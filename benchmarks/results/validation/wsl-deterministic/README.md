# WSL 正确性验证

2026-09-22，Qwen3-0.6B Q8_0 与固定的解量化 F32 参照。模型大小、SHA-256、当前源码快照、执行二进制及每份原始产物摘要见 [evidence.json](evidence.json)。模型、HTTP、sanitizer 和正式性能回放分开验收。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| CPU 产品 CTest | 3/3 套件通过：30 个核心、51 个基准与归档、4 个 GGUF 用例 | [ctest-cpu.xml](ctest-cpu.xml) |
| CUDA 产品 CTest | 相同的 3/3 套件通过 | [ctest-cuda.xml](ctest-cuda.xml) |
| ASan/UBSan 核心构建 | 2/2 套件通过：核心和基准与归档检查 | [ctest-asan.xml](ctest-asan.xml) |
| CPU 模型，1 线程 | 10/10 通过 | [model-cpu-1t-1.json](model-cpu-1t-1.json) |
| CPU 模型，2 线程 | 10/10 通过 | [model-cpu-2t-2.json](model-cpu-2t-2.json) |
| CPU 模型，8 线程 | 连续三次各 10/10 通过 | `model-cpu-8t-3.json`、`model-cpu-8t-4.json`、`model-cpu-8t-5.json` |
| CUDA 数值参照，8 线程 | 10/10 通过；MiniLLM 仍在 CPU 执行 | [model-cuda-8t.json](model-cuda-8t.json) |
| CPU / CUDA HTTP | 各 8/8 通过，测试服务均退出 | [http-cpu.json](http-cpu.json)、[http-cuda.json](http-cuda.json) |
| 基准目录迁移 | 复制归档后仍通过完整离线验收 | [benchmark-relocation.json](benchmark-relocation.json) |

六次模型验证均记录 `mixed_batches=1`、`max_batch_sequences=3`；三组贪心生成及缓存重放与参照一致。测试侧屏障在第一份真实 prefill sample 后注入新请求，检查生产调度器的真实混合执行，不用等待时长猜测批次组成。数值阈值与先前固定的标准相同。

每份模型和 HTTP JSON 都有同名 `.txt` 原始输出。模型词表的 control-token 警告仍保留，见 `ENG-011`。Sanitizer 只覆盖不依赖 llama.cpp 的核心构建，不代表 GGUF、HTTP 或完整模型已经通过 sanitizer。

原有混合批失败报告保留在 `wsl-native` 和 `wsl-environment` 归档中。这里没有扩大为完整 F16/F32 被测模型、长上下文、任意语言或 Windows/远程 CI 的验证结论。
