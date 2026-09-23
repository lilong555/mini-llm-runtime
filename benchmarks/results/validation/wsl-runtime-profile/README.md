# Runtime Profiler 正确性验证

2026-09-22 的 WSL2 原生验证。`evidence.json` 固定当前源码快照、构建配置、CPU/CUDA/ASan 二进制、两份模型来源与摘要，以及本目录各份原始报告的 SHA-256。源码身份关联到 [Runtime 基准 manifest](../../wsl-runtime-profile/scaling/manifest.json)，不依赖 Git 提交号代表未提交的工作区内容。

## 验证结果

| 范围 | 结果 | 原始证据 |
| --- | --- | --- |
| CPU `RelWithDebInfo` CTest | 4/4 套件，121 次用例执行 | [cpu-ctest.xml](cpu-ctest.xml) |
| CUDA 参照构建 CTest | 4/4 套件，121 次用例执行 | [cuda-ctest.xml](cuda-ctest.xml) |
| ASan/UBSan 核心构建 | 3/3 套件，116 次用例执行 | [asan-ctest.xml](asan-ctest.xml) |
| CPU 数值参照，1 线程 | 12/12 | [cpu-model-1.json](cpu-model-1.json) |
| CPU 数值参照，8 线程 | 12/12 | [cpu-model-8.json](cpu-model-8.json) |
| llama.cpp CUDA 数值参照，8 线程 | 12/12 | [cuda-model-8.json](cuda-model-8.json) |
| MiniLLM HTTP | 8/8 | [cpu-http.json](cpu-http.json) |
| llama.cpp CUDA HTTP | 8/8 | [cuda-http.json](cuda-http.json) |
| Runtime 归档迁移 | 两组均通过，验收结果字节一致 | [runtime-relocation.json](runtime-relocation.json) |

CPU/CUDA 构建各执行 32 项核心用例、51 项已有基准与归档用例、34 项 Runtime 基准用例、4 项 GGUF 用例。无 llama 的 ASan/UBSan 构建没有 CLI/GGUF 用例，Runtime 基准套件为 33 项。共 358 次执行，不是 358 个不同用例。

## 检查范围

- profiler 开关前后完整 logits 逐字节相等，三步贪心续写一致；覆盖无 logits 的前缀、多个输出、混合序列及部分尾页共享。
- stage 数量、矩阵形状、parallel count/grain、batch ID、上下文统计及时间等式合法，阶段存储可复用。
- 空输入、非法位置、KV 容量不足后均可恢复；预分配异常不会保留旧的成功状态。`ENG-027` 的确定性回归使用超过 `vector::max_size()` 的申请。
- 固定输入配方、模型/二进制/源码身份、缺失或重复报告、错误类型、错误输出、缺失阶段、错误矩阵/线程统计、KV 驻留量和旧汇总残留均有验收反例。
- 真实模型保持与数值参照的既有容差及生成对照；三份报告均有 `mixed_batches=1`、`max_batch_sequences=3`，KV 最终回收。
- HTTP 覆盖流式/非流式等价、输入检查、八并发、取消、断连回收和超时。临时服务已经退出，端口 8071、8072 无监听。

性能采集与这些检查分开运行；性能进程没有常驻数值参照模型。[阶段基线](../../wsl-runtime-profile/README.md) 保留 36 份报告、558 次测量及退化样本。

## 适用边界

ASan/UBSan 仅覆盖不依赖 llama 的核心，不包含真实模型、GGUF 或 HTTP 的 sanitizer 验收。完整 F16/F32 被测模型、其他模型/平台、在线 profiling 扰动、Windows 和远程 CI 均不在本批验证范围内。MiniLLM 仍在 CPU 上运行，CUDA 只用于上游参照及其 HTTP 后端。

全仓 CTest 归档检查仍明确列出三次历史套件没有提供用例数量；旧截断 XML 仍作为诊断保留，不计为完整通过证据。本目录的 11 次套件执行均有完整计数。
