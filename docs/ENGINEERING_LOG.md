# 工程问题台账

本台账记录可复现的问题、当前处理状态以及解决方案的验证依据。正文统一使用简体中文，问题编号、代码标识符、命令、路径和原始诊断信息保留原文。

状态说明：`已解决`表示在所述范围内完成验证；`已缓解`表示已有可用的替代方案或缓解措施；`待解决`表示仍需进一步处理。本台账不是版本变更日志。

## 问题索引

| 编号 | 状态 | 领域 | 问题 |
| --- | --- | --- | --- |
| ENG-001 | 已解决 | 构建 | MSVC 本地化头文件输出导致 Ninja 规则异常 |
| ENG-002 | 已解决 | 构建 | C++20 设置被 C++17 依赖继承 |
| ENG-003 | 已解决 | GGUF | 将可选的 RoPE 元数据误作必填字段 |
| ENG-004 | 已解决 | 数值验证 | 量化参考后端额外量化激活，影响数值对照 |
| ENG-005 | 已缓解 | 模型下载 | Windows 原生 HTTPS 模型下载失败 |
| ENG-006 | 已解决 | 服务运维 | 时间戳类型不一致，阻止服务正常停止 |
| ENG-007 | 已解决 | 请求生命周期 | 前缀复制失败时可能重复发布终态事件 |
| ENG-008 | 待解决 | 性能 | CPU 混合调度在实测负载上表现更差 |
| ENG-009 | 已解决 | 验证流程 | 手动指定的参考模型文件名不存在 |
| ENG-010 | 待解决 | 第三方依赖 | 固定版本的上游依赖产生 MSVC 编译警告 |
| ENG-011 | 待解决 | 模型词表 | 分词器修正疑似控制 token 的词表条目 |
| ENG-012 | 已解决 | 验证证据 | CTest 截断已通过测试套件的输出 |
| ENG-013 | 已解决 | 版本管理 | 文本换行规范化改变基准回放文件的摘要 |
| ENG-014 | 已解决 | 验证证据 | PowerShell 校验脚本在 XML 访问错误后继续执行 |
| ENG-015 | 已缓解 | 持续集成 | Action 运行时弃用及托管操作系统标签漂移 |
| ENG-016 | 已解决 | SIMD | Attention 的 V 加权累加未使用 SIMD |
| ENG-017 | 待解决 | 模型验证 | 混合批断言受执行时序影响 |

## ENG-001：MSVC 本地化头文件输出

- 状态：已解决。
- 影响：Ninja 无法可靠解析本地化 MSVC 编译器的依赖输出，生成的构建规则可能异常。
- 复现条件：使用 Ninja 配置 MSVC 构建，且 `/showIncludes` 的输出前缀采用 Windows ANSI 代码页，而非 UTF-8。
- 原因：依赖前缀检测对编译器输出使用了错误的解码方式。
- 解决方法：由 `cmake/MSVCIncludes.cmake` 探测一次真实的头文件包含，使用 `ENCODING ANSI` 解码，并设置 `CMAKE_CL_SHOWINCLUDES_PREFIX`；不直接修改生成的 Ninja 文件。
- 验证依据：原生 MSVC 19.44 的 CPU Release 配置及全部 112 个构建步骤完成，两个已注册的 CTest 测试套件均通过；同一工具链下的 CUDA 构建也已完成。非 Windows 构建跳过该探测。

## ENG-002：第三方依赖的语言标准

- 状态：已解决。
- 影响：UTF-8 字符串字面量采用 C++20 的 `char8_t` 语义后，上游分词器编译失败。
- 复现条件：在设置了 `CMAKE_CXX_STANDARD=20` 的父项目中引入固定版本的 llama.cpp 子目录。
- 原因：依赖目标继承了父项目的语言标准设置。
- 解决方法：以 C++17 配置 llama.cpp，再为本项目目标恢复 C++20；上游 HTTP 库目标同样使用 C++17 编译。
- 验证依据：未修改上游源码，CPU 与 CUDA 产品构建均完成；`scripts/Fetch-Dependencies.ps1` 校验依赖工作区干净且提交固定。

## ENG-003：Qwen3 的可选 RoPE 元数据

- 状态：已解决。
- 影响：合法的官方 Qwen3-0.6B GGUF 在模型加载时被拒绝。
- 复现条件：加载 `models/manifest.json` 固定的模型文件，其中不包含 `qwen3.rope.dimension_count`。
- 原因：将可选元数据字段无条件作为必填项。
- 解决方法：使用显式的 attention head 维度，仅在可选字段存在时校验 RoPE 维度；仍然拒绝不兼容的形状和不支持的部分旋转维度。
- 验证依据：真实模型与 CPU、CUDA 参考后端的 logits 和贪心输出对照通过；分块大小为 1、7、16，以及共享 17 个 token 前缀后的分支计算，均保持 MiniLLM logits 一致。

## ENG-004：使用相同有效权重的数值参照

- 状态：已解决。
- 影响：即使权重和模型架构相同，直接与使用 Q8_0 权重的 llama.cpp 比较，logits 误差仍会超过严格阈值。
- 复现条件：运行 `llmserve-model-tests`，将同一 Q8_0 文件同时用于待测实现和参考模型。
- 原因：上游量化矩阵内核还会额外量化激活，而 MiniLLM 使用 Q8_0 权重与 F32 激活相乘，两条计算路径不能直接作为相同算术条件下的数值参照。
- 解决方法：使用固定版本的上游转换器，将同一份 Q8_0 权重解量化为 F32，校验派生文件的哈希，再作为有效权重一致的参照；不通过放宽验收阈值掩盖差异。
- 验证依据：与 CPU、CUDA 的 F32 参照比较时，各有 10 项检查通过。使用固定输入 token 的 CPU 参考 logits 对照中，RMSE 为 0.000963-0.002472。报告见 `benchmarks/results/validation/model-f32-cpu-reference.json` 和 `model-f32-cuda-reference.json`；Q8 诊断报告保留在 `benchmarks/results/diagnostics/`。
- 适用边界：这不代表任意提示词下的 Q8 后端等价，不会恢复量化前的原始权重，也不表示全部模型变体均已验证。

## ENG-005：Windows 原生 HTTPS 模型下载

- 状态：已缓解。
- 影响：在当前环境中，通过 Windows 原生 HTTPS 路径从 Hugging Face 下载模型失败。
- 原因：已观察到原生 TLS 或网络路径故障，但尚未定位机器环境层面的根因。
- 缓解方法：使用 `scripts/Download-Model.ps1 -UseWsl` 调用 WSL curl 下载，在将临时下载文件改为正式文件名之前，根据固定的 manifest 校验大小与 SHA-256；不关闭 TLS 验证。
- 验证依据：官方模型已通过该路径下载，并通过 manifest 校验。
- 下一步：单独复现并诊断原生 TLS 路径。WSL 只是可选的下载方式，不是运行时依赖。

## ENG-006：停止服务时的进程身份校验

- 状态：已解决。
- 影响：即使 PID 和可执行文件正确，停止脚本仍无法依据保存的进程身份停止运行中的服务。
- 复现条件：使用会将 ISO 时间戳反序列化为 `DateTime` 的 PowerShell 版本读取服务 JSON，再将其字符串形式与重新格式化的时间戳比较。
- 原因：JSON 时间戳转换改变了表示形式，但没有改变实际时刻，字符串比较因此误拒绝了正确的进程。
- 解决方法：创建停止标记前，比较 UTC ticks 与可执行文件路径，并校验标记位于项目的 `.run` 目录内；保留进程身份检查，避免误操作复用了相同 PID 的其他进程。
- 验证依据：`scripts/Stop-LLMServe.ps1` 已正常停止六次 CPU 策略实验中的服务；每次实验使用独立的服务进程。

## ENG-007：前缀复制失败时的终态处理

- 状态：已解决。
- 影响：前缀复用期间发生后端异常时，可能进入多条终态处理路径，造成重复完成通知和资源统计。
- 复现条件：可复用前缀进入缓存后，在 `ModelRunner::copy_sequence()` 中注入故障。
- 原因：请求准入失败处理与后端故障停止后的清理流程，都可能终结同一个请求句柄。
- 解决方法：保证终态事件的发布具有幂等性，并在清理时释放预留容量与缓存引用。
- 验证依据：回归用例 `engine_prefix_copy_failure_has_one_terminal_and_no_reservation_leak` 断言仅产生一个终态事件、仅统计一个失败请求、没有第二个事件，且停止后已使用的容量信用为零。该用例已在 28 项 CPU 单元测试中通过，见 `benchmarks/results/validation/windows-cpu-ctest.xml`。

## ENG-008：CPU 混合调度性能退化

- 状态：待解决。
- 影响：在已测 CPU 压力负载下，混合 prefill/decode 没有改善吞吐或延迟。
- 复现条件：使用 `scripts/Benchmark-Policies.ps1 -Backend mini` 回放 `benchmarks/traces/cpu-mixed-s0.jsonl`。每种策略运行三次，每次 24 个请求，到达率为 4 请求/秒，提示词长度为 128/16 个 token，输出长度为 16 个 token；每次均重启服务并执行相同预热。
- 验证证据：见 `benchmarks/results/mini-scheduling/summary.json`。全部 144 个请求成功，未观察到输出 token 序列不一致。

| 三次实验的中位数 | 混合调度 | 预填充优先 |
| --- | ---: | ---: |
| 输出吞吐，token/秒 | 18.40 | 18.59 |
| P95 首 token 延迟（TTFT），毫秒 | 13105.77 | 12290.59 |
| 请求平均每 token 延迟（TPOT）的 P95，毫秒 | 398.89 | 352.85 |
| 满足 SLO 的有效吞吐，请求/秒 | 0 | 0 |

- 原因：尚未确定。CPU prefill 成本、矩阵权重复用和 token 预算组成只是待验证的假设，不是性能剖析结论。
- 下一步：使用固定回放输入采集分阶段 CPU 性能数据，区分矩阵乘、attention、调度和等待开销；比较冷、热前缀负载及更低到达率。针对已定位的一个瓶颈优化，保留不利的基线结果，重新完成数值与端到端检查后再判断是否加速。
- 验收标准：在明确声明的负载上取得可复现收益，输出不变，且不引入不可接受的公平性或尾延迟退化。仅凭 SIMD 点积微基准不能关闭此问题。

## ENG-009：参考模型文件路径

- 状态：已解决。
- 影响：一次模型测试调用在执行任何数值检查之前就失败。
- 复现条件：指定 `Qwen3-0.6B-from-Q8_0-F32.gguf`，但该文件名与参考模型 manifest 中记录的名称不同。
- 原因：手工填写的路径与 `models/reference-manifest.json` 不一致。
- 解决方法：使用 `scripts/Validate-Model.ps1` 从 manifest 解析两个模型路径，并在执行前校验文件大小与 SHA-256；通过 `-Output` 指定报告位置，不手动拼写模型文件名。
- 验证依据：文件缺失时，测试以非零状态退出，诊断报告保留在 `benchmarks/results/diagnostics/reference-path-mismatch.json`。基于 manifest 的验证流程已校验两个模型文件，且 10 项模型检查全部通过，见 `benchmarks/results/validation/model-f32-cpu-only-reference.json`。

## ENG-010：上游编译器警告

- 状态：待解决。
- 影响：固定版本的 llama.cpp 在 CPU Release 构建中产生 MSVC C4297、C4244、C4834 警告，不能宣称依赖构建没有警告。
- 复现条件：使用 MSVC 19.44 执行 `scripts/Build-LLMServe.ps1`。
- 验证证据：警告来自上游的加载模式转换、计算图或模型的整数转换、量化参数转发及采样器代码；构建能够完成。本项目的编译单元在该次构建中没有产生警告。
- 原因：上游存在异常说明、窄化转换及忽略 `nodiscard` 返回值等问题；限定范围内的模型测试尚未出现可归因于这些警告的失败。
- 下一步：检查后续上游版本中的相关修复，重新固定依赖版本，并重新运行构建、模型、HTTP 与基准检查；不直接修改依赖工作区，也不通过全局屏蔽警告掩盖问题。

## ENG-011：模型词表警告

- 状态：待解决。
- 影响：固定版本的分词器提示：token 128247，即 `</s>`，看起来像控制 token，但模型词表未将其声明为控制类型。
- 复现条件：通过 CLI 或模型测试加载固定版本的官方 GGUF。
- 原因：上游启发式规则与模型文件中的 token 元数据不一致；尚未确定应该修改模型文件还是启发式规则。
- 当前行为：固定版本的分词器在加载时修正该 token 类型。MiniLLM 与数值参照共用此分词器；已有的数值、生成和 HTTP 检查通过，但不足以证明完整的特殊 token 兼容性。
- 下一步：增加明确的词表与特殊 token 回归用例，并检查上游模型元数据问题报告；不擅自修改下载的模型文件或其 manifest 哈希。

## ENG-012：CTest 报告截断

- 状态：已解决。
- 影响：成功测试的默认标准输出长度限制，使 JUnit 报告遗漏末尾用例及用例总数。虽然套件通过，但保存的验证证据不完整。
- 复现条件：使用 CTest 默认的 1024 字节成功输出限制，导出包含 28 项用例的单元测试报告。
- 原因：CTest 将整个单元测试可执行文件视为一个测试，并截断捕获的输出。
- 解决方法：保存 JUnit 证据时设置 `--test-output-size-passed 65536`；CI 在测试失败时也上传报告。
- 验证依据：`benchmarks/results/validation/windows-cpu-ctest.xml` 包含全部 28 项单元测试及 4 项 GGUF 测试结果。

## ENG-013：保持基准回放文件的字节一致性

- 状态：已解决。
- 影响：自动将 CRLF 规范化为 LF 会改变回放文件的原始摘要，即使解析得到的请求完全相同。克隆仓库后的文件将与归档基准报告中的摘要不匹配。
- 复现条件：对实测输入 `benchmarks/traces/cpu-mixed-s0.jsonl` 执行 `git hash-object --no-filters`，将结果与暂存区或已提交的 Git 文件对象比较。
- 原因：通用文本规范化规则也作用于 `.jsonl` 回放文件，而 `llmserve-bench` 对包含换行符在内的原始字节计算哈希。
- 解决方法：在 `.gitattributes` 中声明 `*.jsonl -text`，按原始字节暂存回放文件，不进行换行规范化；对这些文件设置 `whitespace=cr-at-eol`，使 `git diff --check` 接受有意保留的 CRLF。普通源码仍保持换行规范化。
- 验证依据：原始文件与 Git 文件对象的对象 ID 均为 `585b02f75cb8cc9242f57aca141099c57e7f5931`；回放请求数据和原始基准报告没有变化。

## ENG-014：验证证据检查遇错即停止

- 状态：已解决。
- 影响：临时归档检查命令误将 `XmlDocument.InnerText` 当作元素正文访问。PowerShell 报告空值方法调用错误后，仍按默认错误策略继续执行后续 Git 命令。远程 CI 任务本身已通过，但额外的本地校验没有完整执行。
- 原因：XML 节点访问方式错误，且脚本错误默认不终止执行；PowerShell 的 XML 适配器还可能使用 XML 属性遮蔽原生属性名。
- 解决方法：`scripts/Test-CtestEvidence.ps1` 使用解析后的 `XmlDocument`、显式元素选择、原生 XML 访问方法、严格模式以及 `$ErrorActionPreference = 'Stop'`；检查套件总数、失败和跳过状态、完整的通过用例计数，以及输出缺失或截断。
- 验证依据：本地与远程归档的六份报告均通过检查，共对应九次套件执行、180 次用例执行；失败和截断的测试样本均被拒绝。这里统计的是重复执行次数，不是 180 个不同的测试用例。

## ENG-015：GitHub Actions 运行时兼容性

- 状态：已缓解。发布版本标签前，固定后的工作流必须通过与目标提交精确对应的 GitHub Actions 验证。
- 影响：运行器提示 `actions/checkout@v4` 与 `actions/upload-artifact@v4` 使用的 Node 20 已弃用，并被强制改用 Node 24；同时提示 `ubuntu-latest` 将在未来迁移。
- 验证证据：见运行 `35354736795` 的注解。C++ 单元测试及内存与未定义行为检查已通过，但 CI 环境尚未完全固定。
- 解决方法：使用官方 v7.0.1 发布版本，其 `action.yml` 明确声明 `node24`。将 checkout 固定到 `3d3c42e5aac5ba805825da76410c181273ba90b1`，将 upload-artifact 固定到 `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a`；使用 `ubuntu-24.04` 与 `windows-2025` 标签，并将测试产物缺失视为失败。
- 验证依据：发布标签、提交 ID 与运行时声明均已对照官方 `actions` 仓库核验；对应提交的工作流运行结果仍是构建与测试的验收条件。
- 适用边界：指定版本名称的托管操作系统仍会收到镜像更新，不宣称工具链镜像能够逐字节完全复现。

## ENG-016：Attention 的 V 加权累加未使用 SIMD

- 状态：已解决。
- 影响：CPU attention 中 K 与 query 的 FP16 点积通过 AVX2/FMA/F16C 执行，但 V 的 FP16 到 F32 加权累加逐元素运行，导致同一 attention 内的 K/V 路径没有一致使用 SIMD 分派。
- 复现条件或证据：`src/minillm/runtime.cpp` 的 K 路径调用 `dot_f16`，原 V 路径直接循环执行 `probability * half_to_float(...)`。
- 原因：现有 kernel 接口只提供 FP16 点积，没有提供 `output += scale * FP16 input` 的向量内核。
- 解决方法：增加 `add_scaled_f16` 的 scalar 与 AVX2/FMA/F16C 实现，保留运行时能力检测和非 8 对齐尾部处理；attention 的 V 路径通过 `KernelMode` 调用该内核。
- 验证依据：`simd_half_value_accumulation_matches_scalar` 覆盖长度 1、7、8、9、31、128、1024、4097，29 项 CPU 单元测试全部通过。真实 Qwen3-0.6B Q8_0 模型的 10 项检查通过，scalar/SIMD logits 的 RMSE 为 `2.2703359845177045e-05`、最大绝对误差为 `0.00013446807861328125`、cosine 为 `0.9999999999725622`，三组贪心生成 token 完全一致，物理 KV 页最终为 0。

## ENG-017：模型验证的混合批断言受执行时序影响

- 状态：待解决。
- 影响：真实模型的数值、生成、KV 和前缀缓存检查均可通过，但默认 8 线程执行可能仅因没有观察到 mixed batch 而使整个验证命令失败。
- 复现条件或证据：使用 `scripts/Validate-Model.ps1` 的默认 8 线程配置连续两次得到原始诊断 `tests\model_tests.cpp:295: stats.mixed_batches > 0`；两次运行在失败前的 scalar/SIMD、chunk boundary、paged tail copy-on-write、生成一致性和 KV 回收检查结果相同。相同二进制使用 `--threads 1` 时记录 `mixed_batches: 1` 并通过全部 10 项检查。
- 原因：尚未确定。当前断言依赖工作线程处理请求和调用线程连续提交请求之间的相对时序，是否形成 prefill/decode 混合批并非由测试输入完全确定。
- 下一步：用可控的 runner 屏障或确定性请求注入构造同时存在 prefill 与 decode 的调度状态，再断言 mixed batch；模型数值验证不应依赖未受控的主机时序。
- 验证：尚未完成确定性回归，因此保持待解决。
