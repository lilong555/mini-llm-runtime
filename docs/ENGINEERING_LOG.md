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
| ENG-008 | 待解决 | 性能 | CPU 混合调度的平均 TPOT 与单次停顿存在权衡 |
| ENG-009 | 已解决 | 验证流程 | 手动指定的参考模型文件名不存在 |
| ENG-010 | 待解决 | 第三方依赖 | 固定版本的上游依赖产生 MSVC 编译警告 |
| ENG-011 | 待解决 | 模型词表 | 分词器修正疑似控制 token 的词表条目 |
| ENG-012 | 已解决 | 验证证据 | CTest 截断已通过测试套件的输出 |
| ENG-013 | 已解决 | 版本管理 | 文本换行规范化改变基准回放文件的摘要 |
| ENG-014 | 已解决 | 验证证据 | PowerShell 校验脚本在 XML 访问错误后继续执行 |
| ENG-015 | 已缓解 | 持续集成 | Action 运行时弃用及托管操作系统标签漂移 |
| ENG-016 | 已解决 | SIMD | Attention 的 V 加权累加未使用 SIMD |
| ENG-017 | 已解决 | 模型验证 | 混合批断言受执行时序影响 |
| ENG-018 | 待解决 | 性能 | CPU PagedKV 的长上下文访问成本高于连续布局与 llama.cpp |
| ENG-019 | 已缓解 | 构建 | 普通 PowerShell 缺少完整 MSVC 开发环境 |
| ENG-020 | 已解决 | 实验验收 | 基准汇总未严格拒绝不完整或不等价的报告 |
| ENG-021 | 已解决 | 环境配置 | WSL 原生开发工具覆盖不完整 |
| ENG-022 | 已解决 | 性能采集 | Nsight Systems 与驱动不兼容导致 GPU 时间线缺失 |
| ENG-023 | 已解决 | 性能采集 | WSL 内 Nsight Compute 无权访问 GPU 性能计数器 |
| ENG-024 | 已解决 | 编辑器 | clangd 未发现 WSL 编译数据库 |
| ENG-025 | 已解决 | 基准编排 | 策略回放脚本依赖 Windows 行为且存在语法错误 |
| ENG-026 | 已解决 | 验证证据 | CTest 归档格式与诊断分类不匹配 |
| ENG-027 | 已解决 | 观测状态 | Profile 预分配失败保留旧的成功状态 |
| ENG-028 | 已解决 | 在线观测 | token 延迟缺少 batch 与模型阶段关联 |
| ENG-029 | 已解决 | 派生分析 | 相对路径与绝对路径混用导致归因汇总失败 |
| ENG-030 | 已解决 | 版本管理 | GitHub 仓库可见性与发布授权不一致 |
| ENG-031 | 已解决 | 版本管理 | WSL 仓库沿用 Windows 凭据助手路径 |

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
- 影响：在已测密集到达负载下，混合 prefill/decode 没有改善吞吐或请求平均 TPOT，但单次 ITL 尾部可以较低，不能将这些指标合并为“所有延迟均退化”。
- 复现条件：使用 `scripts/Benchmark-Policies.ps1 -Backend mini` 回放 `benchmarks/traces/cpu-mixed-s0.jsonl`。每种策略运行三次，每次 24 个请求，到达率为 4 请求/秒，提示词长度为 128/16 个 token，输出长度为 16 个 token；每次均重启服务并执行相同预热。
- 验证证据：见 `benchmarks/results/mini-scheduling/summary.json`。全部 144 个请求成功，未观察到输出 token 序列不一致。

| 三次实验的中位数 | 混合调度 | 预填充优先 |
| --- | ---: | ---: |
| 输出吞吐，token/秒 | 18.40 | 18.59 |
| P95 首 token 延迟（TTFT），毫秒 | 13105.77 | 12290.59 |
| 请求平均每 token 延迟（TPOT）的 P95，毫秒 | 398.89 | 352.85 |
| 满足 SLO 的有效吞吐，请求/秒 | 0 | 0 |

- 在线证据：2026-09-23 的三轮无观测 WSL 对照中，mixed / prefill-first 中位吞吐为 18.41 / 18.66 token/s，P95 请求平均 TPOT 为 395.28 / 333.42 ms，P99 单次 ITL 为 1224.77 / 2480.67 ms。阶段模式中一次 3673.99 ms 的 Engine token 间隔包含六批 prefill 与随后一批 decode，runner 区间交集为 3673.56 ms，scheduler 仅为 0.012814 ms。完整输入、开关成本、低到达率及不利样本见 `benchmarks/results/wsl-batch-telemetry/eng-008-analysis.md`。
- 原因：已测范围内排除 scheduler 自身计算为主要成本，支持继续研究 prefill 模型执行与 decode 停顿的关系；尚未通过受控计算路径切换确定矩阵权重复用、LM head 或线程池的独立贡献。
- 下一步：针对已记录的真实矩阵形状和 attention 成本建立受控对照，补充 worker 时间线与冷、热前缀输入；完成对应数值门槛后实施一个优化，保留不利结果，重新完成模型与 HTTP 检查。
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
- 原始诊断：`load: control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden`。
- 原因：上游启发式规则与模型文件中的 token 元数据不一致；尚未确定应该修改模型文件还是启发式规则。
- 当前行为：固定版本的分词器在加载时修正该 token 类型。MiniLLM 与数值参照共用此分词器；已有的数值、生成和 HTTP 检查通过，但不足以证明完整的特殊 token 兼容性。
- 下一步：检查上游模型元数据问题报告；保留固定权重与词表语义，不修改下载的模型文件或其 manifest 哈希。
- 词表验证：`tests/host_model_tests.cpp` 对 `tests/data/qwen3_validation_cases.json` 的 7 组中文、英文、重复、特殊 token 和生成输入核对 token IDs，并将全部 151936 个 token 的 piece/EOG 与固定上游 vocab-only 参照逐项比较，8/8 host-model 实模型检查通过。原始警告仍存在，见 `benchmarks/results/validation/host-model/host-real.txt`；该验证没有修复模型元数据问题，本项保持待解决。

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

- 状态：已解决，限定于测试中混合批状态的确定性构造。
- 历史工具链证据：2026-09-22，GCC 11.4.0、CUDA 12.8 的独立产品构建及 CTest、CPU/CUDA HTTP 检查通过；`bash scripts/dev.sh validate` 和 `bash scripts/dev.sh cuda validate` 均以 `tests/model_tests.cpp:295: stats.mixed_batches > 0` 失败。各自九项数值与 KV 检查完成，完整失败报告保留在 `benchmarks/results/wsl-environment/cpu-model.json` 和 `cuda-model.json`，不能视为完整模型验收通过。
- 故障复现（2026-09-22）：在 `build/wsl-native` 分支、提交 `a1fe5d327adb3f806c0dfa7042565e42fbaff0d5` 及当时的未提交工作区上完成 CPU 构建检查后，默认 8 线程模型验证再次以 `tests/model_tests.cpp:295: stats.mixed_batches > 0` 失败。九项数值与 KV 检查完成，三组生成 token 与参照一致；本地原始报告为 `.run/environment-model.json`，标准输出与诊断为 `.run/environment-model.log`。该结果不构成完整模型验收通过。
- WSL2 验证证据：Ubuntu GCC 11.4.0 原生 `RelWithDebInfo` 构建运行 `bash scripts/dev.sh validate`，九项数值与 KV 检查及三组生成对照完成，随后在 `/home/li/code/mini-llm-runtime/tests/model_tests.cpp:295: stats.mixed_batches > 0` 失败。完整报告见 `benchmarks/results/wsl-native/model.json`，不计为全套通过。
- 影响：真实模型的数值、生成、KV 和前缀缓存检查均可通过，但默认 8 线程执行可能仅因没有观察到 mixed batch 而使整个验证命令失败。
- 复现条件或证据：使用 `scripts/Validate-Model.ps1` 的默认 8 线程配置连续两次得到原始诊断 `tests\model_tests.cpp:295: stats.mixed_batches > 0`；两次运行在失败前的 scalar/SIMD、chunk boundary、paged tail copy-on-write、生成一致性和 KV 回收检查结果相同。相同二进制使用 `--threads 1` 时记录 `mixed_batches: 1` 并通过全部 10 项检查。
- 原因：连续提交请求没有建立模型执行阶段之间的先后约束；请求长度和提交顺序不能保证下一次调度同时存在 prefill 与 decode。断言因而依赖调用线程与模型线程的相对时序。
- 解决方法：测试侧 `test::GatedRunner` 在真实 runner 产生第一份 prefill sample 后暂停返回；调用线程确认屏障后提交其他请求，再释放屏障。下一轮调度具有已进入 decode 的请求和待 prefill 请求；屏障有有限等待，并在提交异常时释放。生产 Engine、调度策略和数值阈值保持各自原有语义。
- 验证：`benchmarks/results/validation/wsl-deterministic/` 中 CPU 1、2 线程各一次、8 线程连续三次，以及 CUDA 参照的 8 线程一次，全部为 10/10 通过；六份报告均记录 `mixed_batches=1`、`max_batch_sequences=3`，三组生成及缓存重放与参照一致，KV 回收通过。核心回归 `engine_mixed_batch_after_deterministic_request_injection` 每次执行十轮受控注入，在 CPU、CUDA 和 ASan/UBSan 构建中通过。源码、模型和二进制身份见该目录的 `evidence.json`。
- 适用边界：不代表完整 F16/F32 被测模型、长上下文或任意主机负载均已验收；混合调度的性能问题 `ENG-008` 独立保留。

## ENG-018：CPU PagedKV 的长上下文访问成本

- 状态：待解决。
- 影响：同一 Qwen3-0.6B Q8_0、F16 K/V、8 线程 CPU 工作负载下，1536-token prompt 的 MiniLLM 中位 TPOT 为 `49.263715625 ms`，llama.cpp 为 `33.843925 ms`，MiniLLM 高 `45.6%`。从 16 增长到 1536 token 时，两者 TPOT 分别增长 `16.6163625 ms` 与 `10.6811875 ms`。
- 复现条件或证据：原始服务报告、固定 token trace 和汇总见 `benchmarks/results/kv-cache-cpu/` 与 `benchmarks/traces/kv-cpu-*.jsonl`。`mini-kv-cache-bench` 在完全相同的 AVX2/F16C QK、softmax 和 PV 数学下，仅切换真实 `PagedKV` 与按层连续 F16 布局；16、256、1024、1536 token 的耗时比分别为 `1.20x`、`1.26x`、`1.35x`、`1.32x`。
- 模型级证据：`benchmarks/results/wsl-runtime-profile/context/` 的独立 8 线程实验中，无 profiler 的 16/256/1536 KV decode 中位数为 42.98/44.85/58.47 ms；profile 进程的合并 attention 阶段为 2.00/3.27/15.28 ms，LM head 为 6.80/6.51/6.43 ms。两种模式的完整输出摘要一致。该证据缩小了后续归因范围，但未分离 QK、softmax、PV 或页查找，不能替代旧服务对照或证明 allocator 是全部差距来源。
- 在线证据：`benchmarks/results/wsl-batch-telemetry/eng-018-analysis.md` 中，单轮 mixed 阶段模式的初始 16/256/1536 KV 后续 32 次 decode，attention 中位数为 2.04/3.90/15.36 ms，LM head 为 7.41/7.56/6.96 ms；实际 KV 范围为 16–47、256–287、1536–1567。每配置一个独立进程，不视为稳定优化收益，也不与旧平台报告直接计算加速比。
- 原因：隔离基准证明逐 token accessor 与跨页不连续访问具有显著成本；完整差距还混有 llama.cpp 的 AVX512、FlashAttention、计算图和其他 kernel 差异，现有证据不能把全部差距归因于 KV allocator。将 page size 改为 256 的顺序实验发生明显性能漂移，不能据此确认最优页大小。
- 下一步：沿已有在线 batch/token 关联分离 QK、softmax、PV 和 page lookup；对应数值门槛满足后，再对照按物理页访问、online softmax、page slab 或索引缓存。页大小实验必须交替顺序并记录频率与温度，不得把布局微基准等同于端到端收益。
- 验证：差距已由三轮真实服务报告和七轮布局隔离报告复现，但尚未实现或验证优化，因此保持待解决。

## ENG-019：普通 PowerShell 缺少完整 MSVC 开发环境

- 状态：已缓解。
- 影响：直接执行 `cmake --build build/cpu --config Release -j 8` 无法完成增量编译，但旧二进制上的 CTest 仍可运行，容易造成测试通过而源码未重新构建的误判。
- 复现条件或证据：普通 PowerShell 中编译 `src/minillm/parallel.cpp` 报告原始诊断 `fatal error C1083: 无法打开包括文件: “atomic”: No such file or directory`。
- 原因：`cl.exe` 路径存在，但当前进程没有加载 Visual Studio Developer Shell 提供的完整 `INCLUDE`、`LIB` 等环境。
- 解决方法：Windows 构建统一调用 `scripts/Build-LLMServe.ps1`；脚本在需要时通过 `vswhere.exe` 导入 `Microsoft.VisualStudio.DevShell.dll` 并进入 x64 开发环境。
- 验证依据：`scripts/Build-LLMServe.ps1 -BuildDirectory build/cpu -Jobs 8` 随后完成 CPU Release 配置与增量构建，`ctest --test-dir build/cpu --output-on-failure` 的 `unit`、`gguf` 两个套件全部通过。直接调用 CMake 的普通 PowerShell 环境仍未自动修复，因此状态为已缓解。

## ENG-020：基准汇总缺少严格的完整性与等价性验收

- 状态：已解决，限定于同一二进制和模型的 Serving 策略对照。
- 影响：缺失请求或输出 token 不一致的报告仍可能生成汇总；未校验的配置、模型文件或二进制差异可能混入策略比较，不能将汇总生成成功等同于实验验收通过。
- 复现条件或证据：`scripts/Analyze-Benchmarks.ps1` 使用第一份报告构建 `$expected`，只遍历其他报告中存在的请求；同一 ID 写入哈希表会覆盖旧值，没有完整集合或重复 ID 检查。输出差异仅累加 `$mismatches` 并写入 `token_sequence_mismatches_against_first_trial`，没有对应失败分支。跨报告检查只有 `trace_fnv1a64`、`server_before.backend`、`server_before.model`，未核对完整配置、模型文件摘要和二进制身份。
- 原因：脚本承担统计汇总职责，但缺少独立的实验身份、请求完整性与比较条件验收层。
- 解决方法：采集脚本绑定源码快照、构建配置、依赖、服务端与客户端、模型来源和原始 trace；每轮前后核对输入身份。验收器按 trace 检查完整请求集合、大小写敏感 ID、策略轮次、终态和输出长度，按显式参照检查成功请求的 token，并从原始时间戳核对统计。错误或不等价报告保存失败结果，不保留成功汇总；压力实验合法终态必须预先声明。
- 验证：CTest 的 51 项基准与归档检查覆盖删除请求、重复 ID、错误模型或二进制摘要、配置差异、token mismatch、输出截断、缺失轮次、统计错误和旧汇总残留。`benchmarks/results/wsl-policy-validation/` 的六轮原始回放完成全部 144 个请求，token 一致，严格验收通过；整个目录复制到另一位置后，离线验收仍通过。`benchmarks/results/wsl-protocol-validation/` 中真实 HTTP 429 与 SSE 超时均以显式合法失败保留，不计入成功或 goodput。
- 适用边界：历史报告没有被补造 manifest。该实现不覆盖 dot/KV 微基准的统一采集入口或跨源码、跨二进制的优化前后比较；这些工作仍属于 `PLAN-001` 的剩余范围。CPU 混合调度没有由此获得性能收益，原始负结果继续保留。

## ENG-021：WSL 原生开发工具覆盖不完整

- 状态：已解决，限定于工具安装与下述能力验证；GPU 计数器权限另见 `ENG-023`。
- 影响：缺少专项原生工具时，即使 CPU 产品可运行，也无法直接执行 Linux CUDA 编译、原生 PowerShell 基准验收测试及 `perf` 性能采集。
- 复现条件或证据：2026-09-22，Ubuntu 22.04.3 / WSL2 中 `command -v pwsh nvcc perf nsys ncu` 未找到对应工具；`build/wsl-cpu/CMakeCache.txt` 记录 `LLMSERVE_POWERSHELL:FILEPATH=LLMSERVE_POWERSHELL-NOTFOUND`，CTest 仅注册 `unit`、`gguf`。`nvidia-smi` 可识别 RTX 4070 Laptop GPU、8188 MiB 显存及驱动 591.74，但这不证明 Linux CUDA Toolkit 可用。
- 原因：基础 CPU 开发工具与 PowerShell、Linux CUDA、性能采集工具是独立的安装项；Windows 的工具不能直接替代 Linux 原生编译和测试入口。
- 解决方法：使用原生 PowerShell 7.6.6、CUDA Toolkit 12.8、LLVM 19 和可用于当前内核的 perf 5.15.209；CPU/CUDA 使用独立构建目录。Nsight Systems 版本要求见 `ENG-022`。不安装 WSL Linux 显示驱动，不放宽全局 perf 权限。
- 验证：`benchmarks/results/wsl-environment/` 保留三份 3/3 通过的 CTest 报告，均包含 PowerShell fixture；CPU/CUDA HTTP 各 8/8 通过，临时服务已退出。两份模型通过大小与 SHA-256 校验；CUDA 冒烟的 1048593 个结果正确，Compute Sanitizer 为 0 错误、0 泄漏字节；perf 的用户态事件可读，457 个样本丢失 0，MiniLLM 符号可解析。
- 适用边界：系统级 perf、全部 PMU 事件和 Unified Memory 跟踪未验收。`perf report` 仍输出 `(Cannot load tips.txt file, please install perf!)`，报告中的事件和样本可读，帮助资源未验收。该批环境报告中的完整模型失败见 `ENG-017`，不由工具链验收结论代替。

## ENG-022：Nsight Systems 与驱动不兼容导致 GPU 时间线缺失

- 状态：已解决，限定于当前设备的 CUDA kernel 与显式内存传输时间线。
- 影响：采集进程返回 0 且存在 CUDA API 表，但没有 GPU kernel 和内存传输数据，不能用于 GPU 执行耗时归因。
- 复现条件或证据：Nsight Systems 2024.6.2 配合驱动 591.74，使用 `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` 运行 CUDA 向量程序。报告出现 `does not contain CUDA kernel data`；SQLite 诊断包含 `Installed CUDA driver version (13.1) is not supported by this build of Nsight Systems. CUDA trace will be collected using libraries for driver version 12.8`。证据见 `benchmarks/results/wsl-environment/nsys-2024-diagnostics.json`。
- 原因：该 Nsight Systems 版本的采集库与当前驱动接口不兼容；CUDA Toolkit 能编译和执行程序，不代表随附 profiler 能采集当前驱动的 GPU 时间线。
- 解决方法：使用官方固定版本 `nsight-systems-2026.1.3`，校验安装包 SHA-256，在用户目录安装并通过 `~/.local/bin/nsys` 使用。CUDA Toolkit 仍为 12.8，安装包身份见 `benchmarks/results/wsl-environment/environment.json`。
- 验证：登录终端的 `nsys --version` 返回 `2026.1.3.425-261338342291v0`。`scripts/cuda_smoke.cu` 的报告包含全部 64 次 kernel、2 次 H2D 和 1 次 D2H；`nsys-validation.json` 对 SQLite 事件数完成校验，`nsys-stats.txt` 保留原始汇总。
- 适用边界：仍有 `Unified Memory cannot be traced` 诊断；本项只验收显式分配与复制的时间线。GPU 硬件计数器由 `ENG-023` 独立验收，完整模型性能采集不在本项范围内。

## ENG-023：WSL 内 Nsight Compute 无权访问 GPU 性能计数器

- 状态：已解决，限定于当前 Windows/WSL、GPU 和工具版本组合的基础 kernel 指标采集。
- 影响：缺少主机授权时，CUDA 编译、执行、内存检查和 Nsight Systems 时间线可用，但不能采集 Nsight Compute 所需的 GPU 性能计数器。
- 复现条件或证据：`ncu --target-processes all --launch-count 1 --section LaunchStats .run/cuda-smoke` 返回 1，原始诊断为 `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.`。完整输出见 `benchmarks/results/wsl-environment/ncu.txt`。
- 原因：Windows NVIDIA 驱动未向当前用户开放 GPU 性能计数器；WSL 内 root 权限不能替代主机侧的访问控制。
- 解决方法：在 Windows NVIDIA 控制面板启用开发者设置，并在“管理 GPU 性能计数器”中授权访问；WSL 中使用普通用户执行采集。参考 `https://developer.nvidia.com/ERR_NVGPUCTRPERM`，不更换 CUDA Toolkit、模型或数值阈值。
- 验证：2026-09-22，主机授权有效时，相同 CUDA 冒烟二进制的 `--section LaunchStats` 和 `--set basic` 采集均返回 0，没有权限错误。程序的 64 次 kernel 执行与 1048593 个结果校验通过；基础采集包含 1 个 kernel、8 次 profiler replay，CSV 中的耗时、SM/DRAM 周期、占用率和吞吐指标为有效值。原始指标见 `benchmarks/results/wsl-environment/ncu-metrics.csv`，校验结果见 `ncu-validation.json`，采集日志见 `ncu-launch-capture.txt` 和 `ncu-basic-capture.txt`；`.ncu-rep` 的本地路径及 SHA-256 固定在环境元数据中。
- 适用边界：这里只证明基础硬件指标可采集，不是完整模型或端到端性能基线。未授权时的失败证据保留，`ENG-017` 的完整模型验证失败不受本项结论影响。

## ENG-024：clangd 未发现 WSL 编译数据库

- 状态：已解决，限定于编译参数发现与目标文件诊断。
- 影响：CMake 构建通过，但编辑器缺少项目头文件路径及 C++20 参数，产生大量无效诊断。
- 复现条件或证据：没有根目录配置时，`clangd --check=src/minillm/runtime.cpp --log=error` 报告 `[pp_file_not_found] Line 1: 'minillm/runtime.h' file not found`；显式指定 `--compile-commands-dir=build/wsl-cpu` 可以读取构建参数。
- 原因：编译数据库位于嵌套的 `build/wsl-cpu`，clangd 默认搜索路径没有选中该目录。
- 解决方法：根目录 `.clangd` 将 `CompileFlags.CompilationDatabase` 指向 `build/wsl-cpu`，编辑前完成 CPU 配置与构建，不手工复制头文件路径或编译宏。
- 验证：`benchmarks/results/wsl-environment/clangd-kernels.txt` 记录从实际编译数据库读取 C++20、头文件和 SIMD 参数，`kernels.cpp` 检查为 0 错误。`clangd-lsp.json` 记录 `runtime.cpp` 的 LSP 诊断为空且正常退出；该检查关闭后台索引。
- 适用边界：对 `runtime.cpp` 执行完整 `clangd --check` 还会出现 `ExpandDeducedType`、`ExtractFunction` 的重构动作探针错误，不等同于 LSP 编译诊断。后台索引和全部重构功能不在本项验收范围内。

## ENG-025：策略回放脚本的原生兼容性

- 状态：已解决，限定于当前 WSL2、PowerShell 7.6.6 和 GCC/CUDA 产品的原生运行。
- 影响：仅有验收器 fixture 通过，不能保证策略采集入口可解析、可启动服务或可读取源码身份。
- 复现条件或证据：PowerShell AST 解析 `scripts/Benchmark-Policies.ps1` 的 `Read-CMakeSetValue` 时返回 `Unexpected token '(' in expression or statement.`；代码还依赖 `Get-NetTCPConnection`、`.exe` 和 `Start-Process -WindowStyle`。原生采集读取隐藏文件时，`Get-Item` 返回 `Could not find item /home/li/code/mini-llm-runtime/.gitattributes.`。
- 原因：PowerShell 字符串错误地使用了 C++ 风格的引号转义；启动、端口和路径处理依赖 Windows；Linux 的隐藏文件需要显式 `-Force`。通过临时 bind 探测空闲端口还会把 TIME_WAIT 误当成活动监听。
- 解决方法：使用合法的 PowerShell 正则字符串；通过 .NET 枚举活动监听，按平台选择产品目录、可执行文件和启动参数；按进程路径与启动时刻确认服务身份，使用平台路径比较；读取源码元数据时包含隐藏文件。采集前增量构建失败即终止。
- 验证：原生 CPU 六轮策略回放及两类协议回放完成，服务均通过停止脚本退出；CPU/CUDA 的真实 HTTP 各 8/8 通过。基准 fixture 覆盖脚本解析和活动端口探测。原始回放、manifest 和服务回收记录见 `benchmarks/results/wsl-policy-validation/`、`benchmarks/results/wsl-protocol-validation/` 和 `benchmarks/results/validation/wsl-deterministic/`。
- 适用边界：本次没有重跑 Windows 产品或远程 CI；现有 Windows 启动分支不等同于新的 Windows 实测证据。

## ENG-026：CTest 归档格式与诊断分类

- 状态：已解决，限定于证据格式识别及完整报告与诊断报告的分类。
- 影响：CTest 返回成功后，归档校验器仍可能拒绝合法的基准套件输出；历史截断报告混入完整证据目录时会使默认归档门禁失败。
- 复现条件或证据：`scripts/Test-CtestEvidence.ps1` 对含 `43/43 benchmark validation tests passed` 的报告返回 `Truncated or inconsistent test output`。三个环境报告使用旧标记 `benchmark validation fixtures passed`，没有单项数量。旧 WSL XML 明确包含 `[This part of the test output was removed since it exceeds the threshold of 1024 bytes.]`。
- 原因：归档校验器只接受 C++ 套件的计数格式，且假定所有 XML 都是完整的通过证据；旧 WSL 采集使用了 CTest 默认成功输出上限。
- 解决方法：支持带数量的基准完成标记；仅对精确匹配的旧基准套件标记保留“数量未报告”状态，不推测单项数量。正式归档检查将 `diagnostics/` 单独列出；显式检查该诊断目录仍拒绝截断 XML。当前 CTest 使用 `--test-output-size-passed 65536`。
- 验证：完整、旧格式、截断、失败标记、数量不一致和诊断分类均有回归。当前验证目录的三份报告、八次套件执行、251 次用例执行完整通过；这是重复执行数量，不是 251 个不同用例。默认全归档检查列出三个未提供数量的旧套件，并返回 `passed_with_unreported_case_counts`。`benchmarks/results/diagnostics/wsl-native-ctest-truncated.xml` 原始字节保留，SHA-256 仍为 `a94f68493e365823c29f46af9de7fa0a031e21a547cd53e06f3973bf758b7557`，不作为完整验证通过的依据。

## ENG-027：Profile 预分配失败保留旧的成功状态

- 状态：已解决，限定于 profile 预分配异常后的状态失效与复用。
- 影响：重用曾成功的 `ForwardProfile` 时，如果阶段存储扩容失败，调用方可能读到旧的 `completed=true`；该状态不能代表本次 forward 完成。
- 复现条件或证据：`ForwardScope` 在重置 profile 字段前调用 `vector::reserve`。`reserve` 抛出 `std::length_error` 或 `std::bad_alloc` 时，构造函数不会完成，析构函数也不会执行。对应源码快照保留在 `benchmarks/results/diagnostics/runtime-profile-preallocation/`；该次基准在首个进程完成前主动终止，采集状态为失败、完成报告数为 0，不作为性能证据。
- 原因：预分配这一可能抛出异常的操作先于状态失效处理。
- 解决方法：`ForwardProfile::reset` 先清空上次状态并保留 `batch_id` 及预留存储，再执行扩容；失败保留未完成、无计时状态。
- 验证：`forward_profile_reserve_failure_clears_stale_success` 通过超过 `max_size()` 的请求确定性触发预分配异常，检查状态、标识及存储复用，在 CPU、CUDA 参照构建和 ASan/UBSan 核心构建中均通过。当前 CTest 为 4/4、4/4、3/3，共 358 次用例执行；CPU 1/8 线程和 CUDA 数值参照的模型检查各 12/12，包含 profiler 开关、KV 容量失败及恢复。原始结果与源码、模型、二进制身份见 `benchmarks/results/validation/wsl-runtime-profile/`；正式模型基准在独立目录完成 36 个进程、558 次测量验收，没有复用已中止采集作为性能证据。

## ENG-028：在线 token 延迟缺少批次与模型阶段关联

- 状态：已解决，限定于有界在线观测、SSE token 关联及完整性验收。
- 影响：请求平均 TPOT 与单次 ITL 可能表现出不同方向；服务总批次计数和离线 Runtime 计时不足以确定某个输出经历了哪些在线 batch，也无法测量观测开关对组批的扰动。
- 复现条件或证据：`ENG-008` 的历史 mixed/prefill-first 对照具有不同的平均 TPOT 和 P99 ITL；原 `Statistics` 只有累积 `batches`、`mixed_batches`，原 SSE `token_id` 没有对应 batch 标识。
- 原因：Engine 的调度轮次、Runtime 阶段和客户端 token 时钟之间没有共同标识，也没有保存实际 batch 的请求与上下文信息。
- 解决方法：`off / batches / stages` 三种模式使用启动时预分配的有界存储，停服后导出 JSONL；以 batch ID、请求创建顺序、输出序号关联 SSE。分别记录准入、调度、执行、物理 KV 和容量信用；后端未知指标为 `null`，丢记录和未完成 batch 不作为完整性能证据。到达缩放保留 trace 原始字节与摘要。
- 验证：`benchmarks/results/validation/wsl-batch-telemetry/` 中 CPU/CUDA 参照构建的 CTest 均为 5/5，ASan/UBSan 核心为 4/4，合计 463 次用例执行。CPU 1/8 线程及 CUDA 数值参照的模型验证均为 13/13，三组 HTTP 均为 8/8；涵盖混合批、缓存复用、缓冲耗尽、异常、SSE 关联、阶段时间守恒及停服回收。在线验收器的 27 项用例拒绝缺失或伪造证据。
- 回放证据：`benchmarks/results/wsl-batch-telemetry/` 的 42 份独立服务报告、594 个请求、9810 个输出 token 验收通过，28 份观测 JSONL 无丢记录；观测开销和 batch 组成变化均保留。目录迁移后重新验收通过，临时服务全部退出。
- 适用边界：当前不含 sequence 生命周期事件与完整 token 输入，不能用于 Runtime replay；attention 页访问、QK/softmax/PV、worker 调度时间线尚未分离。`ENG-008`、`ENG-018` 的根因和优化收益不由本项观测能力直接证明。

## ENG-029：派生归因表混用相对路径与绝对路径

- 状态：已解决，限定于本地派生归因汇总的路径处理。
- 影响：归因表生成中止；42 份原始服务报告及各自的在线验收已经完成，没有被修改或视为失败采集。
- 复现条件或证据：绝对目录调用 `relative_to` 时传入相对基目录，原始诊断为 `ValueError: '/home/li/code/mini-llm-runtime/benchmarks/results/wsl-batch-telemetry/mixed/trial-0-batches' is not in the subpath of 'benchmarks/results/wsl-batch-telemetry' OR one path is relative and the other is absolute.`。
- 原因：基目录从相对字符串构造，报告中的 `directory` 由采集器保存为绝对路径；两者没有统一形式。
- 解决方法：派生分析先对目录和基目录执行 `Path.resolve()`，再提取相对证据路径。该本地辅助脚本属于 `.run/`，不作为产品或提交内容。
- 验证：`benchmarks/results/wsl-batch-telemetry/hypothesis-table.json` 已生成，包含全部 28 份观测报告的调度占比、六份原始负载的最长停顿区间和六份上下文阶段汇总；原始报告摘要仍与采集验收记录一致。正式验收器的独立目录迁移检查也通过，结果见 `benchmarks/results/validation/wsl-batch-telemetry/relocation.json`。

## ENG-030：GitHub 仓库可见性与发布授权不一致

- 状态：已解决，限定于当前仓库设置和交付文档；不代表历史公开内容已从第三方副本中撤回。
- 影响：仓库可见性和交付文档曾先后采用不同策略，可能导致阶段性成果未按所有者的当前发布决定同步。
- 复现条件或证据：2026-09-23，远端曾返回 `"visibility":"PUBLIC"`，在按默认私有策略收敛后返回 `"visibility":"PRIVATE"`；随后仓库所有者明确要求上传当前阶段成果并公开仓库。
- 原因：默认私有策略与后续的显式公开授权处于不同时间点，远端设置和交付文档需要以最新授权同步更新。
- 解决方法：在检查待上传内容后，将 GitHub 仓库可见性设为公开；README 和版本控制文档统一描述当前公开状态，仍禁止提交凭据、模型权重、依赖 checkout、构建产物和本地运行状态。
- 验证：执行 `gh repo view lilong555/mini-llm-runtime --json nameWithOwner,visibility,url,viewerPermission,defaultBranchRef`，返回仓库 `lilong555/mini-llm-runtime`、`"visibility":"PUBLIC"`、`"viewerPermission":"ADMIN"` 和默认分支 `main`。

## ENG-031：WSL 仓库沿用 Windows 凭据助手路径

- 状态：已解决，限定于当前 WSL 工作区的 GitHub 凭据助手配置。
- 影响：`git fetch origin` 虽返回成功，仍报告 Windows GitHub CLI 路径不存在，凭据获取与保存阶段产生错误诊断。
- 复现条件或证据：原始诊断包含 `C:/Program Files/GitHub CLI/gh.exe: not found`。`git config --show-origin --get-regexp '^credential\..*(helper|useHttpPath)$'` 显示 `.git/config` 将 GitHub 凭据助手设置为 Windows `gh.exe`，覆盖了用户配置中的 Linux `/usr/bin/gh`。
- 原因：仓库本地配置保留跨平台迁移前的绝对可执行文件路径。
- 解决方法：删除仓库本地的 `credential.https://github.com.helper` 覆盖项，使用已有用户配置中的 `!/usr/bin/gh auth git-credential`；凭据和本地配置不纳入提交。
- 验证：`git config --show-origin --get-all credential.https://github.com.helper` 仅返回 `/home/li/.gitconfig` 中的 Linux 配置；重新执行 `git fetch origin` 返回 0，且无错误诊断。

## ENG-032：实验源码快照未进入可独立获取的归档

- 状态：部分解决；`wsl-runtime-profile/context` 的原始快照已核对，归档检查与导出门禁可用。其他历史目录不据此视为完整。
- 影响：原报告虽记录过验收通过，独立 Git 检出缺少必需 ZIP 时无法重新检查源码身份；作者工作区中的文件存在不代表发布内容完整。
- 复现条件或证据：审计基点 `68ac275913207975a88e2090c6617467e351301c` 的 `git archive` 副本中，`benchmarks/results/wsl-runtime-profile/context/manifest.json` 引用的 `source-snapshot.zip` 不存在。`historical-clean-availability.json` 记录 `ARCHIVE_INCOMPLETE`，具体证据位于 `benchmarks/results/validation/evidence-m0/`。
- 原因：`.gitignore` 排除了所有实验源码 ZIP，原始分析器要求它们存在，但没有发布可用性门禁。
- 解决方法或下一步：`Test-EvidenceAvailability.ps1` 检查必需文件、SHA-256 与 ZIP 内各源码；`Export-BenchmarkBundle.ps1` 在副本中执行严格验收、封包与迁移复验。`context` 的本机原 ZIP 与历史摘要 `9b7f243855874a12fb5f42e6053cb3e3b5e205ffb53865e85f89df0b741d689c` 一致，仅为该原件和新基线添加明确的 Git 路径例外。其他历史缺件必须逐一报告，不根据当前源码重建旧快照。
- 验证：`historical-revalidation.json` 记录缺件拒绝、16 个既有文件字节不变，以及加入原件后的完整归档与独立包复验通过。历史 manifest、原始报告、源码状态及旧汇总不变。
- 适用边界：导出用途为 `archive_revalidation`，不含模型权重、二进制或工具链；不能当作相同二进制已在另一台机器重跑的证明。

## ENG-033：分析失败删除既有验收结果

- 状态：已解决，限定于 Runtime 与 Serving 策略分析器的无损失败和发布异常恢复。
- 影响：缺失源码 ZIP、损坏 manifest 或报告校验失败时，复验会先删除已有汇总，破坏旧证据；缺件与原测量失败也容易混淆。
- 复现条件或证据：原 `Analyze-Runtime.ps1`、`Analyze-Benchmarks.ps1` 在读取 manifest 前执行 `Remove-Item`，异常路径再次删除汇总。`historical-clean-check.txt` 和两个基准 fixture 套件覆盖缺件与旧成功记录共存的情况。
- 原因：输出清理先于输入与依赖验证，直接覆盖验收标记，没有区分本次失败与历史结果。
- 解决方法：先检查依赖并完成严格校验；汇总序列化到同目录的临时区域，逐文件原子替换，最后发布成功标记；发布失败恢复原字节。失败诊断单独写入 `analysis-failure.json`。历史复验和导出使用临时副本，调用方以本次退出状态判断结果。
- 验证：两套 fixture 对失败前后所有已有文件核对 SHA-256；缺 ZIP、损坏源码状态、篡改 ZIP 条目、旧成功汇总、跨目录迁移及导出反例通过。`atomic-publication-rollback` 用目标目录冲突触发中途发布错误，确认已替换的文件恢复原摘要。原始 CTest 证据见 `benchmarks/results/validation/evidence-m0/`。
- 适用边界：同一归档目录仅允许单写者；文件级发布及异常恢复不等于跨文件系统事务或断电恢复协议。

## ENG-034：PowerShell 将空备份路径转换为空字符串

- 状态：已解决，限定于 JSON 原子替换和发布异常恢复。
- 影响：首次写入成功，替换已存在的 JSON 时失败，妨碍正常采集与验收。
- 复现条件或证据：`[IO.File]::Replace($temporary, $Path, $null)` 在当前 PowerShell/.NET 绑定中报告 `The value cannot be an empty string. (Parameter 'path')`；`runtime-benchmark-validation` 的已有输入文件替换触发该问题。
- 原因：传入字符串形参的 PowerShell `$null` 被转换为空字符串，未满足 .NET 接口要求的空引用语义。
- 解决方法：无备份路径时使用 `[NullString]::Value`；需要回滚的发布使用真实备份路径。
- 验证：已有 JSON 替换、两种分析器重复验收与发布回滚 fixture 均通过；当前 WSL PowerShell 为 7.6.6。Windows 分支未在本机实测。

## ENG-035：在线观测包重复发布派生汇总导致哈希不符

- 状态：已解决，限定于单组在线观测归档的导出与独立复验。
- 影响：原始报告和 JSONL 可以通过校验，但导出包的迁移复验失败，无法形成可交付文件。
- 复现条件或证据：对 `wsl-batch-telemetry/context-256/trial-0-batches` 导出时，先显式运行 `Analyze-Benchmarks.ps1`，随后 `analyze_telemetry.py` 内部再次运行同一分析器，导出入口报告 `在线观测归档验收失败。`。
- 原因：首次分析重新发布带当前时间戳的 `validation-summary.json`；再次进入 bundle preflight 时，派生文件字节不再匹配封包时的 SHA-256。归档副本可分析一次，不能在同一副本上交错使用旧包摘要与新派生结果。
- 解决方法：观测组仅调用一次 Python 入口，由它完成策略和时间线验收；其他组直接调用对应 PowerShell 入口。每个包独立携带 `verification/` 脚本，包内工具身份与历史测量源码分别固定。
- 验证：`benchmarks/results/evidence-m0/telemetry-bundle.zip` 的两份真实服务报告及对应 JSONL 完成导出、解压和包内工具复验；`telemetry-revalidation.json` 记录成功与文件访问跟踪，原历史目录无字节变化。该结果是历史数据复验，不是新的 Serving 性能采集。

## ENG-036：CUDA 存储验证的旧成功摘要残留

- 状态：已解决，限定于存储验证入口的报告目录和失败状态。
- 影响：复用曾成功的报告目录时，后续验证中途失败可能留下旧的 `passed=true`，不能据此判断当前模型或矩阵通过。
- 复现条件或证据：存储验证入口使用 `create_directories` 接受已存在的目录，只在全部检查结束后写入成功摘要。目录复用反例及逐文件摘要检查见 `benchmarks/results/validation/cuda-storage/negative-checks.json`；错误 checkpoint 的真实失败输出保留在 `rejected-model.txt`。
- 原因：报告目录缺少单次执行的独占约束，验证开始时没有使旧状态失效。
- 解决方法：显式输出目录必须尚不存在；默认命令选择带 UTC 时间和进程号的新目录。新报告先写入 `incomplete`，全部检查通过才写入 `passed`，失败写入 `failed`。固定数值契约的模型 SHA-256 在实模型存储构造前检查，不按文件名或形状近似接受 checkpoint。
- 验证：目录复用命令返回 1，已有文件 SHA-256 不变；将 matched-weight F32 reference 当成 Q8_0 输入时返回 1，报告为 `failed` 且 `passed=false`。10 项存储单测及另含实模型的一组 11/11 检查通过，普通执行与 Compute Sanitizer 的报告一致。进程中断时的 `incomplete` 不能视为成功；本项不代表完整 GPU 模型已验收。

## ENG-037：PowerShell 有序字典的属性汇总失败

- 状态：已解决，限定于本地证据汇总；产品执行与原始测试报告不受影响。
- 影响：CUDA 存储验证全部结束后，辅助归档脚本在汇总 CTest 用例数时退出，无法发布完整的身份与复核记录。
- 复现条件或证据：PowerShell 7.6.6、`Set-StrictMode -Version Latest` 下执行 `@([ordered]@{passed=11}) | Measure-Object -Property passed -Sum`，返回 `Cannot process argument because the value of argument "passed" is not valid. Change the value of the "passed" argument and run the operation again.`。原始复现输出位于 `benchmarks/results/validation/cuda-storage/diagnostics/collection-error.txt`。
- 原因：`OrderedDictionary` 的键访问与对象属性不是同一种管道契约；`Measure-Object -Property` 不能按所需方式取得字典中的计数字段。
- 解决方法：汇总记录使用 `[pscustomobject][ordered]@{...}`，按对象属性求和。辅助脚本位于 `.run/`，不属于产品；归档携带独立 `verify.ps1`，复用源码快照中的既有 CTest 和源码校验工具。
- 验证：四种构建的 28 次 CTest 套件、722 次用例执行与原始 XML 一致；完整归档在独立目录通过复核，缺少源码 ZIP 和矩阵报告篡改的反例均返回 1，原件摘要不变。结果见同目录的 `revalidation.json`，初次汇总失败没有被当作模型或性能失败。

## ENG-038：RMSNorm 有限输入的中间量溢出

- 状态：已解决，限定于自有 CUDA 基础算子的有限极值处理；CPU 数学路径保持原有实现。
- 影响：直接在 FP32 中计算 `x*x`、平方和或 `variance + epsilon` 时，即使输入和 epsilon 有限，中间量仍可能溢出为 Inf，输出错误地退化为零或非有限值。
- 复现条件或证据：`ops_rms_norm_finite_extremes_and_epsilon` 包含幅值 `1e30`、`FLT_MAX`，以及从最小正 subnormal 到 `FLT_MAX` 的 epsilon。`(1e30)^2` 已超过 FP32 最大有限值，按未缩放公式执行不能满足该用例的 FP64 对照。
- 原因：输出尺度可表示不意味着平方或方差中间量可表示；只缩放输入而直接计算 `epsilon / max_abs / max_abs` 还可能在极小输入时溢出。
- 解决方法：使用 `max(max_abs, sqrt(epsilon))` 作为共同缩放因子，在 FP32 中归约缩放后的平方和并计算缩放后的 epsilon。NaN/Inf 输入保留为 NaN，不允许通过 norm 掩盖后成为有效 token。
- 验证：上述极值、普通宽度、多 head、原地与 padding 用例满足固定 `atol=2e-4, rtol=2e-4`；`ops_rms_norm_nonfinite_cannot_become_valid_token` 验证错误行返回 `-1`。11 项算子检查及 memcheck/racecheck/synccheck 均通过，见 `benchmarks/results/validation/cuda-ops/`。此项不表示完整 GPU 模型数值已通过。

## ENG-039：FP16 舍入边界放大真实层的跨后端误差

- 状态：已解决，限定于层级验证中连续误差与 FP16 边界误差的分离；不代表跨后端逐元素等价。
- 影响：把受控算子的逐元素容差直接用于包含独立 GEMM、FP16 舍入和 FFN 的真实整层，会把舍入边界传播与算子实现错误混为一类；该测试失败不能被隐藏或视为整模型已通过。
- 复现条件或证据：第 27 层、2 个 token、context=1，独立 CPU FP64 对照的最大误差为 `0.0018310546875`，RMSE 为 `0.0002094027128162679`。首个逐元素诊断为 `actual=0.43371963501 expected=0.434062957764 absolute=0.000343322753906 limit=0.000286812591553`。失败输出、实际源码快照和二进制身份保留在 `benchmarks/results/validation/cuda-layer/diagnostics/`。
- 原因：V 投影的最大 FP32 差异为 `1.1444091796875e-05`，其中两个元素跨过 FP16 RN-even 的分界；context=1 的 attention 直接读取 V，差异成为 `0.001953125`，继续经过 output/FFN 投影。Q/K/V 的连续浮点误差与 FP16 离散化误差需要分别核对。
- 解决方法或下一步：保留全部原始输入和独立整层对照，以预先固定的模型数值门槛检查真实层；额外使用实际 Q/K/V 作为共享输入，由独立 CPU FP64 计算 attention/FFN，以原 `atol=2e-4, rtol=2e-4` 检查该边界路径。受控 fixture、基础算子和 CPU 旧门槛不变，不修改产品数学实现来追逐某个舍入结果。
- 验证：六组真实层的独立整层与共享 Q/K/V 对照均通过；基础算子和边界路径保持原容差。独立末层混合批最大绝对误差为 `0.12060546875`，最大 RMSE 为 `0.0033648982414092882`，满足固定模型门槛；三组末层不满足直接逐元素算子门槛的事实由 `max_unit_tolerance_ratio` 保留。四种构建共 741 次用例执行，设备与实模型层 memcheck 为 0 错误/0 泄漏，racecheck 为 0 hazards，synccheck 为 0 错误。完整 28 层及生成 token 仍需独立模型验证。
