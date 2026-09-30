# PROJECT_PLAN_V2 — MiniLLM / LLMServe

## 0. 审计身份与执行规则

- Audit HEAD：`68ac275913207975a88e2090c6617467e351301c`
- Default branch：`main`
- HEAD 时间：`2026-09-23T11:44:24Z`
- Audit Date：`2026-09-23`，Asia/Tokyo。
- 本文件是审计后的建议计划，不表示下列新增模块已经实现。
- 开发起点变化时，先比较新 HEAD 与 Audit HEAD；不得把旧报告自动归属于新二进制。
- 现有实验包含 dirty worktree 的 source-state/hash 记录。基点 SHA、工作区状态、二进制身份与发布证据完整性必须分别说明。
- 审计发现至少 `benchmarks/results/wsl-runtime-profile/context/manifest.json` 引用的 `source-snapshot.zip` 不在同路径的当前 GitHub 仓库中。不得把历史 `passed` 当作当前 clean-clone 复验已经通过，也不得伪造历史 ZIP。
- 旧 `docs/PROJECT_PLAN.md` 保留为历史；本文件经负责人采纳后成为下一阶段决策入口，`NEXT_SPEC.md` 是立即执行规范。

任务类型：
- `INFRA`：开发依赖、构建、接口和可交付证据链。
- `MEASUREMENT`：获得信息，不承诺加速。
- `RESEARCH`：有假设、有对照、有停止条件，允许负结果。
- `PRODUCT`：通过相应验证后进入可选择的正式执行路径；不自动取代默认 CPU backend。

工作量使用 S/M/L/XL，不给未经验证的工期承诺。每个任务使用独立 branch、明确依赖和小范围 commit。最多同时推进一条主要产品路线与一项有边界的研究。

## 1. 当前证据登记

下列路径相对于仓库根目录，均在 Audit HEAD 检查；测量归属以各报告 manifest 为准，而不是以本文件所在提交为准。

| ID | 证据 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| E01 | `src/minillm/runtime.cpp`、`kernels*.cpp`、`paged_kv.cpp` | 自有 CPU forward、SIMD、真实 FP16 物理 KV；PV 已使用 `add_scaled_f16` | 新一轮端到端优化已经完成 |
| E02 | `include/minillm/profile.h`、`apps/runtime_bench.cpp`、`docs/RUNTIME_PROFILING.md` | phase profiling、固定输入模型级 benchmark、计时守恒 | attention 内部因果分解或实际硬件带宽 |
| E03 | `benchmarks/results/wsl-runtime-profile/` | 36 份报告、558 次测量；8 线程 prefill-128 投影约 96.16%；长上下文 attention 明显增长 | 558 个独立实验；所有 projection 时间都是纯算术 |
| E04 | `include/llmserve/telemetry.h`、`src/engine.cpp`、`apps/telemetry_output.h` | 有界 batch capture、SSE token 关联、真实 KV 两字段快照 | 完整 Runtime replay、完整物理资源账本 |
| E05 | `benchmarks/results/wsl-batch-telemetry/eng-008-analysis.md` | 已测 scheduler 计算占比极小；特定长停顿可关联到连续 prefill 批次 | mixed 全面更好或全面更坏 |
| E06 | `benchmarks/results/wsl-batch-telemetry/eng-018-analysis.md` | 线上 attention 随 KV length 增长 | 增长全部由页表、allocator 或 page size 导致 |
| E07 | `apps/kv_cache_bench.cpp`、`benchmarks/results/kv-cache-cpu/summary.json` | 共享数学代码下的历史 paged/contiguous 对照 | 当前 WSL 完整 Runtime 的纯 paging 因果贡献 |
| E08 | `tests/model_tests.cpp`、`tests/gated_runner.h`、`benchmarks/results/validation/wsl-batch-telemetry/` | 确定性 mixed batch、profile 开关、错误状态、模型与 HTTP 验证 | 完整 F16/F32 被测模型、多语言、长 context 数值覆盖 |
| E09 | `scripts/Benchmark-Common.ps1`、`Analyze-Benchmarks.ps1`、`Analyze-Runtime.ps1` | 严格身份、请求集合、输出和统计验收实现 | 所有发布归档已具备全部依赖文件；跨 backend 可位级比较 |
| E10 | `scripts/cuda_smoke.cu`、`docs/WSL_DEVELOPMENT.md`、`benchmarks/results/wsl-environment/` | 本机 CUDA、Compute Sanitizer、Nsight、perf 工具具备已记录的使用证据 | 自有 CUDA 模型执行或真实模型 GPU profiler 已完成 |
| E11 | `CMakeLists.txt`、`src/llama_runner.cpp`、`src/mini_runner.cpp` | 当前模型 CUDA 来自 llama.cpp；自有 Runtime 仍是 CPU | 自有 resident CUDA Runtime 已存在 |

## 2. 旧任务迁移，而非重做

| Old ID | 新审计状态 | 已完成 | 仍缺/新处理 |
|---|---|---|---|
| PLAN-001 | PARTIALLY DONE | Serving/Runtime manifest、源码状态、严格验证、失败保留、跨模式对照 | 发布归档依赖闭合；microbenchmark 身份统一；版本化跨实现比较。进入 M0，不能重建整套工具 |
| PLAN-002 | DONE | 自有 Runtime profiler、模型 benchmark、线程扩展与长 context 组、开销及正确性验证 | 不再排“新增 profiler”；后续只扩展具体研究缺口 |
| PLAN-003 | PARTIALLY DONE | batch/SSE 关联、有界缓冲、ENG-008 特定停顿分析 | 非完整 replay；缺 attention 内部分解及更广独立负载。分流到 M3/M5 |
| PLAN-004 | PARTIALLY DONE | 确定性 mixed 验证、profile off/on 位级对照、错误恢复、13 项模型检查 | 13 项不等于完整模型质量覆盖；按 CUDA 修改扩展 targeted oracle，归 M0/M1 |
| PLAN-005 | STILL REQUIRED，条件化 | 实际 M/N/K 已由 profiler 记录，KV layout bench 已有 | 不再做泛用 benchmark 大工程；只为 M3 获准假设补真实 shape microbenchmark |
| PLAN-006 | STILL REQUIRED，条件化 | PV SIMD 是已有产品代码，但早于旧计划基点 | 缺完整 post-profile 优化 A/B 闭环；M3 最多接受一项默认路径优化 |
| PLAN-007 | PARTIALLY DONE | `ModelRunner::resources()` 的 live pages / resident payload 已打通 | shared/pinned/evictable/COW/future reservations 未完成；M4/M5 按需求扩展 |
| PLAN-008 | STILL REQUIRED | 有低到达率和 batch/stall 观察 | 持续公平性、压力、shadow cost model 仍缺，归 M5；不是 CUDA 前置条件 |
| PLAN-009 | BLOCKED，延后 | 保守 reservation 仍工作 | 需要真实容量压力、资源账本、safety/progress 证明；不阻塞 M1/M2 |
| PLAN-010 | SUPERSEDED | 工具链与向量 smoke 已具证据 | 改为现在启动的 M1 模型 vertical slice，而不是再次安装 CUDA 或重复 smoke |

## 3. 战略决策

1. 主要开发方向立即转为自有 CUDA 模型到 token 的完整路径。M0 只修补必要证据与 oracle，不重新发明实验平台。
2. CPU 保持可用的 reference 与独立产品。最多保留两项有预算的研究：
   - CPU-R1：优先验证小任务线程池调度是否值得串行 cutoff；例如 B=1 的 norm。若筛查否定该方向，可用一次真实 shape 的 projection/multi-token reuse 研究替代，而不是再增加第三条主线。
   - CPU-R2：ENG-018 的共享数学、受控访问方式与 coarse attention 分解。允许结论为“没有足够证据值得改布局”。
3. 不做全局最优线程数、不无限优化 CPU GEMM、不为 SIMD 指令集数量扩展 AVX-512/VNNI。
4. Serving 现在继续保护生命周期与观测契约；成本模型、增量准入和抢占等待相应压力证据，不为了把旧 TODO 打满而开发。
5. CUDA 第一版允许 cuBLAS、单 stream、同步完成、连续 FP16 KV、greedy。自研价值必须在模型执行、数据路径、资源生命周期和验证上成立。
6. 负性能结果可以关闭研究；不能用负结果豁免 correctness、资源安全或完整模型路径的验收。

## 4. 共享门禁

### 正确性

保留现有 CPU numerical、chunk/COW、SSE、状态机与 strict archive fixture。跨 GPU/CPU 使用新的、有版本的数值契约，不让现有位级同实现检查失效。有效权重、source/storage/activation/KV/accumulation dtype 必须分别记录。

### 实验

保留 micro / model / serving 三层。每个比较固定 workload、实际二进制、源码状态、配置及比较允许变化字段。初始化、模型加载、prefix setup 与 steady-state 分开。热状态包括 model、allocator、prefix 三种，不能混写。

已有归档复验在临时副本执行，禁止分析器先删除 tracked 汇总。完整实验包需包含 mandatory artifact，或提供可获取且校验 hash 的内容寻址依赖。missing artifact 不能降级成通过。

### 性能与停止

产品路径是否正确和性能是否优于 baseline 是两项结论。任何“加速”需超过预注册噪声门槛，并报告代表性退化项。M1 是首个 GPU baseline，不要求超过 cuBLAS 或 llama.cpp；不得把只有 kernel 成功当作模型路径完成。

## 5. Milestones

### V2-M0 — 证据可交付与针对性 oracle

| 字段 | 规范 |
|---|---|
| Title | 补齐现有证据的发布闭环，不重做 manifest |
| Motivation | 当前严格验证器依赖的 source ZIP 至少在一个发布目录缺失；GPU 即将需要跨算术路径检查 |
| Current Evidence | E08/E09；context manifest、目录清单、`Check-Snapshot` |
| Question To Answer | 独立工作区能否获取完整实验依赖并重新验收？新 GPU 数据路径使用哪份有效权重和 numerical contract？ |
| Dependencies | Audit HEAD；无 GPU 新产品依赖 |
| Scope | 非破坏性 availability 检查、完整 bundle 导出、targeted oracle fixture |
| Non-Goals | 恢复不存在的历史源码；改写历史 hash；全面替换 PowerShell/Python 工具 |
| Implementation Tasks | INFRA：列举缺失/损坏依赖与 export gate；MEASUREMENT：复验一个完整新基线包；INFRA：固定 GPU 验证用例与算术元数据 |
| Affected Files | `scripts/Benchmark-Common.ps1`、两个 Analyze 脚本、相关验证 fixture、`docs/BENCHMARKS.md`、`docs/VALIDATION.md` |
| New Interfaces / Data Structures | `Test-EvidenceAvailability.ps1`、`Export-BenchmarkBundle.ps1`、有版本 validation case manifest |
| Experiments | 故意删除 ZIP、修改 source-state/hash、迁移目录；在无原绝对路径的副本上验收 |
| Correctness Gate | 历史缺件明确失败/不完整；既有原始文件和汇总不得被检查流程修改 |
| Performance Gate | 无加速目标；不在请求或 Runtime 热路径加入检查 |
| Acceptance Criteria | 新包全部 mandatory artifacts 可得；身份与输出验收通过；missing/tamper 反例失败 |
| Failure / Stop Conditions | 历史原 ZIP 不可得时登记 ARCHIVE_INCOMPLETE 并停止追索；改采集新包，不能无限阻塞 GPU |
| Artifacts | `evidence-availability.json`、bundle manifest、复验报告、targeted cases |
| Portfolio Value | 区分“记录存在”与“独立可复核”的实验工程能力 |
| Difficulty | S–M |

### V2-M1 — Own CUDA：GGUF 到真实生成 token

| 字段 | 规范 |
|---|---|
| Title | Resident CUDA Qwen3 vertical slice |
| Motivation | 自有 GPU 模型执行仍是结构性缺口；工具链已经验证，无需等待 CPU 或 admission 完结 |
| Current Evidence | E01/E02/E08/E10/E11；具体执行由 `NEXT_SPEC.md` 定义 |
| Question To Answer | 是否能在不调用上游模型 forward 的条件下，控制 GPU 权重、workspace、连续 KV，执行完整模型并生成 token？ |
| Dependencies | M0 的新实验包与 numerical contract；现有 CPU oracle；本机 CUDA 环境 |
| Scope | 独立 CUDA target、RAII、一次性 FP32 effective-weight residency、cuBLAS、基础自有算子、连续 FP16 KV、完整 28 层、greedy、CLI |
| Non-Goals | PagedAttention、multi-stream、CUDA Graph、Q8 custom GEMM、GPU prefix sharing、异步 Serving |
| Implementation Tasks | INFRA：目标/所有权/共享 host model；PRODUCT：完整 device forward 与 token；MEASUREMENT：正确性、内存、复制与 timeline |
| Affected Files | CMake、`runtime.h/.cpp` 的小范围 model binding 提取、新 `include/minillm/cuda/`、`src/minillm/cuda/`、CUDA CLI/bench/tests |
| New Interfaces / Data Structures | immutable Qwen3 model、Tokenizer adapter、CudaContext、DeviceBuffer、CudaWeights、CudaWorkspace、ContiguousKV、CudaRuntime |
| Experiments | kernel/单层/整模型验证；固定 context decode；real-shape prefill；NSys 检查全模型路径 |
| Correctness Gate | GPU numerical contract 通过；正常/失败生命周期正确；Compute Sanitizer；旧 CPU/HTTP 无回归 |
| Performance Gate | 稳态零逐层权重/hidden H2D-D2H，零自有热路径 device allocation；性能如实报告，不要求击败上游 |
| Acceptance Criteria | 上游 GGML CUDA 关闭时自有 CUDA CLI 仍能生成；完整路径可追踪；S=1 与 S=4 限定配置通过 |
| Failure / Stop Conditions | 数学或资源 gate 失败不得晋升；性能较差保留 baseline 并定位，不顺手写 custom GEMM |
| Artifacts | weight/memory plan、GPU validation、model benchmark、copy counters、NSys 摘要及可获取原始产物身份 |
| Portfolio Value | 自研 CUDA model execution、显存所有权、resident pipeline |
| Difficulty | XL |

### V2-M2 — CUDA 进入已有 LLMServe

| 字段 | 规范 |
|---|---|
| Title | MiniCudaRunner 与真实 online serving |
| Motivation | GPU kernel/model benchmark 不等于在线系统；需要验证生命周期与请求路径 |
| Current Evidence | E04/E05；现有 ModelRunner 已有 resources/profile 扩展点 |
| Question To Answer | 连续 KV 的自有 GPU Runtime 能否在明确容量边界内接入 continuous batching、取消、超时和 SSE？ |
| Dependencies | M1 完整模型 gate；M0 bundle gate |
| Scope | 新 backend 选择、MiniCudaRunner、同步 execute、限定 active/context 容量、HTTP/SSE 与 replay；prefix cache 初版关闭 |
| Non-Goals | GPU prefix alias、假装连续 KV 是物理分页、提前异步化、重写 HTTP |
| Implementation Tasks | INFRA：能力与资源 schema 版本；PRODUCT：adapter 与 server selection；MEASUREMENT：GPU serving/cancel/backpressure |
| Affected Files | `model_runner.h`、`telemetry.h`、新 `mini_cuda_runner.cpp`、`server_main.cpp`、`http_server.cpp`、`telemetry_output.h`、相关分析器/tests |
| New Interfaces / Data Structures | layout-aware resource report、backend capability、GPU timing 字段；复用现有 execute/resource 语义 |
| Experiments | underload/saturation/burst；确定性 mixed；断连/超时/慢消费者；CPU 与 GPU 分开统计 |
| Correctness Gate | 每请求有限且单一终态；execute 完成前不回收设备资源；无 CUDA 错误；prefix 功能不支持时显式拒绝/配置关闭 |
| Performance Gate | Client TTFT/TPOT/ITL/goodput 与 device timeline 分开；无隐式 CPU fallback |
| Acceptance Criteria | 至少一组完整、可复验的 own-CUDA serving trace；所有相关 HTTP 语义检查通过 |
| Failure / Stop Conditions | 需要全局重写 Engine 才接入时缩减 adapter 范围；不因单次吞吐低就放松正确性 |
| Artifacts | `cuda-serving` bundle、HTTP 证据、capability/metric schema、同步生命周期文档 |
| Portfolio Value | GPU 模型与 C++ online serving 的真实整合 |
| Difficulty | L |

### V2-M3 — 有上限的 CPU 研究，不阻塞 GPU

| 字段 | 规范 |
|---|---|
| Title | 最多两个 CPU 闭环：小任务执行与 ENG-018 |
| Motivation | CPU 已具完整观测；继续投入必须回答具体假设，而不是“把 CPU 做完” |
| Current Evidence | E03/E06/E07；单 token norm 的线程池阶段可见成本；attention 仍是聚合阶段 |
| Question To Answer | 小任务是否应 inline 执行？在共享数学下，访问策略/页跨度与 QK-softmax-PV 各贡献什么？ |
| Dependencies | M0；使用稳定 CPU baseline。与 M1/M2 无串行依赖 |
| Scope | CPU-R1 只做一个执行策略候选；CPU-R2 修改现有 accessor-template bench、预分配 scratch、逐元素 oracle、粗粒度分解 |
| Non-Goals | 两套 attention 重写、无限 GEMM/ISA 优化、默认增大页、凭占比宣布 memory-bound |
| Implementation Tasks | MEASUREMENT：真实 count/grain/shape 回放；RESEARCH：同二进制对照与 A/A；PRODUCT：只有通过 gate 的一个候选才可默认启用 |
| Affected Files | `parallel.cpp`、必要的 `runtime.cpp`、`kv_cache_bench.cpp`、`kernel_bench.cpp`、kernel/state/model tests |
| New Interfaces / Data Structures | 最小执行策略选择；benchmark-local access policy 与 phase counters，优先不动公共 Runtime API |
| Experiments | T=1/8/16，B=1/实际小 batch；L=16/256/1024/1536，P=16/64/256；page size 与访问策略分开改变 |
| Correctness Gate | scalar/SIMD/布局逐元素输出、真实 logits、chunk/COW、线程池异常；未使用的尾部填 NaN 仍不污染结果 |
| Performance Gate | 产品候选需超过 A/A 噪声；模型级目标收益建议至少 5%，主要护栏退化不超过预注册阈值；micro-only 收益不默认推广 |
| Acceptance Criteria | 每项都提交 observation/hypothesis/control/result/conclusion；允许 negative 或 inconclusive 后关闭 |
| Failure / Stop Conditions | 每项最多两个主要设计变体；指标非热点或无模型收益则停止；matrix 研究只能替代 CPU-R1，不能增成第三条并行主线 |
| Artifacts | `cpu-r1`、`eng-018-controlled` 完整实验包与停止/推广决定 |
| Portfolio Value | 受证据约束的性能优化与负结果解释 |
| Difficulty | 每项 M–L |

### V2-M4 — GPU PagedKV 与自有 PagedAttention

| 字段 | 规范 |
|---|---|
| Title | 从连续 GPU baseline 进化到真实分页访问 |
| Motivation | 连续 KV 已可作同模型对照；分页必须证明数据路径、共享与安全回收 |
| Current Evidence | 自有 CPU PagedKV 的语义；M1/M2 新 GPU 证据 |
| Question To Answer | device query + device block table + physical GPU KV pool 是否能正确、高效地产生 attention output？ |
| Dependencies | M1、M2；完整 layout-aware resource contract |
| Scope | device pool、host refcount/allocator、device mappings、paged write/read、自有 kernel、共享与安全回收 |
| Non-Goals | CPU gather 全量 KV 后冒充分页 attention；multi-GPU；异步多 stream；强行复制 CPU 页物理布局 |
| Implementation Tasks | INFRA：页资源状态；PRODUCT：paged KV/attention；RESEARCH：有限 page/layout 选择；MEASUREMENT：容量/碎片/复用/latency |
| Affected Files | 新 `gpu_paged_kv.*`、`paged_attention.cu`、CudaRuntime、MiniCudaRunner、资源报告及测试 |
| New Interfaces / Data Structures | GpuKVPool、DeviceBlockTable、PageLease/页引用；保持同步完成契约，异步 lease 仅在以后需要时引入 |
| Experiments | 随机物理页排列、尾块、多个 sequence、GQA、prefix 分叉、取消；相同数学的连续 baseline |
| Correctness Gate | 页面引用守恒、已使用页不可提前复用、完整 logits/生成对照、memcheck |
| Performance Gate | 三层单独报告；允许分页带来小的 latency 成本但显著改善容量/复用，不要求每场景都更快 |
| Acceptance Criteria | 自有 Serving 请求确实经 device block table 访问物理 KV；不存在隐藏连续 gather 路径 |
| Failure / Stop Conditions | 收益不能覆盖复杂度时保留连续可选 baseline；不无限增加 kernel 变体 |
| Artifacts | GPU KV 状态模型、layout 对照、PagedAttention correctness、真实 serving bundle |
| Portfolio Value | GPU memory systems、页表访问、缓存共享与执行生命周期 |
| Difficulty | XL |

### V2-M5 — 面向实测瓶颈的 Serving 策略

| 字段 | 规范 |
|---|---|
| Title | 公平性、SLO、压力与条件性增量准入 |
| Motivation | 调度计算本身极小，价值在改变 batch cost、停顿及资源利用率 |
| Current Evidence | E05 的 TPOT/ITL trade-off；尚无充分持续公平性与容量压力证据 |
| Question To Answer | 哪些请求应入批、何时接纳、什么时候拒绝，才能在有限资源下改善明确的 SLO？ |
| Dependencies | MEASUREMENT 可在稳定 M2 后启动；增量 admission 必须等待真实资源账本及压力证据 |
| Scope | sustained priority/burst/long-short/memory-pressure trace；shadow batch-cost estimator；必要时 safe-state admission |
| Non-Goals | 优化微小 scheduler 排序开销；把每 token 成本视为相同；无限过载下承诺所有请求有限等待；放松现有 all-success 时间线协议冒充过载验证 |
| Implementation Tasks | MEASUREMENT：负载分层；RESEARCH：shadow 预测、简单策略；PRODUCT：有证据才晋升，增量准入有独立 gate |
| Affected Files | `engine.cpp`、`scheduler.cpp`、resource contracts、trace generator、benchmark validators；为允许失败的压力结果增加独立分析模式，保留当前 `analyze_telemetry.py` 的 all-success 契约 |
| New Interfaces / Data Structures | BatchCostEstimate、AdmissionDecision、可回收资源快照；优先数组/O(n) scan |
| Experiments | underload/saturation/overload、持续 priority、共享前缀、小 KV、长 decode；独立 held-out trace |
| Correctness Gate | no-overcommit、单终态、资源守恒；safety 与 progress 分开证明；共享页不能被重复计算为可释放资源 |
| Performance Gate | 预注册 goodput/等待或最大停顿改善；失败不删样本；不能让某一类请求饥饿换吞吐 |
| Acceptance Criteria | 策略适用范围明确；无法提升时继续使用保守 admission 与简单 policy |
| Failure / Stop Conditions | 未观察到容量瓶颈不做增量；预测不稳定不驱动调度；一个 block headroom 不作为完成性证明 |
| Artifacts | fairness/pressure 报告、shadow 误差、资源不变量、策略或停止决定 |
| Portfolio Value | 算法能力与实际系统常数、资源约束和 SLO 的结合 |
| Difficulty | Measurement M；Research/Product L–XL |

## 6. Priority Matrix

| 工作 | Information Gain | Technical Depth | Dependency Unlock | Portfolio Value | 实现/验证成本 | 风险 | 本机可验证 |
|---|---|---|---|---|---|---|---|
| M0 发布证据闭环 | High | Medium | Very High | High | Low/Medium | Low | Yes |
| M1 CUDA 完整模型 | Very High | Very High | Very High | Very High | High/High | High，范围需锁定 | Yes |
| M2 CUDA Serving | High | High | High | Very High | Medium/High | Medium | Yes |
| CPU-R1 小任务执行筛查 | High | High | Medium | High | Low/Medium | Low | Yes |
| CPU-R2 ENG-018 控制实验 | High | High | Medium | High | Medium/High | Medium | Yes |
| 额外 CPU GEMM/packing | Medium | High | Low | High | High/High | Medium | Yes，但条件触发 |
| M4 GPU paging | High | Very High | High | Very High | High/High | High | Yes，M1/M2 后 |
| M5 fairness/pressure | High | High | High | High | Medium/High | Medium | Yes |
| 立即增量 admission | Low，缺压力证据 | High | Low | Medium | High/High | High | 可测试但不应先做 |
| Radix/hash 重写 | Low | Medium | Low | Medium | Medium/High | Medium | Yes，当前不值 |
| multi-GPU/PD | Low | High | Low | 无本机实证时低 | Very High | High | 当前 No |

## 7. Dependency DAG

```text
                 M0 evidence + targeted oracle
                    |                |
                    v                +----> M3 CPU-R1/R2（条件、可负结果关闭）
                 M1 CUDA model
                    |
                    v
                 M2 CUDA Serving
                  /        \
                 v          v
          M4 GPU paging    M5 fairness/pressure measurement
                 \          /
                  \        /
                   v      v
            M5 conditional admission/product policy
```

可并行：M0 后的有限 CPU 研究与 M1；稳定 M2 后的压力观测与 M4。
必须串行：完整 GPU model → GPU online validation；连续 baseline → 分页因果对照。
条件触发：CPU packing、Radix/hash、cost-aware 上线、incremental admission、preemption。
可以删除：没有新证据收益的研究、重复 profiler/manifest/smoke 建设。

## 8. NOT NOW v2

暂缓：完整 OpenAI API、chat template 大工程、随机采样、更多 architecture、MoE、Q4 为做而做、Web UI、RAG/Agent、authentication/multi-tenancy。
作为条件研究而非承诺：Radix Tree、block hash、activation quantization、preemption、speculative decoding、CUDA Graph。
未来：multi-GPU、NVLink、PD disaggregation、distributed serving。
CPU 算法优先 O(n) scan/数组与简单 queue；只有真实规模和 profiler 证明必要时才增加复杂索引。
cuBLAS 可以保留为正式 matrix implementation；不以“所有 GEMM 必须自写”衡量自研程度。

## 9. 立即行动与终态证据

V2-M0 与 `NEXT_SPEC.md` 的 V2-M1 已验收；完整 CUDA 模型、真实 token CLI、模型 A/A/异构基线、Profiler 和可独立复核的证据包均已提供，入口为 [项目执行状态](EXECUTION_STATUS.md)。下一阶段为 V2-M2：将同步完成的自有 CUDA Runtime 接入现有 Serving，验证 HTTP/SSE、取消、超时与资源回收。V2-M2 尚未实施，GPU paging 与自有 PagedAttention 继续按依赖进入；模型基线的测量不确定项不改称已取得加速。

最终只保留六个能被证据证明的卖点：
1. 自有 CPU/GPU 模型执行与 matched-weight numerical validation。
2. 有 profiler、受控对照和 model/serving 验证的一个 CPU 性能闭环。
3. CPU/GPU KV 所有权、物理共享、页表与回收。
4. 自有 resident CUDA 数据路径和可解释的 kernel/memory 行为。
5. 有公平性与 SLO/压力边界的在线 serving。
6. 可独立获取、校验、重放的实验与正确性证据。

任何一项缺代码或证据，就写“未完成/未验证”，不写“高性能”“生产级”替代事实。
