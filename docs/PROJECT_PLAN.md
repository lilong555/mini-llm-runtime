# MiniLLM / LLMServe 项目计划

项目主线：解释并优化现有自研 CPU 推理系统，建立同一 Serving Engine 驱动的自研 CUDA 执行路径，以代码、数值验证和可复现实验支撑每项能力。

本计划定义目标、阶段依赖和验收条件。当前可用功能见 [README](../README.md)，逐项能力验收见 [规范问题清单](PROBLEM_CHECKLIST.md)，实测结果见 [验证记录](VALIDATION.md)，实际工程问题见 [工程台账](ENGINEERING_LOG.md)。计划中的新接口、文件和 CUDA 组件均为待实施设计，不代表当前已有实现。

## 1. 范围与目标

### 当前实现基线

源码基线：`7e5e84f4e63ae12fbbcee5c6365711e229b0cfef`。这是本计划对应的实现检查点；每次开发和实验仍须记录自己的实际源码提交、工作区状态与二进制身份。

当前产品为 C++20、单节点、单模型、纯文本推理与在线服务，使用指定 Qwen3-0.6B Q8_0 模型，支持贪心采样。MiniLLM 自行完成 CPU forward 和物理分页 KV；llama.cpp 提供 GGUF 元数据解析、tokenizer 及独立 CPU/CUDA 对照后端，HTTP 传输依赖 cpp-httplib。详细归属见 [THIRD_PARTY.md](../THIRD_PARTY.md)。

| 模块 | 已有实现 | 后续工作 |
| --- | --- | --- |
| GGUF | 只读 mmap、TensorView、形状与边界检查，上游元数据解析 | 扩充 F16/Q8 fixture、畸形形状和范围检查 |
| CPU kernels | F32/F16/Q8_0 点积，AVX2/FMA/F16C 分派与 scalar fallback | 真实矩阵形状的分块、转换成本与跨 token 权重复用 |
| Attention | 自有 QK、softmax、FP16 V 加权累加；K/V 两条路径均接入 SIMD | 按页遍历、online softmax 和长上下文访问成本 |
| 线程池 | 持久线程、调用线程参与、原子分发、异常传播 | 任务粒度、唤醒、尾部等待与线程扩展测量 |
| Qwen3 Runtime | GQA、Q/K RMSNorm、NeoX RoPE、SwiGLU、FP32 accumulation | 分阶段计时、批量 LM head、真实 F16/F32 模型覆盖 |
| CPU Paged KV | FP16 物理页、页表、引用计数、free list、共享与尾页 COW | 统一物理资源观测，验证布局与访问优化 |
| Prefix cache | token Trie、命名空间、完整块复用、LRU | 冷热前缀与索引成本测量，按证据决定是否研究 Radix/hash |
| Scheduler | 迭代级组批、混合 prefill/decode、chunk、priority aging | 持续负载公平性、batch 成本预测与时间预算 |
| Admission | `BlockPool` 容量信用，按 prompt + 输出上限保守预留 | 资源压力证据、安全增量准入和可前进性证明 |
| HTTP/SSE | 有界队列、取消、超时、断连回收、UTF-8 缓冲、背压 | 更长真实 socket 压力与故障组合验证 |
| 实验与测试 | dot/KV 微基准、HTTP replay、模型对照、跨平台核心 CI | 严格结果验收、模型级 benchmark、确定性混合批验证 |
| CUDA | llama.cpp 的 CPU/CUDA 可切换参照后端 | 自有 device memory、resident weights、forward、GPU KV 和 PagedAttention |

`Runtime::Impl::multiply()` 已具有批量矩阵语义，通过输出行与输入 token 循环调用 dot。该结构存在权重行缓存复用机会，但尚无显式跨 token 的寄存器级权重解码复用。LM head 对每个需要 logits 的 token 单独调用 `multiply(..., count=1)`；它是待测候选热点，尚无阶段 profile 支持瓶颈结论。

### 已有证据与待验证问题

| 事项 | 当前证据 | 计划约束 |
| --- | --- | --- |
| V 加权累加 SIMD | `ENG-016` 有实现、核心测试和真实模型验证记录 | 保留为已有能力，后续研究关注页遍历及模型级收益 |
| 混合调度 | `ENG-008`：历史 CPU trace 上 mixed / prefill-first 吞吐为 18.40 / 18.59 token/s，P95 请求平均 TPOT 为 398.89 / 352.85 ms，goodput 均为 0 | 研究 batch 组成、模型阶段与 ITL，不能推广为 CPU mixed 必然无效 |
| CPU KV 差距 | `ENG-018`：1536-token prompt 下 MiniLLM / llama.cpp 中位 TPOT 为 49.26 / 33.84 ms；等算术布局隔离测得分页耗时约为连续布局的 1.20–1.35 倍 | 分离页查找、QK、softmax、PV 与矩阵成本；完整后端差距还包含 ISA、FlashAttention 等因素 |
| 页大小实验 | 256-token page 的顺序实验存在明显耗时漂移 | 保留原始数据，采用交替顺序并记录频率、温度后再判断 |
| 混合批测试 | `ENG-017`：`mixed_batches > 0` 断言受请求提交与执行时序影响 | 用可控屏障或确定性注入建立混合状态，不能以偶然通过关闭问题 |
| 数值参照 | Q8_0 权重解量化得到 F32 reference，已完成限定输入对照 | reference 的 F32 不等于被测完整 F32 模型已经验证 |
| 实验身份 | 初始 SIMD/调度报告缺精确源码 SHA；CPU KV 汇总已有源码、二进制和模型身份 | 统一 manifest 与严格校验，保留各份历史报告原有证据边界 |

以上数字引用仓库归档，不代表基线提交的重新测量。CPU KV 报告的源码字段为 `600a1b93cdc95aa11dcbcb56d52d73ad6d5ae8ea`，详见 [原始汇总](../benchmarks/results/kv-cache-cpu/summary.json)；其他结果的条件见 [VALIDATION.md](VALIDATION.md)。现有 CI 归档只证明对应提交与测试范围，不能代替当前提交的模型、HTTP 或 GPU 验证。

调度第 0 轮还有不同方向的结果：mixed / prefill-first 的 P99 单次 ITL 为 1217.03 / 2769.41 ms，见 [mixed 原始报告](../benchmarks/results/mini-scheduling/mixed-0.json) 与 [prefill-first 原始报告](../benchmarks/results/mini-scheduling/prefill_first-0.json)。该单轮结果不足以证明稳定的尾延迟优势，但说明请求平均 TPOT 与单次 token 停顿必须分别研究。原短上下文 trace 也不能代替 KV 容量压力实验；解释 `ENG-008` 与评估增量准入需要不同的控制变量。

### 目标系统与所有权

终态为单节点、单模型、单 GPU 的可验证推理研究系统：保留自研 CPU 参考路径，建立自研 CUDA 路径，由统一的请求生命周期和 Serving 语义驱动。llama.cpp 继续作为独立对照后端。

```text
HTTP / SSE / CLI / Replay
             |
      Request Lifecycle
  queue / cancel / timeout
             |
       Serving Engine
 scheduling / admission / prefix
             |
   ModelRunner semantic contract
       /          |          \
 Mini CPU     Mini CUDA     Llama Runner
    |          (规划)           |
 Own forward  Own forward   llama_decode
 SIMD/pool    own ops +     upstream CPU/CUDA
    |         vendor GEMM
 CPU PagedKV      |
              GPU KV pool

共享：模型配置、输入/position/logits 语义、验证与实验字段
独立：tensor layout、allocator、CPU 线程与 GPU stream/event
```

保留 `ModelRunner`，按实际需要增加能力查询与资源快照；不提前引入通用计算图框架。拟议类型包括 `BackendCapabilities`、`RuntimeResourceSnapshot`、`KvAllocationPlan`。后端无法提供的指标须表达为未知或不支持，不能用估算值冒充实际物理页统计。

自研 CUDA 初版采用同步完成契约：`execute()` 返回时，结果可用，本轮访问的资源不再被 GPU 使用。只有时间线证明同步限制吞吐后，才研究 completion ticket 和延迟回收。

## 2. 路线与优先级

| 阶段 | 定位 | 进入条件 | 主要交付 | 难度 |
| --- | --- | --- | --- | --- |
| Phase 0：可信基线 | P0，近期 | 现有 CPU 产品可构建运行 | 统一 manifest、严格验收器、可重复基线和扩展数值验证 | M |
| Phase 1：CPU 性能归因 | P0，近期 | 实验身份与输入可校验 | Runtime benchmark、stage profile、ENG-008/ENG-018 假设报告 | M–L |
| Phase 2：CPU 执行优化 | P1，由测量选择 | 热点明确，相关正确性覆盖可用 | 一到两个有模型级证据的优化或完整负结果 | L |
| Phase 3：资源与调度 | P1，条件推进 | 资源可观测，公平性/压力负载可重复 | 资源不变量、fairness 报告、shadow cost model，条件性增量准入 | L–XL |
| Phase 4：完整 CUDA 路径 | P2，后续主线 | CPU 参考与实验协议稳定 | resident weights、连续 GPU KV、完整 forward、Serving 接入 | XL |
| Phase 5：GPU 分页与 attention | P2，后续主线 | 完整 GPU 模型和连续 KV 对照可用 | device 页池、自研 PagedAttention、真实 HTTP replay | XL |

Phase 0 的相关门槛逐项满足后可进入 Phase 1；任何计算优化必须先满足其数值门槛。Phase 2 与 Phase 3 可按依赖分别推进，CPU 优化后须重新校准调度成本。完成有限的 CPU 优化研究即可进入 Phase 4，不要求 CPU 达到极限，也不以增量准入取得收益为前提。Phase 5 的分页正确性不依赖更激进的 admission 策略。

Prefix 索引属于条件研究：只有索引内存、查询、淘汰成本的测量触发后才进入，不阻塞 CUDA 主线。难度表示工程复杂度，不构成交付日期承诺。

## 3. 分阶段实施与验收

### Phase 0：建立可信实验与正确性基线

目标：每份结果能定位到源码、二进制、模型、输入、配置与算术条件。

工作项：

1. 统一 `BenchmarkRunManifest` 或等价 JSON schema，把已有 CPU KV 实验的身份字段扩展到所有测量入口。
2. 强化 `scripts/Analyze-Benchmarks.ps1`：验证完整请求集合、唯一 ID、终态与输出完整性、模型/trace 身份、关键配置和报告数量；输出 mismatch 必须导致正确性验收失败。
3. 显式声明每次比较允许变化的字段。策略对照使用相同二进制、只改变 policy；优化前后可比较不同源码/二进制，但必须标识 baseline/candidate，固定其余条件。
4. 在实际提交上重建 dot、CPU KV、纯 prefill/decode 和调度基线；保留历史报告，报告多轮波动。
5. 参数化真实模型验证，明确被测 F32/F16/Q8 权重身份、来源及参照算术；扩展长 context、长 decode、多语言、特殊 token、chunk 与页边界。
6. 解决 `ENG-017` 的测试确定性，保留真实混合执行验证；单线程偶然通过不构成完成证据。
7. 保留跨平台核心构建与 sanitizer；模型、HTTP 和 GPU 采用各自明确的本地或 CI 验证矩阵。

主要入口：`apps/bench_main.cpp`、`scripts/Benchmark-Policies.ps1`、`scripts/Analyze-Benchmarks.ps1`、`tests/model_tests.cpp`、`tests/gguf_tests.cpp`、模型 manifests、`docs/VALIDATION.md`。

每次运行至少记录：

```text
git_sha / git_dirty / binary_sha256
compiler_and_flags / dependency_commits
model_sha256 / reference_identity / trace_file_sha256
backend / kernel_mode / weight_dtype / activation_dtype / kv_dtype
threads / batch / chunk / KV capacity / prefix / policy
CPU / GPU / driver / OS / warmup_protocol
start_time / trial / seed / profiler_mode
```

正式对照使用可还原的源码状态；若工作区有未提交改动，须保留其差异身份，不能只记录 `git_dirty=true` 就视为可复现。性能进程不额外常驻 reference model，数值对照单独运行。

验收：缺失或重复请求、token mismatch、不兼容配置、错误 reference、未声明的二进制差异均被拒绝；正常报告通过，所有相关回归通过。压力实验中的预期拒绝/超时可保留为合法终态，但不得遗漏或伪装为成功。本阶段不要求性能提升；无法确定身份或输入等价时停止计算加速比。

### Phase 1：解释 CPU 执行成本

目标：把 Serving 延迟连接到实际 batch、模型阶段和线程行为，验证或排除 `ENG-008`、`ENG-018` 的主要假设。

工作项：

1. 增加轻量 `ProfileScope`、`StageCounters` 和 `BatchTelemetry`；使用固定 stage ID、预分配或 thread-local 缓冲，热循环不拼接字符串或写 JSON。
2. 新增拟议 `apps/runtime_bench.cpp`，直接驱动 Runtime，覆盖纯 prefill、固定 KV 长度 decode 和固定多序列 mixed 输入。
3. 分别计时 embedding、norm、Q/K/V projection、Q/K norm、RoPE、KV store、QK、softmax、PV、output projection、FFN、final norm、LM head 和 sampling；小阶段可合并，LM head、attention、projection 和线程池行为必须可区分。
4. 每轮记录 prefill/decode/logits token 数、sequence 数、context sum/max、matrix shape、调度与执行时间，并关联客户端 token 时间。
5. 比较 1/2/4/8 线程，单独解释 16 SMT 线程；采集函数热点、线程时间线以及环境支持的硬件事件。
6. 重放原调度 trace、低到达率与长 context 负载；CPU KV 采用已有等算术布局基准，并补充交替页大小实验。

主要入口：`src/minillm/runtime.cpp`、`src/minillm/parallel.cpp`、`src/mini_runner.cpp`、`src/engine.cpp`、`src/scheduler.cpp`、`apps/bench_main.cpp`、`apps/kv_cache_bench.cpp`。

| 假设 | 所需证据 | 排除条件 |
| --- | --- | --- |
| LM head 逐 token 执行增加扫描 | 时间占比、logits 数、矩阵调用次数 | 占比低，或批量输出层对照无模型级收益 |
| 小任务唤醒/同步过重 | count/grain 分布、空任务基准、worker 时间线 | 工作线程持续有效计算，固定成本占比低 |
| mixed 形成不利矩阵形状 | 实际 B/M/K 分布、对应 latency | 相同形状成本相近，差异来自其他阶段 |
| KV 页访问放大 attention 成本 | 页查找、QK/PV、context 长度与布局隔离 | 控制布局后模型差距仍在其他阶段 |
| 临时分配或元数据发布较重 | 分配次数/字节、调用栈、复用对照 | 分配占比低或复用无模型收益 |
| scheduler 自身成本较高 | runner 外的调度与准入时间 | 调度只占很小比例 |

验收：固定输入下 profiling 开关不改变数学输出与资源语义；分别报告其时间开销和对在线 batch 组成的扰动。stage 时间加未归类时间能解释 forward wall time，不能将共享矩阵时间同时完整归入 prefill 和 decode。至少验证或排除一个主要假设；只有总 CPU 利用率或采样开销掩盖行为时，不进入大规模优化。

### Phase 2：完成有限的 CPU 优化研究

目标：选择 profiler 支持的一个主优化，形成数值、内核、模型和 Serving 的完整对照后，再决定是否做第二项。

| 候选 | 进入条件 | 首个最小实现 |
| --- | --- | --- |
| 批量 LM head | 输出层占比与重复调用显著 | 汇集 hidden rows，一次矩阵计算，再恢复 sequence/sample 映射 |
| 小任务串行 cutoff | 唤醒和 barrier 固定成本显著 | 按 operation/shape 选择串行或线程池 |
| Q8×F32 多 token 内核 | 权重重复解码和矩阵成本显著 | 一次解码权重块，更新 2/4 个 token accumulator |
| F16/F32 tiled matrix | 真实 prefill batch 有足够复用 | 有限 M/N/K tiling，保留 B=1 路径 |
| Scratch 复用 | 临时分配或初始化成本可见 | 显式 Runtime workspace 与生命周期 |
| 按页 attention / online softmax | 页查找和访问成本显著 | 页内批量访问、减少 accessor 调用，保留现有 V SIMD |

拟议组件：最小 `MatrixView/MatrixShape`、`CpuMatrixPlan`，按需要增加 `matrix_kernels.*`、`RuntimeScratch`、只读 packed weight owner。微基准使用实测 Q/K/V、FFN 和 LM head 形状，覆盖 B=1/2/4/8/16/32/64/128 的适用点、尾块、hot-cache/streaming 与线程扩展。

验收：保留可切回的 baseline；同算术 scalar 对照、非整 tile、真实 logits、chunk/COW、HTTP 输出全部通过。分别报告 kernel、模型 prefill/decode、Serving TTFT/TPOT/ITL/goodput 和峰值内存。只有预先指定的代表性模型指标改善且主要场景无不可接受退化，才切换默认路径。无收益的研究可以完成，不要求保留复杂实现。

Packing 必须计入启动时间和 mmap + packed 副本的总驻留成本。仅当单次收益为正时，摊销调用数为 `N = T_packing / (T_old_call - T_new_call)`；短进程不能只用热稳态结果评价。

Q8 activation quantization 是独立研究项。先定义 scale 粒度、rounding/clipping、整数累加、解量化和质量预算，再分别验证：与同规则 scalar reference 的实现一致性，以及相对 F32 activation 的 logits、teacher-forced loss 和生成变化。不得放宽现有阈值来掩盖算术变化。

### Phase 3：资源可观测与可前进的调度

目标：先证明压力来源和资源不变量，再决定是否改变准入与调度策略。

主要入口：`include/llmserve/model_runner.h`、`src/mini_runner.cpp`、`src/minillm/paged_kv.cpp`、`src/engine.cpp`、`src/scheduler.cpp`、`src/block_pool.cpp`、`src/prefix_index.cpp`。

工作项：

1. 发布 `RuntimeResourceSnapshot`，分别表达逻辑信用、物理唯一页、resident buffers、共享引用、可回收缓存、active/in-flight pin 和 COW 次数/字节。
2. 建立小 KV 池、长 prompt/output、持续到达、priority flood、cache eviction、取消/超时的压力与公平性 trace。
3. 增加 `BatchCostEstimator` 的 shadow prediction，只记录预测；在独立 trace 上确认误差后，才研究 iteration-time budget。
4. 有真实 reservation slack 和容量受限证据后，再实施 `KvAllocationPlan`、`AdmissionDecision` 与安全增量准入原型。

资源口径：`reserved_unique_blocks` 属于容量信用；`PagedKV::used_pages()` 是活跃物理页；`resident_bytes()` 包含已分配并保留的缓冲。`used_pages()==0` 不代表内存归还操作系统。保留 `BlockPool` 的账本职责，不把信用页当成物理分配器。

当前模型 FP16 KV payload 为 `2 × 28 × 8 × 128 × 2 = 114688 bytes/token`，即 112 KiB/token；16-token 页为 1.75 MiB，8192-token 容量为 896 MiB。容量不等于已分配量，进程 Private Bytes 也不等于 KV payload。COW 需计入整物理页复制成本。

对一次追加，资源计划须包含页增长和真实共享状态引起的 COW：

```text
ΔP_i = ceil((length_i + new_tokens_i) / page_tokens)
       - ceil(length_i / page_tokens) + cow_pages_i

构建计划 → 检查可用/可回收页 → 预留本批资源 → 发布映射并执行
                                                    → 提交或完整回滚
```

该不变量只保证受管理 KV 池不超配；底层 CPU/CUDA 内存分配仍可能失败，必须定义错误清理与终态。

Decode headroom 不能单独证明无活锁。先研究保守的完成顺序检查：存在一个请求完成顺序，使每一步的最坏额外需求均可满足，且完成后的保证可回收页足以推进下一请求。共享页仅在最后一个引用结束后释放，cache insertion 也不得破坏安全状态。压力时停止新准入并进入显式 drain mode，优先推进可完成请求。

若 `F` 是保证可用页，`r_i` 是请求额外需求，`q_i` 是其完成后保证释放的页（包含本次额外分配中可释放的部分），模拟条件为 `r_i <= F`、`F_next = F - r_i + q_i`。每步重新计算共享引用；不能将某请求全部持有页直接计入释放量。验证还须覆盖底层分配失败、取消、超时与缓存淘汰。

Batch 成本从以下组成估计：

```text
T_batch ≈ T_projection(B, shape, dtype, threads)
        + T_attention(sum of attended context lengths)
        + T_LM_head(logits_tokens) + T_pool + T_other
```

采用 bucket table、分段线性或简单非负拟合即可。共享权重与线程固定成本使 batch 成本不能直接等于逐 token 成本相加。

公平性分别定义 waiting admission、active prefill、active decode、跨 prefill/decode 类别与 priority 契约。无限持续过载下允许有界拒绝或限流，不能同时承诺全部接纳、有限等待与有界内存。

验收：随机状态和对抗负载证明引用守恒、无部分提交、在用页不回收、有限合法工作集可完成、故障后资源合法。增量准入只在真实压力下改善预注册指标时启用，资源宽裕时不得明显增加成本。预测失效、无法证明进展或策略退化为更复杂的保守预留时停止该研究，继续保留保守策略。

### Phase 4：自研 CUDA 完整模型路径

目标数据路径：`GGUF → resident GPU weights → own Qwen3 forward → GPU KV → greedy sampling → LLMServe`。

拟议目录：`include/minillm/cuda/`、`src/minillm/cuda/`、`src/mini_cuda_runner.cpp`，通过独立 CMake target 构建；CPU-only 产品继续不依赖 CUDA。

| 步骤 | 交付 |
| --- | --- |
| CUDA-A | 构建、错误处理、move-only `DeviceBuffer`、`CudaStream`、`CudaEvent`、单 stream 与算子正确性框架 |
| CUDA-B | `CudaWeights` 和 `CudaWorkspace`、权重一次上传、device resident 数据与 cuBLAS 矩阵基线 |
| CUDA-C | embedding、RMSNorm、RoPE、residual add、SwiGLU、softmax、KV write、LM head、argmax，完成单层对照 |
| CUDA-D | 连续 GPU KV 与完整 `CudaRuntime`，逐层和最终 logits 对照 |
| CUDA-E | 接入 `ModelRunner` 和真实 HTTP/SSE，保持同步 execute 完成语义 |
| CUDA-F | Phase 5 的物理 GPU 页池、页表与共享生命周期 |
| CUDA-G | Phase 5 的自研 PagedAttention 与按实测需要推进的 batching 优化 |

CPU/GPU 共用数学与请求语义，独立维护实际 tensor layout、工作区和资源所有权。cuBLAS 作为 vendor 矩阵基线不改变自有模型执行、KV 与 Serving 的归属；不要求每个手写 GEMM 都超过 vendor 实现。

数值报告固定有效权重、weight/activation/KV dtype、accumulation 与 math/compute mode。调试路径可取回完整 logits；性能路径尽量只回传 token ID 和必要状态，分别记录 startup、H2D/D2H、launch、同步、显存和 steady-state 延迟。

验收：算子、单层、整模型、chunk/sequence 与错误销毁路径通过相应检查；稳态不逐层搬运权重或 hidden states；自有 CUDA forward 真正进入 Serving。只有 standalone kernel、最终仍由 llama.cpp forward、资源销毁时仍被使用，均不满足阶段完成条件。

### Phase 5：GPU 分页 KV 与自研 PagedAttention

目标：在完整连续 GPU KV 模型基线上，测量分页的容量、共享收益与访问代价。

工作项：

1. 实现 `GpuKVPool`、host allocator/refcount、`DeviceBlockTable` 与 paged KV write；CPU/GPU 页 ID 使用明确的类型边界。
2. 根据 GPU 的 head、token、block 访问方向确定布局，不机械复制 CPU 页中容纳所有层的布局。
3. 实现自有 CUDA attention，使 device query 通过 device block table 直接读取非连续物理 KV，输出仍位于 device。
4. 对照连续 KV：随机打乱物理页，覆盖 page size、尾页、因果 mask、GQA、context/batch、prefix share、COW、取消和压力恢复。
5. 初版在单 stream 已知完成点释放页；引入异步或多 stream 时，增加 event/completion 与必要的 `InFlightPageLease`，防止提前复用。
6. 完成真实 HTTP replay，分别报告 attention latency、显存利用率、碎片、映射上传、复制量、TTFT/TPOT/ITL/goodput。

验收：随机非连续映射与连续 reference 数值一致，在用页不回收，失败无重复终态或泄漏。Serving 使用自有 paged data path，且至少在声明的负载上证明容量、复用或性能价值。将全部 KV gather 成连续数组再执行普通 attention 不满足自研 PagedAttention 的验收定义。

## 4. 近期任务队列

以下是待实施任务。顺序表达主依赖，未满足进入条件的研究项保持候选状态，不影响已具备条件的其他阶段。

| ID | 任务 | 依赖与优先级 | 完成定义 |
| --- | --- | --- | --- |
| PLAN-001 | 统一实验 manifest 与严格验收 | P0，立即 | 身份完整；缺失/重复、配置差异与 mismatch 反例被拒绝；正常基线归档 |
| PLAN-002 | Runtime profiler 与模型级 benchmark | P0，依赖 PLAN-001 的实验身份 | prefill/decode/mixed 阶段报告可解释，profiling 开销和输出一致性可核验 |
| PLAN-003 | Batch telemetry 与 ENG-008/ENG-018 归因 | P0，依赖 PLAN-002 | batch 与 token 时间关联，包含 ITL、页访问对照和可证伪假设 |
| PLAN-004 | 扩展模型验证并解决 ENG-017 | P0，可与观测工作并行 | 被测 dtype/reference 明确；长 context、页边界、特殊 token 与确定性混合测试通过 |
| PLAN-005 | 真实 shape 的矩阵与线程池微基准 | P1，依赖 PLAN-002/003 | 真实 projection/LM head、B、线程数、热缓存/streaming 与空任务对照 |
| PLAN-006 | 实施排名第一的 CPU 优化 | P1，依赖 PLAN-003/004/005 | 保留 baseline；kernel/model/serving 分开验收；负结果也完整归档 |
| PLAN-007 | 物理 KV 快照与守恒检查 | P1，依赖实验与相关正确性基线 | live/resident/shared/evictable/COW 口径明确，共享与清理守恒 |
| PLAN-008 | Fairness/pressure replay 与 shadow cost model | P1，依赖 PLAN-002/003/007 | 独立 trace 的预测误差、优先级等待、内存压力与恢复报告；CPU 路径变化后重新校准 |
| PLAN-009 | 安全增量准入原型 | 条件研究，依赖 PLAN-004/007/008 | safe-state/drain 规则与状态测试通过，和保守策略完成压力对照 |
| PLAN-010 | 独立 CUDA target 与 resident 验证入口 | P2，依赖稳定 CPU 数值/测量基线 | CPU-only 独立；CUDA RAII、驻留与错误路径可测，按 CUDA-A～E 推进完整模型 |

近期先执行 PLAN-001、PLAN-002、PLAN-003，PLAN-004 在计算优化前完成对应覆盖。PLAN-009 无收益或研究周期过长不阻塞 PLAN-010；CPU 阶段按一到两个完整研究闭环控制投入。

## 5. 前三项任务的交付规格

### PLAN-001：实验身份与验收器

阅读入口：`apps/bench_main.cpp`、两个 benchmark PowerShell 脚本、`benchmarks/results/environment.json`、CPU KV 汇总、`docs/VALIDATION.md`。

增加源码/二进制/model/trace 身份、完整 `EngineConfig`、线程与 kernel mode、轮次、seed、warmup；这些信息与每份原始报告关联。报告校验使用输入 trace 的预期请求集合，不能只取第一份可能已不完整的结果为真值。比较前验证终态和输出完整性，并列出允许变化的字段。

拟议产物：

```text
benchmarks/results/<run_id>/
    manifest.json
    validation-summary.json
    mixed-0.json
    prefill_first-0.json
    summary.json
```

验收反例至少包括：删除请求、重复 ID、改变模型 hash、改变不允许变化的配置、破坏输出 token。它们均须被拒绝；正常报告通过。验收器缺口对应 `ENG-020`，实现完成前保持待解决。

### PLAN-002：模型分阶段计时

阅读入口：`src/minillm/runtime.cpp`、`src/minillm/parallel.cpp`、`src/mini_runner.cpp`、`include/minillm/runtime.h`。

首版字段：`batch_id`、`stage`、`layer`、`wall_ns`、输入与 logits token 数、matrix M/N/K、线程数、parallel count/grain。使用 Release 加符号，不在 dot 内增加日志或全局高频原子操作。

拟议 `apps/runtime_bench.cpp` 覆盖短/长 prefill、固定 KV 长度 decode、固定 token 的多序列输入；对应 `runtime-prefill.json`、`runtime-decode.json`、`forward-stages.json`、`profiler-overhead.json`。

验收：开启/关闭 profiler 的固定输入输出一致，耗时与未归类部分可解释总时间，小 batch 和大 batch 的阶段组成可比较，开销单独量化。CPU waiting time 与 worker useful work 分开解释。

### PLAN-003：调度与执行时间线

阅读入口：`src/engine.cpp`、`src/scheduler.cpp`、`apps/bench_main.cpp`、`benchmarks/traces/cpu-mixed-s0.jsonl`、调度与 KV 原始结果。

每轮记录 policy、waiting/active 数、prefill/decode/logits token 数、sequence 数、sum/max KV length、scheduler/runner 时间与资源快照。需要模型侧 replay 时，同时记录 sequence create/share/clear 事件；只有 batch size 不足以重建 KV 生命周期。物理快照尚未实现时显式标记缺失，不以信用统计代替。

保留原 trace，支持小数到达率或 arrival-time scaling，增加低负载和长上下文对照。分析同时包含 TTFT、每请求平均 TPOT、汇总 ITL、每请求最大停顿、batch 组成与阶段耗时。

拟议产物：`batch-telemetry.jsonl`、`runtime-replay.jsonl`、`eng-008-analysis.md`、`eng-018-analysis.md`、`hypothesis-table.json`。报告至少排除一个假设或得到可继续验证的主假设，不以必须产生优化代码为完成条件。

## 6. 基准矩阵与统计协议

### 测量层次

| 层次 | 对象 | 计时边界 |
| --- | --- | --- |
| Microbenchmark | dot、matrix、线程池、布局、prefix index、单 CUDA kernel | 输入与缓存状态明确，单独记录 packing/copy |
| Model-level | 自有 Runtime prefill、decode、多序列 batch | 排除 HTTP 与队列；startup 和 steady state 分开 |
| Serving-level | 到达、排队、准入、调度、执行、SSE 与终态 | 客户端 wall-clock；保留拒绝、取消、超时和背压 |

历史 4.29× 仅属于单线程、hot-cache Q8 dot 微基准。CPU/GPU、vendor/custom、不同算术分别归因；模型级优化必须同时检查 Serving 影响。

### CPU 与 KV

| 场景 | 变量 | 指标 |
| --- | --- | --- |
| Dot hot-cache / streaming | F32/F16/Q8、长度、scalar/SIMD；轮换超过有效缓存的权重集合 | ns/dot、误差、有效字节率与硬件事件 |
| Matrix shape suite | 真实 Q/K/V、output、FFN、LM head；token batch B=1～128 的适用点 | latency、packing/workspace、吞吐、尾块表现 |
| 线程池固定成本 | 空/轻/重任务、count/grain、1/2/4/8 线程和单列 16 SMT | dispatch、工作分布、barrier 尾部与扩展效率 |
| Prefill | 16/32/128/512/2048 token，按资源扩展 | time-to-logits、prefill token/s、stage 时间、峰值内存 |
| Decode | KV=16/128/512/2048/8192 的适用点、固定输出步数 | ms/token、LM head/attention、KV 访问 |
| Mixed / batch scaling | token batch、sequence 数、prefill/decode 比例独立变化 | batch latency、共享矩阵成本、单序列延迟 |
| KV 布局与页大小 | 等算术下的分页/连续布局；交替 page size/context | 页查找、QK/PV、驻留/活跃内存、模型 TPOT |

Context 点位须为后续输出留容量，并遵守模型和运行配置上限；8192-token 容量并不允许 8192-token prompt 后继续任意 decode。token batch 不等于 active sequence 数。

### Prefix 与索引

| 场景 | 设计 | 必测指标 |
| --- | --- | --- |
| 冷前缀 | 模型已预热，索引与 prefix KV 为空 | TTFT、实际重算 token、KV 分配 |
| 部分/近全命中 | 不同共享比例，覆盖 P−1/P/P+1，保留 logits 重算 | reused tokens、重算块、logits 一致性 |
| 公共前缀并发 | 固定 prefix、不同 suffix、多并发 | 命中率、TTFT、唯一物理页与共享 |
| 索引压力 | entry 数、前缀长度、分叉深度 | 元数据字节、lookup/insert/erase P50/P99 |
| 淘汰与销毁 | 长链、共享节点、反复插入/淘汰/取消 | 淘汰延迟、销毁栈深/时间、引用守恒 |

Trie/Radix/block hash 比较必须固定 physical KV budget、完整块复用、namespace、eviction policy、输入与 logits 重算规则。Hash identity 包含前缀上下文，并通过碰撞注入验证；索引更换本身不代表命中率或 TTFT 改善。

### 负载与调度

| 场景 | 设计 | 必测指标 |
| --- | --- | --- |
| 低负载/饱和附近/过载 | 先测饱和点，再取约 0.25/0.5、0.8/1.0、1.2/2.0 倍到达率 | 基础延迟、排队拐点、拒绝、队列上界与恢复 |
| 突发与长短混合 | 集中到达后恢复，长 prefill 与短 decode 交错 | 排空时间、短请求 TTFT、decode 最大停顿 |
| 内存压力 | 降低 KV capacity 或增加 context/output | 准入原因、reservation slack、eviction 与进展 |
| 消费者压力 | 正常、慢读、停止读取、断连 | 有界事件缓冲、取消、终态与泄漏 |
| 优先级公平性 | 同优先级、多等级、持续高优先级到达 | 分组 SLO、服务份额、低优先级最大等待 |
| 策略对照 | mixed、prefill-first、候选 cost-aware | 平均 TPOT、ITL 尾部、预测误差、iteration 时间、goodput |
| Drain/admission | 容量接近耗尽，加入取消/超时/共享页 | 不超配、进度、停顿、资源回收与恢复 |

过载实验中的有界拒绝可以是预期策略，队列无界增长或耗尽后无法恢复不能接受。原 24 请求的小样本 trace 保留为回归输入，尾延迟结论需要更多请求与独立重复。

### GPU

Kernel 层使用 resident 输入，明确 CUDA-event 区间及是否包含 packing/copy；模型层权重仅初始化上传，startup 与 steady state 分开；Serving 层回放自有 CUDA runner 的真实请求，包含排队和调度。

对照包含自有 CPU、自有 CUDA + vendor matrix、自有 CUDA + 选定 custom kernel，以及 llama.cpp CUDA。每份报告记录有效权重、精度、KV dtype 和 math mode；不同算术的后端对照单独解释。

### 统一统计规则

1. 保留已有独立进程、相同预热、策略交替顺序；增加配对多轮，按需要随机化，记录频率和温度漂移。
2. 预先固定主要指标、允许退化和 SLO。Goodput 全零时可预定义多组 SLO 曲线，不能事后移动阈值制造收益。
3. 报告重复分布、中位数、尾部分位数及不确定性；不以最佳轮次代表总体。
4. 所有预期请求必须有结果，成功条件下的延迟同时列出失败、拒绝、取消和超时。
5. 每请求平均 TPOT、汇总 token ITL、每请求最大停顿分别计算；单输出 token 的 TPOT 记为不适用，不混入零延迟样本。
6. 分开标注 model warm、prefix warm、allocator warm；服务重启不等于 OS 文件缓存冷启动。
7. 保留 dispatch lag，验证客户端没有成为瓶颈；并发规模变化时重新检查。
8. 数值对照、profiler 和正式性能测量分别运行；原始 trace 保持字节与摘要一致。

## 7. Profiler 方案

### CPU

归档开发环境是 AMD Ryzen 7 7745HX，8 核、16 逻辑处理器；每次采集仍须记录实际机器与工具能力。

| 顺序 | 工具与手段 | 要回答的问题 |
| --- | --- | --- |
| 1 | 自有 wall-clock、manifest、stage counters | 计时边界、重复性与 instrumentation 开销 |
| 2 | Windows Visual Studio CPU Usage；Linux `perf record` | 热点函数、调用路径、self/total 时间 |
| 3 | Concurrency Visualizer / ETW / 可用调度事件 | worker useful work、等待、迁移、唤醒与长尾 |
| 4 | 支持的 `perf stat` / AMD uProf 事件 | cycles、instructions、IPC、cache/branch 与内存行为 |
| 5 | 汇编视图 / `perf annotate` | 指令依赖、转换、load/store 对热点的解释 |
| 6 | scheduler/batch timeline | admission、batch 执行与用户停顿之间的关系 |

先确认 PMU、事件与采集权限；不支持的事件记为不可用，不能填零。使用 Release 加符号；frame-pointer、LTO 或优化设置变化纳入 manifest。不将面向其他处理器的硬件事件当作本机可用能力。

解释约束：wall time 与多线程 CPU time 不能直接相减；包含子调用的 total time 不能重复累加；主线程等待可能对应 worker 有效计算；IPC 高不等于完成模型更快；逻辑 bytes/time 不等于实际 DRAM 带宽；resident bytes 不等于本轮分配量。NUMA 与 false sharing 研究须先有 topology、线程/一致性事件及对照证据。

### GPU

Phase 4 完整模型出现后先用 Nsight Systems，观察 CPU launch、H2D/D2H、stream idle、分配释放、同步等待、每 token launch 数与映射上传。NVTX 范围覆盖 prefill/decode/mixed、layer、projection、attention、LM head、sampling。

Systems 定位重要 kernel 后，再用 Nsight Compute 检查 duration、occupancy、memory throughput、warp stall、active lanes、register/spill、shared memory 和 load/store 效率。以 kernel latency 和模型级收益判断优化，不单独追求 occupancy。

采集可能产生 replay 和额外开销，正式吞吐与延迟另跑无 profiler 的实验。拟议产物组织：

```text
profiles/<run_id>/
    run-manifest.json
    systems-summary.json
    kernel-metrics.json
    analysis.md
    raw-artifact-manifest.json
```

小型摘要、分析与 hash 清单可入库；大型 `.nsys-rep`、`.ncu-rep`、ETW 作为实验产物保存，记录可定位位置和摘要，不连同模型权重、构建目录或依赖 checkout 提交。

## 8. 条件研究与暂缓范围

| 项目 | 定位 | 进入条件 |
| --- | --- | --- |
| CPU GEMV/GEMM、Q8×F32、线程粒度 | P1 候选 | Phase 1 明确真实热点与 shape |
| F16/F32 完整模型覆盖 | P0 | 对应权重来源、数值参照与资源准备明确 |
| Fairness 与物理 KV 观测 | P1 | 基线可重复，资源口径可核验 |
| Cost-aware scheduling、增量准入 | 条件研究 | 预测有效或真实资源压力，满足安全/进展门槛 |
| GPU residency、基础算子、完整 forward | P2 主线 | Phase 4 条件满足 |
| GPU paged KV、自研 PagedAttention | P2 主线 | 连续 GPU KV 完整模型已验证 |
| Q8 activation quantization、Q4 | 条件研究 | 计算/带宽/显存证据与独立质量预算支持 |
| Radix Tree、block hash | 条件研究 | 当前 Trie 的索引内存或操作成本显著 |
| 抢占重算 | 条件研究 | 压力恢复需要，额外计算与回滚可验证 |
| CUDA Graphs、更多 fusion | 暂缓至测量触发 | Launch 开销显著，shape 与生命周期稳定 |
| Chat template、随机采样、完整 API 兼容 | 暂缓 | 主线不依赖这些功能；有明确使用需要时单独评估 |
| 更多模型架构、MoE | 暂缓 | 单模型 CPU/CUDA 的执行深度与证据完成后另立范围 |
| Speculative decoding | 暂缓 | 稳定 GPU decode、draft/verification/rollback 成本与容量证据 |
| Multi-GPU、PD 分离、分布式 Serving | 暂缓 | 具备真实硬件、通信与独立资源池实验条件 |
| 多租户、鉴权 | 暂缓 | 当前产品限定单租户 loopback，namespace 不作为安全隔离 |
| NUMA 专项 | 暂缓至测量触发 | 实际 topology 和跨 NUMA 负载证明需要 |

近期端到端参照以已接入的 llama.cpp 为主。其他系统用于设计比较，不要求部署全部框架：

| 参考对象 | 研究问题 | 本项目证明方式 |
| --- | --- | --- |
| llama.cpp / ggml | 矩阵计划、转换/工作区、buffer 与执行完成边界 | 同算术 matrix bench、模型对照与驻留时间线 |
| vLLM | KV 分配、压力恢复、共享块引用与回收 | 小池长 decode、随机共享/释放和进展测试 |
| SGLang | 压缩前缀索引、使用中引用保护与淘汰 | 相同 KV 预算和缓存策略下比较元数据、查询成本与 TTFT |
| TensorRT-LLM | 利用率与保守准入的权衡、不同阶段的执行形状 | admission/停顿/goodput，以及 B=1、批量 decode、prefill 对照 |

固定源码、论文与资料入口见 [REFERENCES.md](REFERENCES.md)。实施时记录实际阅读的版本，历史设计文档不自动代表当前代码；借鉴结构不意味着复现其性能或拥有其实现。

## 9. 停止条件与成果验收

每项研究开始前记录假设、适用负载、主要指标、允许退化边界、正确性门槛、最多尝试的主要设计变体与停止条件。研究可用负结果完成；切换默认实现需要独立的正确性与收益证据。

| 路线 | 停止条件 | 保留产物 |
| --- | --- | --- |
| SIMD / dot | 非主要成本、只在 hot-cache 有效、模型收益低于噪声或 ISA 分支成本过高 | 稳定 scalar/SIMD 与适用范围 |
| CPU tiling | 只在不真实 shape 获益，packing 抵消收益或 B=1 严重退化 | 简单 GEMV 与有限有效路径 |
| Activation quantization | 质量超预算、量化成本不能摊销、端到端收益不足 | Q8×F32 路径和误差研究 |
| 线程池 | worker 已持续有效，固定成本低，复杂调度无模型收益 | 当前池及已验证 cutoff |
| 调度 | 独立 trace 不改善预注册 SLO、预测不稳或收益来自其他类别饥饿 | 简单 policy、模式与适用边界 |
| 增量准入 | 资源宽裕、安全估计退化为保守策略、有效利用率不升 | 保守 reservation、资源快照与压力报告 |
| Prefix 索引 | 非热点、元数据占比小、TTFT 无显著变化 | Trie，必要的 allocator/销毁改进 |
| 自研 CUDA matrix | 同精度同 shape 的有限迭代仍明显弱于 vendor，且无系统收益 | Vendor baseline 与少量有效 custom kernel |
| GPU attention 优化 | 已非模型主导成本、寄存器/工作区抵消收益、收益超出可验证条件 | 正确 paged data path 与有限优化 |
| Speculative decoding | draft/verification/rollback 成本、显存或接受率使收益不稳定 | 成本报告，保持主线简单 |
| Multi-GPU / PD | 没有实际硬件与通信实证条件 | 明确未实现/未验证的架构研究 |

用 Amdahl 上界约束投入：模块占总时长 `f` 时，即使其耗时归零，总加速也不超过 `1/(1-f)`；5% 占比对应约 1.053×。停止某个无收益 custom GEMM 分支不等于停止自研 CUDA Runtime，模型执行、device ownership、KV 和 attention 仍是独立成果。

核心改动运行 CTest；模型、数值、KV 或 Serving 改动还须运行相关真实模型与 HTTP 检查。保留全部原始轮次、失败、输入和不利结果。已解决状态必须有对应证据，实际问题按 [ENGINEERING_LOG.md](ENGINEERING_LOG.md) 的中文字段记录。

| 最终成果 | 必须具备的证据 |
| --- | --- |
| 自有模型执行与数值验证 | Qwen3 forward，F32/F16/Q8 被测路径，匹配权重参照，chunk/prefix/长 context 对照 |
| 有归因的 CPU 执行优化 | 阶段热点、可证伪假设、实现和 kernel/model/serving 分层报告 |
| 真实物理分页 KV | 数据访问路径、共享/COW/回收不变量、live/resident/slack 与收益/代价 |
| 可验证的在线资源调度 | 有界生命周期、公平性与压力恢复、admission safety/progress、goodput/尾延迟权衡 |
| 自研 resident CUDA 与 PagedAttention | 自有 forward 进入 Serving、权重常驻、device 页表直读、自有 attention 与 HTTP 证据 |
| 可复查的系统实验工程 | 源码/二进制/模型/trace/环境身份，严格验收，正确性与性能分层，完整负结果 |

每项成果关联源码、测试和结果，并区分已完成与目标状态。报告给出指定平台、模型、负载上的阶段时间、模型指标和 Serving 变化；SIMD 微基准和上游 CUDA 性能不能作为自研端到端加速。
