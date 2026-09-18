# MiniLLM / LLMServe 规范问题清单

主题：Mini LLM Runtime（C++、SIMD、GGUF）→ LLM Serving（continuous batching、paged KV、prefix cache）。

共 45 项。P0 是正确性与可交付门槛，P1 是性能研究主线，P2 是扩展。`[x]` 表示当前限定范围内已有实现和验证依据，不代表覆盖任意模型、平台或负载；`[ ]` 是明确待办。

## 1. 范围与构建

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] ENV-001 | P0 | Runtime、Serving 与第三方后端的责任是否明确？ | MiniLLM 自行执行前向与 CPU paged KV；llama.cpp 的 parser、tokenizer、CUDA 能力单列。见 `THIRD_PARTY.md` |
| [x] ENV-002 | P0 | 无 Python、无 CUDA 的机器能否构建和运行主 CPU 产品？ | 独立 CPU Release 构建、真实 CLI、10 项模型与 8 项 HTTP 检查通过；移除 CUDA/Python 环境路径后可运行，PE 导入无相关 DLL |
| [x] ENV-003 | P0 | 第三方依赖是否固定且未修改？ | CMake 核对 llama.cpp SHA；Fetch 脚本拒绝错误提交及脏 checkout |
| [x] ENV-004 | P0 | 模型与参照权重是否具有可验证来源？ | 两份 model manifest 含版本、转换关系、字节数与 SHA-256 |
| [x] ENV-005 | P0 | 不支持的架构、采样和 API 参数是否显式拒绝？ | Qwen3 主范围；greedy-only；参数校验与 HTTP 422/404 |
| [x] ENV-006 | P1 | 跨编译器与 sanitizer 能否保持相同行为？ | GitHub Actions 的 Windows/MSVC、Linux/GCC 完整 CPU 构建与单测通过，Linux ASan/UBSan 核心测试通过；真实模型与 HTTP 仍是本地验证，见 `VALIDATION.md` |

## 2. Mini Runtime

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] RT-001 | P0 | GGUF 加载是否保留正确的形状、类型、row stride 与 mmap 生命周期？ | 官方 Q8_0 GGUF 的真实前向；`GgufModel`、`TensorView` |
| [x] RT-002 | P0 | 损坏、截断、错误类型和越界 GGUF 是否安全失败？ | 4 项 GGUF 回归通过：合法 mmap 行、元数据类型、截断数据、错误 magic、文件缺失与不支持的 tensor 类型；不宣称覆盖任意恶意文件 |
| [x] RT-003 | P0 | FP16 转换是否覆盖 subnormal、舍入、零和无穷？ | 全部有限 half bit patterns 往返，ties-to-even 边界测试 |
| [x] RT-004 | P0 | SIMD 分派是否检查 CPU 和操作系统状态？ | AVX2/FMA/F16C + OSXSAVE/XCR0；独立 scalar 入口；不全局启用 AVX2 |
| [x] RT-005 | P0 | Q8/F16/F32 SIMD 结果是否与标量计算一致？ | 多尺寸、尾元素和随机输入；Q8 解量化对照；最大误差有报告 |
| [x] RT-006 | P0 | 完整模型前向是否有同权重数值 oracle？ | Q8 权重解量化 F32 参照；teacher-forced logits + greedy 序列，见模型验证 JSON |
| [x] RT-007 | P0 | GQA、Q/K norm、RoPE position 和 causal mask 是否在分块下保持语义？ | 33-token prompt，chunk=1/7/16；页尾分支与重算 logits 相等 |
| [ ] RT-008 | P1 | 所有声明的权重类型和目标上下文长度是否都有模型级覆盖？ | F16/F32/Q8 模型、长 prompt、长 decode、多种语言与极端 token 输入 |
| [x] RT-009 | P0 | 多线程矩阵工作分配是否丢任务、重复计算或吞异常？ | 线程池完整覆盖、异常传递与复用测试；生成序列跨 batch 一致 |

## 3. Paged KV

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] KV-001 | P0 | Attention 是否真的按页表访问非连续物理 KV？ | `PagedKV` 存储 FP16 数据；Runtime attention 通过 `key/value` 访问页，不只是维护元数据 |
| [x] KV-002 | P0 | 分配、别名、引用计数和释放是否守恒？ | 满池、共享、释放与复用测试；结束后 live physical pages=0 |
| [x] KV-003 | P0 | 部分页共享后追加能否避免污染原请求？ | 尾页 COW；元数据单测与真实模型 17-token prefix 分支对照 |
| [x] KV-004 | P0 | 容量计算是否使用 KV head 数和显式 head_dim？ | 28 层、8 KV heads、128 head_dim、FP16；112 KiB/token；物理页与信用分开 |
| [ ] KV-005 | P1 | 是否实现增量准入，并为 decode 预留可证明的 headroom？ | 当前物理按需分配、逻辑按 prompt+输出上限保守预留；增量准入需压力与回滚证明 |
| [x] KV-006 | P0 | CPU 自有分页与 llama.cpp GPU KV 是否严格区分？ | 不将 llama.cpp 序列别名或信用页描述为自研 GPU PagedAttention |

## 4. Prefix Cache

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] CACHE-001 | P0 | 前缀身份是否由 token 序列与 namespace 精确决定？ | token Trie 精确匹配，无 hash-only 命中；单模型进程固定模型身份 |
| [x] CACHE-002 | P0 | 完全命中时，首个输出的 logits 从何而来？ | 复用上限小于 prompt 长度，只共享完整块；保留最后 token/块重算 |
| [x] CACHE-003 | P0 | 淘汰是否只释放缓存持有的引用？ | LRU 删除缓存别名；活跃请求的物理页引用继续有效 |
| [x] CACHE-004 | P0 | 查询与使用之间会不会发生陈旧命中？ | 查找、准入、别名复制与淘汰归单一执行线程所有 |
| [ ] CACHE-005 | P2 | Radix Tree 或 block hash 是否优于当前 token Trie？ | 比较索引内存、匹配耗时、长前缀销毁、命中率与端到端效果 |
| [ ] CACHE-006 | P2 | 多租户场景下缓存是否需要认证隔离与侧信道约束？ | 当前仅单租户 loopback；namespace 不等于认证，生产方案独立验收 |

## 5. Scheduler

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] SCH-001 | P0 | 新请求能否在其他请求结束之前动态加入？ | 真实多请求执行；`max_batch_sequences` 与逐轮调度 |
| [x] SCH-002 | P0 | Prefill 和 decode 是否进入同一次前向？ | `BatchPlan` → 单次 `ModelRunner::execute`；真实 `mixed_batches > 0` |
| [x] SCH-003 | P0 | chunk、token budget、位置与 logits 数量是否同时正确？ | 调度边界单测、模型分块对照、非法/重复 sample 检查 |
| [ ] SCH-004 | P1 | Priority aging 是否在持续高优先级流量下保护普通请求？ | 当前有排序与等待保护；持续注入压力下的等待上界和公平性曲线待测 |
| [x] SCH-005 | P0 | 永远放不下、暂时拥塞与可淘汰缓存如何区分？ | 请求长度提前拒绝；容量不足先淘汰缓存再等待；小池可进展；队列有界 |
| [ ] SCH-006 | P2 | 是否需要抢占重算，且其收益能否覆盖额外计算？ | 当前不抢占；需比较重算量、TPOT、吞吐和活锁风险 |
| [ ] SCH-007 | P1 | CPU 上是否应使用不同于 GPU 的 cost-aware token budget？ | 结合 prefill 长度、KV 长度、矩阵复用与 profiler，不能假设所有 token 成本相等 |

## 6. HTTP 与生命周期

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] HTTP-001 | P0 | streaming 和 non-streaming 的文本、token 与 usage 是否一致？ | C++ 在线测试；最终 usage；单个 `[DONE]` |
| [x] HTTP-002 | P0 | UTF-8 跨 token 碎片和 EOS 计数是否有明确语义？ | UTF-8 增量缓冲测试；EOS 计入采样数、不输出特殊 token 文本 |
| [x] HTTP-003 | P0 | 显式取消和两种断连是否回收在途请求？ | 真实流式取消、stream/non-stream 断连；活跃信用归零 |
| [x] HTTP-004 | P0 | 超时是否包括排队，且何时实际生效？ | 创建时计算 deadline；前向边界检查；HTTP 408；不承诺 kernel 中途抢占 |
| [x] HTTP-005 | P0 | 慢消费者能否拖住整个模型线程？ | 独立有界事件缓冲、显式 backpressure 错误、socket write timeout |
| [x] HTTP-006 | P0 | 错误参数、重复 ID、关停与 backend 错误是否有有限终态？ | 严格 JSON 检查、重复在途 ID 拒绝、关停 join、backend fail-stop |

## 7. 实验可信度

| 状态 / ID | 优先级 | 规范问题 | 验收标准 / 依据 |
| --- | --- | --- | --- |
| [x] BENCH-001 | P0 | 是否保留可重放的原始输入和完整输出？ | JSONL trace、精确 token IDs、到达时间、digest、逐请求 token 到达时间 |
| [x] BENCH-002 | P0 | TTFT、TPOT、吞吐和 goodput 的分母是否明确？ | 成功/失败不删样本；完整回放时长；请求均值 TPOT 与逐 token ITL 分开 |
| [ ] BENCH-003 | P1 | 策略比较是否具有重复实验、冷/热缓存和正确性对照？ | CPU 每策略 3 次、共 144 个请求已完成，输出一致，不利结果已保留；独立冷/热前缀实验及 GPU 对照待补 |
| [ ] BENCH-004 | P1 | 是否覆盖过载、共享前缀与长请求公平性，并采集阶段 profiler？ | 不仅测单一平均长度；保留队列、内存和分阶段时间；形成可检验瓶颈结论 |
| [x] BENCH-005 | P0 | 是否将微基准、模型执行、Serving 和上游能力分别归因？ | SIMD dot 的 4.29x 是单线程热缓存微基准；不写成端到端提升 |

## 执行顺序

1. 保持 P0 回归通过，将跨平台构建与 sanitizer 纳入持续验证。
2. 扩展 CPU/GPU 分开的重复实验，补齐冷/热前缀，保留真实输入、失败与输出差异。
3. 用阶段 profiling 选择一个 Runtime 优化点，先验证数值，再验证端到端。
4. 在有资源守恒证据的基础上推进增量准入与公平调度。
5. 自有 CUDA paged attention、Radix Tree 和抢占作为独立扩展，不与主线成绩混合。

实际遇到的问题、原因、处理方法与验证状态见 [ENGINEERING_LOG.md](ENGINEERING_LOG.md)。
