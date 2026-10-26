# PROJECT_PLAN_V4 — GPU Memory Systems 收尾规划

> Historical（历史计划）。分页功能和性能研究已完成，当前决定以 `GPU_KV_DECISION.md` 为准。

> Proposal ID：PLAN-V4-KV-20260928  
> 审计日期：2026-09-28，Asia/Tokyo  
> 固定审计基点：`6ca7d2dfeccb38b11b7596a552046651afcb693e`  
> 指定分支：`feat/own-cuda-serving`  
> 下一技术规范：`NEXT_SPEC_V3.md` / `GPU-KV-001`  
> 状态：独立复审提案；不表示 GPU 分页已经实现。

**版本保护。** 文件名按本轮要求提供。先前的 V3、NEXT_SPEC_V2、精度研究用 V4/NEXT_OPT_SPEC 均是历史输入，不删除、不覆盖。若目标工作区已有同名 V4，先将本提案作为独立评审版本，采纳时明确活跃 Plan ID 并保留原稿的提交引用；不能同时让两份相互冲突的路线成为编码入口。本轮仅生成这份文档及 NEXT_SPEC_V3，不修改仓库代码、分支、可见性、tag 或 Release。

**审计边界。** 本轮复核了远端 refs、指定源码、旧计划、关键测试实现、归档统计、CI 与后续分支。没有在作者 GPU 上重新执行模型/HTTP/Profiler，没有独立重跑所有归档验证器。历史测试通过、性能数字和后续分支实验均按其真实采集身份引用；不能标成 6ca7d2d clean build 的新实测。

## 1. Executive Summary

项目已不是 loader 或 wrapper。固定基点具有自研 CPU Qwen3 forward、SIMD/线程池、真实 CPU FP16 分页 KV、Trie prefix cache，以及自研 CUDA 完整模型和在线 HTTP/SSE Serving。GPU 大矩阵由 cuBLAS 计算，归约使用 CUB；模型执行、显存、KV、组批和生命周期由项目控制。

当前最强价值是 CPU→GPU→在线服务的完整可解释链路。主要缺口不是 API 数量，而是 GPU 物理 KV 容量仍绑定 `S × Lmax`、默认 main 展示落后，以及后续发现的停服传输排空缺陷尚不属于固定基点。新发现的 FP16 分支已执行研究并因原数值门槛失败停止，不能再次规划“从零实现 FP16”。

建议：先整合已有停服修复和最小发布展示；只再做一条系统主线——**共享 GPU 页池、device block table、直接分页访问的 attention**。保留 F32 权重、当前数学、单 stream、同步执行和保守 admission；首版不做 GPU prefix sharing、fusion、抢占或并发扩容。用同容量测执行代价、同预算测容量利用，完成或触发停止条件后冻结主要功能。

### 1.1 固定基点与实时增量必须分开

| 对象 | 本轮确认 | 解释 |
|---|---|---|
| main | `68ac275913207975a88e2090c6617467e351301c` | 默认 README 仍写 MiniLLM 仅 CPU、自有 CUDA 尚无 |
| Serving 分支 | `6ca7d2dfeccb38b11b7596a552046651afcb693e` | 相对 main ahead 17 / behind 0；merge base 为 main |
| 基点 CI | run `36244664620`，success | 覆盖该 SHA；不能外推为 GPU 远程测试 |
| 已有 Release | `cuda-serve-001-20260926`，target 6ca7d2d | 不是没有发布物；问题是默认入口落后 |
| 已有停服修复 | `57268f93aecc7c947146d60a3d61a03d84f46a1b` | 比 6ca 多一个提交；CI run `36406587106` success |
| FP16 研究分支 | `103070a91f1451eee3ee92e9c691e519ba733539` | 补充证据，不属于 6ca；F16 Serving 仍被源码拒绝 |
| 打开的 PR | 本次查询为空 | 本轮不创建或合并 PR |

FP16 分支文档记录 owned device bytes 从 3,449,229,312 降至 2,258,046,976；但 repeated/1536 的 32-token 续写在 step 19 得到 cosine 0.999885866，未达到冻结的 0.9999。应记录为 `blocked_correctness`，而不是因为 token 尚未分歧就改称成功。这里只复核了分支文档与 Serving 拒绝路径，没有独立执行该实验。[R9][R10]

## 2. Current Architecture

```text
                    HTTP / SSE / Replay
                            |
                  RequestHandle / Engine
          bounded queue / cancel / timeout / finality
                            |
                  schedule_batch / admission
                            |
                        ModelRunner
                 /             |              \
          MiniRunner      MiniCudaRunner     LlamaRunner
               |               |                  |
        own CPU Runtime    own CudaRuntime      llama_decode
               |               |              upstream runtime
          SIMD / pool     cuBLAS + own ops
               |               |
        CPU PagedKV      contiguous FP16 KV
        refcount / COW     fixed S x Lmax slots

         Immutable Qwen3Model + GGUF mapped tensor views
              Tokenizer adapter: upstream vocabulary
```

拟新增且尚未实现：

```text
CudaRuntime (same mathematics / same synchronous contract)
       |
       +-- contiguous access policy (retained reference)
       |
       +-- paged access policy
              |
       device block table
              |
       shared physical GPU KV slab
```

Host 管理页分配及映射；device kernel 直接读取真实 block table。不能在每次 attention 前 gather 全量 KV 为连续数组冒充分页。

CPU 与 GPU 共享数学配置、token/position 语义、完成/错误契约、测试输入；不强行共享物理布局、allocator 实现、同步机制或最优 kernel shape。沿用现有 ModelRunner，不增加 UniversalBackend/GenericMemoryProvider。

## 3. Current Capability Matrix

### 3.1 模块事实

| 范围 | 已完成 | 部分完成/缺失 | 归属 |
|---|---|---|---|
| CPU 模型 | Qwen3 embedding、Q/K norm、GQA、RoPE、SwiGLU、LM head | 大量 architecture/完整模型质量面未覆盖 | 自研执行；parser/tokenizer 部分上游 |
| CPU compute | scalar/AVX2/FMA/F16C、Q8/F16/F32 dot、V SIMD、线程池 | 没有通用高效 GEMM；矩阵仍 row-dot | 自研 |
| CPU KV | FP16 page、free list、真实页表、引用、share、tail COW | 不是 GPU 页表实现 | 自研数据路径 |
| CUDA 模型 | 完整层/完整模型、resident weights/workspace、own attention/argmax | 无 native Q8；6ca 无 F16 matrix 候选 | 自研执行 + cuBLAS/CUB 原语 |
| CUDA KV | 连续 FP16 device allocation、多序列、clear/reuse | 无 GPU pool paging/共享/COW | 自研，但静态容量 |
| Serving | 每轮重组 batch、dynamic join、mixed/chunk、优先级/aging、bounded queue | 持续优先级公平性/完整 overload 研究未完成 | 自研 |
| Admission | prompt+max_tokens 保守信用 | 非增量 admission；信用不等于物理页 | 自研 control path |
| HTTP | streaming/non-stream、UTF-8、cancel、timeout、disconnect、背压 | 6ca 的 shutdown response drain 有后续已知缺陷 | httplib transport + 自研生命周期 |
| Measurement | micro/model/serving、source/binary/model/trace identity、NSys/NCU、batch/SSE | 不能把旧数据当新版本性能或完整 replay | 自研工具，NVIDIA profiler |
| GPU prefix | 明确禁用，copy_sequence 拒绝 | 不应声称支持 GPU prefix reuse | 尚未实现 |

CPU 的 `TensorView` 不是空接口；Runtime 确实读它并计算。Serving 的 BlockPool 则主要是容量信用。MiniCudaRunner 的 `resources()` 返回真实 live token/固定 resident 数，不将 credit block 假称 GPU page。[R2][R3]

### 3.2 十五维成熟度

| 维度 | 判定 | 原因/缺口 | 求职价值与后续投入 |
|---|---|---|---|
| CPU inference runtime | 已完成限定版本 | 单一主要架构，非任意 GGUF | 高；冻结为独立 backend 与参照 |
| CUDA inference runtime | 已完成限定版本 | F32 matrices、单 stream，不是 native Q8 | 高；不重做 build/forward |
| LLM serving | 已完成主链，边界部分待补 | 真实 iteration batching；shutdown drain 后续修复 | 高；纳入已有修复，不重写 |
| GPU memory management | 部分完成 | RAII/arena/budget 完整，KV 静态 slots | 高；唯一新增系统主线 |
| Scheduler | 已完成基础，研究部分完成 | policy/aging 不等于任意持续压力公平性证明 | 高；暂不写复杂 cost model |
| Concurrency/lifecycle | 部分完成 | fail-stop/poisoned 有证据；逻辑终态与网络排空曾脱节 | 很高；已有修复验收是发布前置 |
| Correctness | 分层具备但非穷尽 | 数值/状态/HTTP 均有；固定语料不是全模型质量保证 | 很高；保护被修改的数据路径 |
| Testing | 具备 | 测试数量与独立覆盖不同；GPU CI 单列 | 高；不新增证明层 |
| Performance engineering | 具备研究基础 | profiler 与负结果真实，仍需受控系统能力闭环 | 很高；只做一项 |
| Observability | 已完成必要部分 | CPU stages、batch/SSE、GPU trace；非完整 replay | 高；只补 page 指标 |
| Benchmark methodology | 具备 | 分层/身份/配对/负结果；部分历史归档不可独立重跑 | 高；冻结规模 |
| Code architecture | 基本合理 | immutable model、runner、storage 边界已分开 | 高；仅新增必要页状态/访问策略 |
| Repository engineering | 部分完成 | main 过时、旧 artifact 多，新包已外置 | 中高；发布整理，不重写历史 |
| Documentation | 内容丰富，入口需收敛 | 历史计划、实验 cohort 与活跃计划易混淆 | 高；一个入口、一套状态说明 |
| Portfolio presentation | 部分完成 | 已可展示，但默认首页没有完整成果 | 很高；必须修复，不等所有新功能 |

阶段分类：**C 已扎实成立，最接近 D 的受限单 GPU Serving 研究原型**。若必须单选，选择 D 并保留“单模型、单 stream、同步、greedy、连续 GPU KV、限定负载验证”的限定。不是 E，不标 production-grade。

## 4. Portfolio Assessment

最强三点：
1. 自有 CPU forward → CPU memory systems → CUDA forward → online serving 的完整链路。
2. 资源生命周期的具体实现：ownership、committed/pending、poisoned、整批验证后发布。
3. 数值参照、Profiler 和不利结果，而不是只写“高性能”。

最弱三点：
1. Git 默认入口与真实成果分离，且发布前应纳入已知 transport-drain 修复。
2. GPU 物理容量不能随实际请求长度在多个 slot 之间共享；source Q8 展开成 F32。
3. 工程材料多、故事分散；局部验证通过容易被误读成全场景正确。

现在即可写入简历，但只写已有/已验证能力。候选人仍需能现场说明与修改关键代码；仓库里有代码不自动证明作者掌握。本文不是正式简历，不生成未发生的改善数字。

### 4.1 Portfolio Release Gate（最小且不自动执行）

```text
复核 6ca + 已有 57268f9 修复
    -> 相关测试与候选 SHA 的 CI
    -> 用户审阅并将已完成成果整合进 main
    -> main/最终 tag 的一次 smoke
    -> README 明确 CPU / own CUDA / upstream 边界
    -> 保留历史 cuda-serve-001-20260926 标签，不移动
```

已有 Release 可以直接作为当前作品入口，不应描述成“项目尚无任何可分发版本”。新 release/tag 必须引用新的候选；不得用旧 benchmark 冒充新构建实测。若合并已有研究分支，F16 candidate 的 Serving 拒绝行为和失败记录必须保留；不要为方便合并而放宽数值合同。[R7][R8][R9]

本轮不执行 merge、tag、公开性调整或上传 Release。

## 5. Current Technical Debt

只保留会影响下一阶段/展示的债务。

| Debt | 事实/风险 | 本轮处理 |
|---|---|---|
| HTTP shutdown drain | 6ca 顺序为 engine.stop 后 server.stop；后续 ENG-064 表明尚未发完的 SSE 可能截断 | 使用已有 57268f9，不重新设计 drain |
| 固定物理 KV | CudaStorage 按 S×Lmax 分配，--context 只约束信用 | GPU-KV-001 |
| 归档与活跃文档 | main 未展示 own GPU；6ca 的“private unless”规则与公开事实不一致，后续 policy 分支已修正 | 保留原历史，整合时采用已授权的公开规则；不自动改仓库可见性 |
| 数值研究候选 | FP16 在冻结合同下失败，并非“FP16 一般不可用” | 关闭该候选本轮投入；不重选阈值 |
| 资源统计 | contiguous pages=null；paging 需要实际页数、table bytes、assigned vs free | 复用 RunnerResources，少量增量 |
| Profiling 解释 | API 时间重叠；D2H 可能等待；GPU busy 不等于 SM occupancy | 修正叙述，不新建 profiler |
| 历史大 artifact | 部分 M1 源快照/ZIP 与原件重复 | 优先未来外置；不先删唯一证据 |

### 5.1 实现和 artifact 规模

按 6ca 的 Git tree blob 元数据计算：
- `src/`：33 个文件，224,632 bytes。
- `include/`：23 个文件，40,134 bytes。
- 二者共 56 个文件、264,766 bytes（258.56 KiB）。
- 这不是整个项目 LOC：不包含 apps、tests、scripts、docs，也没有把数据文件计作产品代码。
- 仓库 API 的 `size` 本次返回 115525；这里只保留 API 原始口径，不称为本地 clone/pack 实测大小。
- M1 完整包：56,872,322 bytes、1050 文件，旧组件和诊断结构偏重。
- M3-1 外部 canonical 包：14,130,560 bytes、171 文件；Release asset digest 与 evidence 索引相同。

没有独立取得全部历史 blob/完整本地 Git pack，不编造其总量或精确产品/测试 LOC 百分比。[R6][R7]

### 5.2 Artifact Policy

Git：source、tests、固定输入与小 fixture、schema、现有 analyzer、结论、小摘要和 manifest/locator。

Release/外部单一 canonical 包：大 raw JSON、NSys/NCU/SQLite、CI ZIP、必要 dirty source snapshot、详细负结果。

Local/regeneratable：解包副本、临时日志、重复 roundtrip、由 canonical raw 可重算的派生中间表。

迁移顺序为“目标可得→hash→一次已有复验→更新索引→去掉重复工作树副本”。不自动 rewrite public history；删当前文件不等于删历史 blob。不得生成下一层 bundle validator。对于 FP16 后续分支只在 `.run` 中引用的结果，展示时可补一个小型公开摘要/必要原始切片，不要求重封完整平台。

## 6. Industry Mapping

外部资料访问日期均为 2026-09-28；这是概念映射，不声称实现等价、性能可比或追踪了对方全部最新源码。

| MiniLLM 问题 | 官方项目对应 | 现在借鉴什么 | 不借鉴什么 | 验证/停止 |
|---|---|---|---|---|
| buffer owner、设备视图、完成点 | ggml 的 buffer/device/backend/event 边界 [I1] | owning storage 与 non-owning view、显式同步边界 | 完整 graph/backend registry | 单节点下简单边界已够，不继续抽象 |
| logical blocks→physical KV、free pool | vLLM V1 BlockPool/请求 blocks [I2][I3] | 初始化 pool、真实映射、分配/回收不变量 | 共享缓存 hash、复杂 hybrid groups、抢占 | 同预算容量与映射正确；没有收益即记录 |
| prefix index 与在用资源保护 | SGLang Radix cache eviction [I4] | “可驱逐”与“在途锁定”区别 | 因名字热门替换 CPU Trie；本轮 GPU prefix | 未测索引瓶颈不重写 |
| KV budget、按需分派与复用区别 | TensorRT-LLM KV Cache System [I5] | 固定预算池与分派、请求长度分布、实际复用和调度估计分开 | 多池/host offload/PD 集群 | 不将按需分派说成 cudaMalloc 按需 |
| chunk 与吞吐/内存调参 | SGLang tuning [I6] | 结合 queue/token usage 解释容量和负载 | 照搬大卡的默认内存余量与并发值 | 本机固定模型与负载，不套 H100 参数 |
| Async API 与 host 等待 | CUDA 12.8 API sync rules [I7] | 检查拷贝方向、host memory 和完成点 | 看到 Async 就断言不阻塞 | 只有关键路径新问题才新采集 |

vLLM 的 Paged Attention 设计页本身标注为历史说明、不再描述当前实际执行代码。[I8] 可用于理解原始原理，不可据此声称“当前 vLLM 必定使用这里的 kernel”。

本 portfolio 只再选择两个紧密相关的概念：**可共享容量的 GPU 页池**与**直接按 device block table 读取的 attention**。它们构成一个数据路径功能，不是同时加两套系统。FP16/BF16/FP8/INT8/INT4、CUDA Graph、overlap、speculative decoding、TP、PD 与 KV transfer 是独立的精度/执行/分布式问题，不自动加入 roadmap。

## 7. Strategic Decisions

### 7.1 唯一技术主线

**GPU Paged KV + Device Block Table + Page-aware CUDA Attention，容量优先，保持数学不变。**

选择理由：
- 6ca 的物理容量确实绑定 S×Lmax，而非全局 workload 的实际长度。
- 当前 CPU 已有分页，GPU 缺少真实地址映射与设备访问；新增能力能补齐 memory-systems 深度。
- 自研 GPU model/Serving、资源指标和完整数据路径已经具备，不再受旧依赖阻塞。
- 单 GPU 上能够做相同 KV 预算的受控实验。
- 新发现的 FP16 候选已执行并被数值合同阻止，不再当作未开始任务。
- 保留相同精度、相同 QK/softmax/PV 计算次序，可将研究集中在内存组织，而非同时承担精度变化。
- 不能据当前低占用 trace 推断 paging 必然更快；主指标是容量/预算，latency 为独立代价。

这里不是“FP16 失败所以必须做 paging”，也不是“vLLM 有所以跟进”。若剩余投入不足，直接执行发布与冻结也是合理结束；GPU 分页不是投简历的前置资格。

### 7.2 GPU KV 量化模型

目标 Qwen3：28层、8 KV heads、head_dim=128、K/V各FP16。
`bytes_per_token = 2 × 28 × 8 × 128 × 2 = 114688 = 112 KiB`。

当前S=4、Lmax=2048：
- 8192 tokens ×112 KiB =896 MiB resident KV。
- 一次NSys峰值1255tokens对应137.27MiB有效payload，占15.32%。
- 其余约758.73MiB是该观测下的预留空闲/尾部容量，不是已测得可以无代价删除的内存。

拟设P=16：
- 一个物理逻辑块覆盖所有28层的16个token，payload=1.75MiB。
- live pages=`sum ceil(length_i/P)`，第一版不共享。
- 每请求末页浪费<16token，4个活动序列的尾页浪费≤60token=6.5625MiB。
- 没有分派的pool页是保留容量，不计为尾页碎片。
- Pool只在启动cudaMalloc一次；按需的是page assignment，不是物理显存返回操作系统。

同容量8192tokens测执行代价；不同长度负载下以**288MiB KV子系统预算**测容量。候选160页=2560tokens=280MiB，表和对齐也计入预算。当前静态模型Lmax=2048在该预算下最多一个224MiB slot；两个slot要448MiB。具体实验仍须执行，不把推导写成已测性能。

### 7.3 Weight path 与执行决策

| 路径 | 价值 | 风险/状态 | 决定 |
|---|---|---|---|
| 保持F32 | 现有数值/Serving基线可靠，隔离KV变量 | 占显存、某些shape性能不是最优 | 本轮保持 |
| F16/BF16矩阵 | 低精度存储/计算/Tensor Core有研究价值 | F16候选已在原合同失败；不代表全部FP16方案无效；BF16未验证 | 本轮不重开，不放宽阈值 |
| Native Q8 | 带宽与GPU kernel深度高 | scale、layout、M=1与prefill路径都需新实现/验证 | 不加入本轮 |
| 单stream同步 | 完成点、清理与发布容易证明 | 主执行线程需等待；不等于无batching | 保持 |
| Async/Graph | 可能减少host提交/提高overlap | 需要在途资源、图签名、复用率证据 | 不自动加入 |

后续 FP16 分支对旧 Serving trace 的离线分解记录：matrix占kernel time约64.62%、attention约30.51%；copy API的大额host elapsed主要位于D2H，而不是H2D。它只强化“不能靠API总时间归因”的结论，不证明分页是最大时间热点。[R10] 本轮选择的是容量系统问题，不宣称正在修复全部GPU耗时。

### 7.4 Portfolio ROI（审阅判断，不是统计评分）

成本尺度：S=小范围现有模块补丁；M=1–3个核心职责及接线；L=跨模块数据路径；XL=超出当前本机/验证预算。列中的代码量是设计范围估计，不是承诺行数。

| 候选 | Technical / Infra | C++ / CUDA / Systems | 面试与简历增量 | 实现/代码范围 | Debug / Benchmark成本 | 依赖与决定 |
|---|---|---|---|---|---|---|
| GPU PagedKV/direct attention | 高/高 | 高/高/很高 | 高，明确容量与地址映射故事 | L；最多3职责 | L / M | DO NEXT，唯一新增主线 |
| GPU prefix reuse | 高/高 | 高/高/高 | 高但CPU已有主题 | L | L/L | OPTIONAL；本轮不做 |
| FP16 weights | 高/高 | 高/高/高 | 已有有效负结果 | 候选已存在 | 已付出，不追加大矩阵 | OPTIONAL；停止当前候选 |
| Native Q8 CUDA | 高/高 | 高/很高/高 | 高 | L–XL，多算术路径 | 高/高 | OPTIONAL；非剩余必做 |
| CUDA Graph | 中高/高 | 中/高/高 | 有实证才高 | M–L | 高/中 | OPTIONAL；先需复用率/关键路径 |
| Async execution | 高/高 | 高/高/很高 | 高但风险大 | L | 很高/高 | DO NOT DO FOR THIS PORTFOLIO |
| Scheduler优化 | 高/高 | 高/中/很高 | 已有较多相关证据 | M–L | 高/高 | OPTIONAL；不与分页一起改 |
| Incremental admission | 高/高 | 高/中/很高 | 需真正容量压力与progress证明 | L | 很高/高 | OPTIONAL；保持保守方案 |
| Speculative decoding | 高/高 | 高/高/高 | 范围过大 | XL | 很高/很高 | DO NOT DO FOR THIS PORTFOLIO |
| Multi-GPU | 高/高 | 高/很高/很高 | 无硬件实证不能兑现 | XL | 很高/很高 | DO NOT DO FOR THIS PORTFOLIO |
| PD disaggregation | 很高/高 | 高/高/很高 | 无集群实证不能兑现 | XL | 很高/很高 | DO NOT DO FOR THIS PORTFOLIO |
| 更多architecture | 中/中 | 中/中/中 | 相对已有能力增量低 | L，验证面宽 | 高/高 | DO NOT DO FOR THIS PORTFOLIO |
| OpenAI API completeness | 低/中 | 中/低/中 | 低于核心问题 | M–L | 中/中 | DO NOT DO FOR THIS PORTFOLIO |
| 发布、修复与作品表达 | 中/高 | 已有能力的可见证据 | 很高 | S–M | 低/低 | DO NOW，支持性门禁 |

剩余主要新功能：**最多1项**。不设置第二个无条件major feature。Optional意味着可以永远不做，不是下一轮的TODO队列。

## 8. Milestones

### M4-0 — Portfolio Release / Correctness Consolidation

- Type：INFRA / DOCUMENTATION，支持性前置。
- Goal：真实成果可见；已知停服缺陷不带入新研究。
- Why Now：main落后17提交；修复已经存在，不需要新架构。
- Evidence：R1、R7、R8、R9。
- Dependencies：确认57268f9与所选integration SHA。
- Scope：纳入已有修复、候选CI、默认入口/Release说明、一次非性能smoke。
- Non-goals：重跑M1、扩大证据包、自动修改公开性、重写历史、把F16候选变成默认。
- Implementation：用户审阅整合；README清楚链接已验证的backend、局限与正确版本；保留历史tag。
- Files：实施期只需相关文档/索引与已有修复的整合，不重写修复代码。
- Experiments：无新benchmark；启动→token→SSE→停服排空smoke，完整GPU检查使用已有入口。
- Correctness Gate：新候选自身CI；原shutdown反例通过；慢/断连客户端的有限超时边界明确。
- Performance Gate：无性能目标；旧数据保持原采集身份。
- Exit：真实成果可以从默认入口找到，候选/证据一致。
- Stop：现成修复和入口闭合后立即停止，不把release变成平台。
- Portfolio value：避免招聘者只看到旧CPU项目或“已完成”却复现失败。
- Difficulty：S。
- Budget：零新验证框架、零新模型feature、一次smoke。

### M4-1 — Shared GPU KV Pool + Direct Paged Attention

- Type：INFRA / PRODUCT / RESEARCH，唯一新增主线。
- Goal：将物理KV容量与S×Lmax解耦，并保持设备直接访问和模型语义。
- Why Now：完整GPU模型/Serving/资源/基线都在；精度研究不重复。
- Evidence：R2/R3/R4/R10；详见GPU-KV-001。
- Dependencies：M4-0技术正确性前置；稳定F32 baseline；本机CUDA12.8。
- Scope：单GPU页池、host页状态、device table、paged KV write/QK/PV、旧Engine接线。
- Non-goals：GPUprefix/refcount/COW/fusion/precision/Graph/async/preemption/新增scheduler算法/S>4。
- Implementation：复用CudaStorage、BatchState、LayerExecutor、attention数学；只替换访问策略。
- Files：`src/minillm/cuda/{storage,batch_state,layer,runtime,attention}.*`、新host页状态与访问策略；MiniCudaRunner/config最小接线。
- New abstractions：Host PageTableState、device view、两种编译期访问policy；不新增通用memory framework。
- Experiments：同容量8192tokens的layout A/B；同288MiB KV预算的异长请求；必要边界/压力测试。
- Metrics：live/assigned/resident、tail waste、max simultaneously resident requests、TTFT/TPOT/throughput、table H2D、kernel/model latency。
- Correctness Gate：真页映射、NaN尾部防读、同数学oracle、原数值合同、事务/poisoned与HTTP回归。
- Performance Gate：容量能力与执行代价分别报告；不要求token/s改善。
- Exit：request→device table→physical KV→attention→token完整，固定实验得出结论。
- Stop：两种主要实现变体仍无价值、需扩大precision/async/前缀才能成立、无法维护状态不变量时结束；保留baseline与负结果。
- Portfolio value：GPU地址计算、共享容量、资源信用与物理分配、failure lifetime的完整故事。
- Difficulty：L。
- Budget：最多5个主要commit；最多3核心职责；不新增大型benchmark/validator；只一个canonical包。

### M4-2 — Capacity/Latency Conclusion and Feature Freeze

- Type：MEASUREMENT / DOCUMENTATION，不是另一个feature。
- Goal：区分有用容量提升、简单调参可替代的收益和额外延迟成本。
- Why Now：实现不能自动成为默认；要有独立的采纳/拒绝决定。
- Evidence：M4-1原始记录、现有固定输入和状态测试。
- Dependencies：M4-1正确性，或其明确abort结论。
- Scope：完成预算内的micro/model/serving；一份analysis；修正可宣传范围。
- Non-goals：补trial直到显著；为同预算结果发明“4×吞吐”；添加prefix以让数据好看。
- Implementation：使用已有analyzer/CLI，新增少数layout与预算字段，结论与原raw绑定。
- Correctness Gate：所有baseline/candidate身份完整；未通过不称为product-ready。
- Performance Gate：同容量代价超预注册护栏时保留实验性模式；同预算收益不等于kernel提速。
- Exit：accepted_capacity_mode / experimental_negative / measurement_inconclusive之一；功能冻结。
- Stop：预算耗尽即出结论；失败也不再接第三条重大路线。
- Portfolio value：证明实验与取舍，而不是feature统计。
- Difficulty：M。
- Budget：6个micro shape、4个model workload、2条Serving trace×2臂×3trial=12正式进程；最多1次新NSys。

### M4-3 — Portfolio Finalization / Upstream

- Type：DOCUMENTATION / LEARNING。
- Goal：把已有深度变成可解释、可复现、可审查的求职作品。
- Scope：一页README入口、架构/所有权图、一篇有负结果的技术报告、5–8个追问主题、基于真实问题的upstream工作。
- Non-goals：正式简历虚构数字、为PR数量制造低价值改动、开新个人推理框架补功能。
- Gate：能解释page address、credit invariant、FP16失败、SSE drain、profiling时间边界；能实际修改一段核心实现。
- Exit：读者能定位代码、复现命令、证据和限制；主要feature开发结束。
- Stop：展示材料足够清楚，不将写作扩为新文档平台。
- Difficulty：M。
- Budget：1技术报告、1主架构图、1复现入口；其余尽量链接现有证据。

## 9. Entry Gates

1. 固定6ca事实与实时分支增量分别记录，旧V3/V2只作历史。
2. 集成基点含已有HTTP drain修复，且自身CI/相关回归通过。
3. FP32 baseline仍可单独运行；不依赖被数值门禁阻止的F16候选。
4. 先写pool bytes、table shape、失败状态、保守credit证明，再写CUDA访问。
5. 冻结两类对照：同容量/同设置；同预算/不同可承载slot。
6. 新benchmark字段只是解释本次功能，不得变成另一个采集/封包系统。
7. GPUprefix capability保持false；不会因后端内部出现页表自动打开CPU prefix索引。

## 10. Exit Gates

Research Done：
- 真实实现或明确的原型失败；
- 单位/状态/模型/服务所需检查完成；
- 固定实验有原始结果、限制、采纳/停止决定。

Product Eligible：
- 数值与资源安全通过；
- 无silent fallback、无已知未关闭关键bug；
- 按页直接设备访问；
- capacity实验满足合同，代价在可接受范围或明确保留为opt-in模式；
- 同容量比较没有被同预算的并发差异替代。

Default Promotion不属于本轮目标；连续baseline必须保留。速度没提升允许Research Done，不允许用其豁免correctness。

## 11. Measurement Budget

| 层次 | 上限/边界 |
|---|---|
| Host state | 固定seed随机操作，1000–10000步为小测试，不跑大模型 |
| Micro | 6个真实attention shape，P固定16；page boundary更多点属于correctness，不全部benchmark |
| Model | 4个固定workload；baseline/candidate各3独立trial，setup与measured分开 |
| Serving | 现有mixed-length同容量 + 新异长容量trace；每条2臂各3trial，共12正式服务进程 |
| Pressure | 小型功能测试，有限请求；不当性能样本 |
| Profiler | 优先复用已有数据；必要1次完整paged-serving NSys；NCU仅有明确kernel问题才1次 |
| Artifact | 1个canonical包；Git小摘要；不重复封包 |

三轮只提供本机描述性差异；不承诺生产P99。保留失败、超时、不达SLO和无收益；不改历史SLO，不增加trial追正结果。

## 12. Risk Register

| 风险 | 后果 | 控制 |
|---|---|---|
| 把固定基点和新分支混为一谈 | 重复开发/错误履历 | 两层证据、明确SHA |
| main迟迟不更新 | 招聘者看不到真实能力 | M4-0，不等paging完成再展示 |
| credit/page不一致 | 已准入请求中途耗尽 | 相同P，容量≤物理池，保守承诺不变量 |
| prepare失败留下半份映射 | 重用错页/泄漏 | preflight journal、commit/noexcept、post-launch quarantine |
| device映射过早回收 | OOB/串请求 | 单stream完成后clear；poisoned不复用 |
| tail读取NaN/未来KV | 数值错误 | 先mask再load；污染测试 |
| 同时改精度和布局 | 无法归因 | F32/CUBLAS/PV顺序冻结 |
| 同预算结果偷换kernel收益 | 夸大性能 | 两套对照独立命名 |
| 为了paging提前写共享 | scope失控 | first version exclusive ownership，无refcount/COW |
| 表上传/间接索引代价高 | latency退化 | 有限对照；最多一次局部优化后可停止 |
| 只降低pool大小即称“省显存” | 简单调参被当算法收益 | 比较当前静态方案可承载合同，披露alternative |
| 归档再次膨胀 | 项目变成验证平台 | 一个包、已有工具、禁止meta-validator |
| GPU性能/数值不稳定 | 结论不可信 | 固定配置、原始样本、标inconclusive，不刷轮次 |

## 13. Portfolio Definition of Done

现在已经可以求职，分页不是资格考试。以下是建议的最终收尾标准：

- 默认入口或清楚的Release展示自有CPU/CUDA/Serving；已知关键修复整合。
- 数学、资源、服务三层边界清楚，Q8 source不等于Q8 CUDA。
- 至少一个完整系统研究闭环；本轮分页或已有FP16负结果均如实表达。
- 技术数字可定位到代码、binary、模型、固定输入、raw与摘要。
- 不将GPU页池容量分派称成OS级按需显存释放，不将credit称成physical pages。
- 能讲清CPU/GPU布局、错误状态、chunk/batch、GGUF依赖与vendor primitives。
- 能解释一个没有收益/没有通过门槛的决定。
- 验证关键shutdown、poisoned、错误输出、资源回收，无已知严重问题被隐藏。
- 读者可运行最小smoke；不需要浏览成千上万文件才能看懂结果。
- 一个新增主线结束后，只维护bug/兼容性；转向upstream问题与面试准备，不自动开始Q8、Graph或多卡。

## 14. Resume Story（开发价值定义，非现成简历成绩）

我们自行实现了一个有真实数学与内存数据路径的C++ CPU推理Runtime。
它具有SIMD、线程池和CPU物理分页KV，而不是只调用第三方模型接口。
随后建立了常驻权重、显式workspace和错误状态的自有CUDA模型执行。
同一Runtime进入在线Engine，支持动态组批、流式输出和受限生命周期契约。
我们能够把延迟关联到batch、kernel和资源，而不是用microbenchmark代替服务指标。
精度研究未通过门槛时保留负结果；GPU分页研究则隔离数学与内存变量。
最终用同容量代价、同预算容量和真实Serving解释设计取舍，并在足够时停止扩功能。

## Evidence Index

### Repository sources（固定引用，除明确标注的后续分支）
- [R1 — main README](https://github.com/lilong555/mini-llm-runtime/blob/68ac275913207975a88e2090c6617467e351301c/README.md)
- [R2 — CPU Runtime](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/src/minillm/runtime.cpp)；[CPU PagedKV](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/src/minillm/paged_kv.cpp)
- [R3 — CUDA Runtime](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/src/minillm/cuda/runtime.cpp)；[Storage](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/src/minillm/cuda/storage.cpp)；[MiniCudaRunner](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/src/mini_cuda_runner.cpp)
- [R4 — Serving analysis](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/benchmarks/results/cuda-serving-001/analysis.md)；同目录summary/protocol/validation/evidence分别约束各类数字。
- [R5 — old V3](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/docs/PROJECT_PLAN_V3.md)；[old NEXT_SPEC_V2](https://github.com/lilong555/mini-llm-runtime/blob/6ca7d2dfeccb38b11b7596a552046651afcb693e/docs/NEXT_SPEC_V2.md)
- [R6 — source tree](https://github.com/lilong555/mini-llm-runtime/tree/6ca7d2dfeccb38b11b7596a552046651afcb693e/src)；[include tree](https://github.com/lilong555/mini-llm-runtime/tree/6ca7d2dfeccb38b11b7596a552046651afcb693e/include)
- [R7 — existing Release](https://github.com/lilong555/mini-llm-runtime/releases/tag/cuda-serve-001-20260926)
- [R8 — base CI](https://github.com/lilong555/mini-llm-runtime/actions/runs/36244664620)
- [R9 — subsequent shutdown fix](https://github.com/lilong555/mini-llm-runtime/commit/57268f93aecc7c947146d60a3d61a03d84f46a1b)；[its CI](https://github.com/lilong555/mini-llm-runtime/actions/runs/36406587106)
- [R10 — subsequent precision study](https://github.com/lilong555/mini-llm-runtime/blob/103070a91f1451eee3ee92e9c691e519ba733539/docs/PRECISION_STUDY.md)；[Serving rejection](https://github.com/lilong555/mini-llm-runtime/blob/103070a91f1451eee3ee92e9c691e519ba733539/src/config.cpp)

### Official external material（访问2026-09-28；不是项目自己的实现证据）
- [I1 — ggml backend API](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml-backend.h)
- [I2 — vLLM V1 BlockPool](https://docs.vllm.ai/en/latest/api/vllm/v1/core/block_pool/)
- [I3 — vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [I4 — SGLang radix eviction](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)
- [I5 — TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html)
- [I6 — SGLang tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)
- [I7 — CUDA 12.8 synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html)
- [I8 — vLLM historical Paged Attention note](https://docs.vllm.ai/en/latest/design/paged_attention/)
