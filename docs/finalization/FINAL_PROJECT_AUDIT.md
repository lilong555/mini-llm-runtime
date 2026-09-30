# FINAL_PROJECT_AUDIT

## 1. Executive Verdict

**结论：停止扩功能。当前最值得交付的是一套可解释的 C++20 CPU/CUDA Qwen3 Runtime 与同步在线 Serving，而不是一个覆盖行业所有能力的通用推理平台。**

- Audit Date：2026-09-30，Asia/Tokyo。
- Repository：`lilong555/mini-llm-runtime`。
- Default branch：`main`。
- Main HEAD / Portfolio Candidate HEAD：`9a571a5d7a6570c6dfeeac9062adefca1840534e`。
- Open PR：0；PR #2、#3 已合并。
- Main CI：run `36709597589`，completed / success。工作流为 Windows/Linux core、CPU product，以及 Linux ASan/UBSan；不是 GPU CI。
- 已有研究发布：`cuda-serve-001-20260926`、`cuda-prec-001-20260928`、`gpu-kv-001-20260930`。
- 已有版本标签：`v0.1.0`；不可重新绑定它。

**审计范围与可信度。** 本次读取了 GitHub 固定 SHA 的主要执行链源码、构建配置、关键测试、文档、实验摘要与仓库元数据。没有逐行读完全部头文件、测试、脚本、历史计划和二进制证据包，因此不是“全仓库逐文件无遗漏审查”的认证。源码直接观察、仓库历史报告与审计判断在本文中分别标识。

Remote Desktop Commander 查询结果为已安装，但当前对话未提供该应用的终端/文件工具，实际访问未成功。容器 Git 请求也因无法解析 github.com 失败。本次没有进入用户 WSL，没有执行构建、模型、HTTP、GPU sanitizer 或性能测试，也没有下载并重验 Release ZIP。未修改远程代码、分支、标签或 Release。

**Resume readiness：TECHNICALLY READY, PRESENTATION NOT READY。** 这是基于源码与历史验证记录的作品评估，不代表本次独立复现已经完成。仓库值得放入简历；建议先修复当前事实矛盾与快速启动入口，再把它作为无需口头解释的最终版本发送。没有理由因分页或 FP16 的负结果而撤掉项目。

## 2. What The Project Actually Is

Mini LLM Runtime is a bounded, inspectable C++20 CPU/CUDA Qwen3 inference runtime connected to an online serving engine, with explicit resource ownership and evidence-backed engineering decisions.

It is NOT a production-scale vLLM replacement, an original model architecture, a self-written GEMM library, or an API-wrapper application.

Primary identity：**LLM Runtime → Serving 的系统实现作品**。
Supporting systems：CPU SIMD/线程池、调度、HTTP/SSE、测试与测量。
Research case studies：GPU paging 的容量—延迟取舍；FP16 矩阵候选被数值门槛阻止。
Reference paths：llama.cpp 后端和 matched-weight 数值参照。

真实使用价值不是让业务方抛弃成熟框架，而是让工程师在受控模型与硬件范围内检查执行、所有权、失败语义和实验决策。是否具有生产性能或外部用户价值，不能由代码量推断。

## 3. Current Architecture

初始化与请求执行应分开画，不能把 checkpoint、scheduler、KV 当成同一级串行步骤。

```text
初始化：
固定 GGUF → 上游 metadata parser + 项目 mmap/TensorView/Qwen3Model
                         ├─ CPU：Q8_0/F16/F32 权重视图、线程池、物理分页 KV
                         └─ CUDA：resident F32 weights、workspace、连续 FP16 KV

默认作品展示请求链：
Client → HTTP/SSE → Engine admission / 保守容量信用
       → schedule_batch（每轮重新组批，chunked prefill + decode）
       → MiniCudaRunner → CudaRuntime
       → cuBLAS matrices + 项目 RMSNorm/RoPE/GQA/SwiGLU/KV/greedy
       → checked stream completion → 长度提交 → 整批 sample 校验
       → RequestHandle 有界事件队列 → SSE → Client metrics

观测链：
固定 workload → 独立进程 benchmark / 独立 profiler
             → raw + identity + summary → 限定范围的结论
```

软件快速入口默认仍是 CPU `mini`。推荐的 CUDA 作品展示路径为 `mini-cuda`，不是要求把整个软件改成 GPU 默认。

CUDA 默认：F32 pedantic 矩阵、F32 主体激活、contiguous FP16 KV、单 stream、同步 execute、greedy、prefix cache 关闭。验证范围为 Qwen3-0.6B，S≤4、L≤2048、B≤128。CPU 路径不继承所有 CUDA 上限。

研究支路：CPU prefix/COW 是 CPU 已有能力，不是未完成实验；GPU paged 为显式 opt-in 的容量研究模式；FP16 matrix 只保留研究入口，Serving 拒绝。

## 4. Self-developed vs Upstream Boundary

| 内容 | 项目实现 | 复用或限制 |
|---|---|---|
| GGUF 加载 | Windows/Linux 文件映射、视图、bounds/shape/stride 检查 | metadata parser 来自固定 llama.cpp |
| Tokenizer | 接口封装、生命周期与并发使用约束 | vocab/tokenization 来自 llama.cpp，不是自研 BPE |
| CPU forward | Qwen3 执行、SIMD/scalar、线程池、CPU KV | Qwen3/RoPE/GQA 等机制不是原创算法 |
| CUDA forward | 权重常驻、内存计划、算子串接、attention/KV、greedy、状态提交 | GEMM 用 cuBLAS；block/warp reduction 用 CUB |
| Serving | Engine、调度、容量信用、请求生命周期、流式适配 | HTTP 传输用 cpp-httplib；JSON 用 nlohmann/json |
| Reference | LlamaRunner 适配与对照协议 | llama_decode 的模型执行属于上游 |

源码确认自有 CPU/CUDA 路径执行自己的 forward；MiniCudaRunner 不启动 CLI 子进程。源文件 Q8_0 在 CUDA 初始化时解量化为 F32，不是 native Q8 CUDA GEMM。合理复用 vendor primitive 不削弱系统实现价值，但必须准确署明边界。

## 5. End-to-End Value Chain

以下评级针对已读取源码与记录，不表示本次上板复现通过。

| 环节 | 评级 | 判断 |
|---|---|---|
| GGUF → HostModel / TensorView | COMPLETE | 映射与上下界检查有真实实现 |
| CPU/CUDA → Qwen3 forward | COMPLETE | 两条独立执行路径，而非 reference backend 改名 |
| KV ownership / state | GOOD ENOUGH | CPU 共享/COW；GPU ready/prepare/execute/commit/poison 边界清楚 |
| ModelRunner → Engine / scheduler | COMPLETE | 动态 mixed batch 与 sample 映射有源码和测试支持 |
| Engine → HTTP/SSE | GOOD ENOUGH | 有界队列、取消、超时、背压与停服排空；不是公网生产服务 |
| Benchmark / profiler → conclusion | GOOD ENOUGH | 分层时钟、独立 trial 和负结果；本次未重验包内容 |
| clean clone → run | WEAK | README 本机路径、前置条件、平台命令与两终端步骤未收束 |
| README → 主性能结果 → 代码 | WEAK | 强证据藏在深目录，性能页被最后一项研究主导 |
| planning / historical evidence 展示 | OVERBUILT | 历史内容仍像活跃入口；不等于需要删测试或重构产品 |

主要断点发生在交付和解释层，而不是“尚缺另一个核心 Feature”。

## 6. Strongest Technical Evidence

### 四个核心卖点与 Evidence Triangle

| 卖点 | Code | Correctness | Performance / System Evidence |
|---|---|---|---|
| CPU/CUDA 模型执行 | `src/minillm/runtime.cpp`、`cuda/runtime.cpp`、`cuda/layer.cpp` | 模型对照、固定输入与数值门槛；历史全量 12528 次比较 | 历史 70 进程 CPU8/CPU16/CUDA 模型基线 |
| 在线组批与流式服务 | `engine.cpp`、`scheduler.cpp`、`mini_cuda_runner.cpp`、`http_server.cpp` | 动态加入、槽复用、并发 tokenizer、HTTP 终态 | CUDA-SERVE-001 的 12 进程、288 请求及 NSys 时间线 |
| 内存与失败语义 | `paged_kv.cpp`、`cuda/storage.cpp`、`cuda/page_table.cpp` | 页集合 oracle 的一万次操作；预分配检查；执行后故障注入 | live/resident/credit 分离；故障隔离；稳态设备分配与传输记录 |
| 性能研究与停止规则 | 固定协议、分析器、数值检查入口 | 拒绝缺样本/错身份/错误门槛等反例 | A/A 噪声、不确定结果、paging 与 FP16 的停止决定 |

数字是各自历史采集身份的结果，不可统一标成 `9a571a5` clean build 重新测得。

### 建议公开展示的六行结果

| 层次 | Workload / baseline | 记录结果 | 必须带的限制 |
|---|---|---|---|
| CPU micro | 单线程热缓存 Q8_0×F32 点积；scalar 对 auto SIMD | 4.2921×，661.53→154.13 ns/dot | 不是模型或 Serving 加速 |
| CUDA model | 固定 12 workload，分别对 CPU8/CPU16；70 独立进程 | 24 比较中 14 faster、10 inconclusive | 对照是项目 CPU；不能宣称全面更快 |
| CUDA Serving | mixed-length，mixed 对 prefill_first；每臂三进程 | 130.68 对 122.03 token/s；中位数之比约 +7.09% | 固定旧 trace；非显著性结论；burst 差异不确定 |
| GPU paging | 同容量 cap8192 / 896 MiB KV | 吞吐配对变化中位数 -15.73% | 没有同容量显存节省 |
| GPU paging | 同 288 MiB KV 子预算 | 吞吐配对变化中位数 +4.31% | contiguous S1/224 MiB；paged S4/280 MiB+table；ITL 更差 |
| FP16 research | F16 matrix 对 F32，固定数值契约 | owned device memory -34.53%；blocked_correctness | 模型/Serving 性能未采集；Tensor Core 使用未验证 |

同预算结果是“预算内配置”的系统比较，不是只改变地址布局的 kernel 比较。不能用两个表格中位数重新替代逐 trial 配对统计。

长续写 FP16 失败行为具体是 cosine `0.9998858663580449 < 0.9999`，该行 argmax 一致；不是宣称模型所有输出错误。该预设门槛不一定等于业务质量门槛，但不能在看到结果后临时放宽来给候选补资格。

GPU paging 长 decode 的瓶颈解释必须限定到当前分离 QK/softmax/PV 实现。不能由此推断分页技术本身普遍更慢。

## 7. Weakest Links

限制求职价值的 Top 5：

1. **当前事实相互矛盾。** `THIRD_PARTY.md` 仍称没有自有 GPU Serving；GPU 研究页开头仍称采用护栏未测、PR 待合并；CUDA_SERVING 尾部仍称 paging 尚未采集。
2. **陌生人 Golden Path 不完整。** README 从作者绝对路径开始；`dev.sh build` 强制 Python/PowerShell 完整测试依赖；CUDA_ARCHITECTURES 默认 89；这些条件需要提前说明。
3. **性能主线失衡。** PERFORMANCE.md 几乎是 GPU paging 的结论页，CPU/CUDA 执行与已有 Serving 正结果不够醒目。
4. **历史与当前阅读面没有彻底分开。** 计划、状态日志和旧实验阶段描述仍占据技术阅读路径。
5. **最终候选复现证据尚未在本次获得。** main CI 成功不能替代用户 WSL 的 CUDA/HTTP smoke，也不能替代 Release 字节级下载与复核。

第 5 项是本次审计的确认缺口，不可直接说成项目从来没有本机验证。

## 8. Overbuilt / Redundant Areas

| Component | Role | Current value | Action |
|---|---|---|---|
| CPU/CUDA Runtime | CORE | 项目成立的基础 | KEEP / HIGHLIGHT |
| MiniCudaRunner + Engine | CORE | 模型接入在线系统与失败边界 | KEEP / HIGHLIGHT |
| Scheduler / HTTP | SUPPORTING | 形成端到端链路 | KEEP；不扩协议或策略 |
| CPU prefix / physical KV / COW | SUPPORTING | C++、数据结构和内存深度 | KEEP，GPU 主线中不平级堆砌 |
| GPU paged | RESEARCH | 容量能力与负结果 | KEEP / DEMOTE，显式 opt-in |
| FP16 matrix | RESEARCH | 数值停止案例 | KEEP / DEMOTE，不开放 Serving |
| LlamaRunner | REFERENCE | 功能与数值对照 | KEEP / DEMOTE |
| 核心、CUDA、HTTP 回归测试 | SUPPORTING | 保护真实 invariant | KEEP |
| validators / evidence packaging | SUPPORTING 或历史复现 | 保护证据真实性 | 冻结；不再创建“验证器的验证器” |
| PROJECT_PLAN* / NEXT_SPEC* / NEXT_OPT_SPEC | HISTORICAL | 设计演进记录 | ARCHIVE，保持引用可追溯 |
| ENGINEERING_LOG | 历史 + 当前问题入口 | 保留 failure 与原因 | SIMPLIFY 入口；不删除记录 |
| `cuda-vs-001.zip` | HISTORICAL EVIDENCE | 56,872,322 bytes 的已跟踪包 | 可迁移 Release；替代未验证前 KEEP |

未完成全库重复/引用分析，**没有一项产品源码获准直接 DELETE**。

## 9. Portfolio Bugs

PB-01：自研边界文档否认已存在的 GPU Serving，优先级 P0。
PB-02：研究页当前/历史状态混杂，优先级 P0。
PB-03：快速开始依赖作者路径和隐含工具条件，优先级 P0。
PB-04：GPU paging 负结果占据主性能入口，优先级 P1。
PB-05：README 能力表、构建方法、API、实验命令过密，优先级 P1。
PB-06：README 代码导航未突出 MiniCudaRunner/CUDA Runtime，优先级 P1。
PB-07：活跃计划措辞与 FEATURE FREEZE 冲突，优先级 P1。
PB-08：历史大包和活跃阅读面混在一起，优先级 P1；不是立即删证据的理由。

不把零 star、模型较小、S≤4、使用 cuBLAS、负结果或测试多列成项目缺陷。

## 10. Code / Documentation / Repo Hygiene

### 代码边界

`BatchState` 是 CUDA 长度账本，`PageTableState` 是映射与页归属账本，两者并非重复抽象。`BlockPool` 是 Engine 信用账本，不是 GPU allocator。不要为了减少类数量把三者合并。

GPU 完成点先校验 stream/status，再无分配提交页表和长度；preflight discard 与 post-launch poison 不等价。Engine 先验证所有 samples，避免非法 sample 导致本批部分发布；这不是整个网络传输或任意 host 分配失败下的事务保证。

CPU forward 在模型运算前追加 KV；因此不能将 CUDA 的事务状态保证套给 CPU，或宣称所有 Runtime 异常都可无损重试。现有 Serving 的 fail-stop 使用方式应保留。

CPU forward、Engine 与 result 容器仍有 host 动态分配。历史“稳态零分配”只适用于限定的项目设备分配观测，不适用于整个服务的 CPU heap。

一个低风险 P2 候选是让内部 `MappedFile` 显式禁止复制，表达 OS handle ownership。当前已读取路径中未观察到复制调用，不能将其宣传为已确认的 double-close bug。不做也不阻塞求职。

### 文档

外部主入口维持已有五份：README、ARCHITECTURE、PERFORMANCE、GPU_KV_STUDY、VALIDATION。深层实现文档继续保留，不追求全仓库只剩五份文档。

优先移动历史计划，不批量移动全部 implementation docs。迁移前检查脚本、测试、相对链接和锚点；冻结 source snapshots 不改，历史结果文件不做全文替换。

本次三份审计交付是一次性执行材料，不应同时成为 README 新的三条核心阅读入口。实施结束后归入历史或仅在个人工作目录保留。

### 仓库体量与分支

GitHub 元数据 size 为 116712（接口统计口径），不等于实际网络 clone 字节或本地 `.git` 大小；本次未测实际 clone size。已确认 `cuda-vs-001.zip` 56,872,322 bytes；README 15,045 bytes；ENGINEERING_LOG 114,184 bytes。没有可靠的全库 LOC / test-to-product 比值，本次不虚构。

对当前 main 的 compare 结果：

| Branch | Ahead | Behind | 建议 |
|---|---:|---:|---|
| feat/cuda-paged-kv | 0 | 2 | 已整合；保留研究 tag，分支可后续删除 |
| release/own-cuda-serving-public | 0 | 16 | 已整合；不再当活跃候选 |
| perf/cuda-f16-matrix-path | 0 | 17 | 研究 tag 保留；分支可归档 |
| feat/own-cuda-serving | 0 | 24 | 已整合；可归档 |
| feat/own-cuda-vertical-slice | 0 | 30 | 已整合；可归档 |
| fix/http-shutdown-drain | 0 | 23 | 已整合；可清理分支引用 |
| fix/windows-ci | 0 | 28 | 已整合；可清理分支引用 |
| build/wsl-native | 0 | 42 | 已整合；可清理分支引用 |
| fix/v-value-simd | 0 | 45 | 已整合；可清理分支引用 |

本次没有删除任何分支。执行时重新 fetch/compare；只有远程零独有提交还不够，还要排除本地未推送工作和未提交改动。禁止 force-push、历史重写和移动已发布标签。

### Evidence policy

Git：固定协议、输入、小摘要、分析、复现命令和必要代表样本。
Release：完整 raw、profiler、必要 source snapshot 与失败记录。
Local：临时执行、重复解包与可从 canonical raw 生成的中间文件。

已有 ARTIFACT_POLICY 基本满足此设计，应该执行而不是再发明一套政策。迁移包必须先下载、核 hash、解包并通过既有复核器；不能只看到 URL 存在就删 Git 原件。移除当前大文件不等于清除了历史大 blob。

## 11. Recruiter Readability

30 秒：首屏有项目定位，但能力列表过密，缺一个产品主结果表。将四个卖点和受限支持范围前置，研究结论后置。

5 分钟：ARCHITECTURE 的 ownership 说明值得保留；修复 THIRD_PARTY 的否定旧句，添加 CPU/CUDA/上游三路代码导航。

30 分钟：已有足够深度；把请求的三个完成点、KV 三种计数、CPU 与 GPU 共享差异、数值停止线连成故事。

主观评分不是行业认证或录用概率：

| Dimension | /10 | Reason |
|---|---:|---|
| C++ depth | 8 | 真实资源管理、线程池、视图、生命周期；非所有异常均有强保证 |
| CUDA depth | 7 | 自有算子、attention、显存与执行管理；非前沿融合/自研 GEMM |
| Systems depth | 8 | 准入、组批、背压、状态与停服边界 |
| LLM inference relevance | 8 | 真正 forward 与 Serving；模型和部署范围受限 |
| Performance engineering | 8 | 分层基线、噪声与停止规则；本次未复核 raw |
| Correctness discipline | 8 | 数值、状态机、故障与 oracle；无远程 GPU CI |
| Architecture clarity | 7 | 结构清楚，但公开表述仍有历史冲突 |
| Reproducibility | 6 | 固定依赖和身份好；陌生人 Golden Path 未独立验证 |
| README clarity | 5 | 主线有，但说明书和实验流水账偏重 |
| Portfolio differentiation | 8 | 可追问的系统链，不只是 API 调用 |
| Scope control | 6 | 已冻结但历史规划阅读面仍大 |
| Interview depth | 8 | 内容足够；不代表作者已能独立讲清全部细节 |

## 12. Interview Depth

Primary story：从 CPU 模型执行到自有 CUDA Serving，解释“如何使每个 token 的计算、所有权和交付都可信”。
Side story 1：相同容量和相同预算是两个不同问题；GPU paging 的收益为什么不够支持默认采用。
Side story 2：FP16 显存下降但数值门槛未过，为什么停止，而不是降低门槛追求漂亮数字。

### 明天 45 分钟必须能讲清的十题

1. 一个请求从 HTTP 到 token 返回的完整执行链，以及代码入口。
2. 自研、vendor primitive 和 reference 的边界；Q8_0 文件为何产生 F32 GPU 权重。
3. Tensor row/column/stride 与 cuBLAS `Y=XW^T` 的布局映射。
4. CPU SIMD/线程池如何工作；为什么 4.29× 不可外推。
5. KV 的 112 KiB/token 从何而来；credit、assigned、resident 为什么不同。
6. CPU prefix/COW 和 GPU 独占页池有什么根本区别。
7. continuous batching 如何混合 prefill/decode；取消在什么边界生效。
8. CUDA preflight、post-launch poison 和 HTTP 排空三个边界为什么不能混淆。
9. TTFT、mean TPOT、ITL、goodput 的时间边界和分母是什么。
10. paging/FP16 为什么没有成为默认；哪些结论只是限定实验而不是普适定理。

### 分层问题库

Basic：GGUF parser 与 mmap 分工；Q8_0 的块与尺度；RMSNorm/RoPE/GQA；KV 字节公式；TTFT/TPOT/ITL；cuBLAS/CUB 的贡献边界。

Intermediate：row-major 到 cuBLAS 的转置与 stride；完整页共享和尾页 COW；线程池 generation/condition variable/异常回传；整批 B 上限与 chunked prefill；typed view 的 owner 寿命；clear 为什么不等于 cudaFree。

Deep Dive：页映射与长度的双提交不变量；metadata staging 的异步借用寿命；device block table 与直接 QK/PV 地址解析；整批校验为何不等于网络事务；A/A 与跨 cohort 基线漂移；同预算不同 S/capacity 的因果边界。

Challenge：为什么不用 llama.cpp 作为唯一实现；argmax 相同为何仍不能放过 cosine 失败；当前代码能否声称零 heap 分配；三进程约 7% 是否就是显著提升；CPU forward 异常后是否支持无损重试；哪些结论不能外推到其它 GPU 或模型。

问题库只用于个人准备，不全部塞进公开文档。

## 13. Final Value Proposition

一句话：**自研 C++/CUDA 大模型运行时与在线服务，以数值验证和性能实验解释内存、延迟及容量取舍。**

30-second pitch：

我实现了一条从 GGUF 权重到在线 token 返回的 C++20 推理链，CPU 和 CUDA 都执行自己的 Qwen3 forward，再通过同一个 Engine 提供连续组批和 SSE。项目重点不是 Feature 数量，而是权重与 KV 所有权、执行失败后的隔离，以及从模型到请求的测量边界。GPU 分页和 FP16 都经过实验，但没有为了漂亮数字强行成为默认。

3-minute technical pitch：

这个项目首先解决的是“我是否真正理解一次模型前向”，而不是“能否调用一个大模型接口”。GGUF 的元数据和 tokenizer 复用固定 llama.cpp；文件映射、张量视图、CPU SIMD/线程池、Qwen3 forward 和 CPU 物理分页 KV 是项目实现。随后加入自有 CUDA Runtime，在初始化时把有效权重解量化并常驻 GPU，矩阵使用 cuBLAS，归一化、RoPE、attention、KV 与执行管理由项目实现。

第二层问题是把模型变成一个有明确生命周期的服务。Engine 在请求准入时保守预留容量，每轮把 prefill 和 decode 重新组批；MiniCudaRunner 将批次直接传给 Runtime，不调用 CLI 子进程。模型线程完成执行并校验整批 sample 后才发布事件；取消、超时和背压在定义好的边界收尾。GPU 执行后失败会 poison，逻辑信用释放不意味着物理 KV 回滚。HTTP 还有独立的响应排空，因为结果已经入队不代表客户端已经收到。

第三层是验证这些设计究竟得到什么。模型数值用相同有效权重的参考，而不是随意拿不同精度结果比较；micro、完整模型和 Serving 分别测量。历史 Serving 的固定混合负载中，mixed 比 prefill_first 的吞吐中位数高约 7.09%，但突发负载没有稳定优势。GPU paging 在同容量下反而更慢，在同预算异长请求下有有限吞吐收益但 token 停顿更大，所以保持 contiguous 默认。FP16 矩阵候选降低了 owned 显存，但长续写 cosine 未过原门槛，于是没有开放 Serving。这个项目最终证明的是：我能建立完整执行链、检查资源和失败边界，并对优化作出可复核的采用或停止决定。

简历只保留一版，四条：

- 实现 C++20 CPU/CUDA Qwen3 推理链，复用 GGUF parser/tokenizer 与 cuBLAS，独立实现 forward、SIMD 和 KV；以固定输入数值对照验证，明确非 native Q8 CUDA GEMM。
- 将自有 CUDA 接入连续组批与 SSE，完成动态请求加入、槽复用及有界背压；固定混合负载吞吐中位数较 prefill_first 提升约 7.09%，保留突发负载的不确定结果。
- 实现 CPU 页共享/COW 与 GPU 页池事务，区分容量信用、live KV 和 resident allocation；以一万次页状态操作及故障注入保护回收、提交和 poisoned 隔离边界。
- 建立模型/服务分层实验与数值门槛：GPU paging 同预算有有限收益但延迟更差，FP16 显存下降仍因数值门槛未过而停止；保留原始身份、失败及复核入口。

技术栈：C++20、CUDA、cuBLAS/CUB、AVX2/FMA/F16C、CMake/CTest、Nsight。

## 14. Feature Freeze Verdict

不增加 Q4/native Q8 CUDA、FlashAttention、Graph、多 stream、异步执行、speculative decoding、GPU prefix、多模型、多 GPU、PD 或 Web UI。现有独立 Runtime、Serving、测试和研究已足够形成作品。

DO NOT TOUCH：Runtime math、scheduler 策略语义、HTTP 生命周期、CUDA 单调用者/单 stream/sync/poison 合同、冻结输入与阈值、既有 raw 和历史 source identity。

HIGHLIGHT：两条 forward；MiniCudaRunner；Engine iteration 的整批校验；HTTP ResponseLease；CPU PagedKV；GPU PageTableState；CUDA-SERVE-001 分析与独立时间线。

ARCHIVE：旧 PROJECT_PLAN/NEXT_SPEC/NEXT_OPT_SPEC、完成阶段的状态说明、已合并分支引用。保留设计原因和失败记录。

DELETE：现在没有获准删除的产品代码。只有已经确认具有可下载、可校验、可复核替代品的重复产物，或已整合且无本地独有工作的分支引用，才可进入后续删除清单。

完成最多三个收尾阶段后，将新增投入转向真实上游贡献。对当前背景而言，接受外部维护者评审、修复真实问题、证明在别人代码库里工作，预计比继续增加第 N 个个人项目 Feature 有更高边际收益。这是求职策略判断，不是录用保证；不在这里自动规划或提交上游 PR。

## Appendix A. 行业概念对照与岗位校准

| MiniLLM concept | Industry analogue | Same? | Important difference |
|---|---|---|---|
| 每轮 token-budget mixed batching | vLLM scheduler、TensorRT-LLM IFB | 同类概念 | 本项目同步、S≤4，没有同等规模和通用性 |
| ModelRunner | vLLM model runner / worker execution | 职责相近 | 不是相同接口；vLLM 的异步与 GPU-native 执行不能归给本项目 |
| CPU prefix Trie / KV reuse | SGLang RadixAttention 的 prefix reuse | 目标部分相似 | 本项目 CPU 路径；GPU 不共享；不能说实现了完整 RadixAttention |
| GPU page table + direct attention | FlashInfer paged-KV attention | 数据访问概念相近 | FlashInfer 本身不负责 allocator 策略；本项目密集表和分离算子不是其优化实现 |
| 保守信用与 cache pool | TensorRT-LLM KV cache manager | 同类资源问题 | 不等于相同调度、逐步 admission 或 preemption |
| LlamaRunner | llama.cpp | 直接复用 | 对照执行不算作者自研模型路径 |

当前可读官方岗位中，OpenAI Inference Performance Optimization 强调端到端 profiling、benchmark 和延迟/容量/成本分析；NVIDIA 的 inference performance 岗位强调 C++/GPU 编程、Nsight、受控实验与可复现结果。项目已经具有对应实现证据，但没有证明生产集群、跨团队交付或多卡通信能力。

NVIDIA 的 New College Grad 页面也列出推理 benchmark 与系统优化，说明这类工程能力不只属于高级岗位；具体任职资格应逐 JD 判断，不能据此宣称本科学历自动满足。Fireworks 官方 careers 可确认 Performance Optimization 和 LLM Infrastructure 岗位类别，但其 JS 正文未获取；字节官方职位正文本次也未获取，因此不伪造两者的精确任职要求。

## Appendix B. 主要证据位置

所有项目路径默认固定到 `9a571a5d7a6570c6dfeeac9062adefca1840534e`，历史报告内部记录的采集身份保持原样。

- 元数据：branches、pulls、compare、tags、releases；Actions run `36709597589`。
- 执行：`src/minillm/gguf_model.cpp`、`runtime.cpp`、`kernels_avx2.cpp`、`parallel.cpp`、`paged_kv.cpp`。
- GPU：`src/minillm/cuda/{runtime,storage,layer,matrix,page_table}.cpp`、`attention.cu`。
- Serving：`src/{engine,scheduler,mini_runner,mini_cuda_runner,http_server,config}.cpp`、`include/llmserve/model_runner.h`。
- 测试：`tests/gpu_page_table_tests.cpp`、`tests/cuda_serving_tests.cpp`；其余测试以构建注册、文档和选读交叉核查，未声称全部逐行阅读。
- 构建：`CMakeLists.txt`、`.github/workflows/ci.yml`、`scripts/dev.sh`、`AGENTS.md`。
- 证据：`benchmarks/results/{cuda-model-baseline,cuda-serving-001,cuda-precision-001,gpu-kv-001}/`、`simd-q8-dot.json`。
- 文档：README、ARCHITECTURE、PERFORMANCE、VALIDATION、GPU_KV_STUDY、CUDA_SERVING、CUDA_NUMERICS、THIRD_PARTY、ARTIFACT_POLICY、VERSION_CONTROL。

官方对照资料：
- https://docs.flashinfer.ai/tutorials/kv_layout.html
- https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/
- https://vllm.ai/blog/2026-03-24-mrv2
- https://docs.sglang.io/
- https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/features/paged-attention-ifb-scheduler.html
- https://openai.com/careers/software-engineer-inference-performance-optimization-san-francisco/
- https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer---AI-Inference-Performance_JR2024262
- https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Inference-Performance-Engineer---New-College-Grad-2026_JR2014441
- https://fireworks.ai/careers

职位页面仅用于能力校准，不保证申请状态、学历适配或录用可能性。
