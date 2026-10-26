# PROJECT_PLAN_V4 — 一项精度/性能优化，然后冻结功能

> Historical（历史计划）。FP16 候选已停止，当前收尾状态以 `GPU_KV_DECISION.md` 为准。

## 0. 文档身份与最终决定

- Audit Date：2026-09-26，Asia/Tokyo。
- Audit Branch：`feat/own-cuda-serving`。
- Audit HEAD：`6ca7d2dfeccb38b11b7596a552046651afcb693e`。
- Main HEAD：`68ac275913207975a88e2090c6617467e351301c`。
- Previous Reference：`5a4508d6ace7777e39460675de9ed770fcb66d43`。
- 当前提交自己的 CI：run `36244664620`，success。
- 本文件是后续决策，不表示 FP16 路径已经实现或已经取得性能提升。
- 唯一 Primary：**FP16 常驻大矩阵权重 + FP16 矩阵输入 + FP32 accumulation/output 的精度路径研究**。
- 唯一 Backup：**contiguous KV 上的 fused/online-softmax attention**；仅在 Primary 被停止且 attention entry gate 成立时进入。
- Recommended Remaining Major Features：**计划 1 项；最多尝试 2 条路线，不并行；Primary 成功后不再自动启动 Backup。**
- CPU、当前 F32 CUDA 模型、GPU Serving、现有数值语料、M1/M3-1 原始实验全部作为已完成 baseline，不重做。
- 后续逐步实验只使用当前服务器与现有 benchmark/validator；不新建 evidence framework。
- 这是个人 C++/CUDA/LLM Systems 作品的收尾计划，不是 vLLM clone 的路线图。

执行前重新读取 refs。若代码有更新，只审计 delta，记录实际候选 SHA；不把新代码的结论附会到本次审计 HEAD。

## 1. 已核实事实与证据边界

### 1.1 证据索引

以下路径均相对于仓库根目录，按 Audit HEAD 固定：

| ID | 路径 / 来源 | 支持的结论 |
|---|---|---|
| E1 | `src/mini_cuda_runner.cpp`、`include/llmserve/model_runner.h` | own-CUDA adapter、同步执行、sample 映射、capability、poisoned 隔离 |
| E2 | `src/engine.cpp`、`apps/server_main.cpp`、`src/http_server.cpp` | 真实 Engine/HTTP/SSE 接入；保守 credit；整批样本验证后发布 |
| E3 | `src/minillm/cuda/runtime.cpp`、`layer.cpp`、`storage.cpp` | 常驻权重、完整模型、workspace、连续 FP16 KV、紧凑输出 |
| E4 | `src/minillm/cuda/matrix.cpp`、`context.cpp` | 当前矩阵全 F32，`CUBLAS_COMPUTE_32F_PEDANTIC`，handle 使用 `CUBLAS_PEDANTIC_MATH` |
| E5 | `src/minillm/cuda/attention.cu` | 独立 QK / softmax / PV，materialized scores/probabilities，非 FlashAttention、非 PagedAttention |
| E6 | `benchmarks/results/cuda-serving-001/analysis.md`、`summary.json`、`protocol.json` | 两 trace、两 policy、三 trial 的 Serving 观察 |
| E7 | 同目录 `validation.json`、`evidence.json`，canonical Release bundle | 生命周期、单次 NSys、采集身份和原始产物索引 |
| E8 | `tests/cuda_serving_tests.cpp`、`tests/cuda_model_tests.cpp`、`tests/cuda_full_validation.cpp` | adapter、确定性 mixed/reuse/fault、模型数值检验入口 |
| E9 | `src/minillm/qwen3_model.cpp`、`include/minillm/model_types.h`、既有 weight/memory plan | 真实 Qwen3 结构和存储口径 |
| E10 | `docs/PROJECT_PLAN_V3.md`、`docs/NEXT_SPEC_V2.md`、`docs/EXECUTION_STATUS.md` | 旧范围与完成状态；状态声明必须结合 E1–E8，而非单独采信 |

审计依据是代码、提交记录、发布的逐轮摘要、验证记录及 profiler 分析；没有在本机重新运行用户 GPU，也没有独立执行完整 Release 原始包的重算。本计划的第一步允许用已有包/SQLite补齐决策所需的小型分解，不要求重新采集整套 baseline。

### 1.2 当前应保留的实测结论

- 12 个正式服务进程，288 请求成功，9216 输出 token。
- mixed-length：mixed 130.68 token/s、goodput 4.084 req/s；prefill_first 122.03、3.496。
- 上述吞吐约 +7.09%、goodput 约 +16.82%，是本机固定 trace 的三轮描述，不是统计上普适保证。
- burst-reuse：70.02 与 70.22 token/s，方向随 trial 改变，保持 measurement_inconclusive；固定空闲间隔在吞吐分母中。
- 最大单 token ITL：mixed-length 31.14 vs 340.46 ms；burst-reuse 47.77 vs 466.71 ms。不能用 request mean TPOT 隐藏长停顿。
- 现有较差策略不是新优化唯一 baseline；后续固定 mixed，避免“击败已知较差策略”夸大新 kernel 收益。
- GPU owned device allocation 3,449,229,312 bytes；其中连续 KV 939,524,096 bytes，容量 8192 token。
- 1255 peak live tokens 只来自一条 NSys trace，不代表所有正式进程的峰值。
- NSys 288 forwards 含 8 warmup、280 measured；201216 kernels 的平均值约 699/forward，不能除以280。
- 单次测量 device span 5702.04 ms、busy union 5211.49 ms、gap 490.56 ms。busy fraction 约91.4%，不是 SM occupancy 或整个服务窗口利用率。
- API elapsed 包含排队/等待与 instrumentation；4205.71ms cudaMemcpyAsync 不能当成75KB H2D的纯复制成本。
- 启动约10秒，不能与 steady-state throughput 混为同一目标。
- 正式采集对应 `b1ced89... + dirty snapshot`，不是 `6ca7d2d` clean build 的重新实测；最终发布候选 CI 属于6ca7d2d。

### 1.3 尚未证明

- 当前 Serving 各 batch 的 matrix/attention/launch 因果份额。
- 当前 workload 的 DRAM 是否饱和。
- FP16 在 M=1 上是否使用 Tensor Core，以及转换成本是否抵消收益。
- Paging 是否实际增加当前 hard-coded S=4 的并发上限。
- pinned metadata 是否减少关键路径，而非只是把等待移动到同步 API。
- 跨不同 workload、显卡、驱动的普适加速。
- 完整模型质量：固定 logits/token 检查不等于语义能力评估。

## 2. 为什么选精度路径，而不是先做分页或再改调度

当前模型维度为 layers=28、hidden=1024、FFN=3072、query heads=16、KV heads=8、head_dim=128、vocab=151936。query width=2048，不是 hidden/heads 推导出的64维head。

大矩阵参数：
- transformer body：440,401,920。
- tied embedding/LM head：155,582,464，只计一份物理权重。
- norm vectors：65,536，保持F32。
- 当前F32权重：2,384,199,680 bytes。
- 把大矩阵与embedding改为F16、norm仍F32：预计1,192,230,912 bytes。
- 预计权重减少1,191,968,768 bytes，约1.110GiB。
- 加一个最大128×3072×2的cast scratch，预计总owned约2,258,046,976 bytes；相对当前减少约34.53%，最终以对齐后的计划和实际分配为准。

这些是静态推导，不是已取得的实验结果。

相比之下，当前全部KV预留仅896MiB。1255/8192≈15.32%是单trace的占用观察；相同大小的paged pool仍然会预留同样显存。要让分页增加可服务并发，还必须处理S=4限制、pool budget和准入，不是换一次accessor就完成。

精度路径可以在不改变KV、scheduler和Serving生命周期的前提下，研究：
1. 存储减少是否兑现；
2. 转换后的矩阵路径是否改善prefill/小batch；
3. 数值误差是否可接受；
4. 为什么model收益可能没有完全传导到Serving。

不声称FP16必然是当前已证实的主要性能瓶颈。确定的是F32存储成本；计算收益由第一阶段受控探针与后续实验决定。

## 3. 优先级与候选决策

| 排名 | Candidate | Engineering ROI | Portfolio ROI | 决策 |
|---:|---|---|---|---|
| 1 | B 精度/FP16 | 高，内存收益可计算；性能待证 | 高，增加numerical/architecture闭环 | PRIMARY |
| 2 | L 完成一个优化后冻结 | 高，直接防止无限扩张 | 很高，提高可讲性与外部评审投入 | 全程约束与收尾 |
| 3 | D contiguous attention fusion | 中高，需证明目标attention份额 | 很高，CUDA/算法/IO较深 | 唯一条件BACKUP |
| 4 | A GPU paging/PagedAttention | 中，已有占用证据但缺容量压力 | 很高，memory systems/ACM契合 | 本轮不做 |
| 5 | C native Q8 small-M | 潜力高，双路径与验证成本高 | 高，量化kernel差异化 | 本轮不做 |
| 6 | F scheduler/tail | 已有真实现象，但mixed已解决部分 | 中高，已有卖点较多 | 保持现policy |
| 7 | E host/launch小优化 | 小范围可能有效，因果未证 | 中，单独难成主卖点 | 只解释，不顺手实施 |
| 8 | J CUDA Graph | shape/捕获/指针/生命周期成本高 | 中高 | 延后 |
| 9 | G incremental admission | 当前没有资源瓶颈证据 | 高但证据不足 | 不做 |
| 10 | H GPU prefix | 依赖共享/引用管理 | 中高 | 不做 |
| 11 | I async/multistream | overlap机会未证明 | 高但复杂度大 | 不做 |
| 12 | K大量CPU优化 | 新增证据边际价值较低 | 中 | 冻结为reference/独立CPU产品 |

等级是针对本项目当前状态的审阅判断，不是行业通用排名；不赋予虚假精确分数。

## 4. M4-0 — 决策冻结与小范围入口检查

| 字段 | 内容 |
|---|---|
| ID | M4-0 |
| Type | MEASUREMENT / DECISION |
| Goal | 固定Baseline、precision contract、对照与停止条件；只补决策缺口 |
| Why Now | M3-1已完成，不需要再建设Runtime或Serving |
| Evidence | E1–E10 |
| Portfolio Rationale | 显示能够从现有证据决定研究，而不是按热门词扩功能 |
| Dependencies | 最新HEAD差异检查；现有Release原始包 |
| Scope | 读取已有NSys/SQLite；按measured forward拆matrix、QK/softmax/PV、其他kernel；列M/R/context与复用次数；检查现有validator的精度假设 |
| Non-Goals | 重采M1、补trial直到显著、新profiling框架、跑所有375shape |
| Implementation | 保存一份decision-note；冻结接口/阈值/工作负载；提取12–16个真实矩阵shape供M4-1 |
| Benchmark | 无新增全套baseline；后续16shape探针包含在M4-1总预算内 |
| Correctness Gate | 历史身份不重写；不把摘要推测当硬件计数器；明确pre/post-append context |
| Performance Gate | 先用已有kernel fraction与探针做Amdahl判断；不因目标收益小就改主指标 |
| Resume Value | 性能建模与实验设计 |
| Interview Value | 为什么先选这个，而不是paging/graph？ |
| Difficulty | S |
| Stop Condition | 入口记录与预注册完成即停；包不可得则标未知，以同预算内小探针继续，不搭新下载平台 |
| Anti-Overengineering Budget | 1份decision-note；0新公共框架；不额外采NSys；不得新增验证验证器 |

## 5. M4-1 — 唯一主要优化：FP16矩阵边界、FP32 Runtime主体

| 字段 | 内容 |
|---|---|
| ID | M4-1 |
| Type | RESEARCH + INFRA；通过门禁后PRODUCT |
| Goal | 在同一自有Runtime中得到可选择的F16矩阵路径，并量化内存/精度/三层性能 |
| Why Now | 当前全部F32矩阵与2.384GB权重已核实；GPU Serving已能作为真实验证终点 |
| Evidence | E3–E9；第2节静态模型 |
| Portfolio Rationale | 增加numerical systems、typed layout、Tensor Core实证与端到端收益归因，而不是再加backend |
| Dependencies | M4-0；现有F32reference与CUDA测试 |
| Scope | 大矩阵/embedding F16常驻；矩阵输入F16；FP32 accumulation/output；其余math、KV、scheduler不变 |
| Non-Goals | BF16/TF32自动扫参、Q8native、genericGEMM、全模型half激活、pagedKV、graph、async |
| Implementation | 一个precision enum；WeightRecord显式device dtype；半精度gather；一个可复用cast scratch；cuBLAS受控入口；沿用原forward和adapter |
| Benchmark | micro≤16shape；model≤6workload；两条现有Serving trace，仅mixed，两模式各3trial；新NSys≤1，NCU≤1 |
| Correctness Gate | 转换/算子oracle与模型质量gate分开；固定teacher forcing、短golden、chunk/mixed/state/fault通过；F32原路径不回归 |
| Performance Gate | 预注册主model指标为prefill-128 latency，目标≥10%；owned减少≥30%；decode/mixed与Serving关键护栏无超过预注册噪声/退化；三层分别解释 |
| Resume Value | 用真实数字说明F32→F16边界、显存变化、模型收益与Serving适用范围 |
| Interview Value | 存储/计算/累加如何区分；何时TC不生效；误差如何定位；为何无Serving收益 |
| Difficulty | L |
| Stop Condition | 见NEXT_OPT_SPEC；正确性失败不放宽门槛；最多一个实现候选与一次有证据的修订；预算耗尽保留负结果 |
| Anti-Overengineering Budget | 1个precision enum；复用原所有者；最多4个主要commit；0新backend/Server/统计框架；最多1个canonical bundle |

产品晋升不等于把默认立即改为F16。默认保留F32，候选通过范围内明确可选。

结果必须按以下状态收尾：
- `performance_success`：正确、内存目标与主性能目标通过，护栏通过。
- `memory_only_success`：正确且显存减少，性能无显著提升但护栏通过。仍是有效成果，不能宣传加速；不自动启动Backup。
- `research_negative`：有可靠负结果，保留报告，候选不晋升。
- `measurement_inconclusive`：预算内噪声阻止性能判断，不能改成“无退化”。
- `blocked_correctness`：不能满足数值/生命周期，停止，不继续包装成完成产品。

## 6. M4-2 — 唯一条件Backup：contiguous fused attention

| 字段 | 内容 |
|---|---|
| ID | M4-2 |
| Type | CONDITIONAL RESEARCH |
| Goal | 仅在Primary不能形成可采用结果时，闭合一个attention IO/执行优化问题 |
| Why Now | 不是现在启动；只有Primary停止后才重新判定 |
| Evidence | E5、已有长context模型profile；必须补目标workload的attention时间份额 |
| Portfolio Rationale | 在线softmax、tiling、数值稳定性、KV访问和GPU IO具有较深解释价值 |
| Dependencies | M4-1已停止；备选不并行；目标workload中attention占比≥25%，且有可复验baseline |
| Scope | 同一contiguous FP16 KV上的融合attention；保留原QK-softmax-PV reference |
| Non-Goals | 同时分页、prefix共享、dtype变化、改scheduler、FlashAttention品牌性能承诺 |
| Implementation | 开始前另写一个小SPEC；最多两个tile候选；复用Q/K/V与state；device-only读取 |
| Benchmark | ≤8attention shape、≤4model case、1Serving trace两模式各3trial、≤1NSys/必要时1NCU |
| Correctness Gate | 因果/GQA/尾部、在线softmax数值、长context logits、现有Serving状态通过 |
| Performance Gate | 预注册主model指标；不能仅靠micro收益；必要时接受有限工作负载收益 |
| Resume Value | 可讲述IO与算术/occupancy/launch权衡 |
| Interview Value | 在线softmax为何稳定；score不落全局内存；为何复用不等于实现paging |
| Difficulty | L–XL |
| Stop Condition | entry不成立直接feature freeze；两种变体均无模型价值或验证成本过高则停；不再选第三条大路线 |
| Anti-Overengineering Budget | 只启动一条attention路径；不建通用kernel生成器；无新evidence层；完成后不得自动做GPU分页 |

Primary成功后该milestone删除或保持未执行，不为了计划完整而实现。

## 7. M4-3 — Feature Freeze、作品表达与Upstream

| 字段 | 内容 |
|---|---|
| ID | M4-3 |
| Type | PORTFOLIO / DOCUMENTATION / UPSTREAM |
| Goal | 将已验证系统和一个研究闭环收敛成可面试、可审查的作品 |
| Why Now | CPU/GPU Runtime与Serving已完整，再加广度的边际收益低于形成可解释证据 |
| Evidence | M1、M3-1、M4-1或条件M4-2的实际结果 |
| Portfolio Rationale | 个人能力需要现场解释与外部review证明，不能由代码/测试数量代替 |
| Dependencies | 一项研究完成，或Primary与唯一Backup均到停止线 |
| Scope | 可运行主分支、README边界、架构图、三层结果表、技术复盘、4–6个卖点、一个窄upstream问题 |
| Non-Goals | 新模型、UI/RAG、多卡/分布式、更多相似独立项目、为了PR数量拆分无意义提交 |
| Implementation | 把成功与负结果都写入主故事；整理可复现命令；选择相关上游bug/性能问题，不假设一定被merge |
| Benchmark | 不新增完整benchmark；文档勘误只复核必要已有样本 |
| Correctness Gate | 文案、数字、SHA、支持范围对应；未验证能力不写入简历 |
| Performance Gate | 只引用通过的对应层指标；不把model收益称Serving收益 |
| Resume Value | 4–6个可证技术点，而不是feature墙 |
| Interview Value | 能解释至少10个深入问题，并现场定位/修改一个关键函数 |
| Difficulty | M |
| Stop Condition | 可展示入口、报告、复现路径及面试讲解完成后，个人项目仅维护bug/兼容性 |
| Anti-Overengineering Budget | README、1篇技术报告、1张架构图、4–6个技术点；不以写文档为名建网站或新平台 |

### Stop Project Gate

满足以下即进入FEATURE FREEZE：
1. main或明确发布tag能构建CPU/GPU并运行已支持模型和HTTP；分支能力不能永远隐藏在非默认分支。
2. 自研/vendor边界、F32/F16/KV精度和限制明确。
3. 一个受控研究有假设、对照、数值、profiler及micro/model/Serving结果；负结果也允许收尾。
4. 关键内存与请求生命周期检查通过，CI与实际提交对应。
5. 数据可获取、复现命令完整，但没有递归封包。
6. 用户能解释自己的实现、trade-off与失败原因，而不只是复述AI生成文档。

现有项目已经可以用于求职；不必等待FP16或PagedAttention才能投递。此计划不是招聘资格保证。

## 8. 为什么转向Upstream

当前可访问官方岗位强调systems programming、prefill/decode性能模型、内存/KV、可测量优化与可靠性。它们也涉及多人协作、真实生产和更广硬件；单机作品不能证明全部要求。

个人项目再加一个feature，主要增加可控环境内的实现证据。一个有复现、测试与review讨论的upstream贡献，补充的是在陌生代码库、兼容约束和外部评审下解决问题的能力。

用户已有开源贡献经验；本计划不要求从文档小修重新起步。建议收尾后选择与本次实际发现相关的数值、边界、内存或CUDA问题；具体repo/issue需要另行审计，不能承诺merge、不能把个人实验结果直接推广到上游。

官方参考（本次审计读取；招聘状态可能变化）：

```text
NVIDIA TensorRT-LLM JR2022327
https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/AI-Computing-Software-Development-Engineer--TensorRT-LLM_JR2022327

OpenAI Model Runtime
https://openai.com/careers/software-engineer-model-runtime-san-francisco/

OpenAI Model Inference
https://openai.com/careers/software-engineer-model-inference-san-francisco/

Anthropic Performance Engineer, Inference Engine
https://job-boards.greenhouse.io/anthropic/jobs/5418323008

CUDA 12.8 cuBLAS
https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html

CUDA 12.8 API synchronization behavior
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html

FlashAttention original paper
https://arxiv.org/abs/2205.14135
```

## 9. Recommended Next Development Path

1. 冻结6ca7d2d及其原始Serving结论，检查新HEAD delta。
2. 用已有timeline和有限真实shape核对精度路径的性能潜力，记录未知。
3. 依NEXT_OPT_SPEC实施一个F16矩阵边界方案，保持原F32 reference。
4. 完成数值、memory与三层实验，按预注册门禁决定晋升、负结果或不确定。
5. 成功或memory-only成功后直接feature freeze；只有失败且entry成立才考虑唯一Backup。
6. 整理作品与面试讲解，把主要开发时间转向真实upstream贡献。

## 10. Not Now

不做GPU paging/PagedAttention、nativeQ8、GPU prefix、复杂scheduler/admission、CUDA Graph、async/multistream、更多CPU优化、更多architecture、MoE、多卡/PD/分布式、RAG/Agent/UI；不扩验证框架，不追加trial追求显著，不把旧计划剩余任务当作债务。
