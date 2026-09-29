# GPU KV 共享容量研究

活跃计划为 [PLAN-V4-KV-20260928](PROJECT_PLAN_V4_KV.md)，规范为
[GPU-KV-001](NEXT_SPEC_V3.md)。本研究仅改变 KV 内存组织，保持 F32 矩阵、
F16 KV、现有 attention 数学、单 stream 与同步完成；不含 prefix sharing、
fusion、Graph、async 或新的精度路径。

## 当前能力

| 范围 | 状态 |
| --- | --- |
| 连续 CUDA 模型与 Serving | 可执行，默认不变 |
| Host 页表状态 | 已实现；容量共享，每个已分派页独占 |
| 内部配置 | `CudaKvLayout`、capacity、P=16，边界校验不依赖 CUDA |
| 长度账本 | 唯一 `BatchState`，提供只读 pending lengths |
| 设备 slab/table 与分页 store | 已实现；同一 store 数学使用不同地址策略 |
| 分页 QK/PV、完整层与 Runtime 事务 | 已实现；与连续布局共用数学，checked completion 后提交 |
| Paged C++ 模型 | 可执行，仅 F32 矩阵；S1/S4 真实模型逐位对照通过 |
| Paged CLI/Serving | 已接入；显式选择布局，Serving 信用与物理容量一致，默认仍连续 |
| 指定预算的请求共存与压力进展 | 确定性真实模型验证通过；不代表吞吐结论 |
| 采用决定 | 正式性能护栏尚未验证，不判为产品可采用 |

`PageTableState` 只持有 active/pending table、free IDs 和本批 journal。
所有容器在构造时预留；不持有设备地址，不保存第二份 token length，
不增加 refcount、COW、通用 allocator 或独立调度器。

## 状态合同

| 阶段 | 映射与页归属 |
| --- | --- |
| ready | active 对应 committed length；assigned + free = N |
| prepare | 校验所有长度与总需求，生成 pending；不消耗 free IDs，不修改 active |
| begin_execution | 无分配地消费已验证的新页；pending 借用保持到完成点 |
| commit | 调用方已检查 stream/status；无分配交换映射，清除 dirty |
| discard_prepared | 不改变 active/free；保留此前 clear 的 dirty |
| poison | 隔离旧页与本批新页；不能通过 clear、commit 或再次 prepare 恢复 |
| clear | 只允许 ready；归还该 sequence 的页，清 host table，标记 dirty |

Host 计数不是 GPU 显存释放。clear 归还的是池内 page ID，
resident slab 仍由 `CudaStorage` 持有。host、设备、模型与 Serving 验证分别记录，
不将某一层的通过扩大为所有层的保证。

## 发布候选

已有能力的整合候选为 `e7e2ced73cb79b82e3025db9e9035982bd858b58`，
位于 `release/own-cuda-serving-public`，包含 `57268f9` 的 HTTP 排空修复。
[PR #2](https://github.com/lilong555/mini-llm-runtime/pull/2) 等待用户审阅；
未合并 main，未移动历史标签。分页实施位于独立 `feat/cuda-paged-kv` 分支。

- 候选自身 CI run `36517808344` 五任务成功；PR CI run `36517815149` 五任务成功。
- 本机 own-CUDA 构建与 22/22 CTest 通过，8-token CLI 与冻结 golden 相同。
- HTTP 12/12 通过，包含 SSE、断连、背压、超时与停服终态。
- 停服时 3 active + 3 queued，六响应各一个取消终态及一个 `[DONE]`；
  服务正常退出，无排空等待超时。

记录位于 `.run/gpu-kv-001/release-smoke-e7e2ced/`。`identity.json` 绑定干净源码、
二进制与模型；`verification.json` 的 SHA-256 为
`601cc4be4e8b9c675fd9ba254ad7070ed13a951e2fd24843aae58ea0093c088f`。
该 smoke 没有产生新的性能基线。

## Host 验收

第一组验证源码为 `e7e2ced` 加固定 dirty snapshot，不是最终提交的 clean build 实测。
原始输出在 `.run/gpu-kv-001/host-state/`，与发布 smoke 分开标识。
第一组提交 `48d43ad` 自身 CI run `36520276353` 五任务通过，不替代本机 GPU 检查。

| 检查 | 结果 |
| --- | --- |
| 无 CUDA/llama 的独立核心 CTest | 12/12 |
| 自有 CUDA CTest | 23/23 |
| 独立核心 ASan/UBSan CTest | 12/12 |
| Host 页状态用例 | 9/9，seed=20260928，10,000 次状态操作 |
| 事务分配 | prepare/start/commit/discard/clear 合法路径在禁用普通 new 时通过 |
| 原 F32 实模型 | S1/S4 2/2，通过既有数值与短 golden 合同 |
| 原 F32 HTTP | 12/12；停服时 2 active + 4 queued，六个完整单终态 |
| 第一组当时的 Paged 拒绝 | Runtime 在模型加载前拒绝；Storage 在设备分配前拒绝 |
| 新设备 memcheck | 未执行；本组没有新增设备 kernel |

Host tests 比较页集合、边界与既有映射，不仅比较总数：覆盖 15/16/17、
31/32/33、容量不足、跨 sequence 页重用、预检撤销、clear dirty 保留和执行后隔离。
`ldd` 确认 CPU-only 测试没有 CUDA 动态依赖；三种构建均无新增编译告警。

| 身份文件 | SHA-256 |
| --- | --- |
| `source-state.json` | `efd5cac73e8048e8b5ff252eedda91c1ea718284cee852a11424b2b67c01d19d` |
| `source-snapshot.zip` | `d2e31c02cb5f7178afdf977c5b6b57d31259d651e745002f5e282efb3cdcb4f1` |
| `execution-identity.json` | `68457b8efa330fa2d674a3b365ac5bf075627ddc2f71f1a64a0046f04a8d6521` |

复验使用既有入口，不新增测试平台：

```bash
cmake -S . -B build/wsl-core -G Ninja -DLLMSERVE_WITH_LLAMA=OFF \
  -DLLMSERVE_REQUIRE_TEST_TOOLS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/wsl-core --parallel 4
ctest --test-dir build/wsl-core --output-on-failure --no-tests=error
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda model-check .run/gpu-kv-model-check
bash scripts/dev.sh own-cuda check-http 8015 .run/gpu-kv-http-check.json
```

报告路径必须尚不存在。无 GPU CI 验证 host 状态和产品构建，不代替设备数值检查。

## 设备存储验收

`CudaStorage` 持有 `[layer][K_or_V][physical_page][token_in_page][kv_width]`
的 FP16 slab；`[S,ceil(L/P)]` 的 I32 table 位于现有 workspace arena。
table 初始化为 -1，slab 初始化为 FP16 NaN。`kv_table_bytes` 属于 metadata/workspace
计数，不重复计入 total；项目 owning allocations 仍为四个。

`ContiguousKvAccess` 与 `PagedKvAccess` 供同一 `store_kernel` 使用：
转换、finite 检查与输出类型不变，分页地址读取前检查 logical block 与物理页号。
连续与分页 slab 使用不同 getter，不能互相解释布局；paged Storage 仅允许 F32 矩阵。
`upload_page_table()` 在同一 stream 入队，host table 借用持续到 checked completion；
是否上传由 host 页状态的 dirty/journal 决定，Storage 不建立第二份映射账本。

验证身份为 `48d43ad` 加固定 dirty snapshot，raw 位于
`.run/gpu-kv-001/device-storage/`。没有把它重标为最终提交的 clean build 测量。
第二组提交 `c27da9b` 自身 CI run `36527206669` 五任务通过。

| 检查 | 结果 |
| --- | --- |
| 独立核心 / own-CUDA CTest | 12/12、23/23 |
| Storage / layer 单测 | 19/19、12/12 |
| 真实 Qwen3 页池入口 | 20/20，包含 19 项单测 |
| 存储及真实页池 memcheck | 20/20，0 错误、0 泄漏 |
| 层级 memcheck | 12/12，0 错误、0 泄漏 |
| 原 F32 模型 | S1/S4 2/2；128 条比较记录的 GPU logits SHA 与上一组一致，六组 golden 相同 |
| 原 F32 HTTP | 12/12；停服时 4 active + 2 queued，六个完整单终态 |

写入测试覆盖随机非连续页号、D=128/Hkv=8、首尾层、15/16/17 与末页、
slab/table padding 和 guard、未写区域 NaN、非法页号与 metadata、错误布局、
clear/reuse、无需上传的同页追加，以及分配失败后的 RAII 清理。
非法页号先记录 status 并屏蔽写入；没有通过回读/重排历史 KV 构造产品执行路径。
初始 getter 错误类型不一致及失败源码保留，见 `ENG-072`。

### 真实存储量

固定 Qwen3、S=4、Lmax=2048、B=128、P=16：

| 项目 | 数值 |
| --- | ---: |
| 物理池 capacity | 2560 tokens / 160 pages |
| KV payload | 293,601,280 bytes，280 MiB |
| Device table | 2,048 bytes |
| Workspace，含 table | 121,313,280 bytes |
| 项目 owned allocation | 2,803,308,544 bytes |
| 本次 table H2D | 2,048 bytes |
| 页映射检查的 assigned / free | 128 / 32 |
| 首尾层定点回读检查 | 16,384 个 FP16 值 |
| 写入阶段新增项目设备分配 | 0 |

普通检查与 memcheck 的存储量报告一致。连续 S4/8192-token 的同版本计划为
3,449,229,312 owned bytes、896 MiB KV；两者物理容量不同，
不能据此声称同等最长请求并发能力下的显存收益。
这里没有接纳真实请求，也没有初始化所有 assigned 页的全部 token；
assigned/free 是页映射测试状态，不是 Serving live KV 观测。

真实模型只加载一套 storage，回读首尾层指定位置，不运行完整 forward 或重跑旧矩阵 sweep：

```bash
build/wsl-own-cuda/bin/minillm-cuda-storage-tests \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --contract tests/data/qwen3_validation_cases.json \
  --output .run/gpu-kv-storage-check --paged-kv
```

| 身份文件 | SHA-256 |
| --- | --- |
| `source-state.json` | `425920c44c2a166145a84309402067a1562ad31535ba41b22a18eb709b38fc74` |
| `source-snapshot.zip` | `e6ce8454de69a72ef522bc86bb7f001ccca77e476191ab9604b334a00341376d` |
| `execution-identity.json` | `6e1ec95642d6d206040d7b7c74d6ae0c3b6082d03f18285287f96aabd7039038` |

## 分页模型验收

`ContiguousKvAccess` 和 `PagedKvAccess` 为同一 store/QK/PV 的编译期地址策略。
分页 kernel 直接读取 block table；没有将历史 KV gather 为连续数组。
QK 先处理 causal mask，再解析可见页；PV 只访问当前 query 可见的 token。
非法页号累计 status 并屏蔽非法读写，保留所有必要的 warp 归约。
softmax、FMA 顺序、权重精度和同步完成点不变。

`CudaRuntime` 复用 `BatchState` 的 pending lengths 准备页表，在开始执行时保留页，
按需上传映射；完成 stream/status 检查后无分配地提交页和长度。
clear 同时归还页和清长度，设备表延迟至下次执行上传。
初始化、预检与执行后错误不混淆；执行后失败使两种状态同时 poison，
相关页不再复用，resident 保留至 owner 析构。

验证身份为 `c27da9b` 加固定 dirty snapshot，raw 位于
`.run/gpu-kv-001/runtime/`。这是本机正确性验证，不是最终提交的 clean build 测量，
也没有使用正式性能预算。
第三组提交 `33a8fb7` 自身 CI run `36537742096` 五任务通过。

| 检查 | 结果 |
| --- | --- |
| 独立核心 / own-CUDA CTest | 12/12、23/23 |
| Layer / Runtime 单测 | 13/13、15/15 |
| Layer / Runtime memcheck | 均为 0 错误、0 泄漏 |
| 真实 Qwen3 连续/分页对照 | S1/S4 2/2；16 case、173 行完整 logits 逐位相同 |
| 真实模型 memcheck | 同一 16 case、173 行对照通过，0 错误、0 泄漏；首次错误路径调用保留，见 `ENG-073` |
| 长上下文 | 1536-token 前缀的 32-token 冻结续写、2048-token 边界及越界预检通过 |
| 多序列与复用 | mixed、交错 S4、case 间 clear/reuse、六组短 golden 通过 |
| 原 F32 模型 | S1/S4 2/2；128 条 GPU SHA 比较记录与第二组一致，六组 golden 相同 |
| 原 F32 HTTP | 12/12；停服时 3 active + 3 queued，六个完整单终态 |

真实两臂按同一 S 顺序构造，不同时常驻两套模型；基准臂生成续写轨迹，
分页臂重放相同输入和批次形状。S1 的两臂 capacity 均为 2048；
S4 的连续/paged capacity 分别为 8192/2560，仅用于数值和资源检查，
不能据此宣称同等最长并发能力或显存收益。
两臂共 282 次完整 forward，稳态无新增项目设备分配、权重或 hidden 往返。
分页 S1/S4 的 table H2D 分别为 27,648/16,384 bytes；clear 后 live pages 为零，
owned allocation 保持不变。

单测另覆盖随机页号、15/16/17、127/128/129、1536/2048、未写尾部 NaN、
非法页/未来页、表与输出重叠，以及页池不足后的可继续执行。
四类 post-launch 故障为完成检查、首次 table H2D、后续 metadata H2D、
损坏设备表；均不提交本批长度或结果，live pages 为 null，拒绝 clear/forward，
owner 析构后项目分配与释放配对。这些是受控注入，不是致命设备错误的恢复承诺。

模型入口及普通检查命令见 [CUDA Runtime](CUDA_RUNTIME.md#分页研究接口)。
第三组不包含 Serving 分页配置、容量/信用与资源发布的验收，后者独立记录如下。

`verification.json` 登记 27 个原始文件；检查源码快照、二进制未变、
F32 输出未变以及普通/memcheck 的 173 行分页摘要一致。

| 身份文件 | SHA-256 |
| --- | --- |
| `source-state.json` | `23cce7dd7242f7f9629a782eb7b0f19d37cac3b83c20a22f79ed98b3df0b181c` |
| `source-snapshot.zip` | `02208b7868628d06d35debb5f7fff69e334236c554eb85cad10159dcc248570f` |
| `execution-identity.json` | `f8dfa67d09c9be9997a3150ccb26a01eb7356e5b9e469a5a35762a7abc3aa54f` |
| `verification.json` | `bb637ed8a8cef6241351fc7cbd2d26e6fda5f8feec6ef56b70b511fba0e07227` |

## Serving 验收

`mini-cuda --kv-layout paged` 将 `context_tokens` 同时传给 Engine 信用池与 Runtime 物理池。
P 固定 16；`BackendCapabilities::kv_page_tokens` 表达页池对齐合同，
Engine 在启动线程前检查页大小和总容量完全一致，不修改 `admit` 或 `schedule_batch`。
连续模式保留原来的信用/静态槽语义；CPU 和上游后端拒绝 CUDA 布局参数。

资源查询仍为无分配标量读取，由模型线程发布；HTTP 和 batch JSON 增加设备表字节数，
不重复计入 owned。实际分派页数来自 Runtime，poisoned 后 live tokens/pages 为 null，
resident 保留至 owner 析构。schema v2 的历史连续记录仍可读；
分析器明确校验分页布局、容量、页大小、表大小、尾页范围和每轮页增长。

验证身份为 `33a8fb7` 加固定 dirty snapshot，原始输出位于
`.run/gpu-kv-001/serving/`。源码范围包含执行核心、测试和脚本，共 143 个文件；
`verification.json` 登记 37 个原始文件，确认验证期间源码和二进制未变。
第四组提交 `6d3d87d` 自身 CI run `36546952201` 五任务通过。

| 检查 | 结果 |
| --- | --- |
| 独立核心 / own-CUDA CTest | 12/12、23/23 |
| Serving 设备小模型 | 13/13，包含两布局、页池压力、错配拒绝和执行后故障 |
| 设备小模型 Serving memcheck | 13/13，0 错误、0 泄漏；不是本组全量实模型 sanitizer |
| 真实分页 Serving | S1/S4 共 80 个输出与独立 Runtime 相同，S4 有 3 个确定性 mixed batch |
| 长输入压力 | 128 页池、两请求共承诺 148 页；等待 31 batch 后均完成，输出共 64 token 与独立参照相同 |
| 真实执行后故障 | 4 active + 2 queued，6 个 backend_error；故障 batch 无 token，信用回收，resident 隔离 |
| 分页 HTTP | S1/S4 各 12/12，含取消、超时、断连、背压及有界排空 |
| 连续 CUDA / CPU HTTP | 各 12/12 |
| 原 F32 模型 | S1/S4 2/2；128 条 GPU 摘要与第三组一致，六组 golden 相同 |
| CPU 模型 | 13/13 |
| 分页 CLI | S1 的 8-token golden 相同，greedy 无全词表下载，clear 后页数为零 |

压力检查使用 1536/768-token prompt，各生成 32 token；任意记录中仅有一个 active request，
但逻辑 slot 上限为 4，证明等待由页信用约束而非 slot 数量不足造成。
峰值 assigned 为 98 页，已有请求正常完成后，等待请求获得信用和页并继续执行。
这不是有限预算下的吞吐比较，也不是持续过载的公平性证明。

分页 S4 HTTP 的 85 个 batch 逐轮满足 `assigned_pages <= reserved_credit_pages <= 160`，
页数增长等于各 slice 跨过的页边界，最终 live tokens/pages 为零。
停服时 S4 为 3 active + 3 queued、S1 为 1 active + 2 queued，均收到完整单终态。
S4/2560-token 的 resident KV 为 293,601,280 bytes，表为 2048 bytes，
owned 为 2,803,308,544 bytes；清理只归还页 ID，不释放 slab。

| 身份文件 | SHA-256 |
| --- | --- |
| `source-state.json` | `cebe12e7b5d5392323f693986966f157ffe0ef53062f6ff90462b86da1e27794` |
| `source-snapshot.zip` | `1d54fa4ce1a0833d4806fcff323c8d8fe1e77e3a4650d0914d30c5defee35692` |
| `execution-identity.json` | `7e160fd624874014050ba4c0fb12dcd5aaef244073f5cc5fb14addde17f3cc40` |
| `verification.json` | `bb7200dbd6d1acaaead64ddd024abad1dddc9de4e616ebec32ed310b94ca5e77` |

## 实验协议与入口预检

[冻结输入](../benchmarks/runtime-inputs/qwen3-gpu-kv-v1.json) 的协议为
`gpu-kv-experiment-v1`，SHA-256 为
`77b44ce8578e73e05889c21e4aa167b5cff5f858110fc4f62981cc49e002bb5e`。
它固定六类 attention、四个模型 workload、两条 Serving trace、顺序、计时边界、
288 MiB KV 子系统预算和 10% 性能护栏。旧协议、旧输入与既有 SLO 不变。

`mini-cuda-kernel-bench` 和 `mini-cuda-runtime-bench` 接受这份输入及
`--kv-layout contiguous|paged`；仅允许 F32。没有 manifest 的运行明确标为
`diagnostic_single_process`，不能计作三轮正式结果。旧协议拒绝布局参数，
CPU 模型基准拒绝新协议，FP16 不能进入本研究。

预检身份为 `6d3d87d` 加固定 dirty snapshot，记录位于
`.run/gpu-kv-001/experiment-preflight/`。入口执行身份见 `execution-identity.json`；
CLI 拒绝测试另有初始和最终源码快照，基准二进制不变。

| 检查 | 结果 |
| --- | --- |
| 独立核心 / own-CUDA CTest | 12/12、23/23；首次文案断言失败保留，见 `ENG-074` |
| Micro | 两布局各六类、各 30 个样本，输出 SHA 逐一相同 |
| 分页 micro memcheck | 0 错误、0 泄漏；该计时不与普通连续运行比较 |
| Model | 两布局各四项、各 175 次 forward；输入摘要、上下文和 token 完全一致 |
| 模型执行边界 | 稳态无新增设备分配、无权重或 hidden 往返；setup 与计时区间分开 |
| 同容量 allocation | 连续 3,449,229,312 bytes；分页 3,449,231,360 bytes，仅多 2 KiB table |
| 页表传输 | micro 单列每次 2 KiB 上传；模型进程合计 358,400 bytes，含 setup |
| 既有回归身份 | 63 个产品源码文件、5 个既有二进制与第四组相同；模型/HTTP 不冒充本轮重跑 |

M4 的 micro 使用倒序物理页号直接读取分页 KV。M32 的 query 位置为
`L-32...L-1`，不是所有 query 都读取长度 L。模型侧四项的计时区间页表传输
分别为 10,240、122,880、10,240、10,240 bytes，均包含两次预热和三次测量重复；
两个 decode workload 的 prefix setup 另有 122,880、81,920 bytes。
页分配、映射维护和必要上传均在对应 forward 计时内，不只报告独立上传 probe。

### 同预算功能结果

指定组为 `[1536,128,128,128]` 个 prompt token，各输出 32 token。
每次仅常驻一个 Runtime；独立 reference、连续臂、分页臂、压力反例顺序执行。
三组各四个请求全部完成，共 384 个 Serving 输出 token 与独立参照相同。

| 配置 | KV 子系统实分配 | 非 KV owned | 观察 |
| --- | ---: | ---: | --- |
| 连续 S1 / capacity 2048 | 234,881,024 bytes | 2,509,705,216 bytes | 单 slot 顺序完成，有 101 个等待 batch |
| 分页 S4 / capacity 2560 | 293,603,328 bytes | 2,509,705,216 bytes | 四请求同时 live，全部完成 |
| 分页压力组 | 同上 | 同上 | 承诺 3456 token 超过池容量，等待 31 batch 后全部完成 |

两臂均在 301,989,888-byte 预算内。分页第 49 批有四个 active request，
长度为 `[1537,32,32,32]`，live tokens 为 1633、assigned 为 103 页、reserved 为 128 页。
普通组峰值 assigned 为 128 页、最大末页空隙为 56 token、最少 free pool 为 32 页；
压力组对应为 118 页、41 token、42 页。三个极值不要求出现在同一批。
末页空隙与空闲池分开统计；停止后 live tokens/pages 归零，resident 保留。

这是特定异长请求组合的功能与容量证据，不是“四倍吞吐”，也不保留四条
2048-token 请求同时执行的能力。分页臂在此比较中实际分配更多 KV；
结论针对当前等长静态槽，不证明优于所有连续分配器。

`verification.json` 登记 28 个文件，SHA-256 为
`1ab7f7f3ea87bfdceb776b897ffcb584469b924320af36c831663a83231a2350`。
本组原始计时保留，但不用于正式收益或性能护栏判定。

## 同容量微基准

`Benchmark-CudaMicro.ps1 -GpuKvStudy` 使用冻结的六类 shape 和六个独立进程，
按 AB、BA、AB 执行；第二轮用例逆序。每项两次预热、三次测量，
每样本 32 次 attention API。分析器核对倒序物理页映射、输出摘要、F32 数学、
实际 allocation 和独立 table probe，不将诊断进程当作正式 trial。
旧的 375-shape 协议与精度微基准仍独立校验。

正式记录位于 `.run/gpu-kv-001/micro/`，采集身份为 `825ae1b` 加固定 dirty snapshot；
不是后续提交的 clean build 测量。六个进程全部成功，共 180 个原始样本、
108 个测量样本、5760 次 attention API 调用。所有布局和 trial 的输出相同，
无计时区间设备分配或显式传输；表上传不包含在 attention micro 计时中。

下表为三对独立 trial 中位数的配对相对变化，`100*(paged/contiguous-1)`；
正数表示分页更慢，不是统计显著性判断。

| 形状 | Host 延迟变化 | 三轮 host 变化 | Event 区间变化 |
| --- | ---: | --- | ---: |
| M1 / L128 | +157.38% | +172.74%、+129.31%、+157.38% | +161.29% |
| M1 / L1536 | +410.27% | +488.83%、+410.27%、+323.12% | +413.61% |
| M4 / L128 | +130.62% | +189.34%、+130.62%、+122.17% | +133.22% |
| M4 / L1536 | +229.46% | +229.46%、+247.45%、+99.03% | +230.13% |
| M32 / L128 | +61.63% | +61.63%、+89.57%、+35.79% | +62.43% |
| M32 / L1536 | +120.09% | +132.22%、+58.94%、+120.09% | +120.19% |

这是明确的不利观察，见 `ENG-076`；不能由此推算完整模型或 HTTP 退化比例。
设备表已在计时前上传，独立 2 KiB table probe 的 host/event 每次均摊中位数为
6.393/5.470 us，不能把它当作上述 attention 退化的解释。
两臂 KV 均为 896 MiB，分页 owned 仅多 2048 bytes，没有同容量显存收益。

进程边界采样的温度为 52～66°C，SM clock 为 1980/2505 MHz；未锁频，
没有连续频率或 kernel 指令/访存采样，原因尚未定位。
保留正序、逆序和全部不利结果，不追加 micro trial 追求有利结论。
下一层模型计时仍需包含页分配、映射维护、必要上传和同步。

工具回归为 own-CUDA 23/23、独立核心 12/12；微基准验证 16/16，
含错误页号摘要、传输漏计、重复 trial、输出分叉和原始命令布局不符等反例。
归档内复核入口从独立工作目录复验通过，31 个必需文件可读；
采集源码及六个既有二进制摘要未变。记录在 `.run/gpu-kv-001/experiment-tools/`。

| 身份 | SHA-256 |
| --- | --- |
| `manifest.json` | `a1201eae4b813a93757643e2773ec0ea0b942fdd7610250de688e09dca1cae30` |
| `source-snapshot.zip` | `2569e255176979548a17d3cd77462f5d72d28a92b56fb596e81208ffc4f4ec33` |
| `summary.json` | `50a2d8c193081a5c5761273bab3115659085c01a10e312d87df0e778c7476d27` |

## 剩余门禁

模型与 Serving 分页路径、固定实验入口和指定容量功能检查已通过。
微基准采集和分析已完成；模型与 Serving 的正式工具尚未支持本协议，
不能直接运行旧模型 70 进程协议或 mixed/prefill_first 对照。
先量化模型层影响，再按规范决定是否使用唯一一次局部地址/table 改进；
不能用 micro 的百分比替代模型及 Serving 护栏，也不扩大正式实验预算。
保持 mixed、F32、单 stream、同步完成与保守 reservation；不引入 incremental admission、
prefix sharing、fusion 或新的 scheduler。可执行不等于已通过最终采用门槛。

正式性能预算已用 micro 6/6、model 0/6、Serving 0/12；新增 NSys 0/1、NCU 0/1。
同容量 8192-token 的执行代价与同 288 MiB KV 预算的异长请求能力分别验收。
不将 host 状态测试或较小的容量参数称为显存节省或吞吐提升。

当前原始输出保留在本地；研究收束时使用一个 canonical bundle，
Git 保留本说明、固定规范、小摘要与最终证据索引，不为本阶段另立组件包。
