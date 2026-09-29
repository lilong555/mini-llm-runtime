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
| Paged CLI/Serving | 尚未接入；默认继续使用连续布局 |
| 采用决定 | 尚未具备产品资格，无容量或性能实测结论 |

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
resident slab 仍由 `CudaStorage` 持有。host、设备和模型验证分别记录，
不能替代尚未接通的 Serving 页池验证。

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
Serving 分页配置、容量/信用与资源发布仍属于下一组，不继承上述模型验收结论。

`verification.json` 登记 27 个原始文件；检查源码快照、二进制未变、
F32 输出未变以及普通/memcheck 的 173 行分页摘要一致。

| 身份文件 | SHA-256 |
| --- | --- |
| `source-state.json` | `23cce7dd7242f7f9629a782eb7b0f19d37cac3b83c20a22f79ed98b3df0b181c` |
| `source-snapshot.zip` | `02208b7868628d06d35debb5f7fff69e334236c554eb85cad10159dcc248570f` |
| `execution-identity.json` | `f8dfa67d09c9be9997a3150ccb26a01eb7356e5b9e469a5a35762a7abc3aa54f` |
| `verification.json` | `bb637ed8a8cef6241351fc7cbd2d26e6fda5f8feec6ef56b70b511fba0e07227` |

## 剩余门禁

Serving 容量/信用接线、布局配置与页指标尚未完成。
下一项复用当前 Engine 的保守 reservation，并使逻辑 credit 与物理页池容量一致；
不引入 incremental admission、prefix sharing 或新的 scheduler。

正式性能预算仍为 micro 0/6、model 0/6、Serving 0/12；新增 NSys 0/1、NCU 0/1。
同容量 8192-token 的执行代价与同 288 MiB KV 预算的异长请求能力分别验收。
不将 host 状态测试或较小的容量参数称为显存节省或吞吐提升。

当前原始输出保留在本地；研究收束时使用一个 canonical bundle，
Git 保留本说明、固定规范、小摘要与最终证据索引，不为本阶段另立组件包。
