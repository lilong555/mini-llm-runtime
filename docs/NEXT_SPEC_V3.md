# NEXT_SPEC_V3 — 共享 GPU KV 页池与直接分页 Attention

> Spec ID：GPU-KV-001  
> 对应提案：PROJECT_PLAN_V4 / PLAN-V4-KV-20260928  
> 审计日期：2026-09-28，Asia/Tokyo  
> 固定功能基点：`6ca7d2dfeccb38b11b7596a552046651afcb693e`  
> 建议实现基点：包含已有 HTTP 停服修复 `57268f93aecc7c947146d60a3d61a03d84f46a1b` 的候选  
> 状态：待实施；本文的 GPU 页池、block table 和分页执行尚未存在于固定基点。

这是一个数据路径能力规范，不是性能承诺。只解决“GPU KV 容量如何在不同长度请求间共享，并被 attention 直接消费”。不同时改变精度、调度策略、attention 数学或异步执行契约。

本轮交付只是本规范和 Plan，不执行以下代码任务、不合并分支、不移动 tag。实现者开始前重新确认工作区/ref；如果已存在同一能力，先审计 delta，不能重复实现。已有 FP16 研究分支、旧 NEXT_OPT_SPEC 和实验记录保留，但不是本阶段的可变精度输入。

## 1. Problem

### 1.1 当前具体缺口

`CudaStorage::make_memory_plan()` 依据：

```text
max_sequences × layers × 2 × max_model_len × kv_width × sizeof(FP16)
```

预留全部连续 GPU KV。`MiniCudaRunner` 将 Engine 的 max_active/max_model_len/batch_tokens 传给 Runtime，但 `context_tokens` 只限制 Serving 的逻辑信用，没有独立控制真实连续 KV allocation。

在固定 Qwen3-0.6B、S=4、Lmax=2048 下：
- GPU KV payload：939,524,096 bytes，即 896 MiB；
- 每个 token 的全部层 K/V：114,688 bytes，即 112 KiB；
- 每个独立 slot 预留 224 MiB，无论请求最终使用 128 还是 1536 token；
- 四个 slot 不会把一个短请求未使用的容量借给其他 slot。

这不是 allocator 泄漏，也不是已证明的 throughput bottleneck，而是固定容量划分造成的资源约束。当前没有 native GPU page table/free pool，CPU 的 PagedKV 不能代替这一缺口。

### 1.2 唯一研究问题

> 保持模型数学、权重精度、同步完成、S≤4 和现有保守 admission，能否把 GPU KV 从固定 slot 变为一个共享物理页池，并在同一 KV 子系统预算下支持更有用的异长请求组合？代价是多少？

“分页管理”和“分页 attention”分别是资源问题与执行问题；本规范必须打通两者，不能以仅有 metadata 或一个未被 Runtime 使用的 kernel 宣布完成。

## 2. Evidence

### 2.1 固定基点证据

| Evidence | 结论 | 限定 |
|---|---|---|
| `src/minillm/cuda/storage.cpp` | KV 真实常驻且按 S×Lmax 预分配 | 不是每请求 cudaMalloc |
| `src/minillm/cuda/runtime.cpp` | 完整模型、一次 forward 的同步完成、pending/committed/poisoned | 失败后不支持物理 rollback |
| `src/minillm/cuda/attention.cu` | QK、softmax、PV 真实读 contiguous FP16 KV | 不是 FlashAttention 或 PagedAttention |
| `src/mini_cuda_runner.cpp` | own CUDA 已进入 Engine，资源快照真实 | prefix copy/share 明确不支持 |
| `src/engine.cpp` | prompt+max_tokens 保守信用，整批结果验证后发布 | credit block 不是 GPU physical page |
| `benchmarks/results/cuda-serving-001/analysis.md` | 一次时间线峰值 live KV 1255/8192 tokens | 单次观察，不能声称普遍节省 85% |
| 同目录 summary/validation/evidence | 12 个正式进程、288 请求、9216 token，有真实 Serving baseline | 历史采集身份不是本次新实测 |

### 2.2 影响进入条件的后续证据

- `57268f9` 已修正 Engine 终态入队后 HTTP provider 尚未排空的问题，CI run `36406587106` 成功。后续实现应纳入此现成修复，不重写 drain；不把它算作分页收益。
- `perf/cuda-f16-matrix-path` 的 `103070a...` 文档记录 `blocked_correctness`，实际 Serving 配置也拒绝 F16 候选。保持旧阈值和 F32 默认，不重做或偷偷启用这一候选。
- 这些后续事实不是 6ca 已有能力。采用 57268f9 后，新的连续/paged 两臂必须都包含它。

参考外部原理而非复制完整系统：
- vLLM V1 pool/physical block management；
- TensorRT-LLM 的 KV budget 与按需分派；
- ggml 的 owner/view/device/completion 边界。
详见 Plan 的官方资料索引，访问日期 2026-09-28。vLLM 的旧 Paged Attention 说明页有历史警告，不能当作当前完整实现规范。

## 3. Goal

以下全部成立才形成目标能力：

```text
existing HTTP / Engine / Scheduler
               |
          MiniCudaRunner
               |
          CudaRuntime
       host page plan / commit
               |
      device block table H2D
               |
  own store / QK / softmax / PV
               |
  physical GPU FP16 pages → output
```

具体目标：
1. `contiguous` 原路径完整保留，默认不改变。
2. 增加一个 opt-in `paged` KV layout，独立指定全池 token capacity。
3. 每个物理页首版由一个 sequence 独占；“共享”指页池容量供多个请求使用，不是多个请求共用同一页的数据。按实际 append 分派页，不按请求最坏长度立即分派全部物理页。
4. 注意第 3 条是池内分派：device slab 仍在初始化时按预算一次分配，不宣称 cudaMalloc 按 token 动态申请。
5. 所有层的 K/V 写入与 attention 均直接通过 device block table 定位，不 gather 全量 KV。
6. 维持原数学和精度：source Q8_0→device F32、activation F32、KV F16、FP32 QK/PV、现有 softmax 分母算法、greedy。
7. 分别完成同容量 latency 对照、同预算异长请求容量实验。
8. 结果可以是有容量价值但速度没有提升；停止条件允许研究以负结果完成。

## 4. Non-goals

本次明确不做：
- GPU prefix caching、页共享、COW、refcount > 1；CPU 的这些功能保留不动。
- 新 Q8/F16/BF16/FP8/INT4 compute path、Tensor Core 调参。
- Fusion、online softmax、FlashAttention、paged fused attention 的性能追逐。
- 增量 admission、抢占、重算、swap/offload、新 scheduler/cost model。
- S>4、Lmax>2048、B>128 的产品扩容。
- 多 stream、异步 execute、CUDA Graph、event-driven in-flight allocator。
- 多 GPU、PD、KV transfer、multi-tenancy。
- 通用 tensor/allocator/registry/backend framework。
- 第二套 Engine、HTTP/SSE、benchmark/manifest/统计/封包框架。
- 全部旧 375-case micro 或 70-process model 实验重新采集。
- 为保证“最后一定快”反复改变 workload、SLO、阈值或增加 trial。

不能把 GPU 分页实现包装成与当前 vLLM/SGLang kernel 等价；第一版是非融合、数学可对照的直接分页执行。

## 5. Existing Components Reused

### 5.1 真实接口

| 当前组件 | 复用方法 |
|---|---|
| `CudaRuntime::forward()` | 仍是唯一模型执行入口；增加 mapping 的 prepare/commit |
| `BatchState::prepare/start/commit/discard_prepared/poison/clear` | 继续管理逻辑 sequence length 和执行 phase，不新增第二套长度账本 |
| `CudaStorage` | 继续独占 device weight/workspace/KV；增加 layout plan 和小型 device table |
| `LayerExecutor::enqueue()` | 保留全部层数学，只改变 KV view/accessor 的选择 |
| `store_kv()`、`causal_attention()` | 保留数学，加入 layout-specific address policy |
| `DeviceBuffer`、`CudaContext`、`DeviceScope` | 所有分配和完成/销毁沿用 RAII |
| `MiniCudaRunner` | 仍使用 greedy、验证 samples、fail-stop；扩展真实页指标 |
| `BlockPool` | 继续只是保守 credit；不直接当作 GPU allocator |
| `Engine::admit/iteration/publish` | 策略不变；配置和资源快照适配 |
| `GatedRunner`、CUDA layer/runtime/serving tests | 复用受控组批、状态、故障与模型 reference |
| Runtime/kernel/Serving benchmark | 增加小型 layout 子协议，保留旧输入及解析 |
| 既有 analyzer / evidence bundle | 只补 layout/capacity 字段与必要检查，不新增验证层 |

### 5.2 源码调用链

```text
Engine::Impl::iteration
  -> MiniCudaRunner::execute
  -> CudaRuntime::Impl::forward
       BatchState::prepare
       [new] PageTableState::prepare
       BatchState::start
       [new] 页预留 + 上传 pending table
       LayerExecutor::enqueue x 28
           store_kv<Access>
           causal_attention<Access>
       status / token D2H
       CudaContext::synchronize
       [new] page mapping commit
       BatchState::commit
  -> Engine 批次校验
  -> token / SSE
```

预检失败保持 ready；设备执行开始后的失败沿用 poisoned。不能在某层中途发布长度、回收新页或继续使用实例。

## 6. Proposed Architecture / Ownership

### 6.1 三种必要职责

**Host PageTableState：** 分配状态、active/pending table 和事务。尽量是无 CUDA 头文件的普通 C++20 类型，可在现有 CPU CI 验证。

**CudaStorage 的 device slab/table：** 持有真实内存。PageTableState 不持有 device pointer，不负责 cudaMalloc。

**Address policies：** 分别为连续与分页生成地址，供同一份 store/QK/PV 数学使用。非 owning POD，编译期选择，无 device 虚函数和通用 factory。

建议新增：
```text
src/minillm/cuda/page_table.h
src/minillm/cuda/page_table.cpp
src/minillm/cuda/kv_access.cuh
tests/gpu_page_table_tests.cpp
```
可以按现有构建依赖微调内部文件位置，但不能为此建立独立 allocator subsystem。

### 6.2 拟议 Host 接口

以下为设计示意，不是基点已经存在的 API：

```cpp
using PhysicalPageId = std::int32_t;
inline constexpr PhysicalPageId invalid_page = -1;

struct PageTableLimits {
    std::size_t sequences;
    std::size_t max_model_len;
    std::size_t page_tokens;
    std::size_t physical_pages;
};

class PageTableState {
public:
    explicit PageTableState(PageTableLimits);

    // 输入 span 来自 BatchState；不再拥有另一份逻辑长度。
    void prepare(std::span<const std::size_t> committed_lengths,
                 std::span<const std::size_t> pending_lengths);
    void begin_execution() noexcept;
    void commit() noexcept;
    void discard_prepared() noexcept;
    void poison() noexcept;

    void clear(std::size_t sequence); // 仅 ready，先完整验证
    std::span<const PhysicalPageId> device_table_for_upload() const noexcept;
    bool upload_required() const noexcept;
    std::size_t assigned_pages() const noexcept;
    std::size_t free_pages() const noexcept;
};
```

必要内部容器：
- active table：长度 `S * ceil(Lmax/P)`；
- pending table：相同尺寸，预分配；
- free IDs：最大 N 个，初始化 reserve，运行中不扩容；
- 本 batch 新页 journal：容量 N，或其它有严格上界的预留空间；
- 可选 debug owner vector：N 个条目，不用实际 refcount 模拟尚不存在的共享。

S≤4，直接 O(S+新增页数) 或复制整个小表更易验证。没有理由使用平衡树、哈希 allocator 或锁自由页池。

为 BatchState 增加 `pending_lengths() const noexcept` 借用 span。该对象仍是长度状态的唯一事实源；其它组件不能独立推进相同 sequence 的 length。

### 6.3 事务与失败

**prepare：**
1. BatchState 先验证完整输入的 sequence/token/position/连续性/上界。
2. 按旧、新长度计算 `ceil(new/P)-ceil(old/P)`，检查全部新增页总数。
3. 复制 active→pending，确定需要的新 page IDs 和 pending mappings。
4. 所有可能分配的 host 结果缓冲、kernel geometry 溢出检查都在第一条 device 写操作前完成。
5. 首版 prepare 不从 free list 正式消费页；以 journal 暂记候选。Runtime 不可重入，不存在其它分配者抢占。

**start：**
1. `BatchState::start()`。
2. `PageTableState::begin_execution()` 无分配、无失败地消费已经验证的 IDs。
3. 将需要更新的 pending table 上传同一 stream。
4. 再排队 KV writes 和 attention。

将状态切为 executing 后，即使第一次 H2D 失败，也可保守 fail-stop；不靠推测“这次肯定没写”恢复。

**commit：**
1. 同一 stream 已完成；device status、token 范围和 Runtime 内部结果检查均通过。
2. PageTableState 以 swap/固定赋值提交 mappings。
3. BatchState 提交 lengths。
4. 提交区域不得动态分配或调用可能失败的 CUDA API；两部分作为单调用者下的一个 no-throw 发布区。
5. 最后向 caller 返回结果。

这里的原子性来自单调用者、发布区不对外观察和预检后的无失败操作，不是宣称两个 C++ 对象的 swap 是硬件原子事务。现有 BatchState::commit 会检查 phase；必须在进入该区域前验证双方 phase，避免先提交一方再进入可预见异常。如果发生内部不变量损坏，仍走 fail-stop、对外资源标为未知，不能向调用方宣称恢复成功。

**prelaunch failure：**
没有开始 device work时，丢弃 plan/journal，free IDs、active mappings、committed lengths 均不变；BatchState 撤销 prepared。已经因前次 clear 而设置的 table dirty 标记不得丢失。

**postlaunch failure：**
state 和 page state 都 poisoned；当前执行可能已经写入旧页尾部或新页。所有已分派/待提交页都被隔离，不能返还 free pool 后继续运行。完成 best-effort synchronization，然后抛错；无本批 token 发布。保留 resident allocation，直至 owner 析构。明确这是 fail-stop，不是 device byte rollback。

**clear：**
只在 ready 完成点、合法 sequence 下执行；先验证再释放该 sequence 所有页、清 host table、重置 BatchState length，并标 device table dirty。预留容器保证 clear 的合法路径不需要扩容。下一次 forward 在 KV 访问前刷新 device table。Poisoned 时沿用 MiniCudaRunner 的不可复用逻辑，不能通过 clear 恢复实例。

### 6.4 生命周期表

| 对象 | Owner | 何时创建/复用 | 何时释放 |
|---|---|---|---|
| F32 weights | 现有 CudaStorage | 初始化；层与请求间共享只读 | Runtime 完成/销毁 |
| FP16 page slab | CudaStorage | 初始化按预算一次分配；池内分派 | Runtime 销毁，不随请求释放 cuda memory |
| Device table | CudaStorage | 初始化，映射变化时上传 | 最后一次访问完成后 |
| Host active/pending table | PageTableState | 初始化；每 batch 复用 | Runtime 销毁 |
| Page ID ownership | PageTableState | prepare/start/commit | 正常 clear；poisoned 时隔离 |
| KV view/accessor | 栈/LayerExecutor 借用 | 每个 dispatch 创建小 POD | 不单独释放内存 |
| Telemetry snapshot | Engine | 模型线程在完成点发布 | 沿用已有 snapshot 生命周期 |

异步 H2D 的 host table buffer 必须保持到 checked completion，不能中途 swap 后被其它线程复用。首版的单执行线程、单 stream、同步返回使这件事可直接保证。

## 7. Memory Layout / Capacity / Invariants

### 7.1 Slab 布局

物理 FP16 元素布局固定为：

```text
[layer][K_or_V][physical_page][token_in_page][kv_head][head_dim]
```

设：
```text
L = layers
N = physical_pages
P = page_tokens
W = kv_heads * head_dim
T = ceil(max_model_len / P)
```

逻辑位置：
```text
logical_block = position / P
offset_in_page = position % P
page = device_table[sequence * T + logical_block]
```

物理元素索引：
```text
((((layer * 2 + kind) * N + page) * P + offset_in_page) * W)
    + kv_head * head_dim + channel
```

`kind=0` 是 K、`kind=1` 是 V。一个 physical page ID 对应同一 logical block 的全部层 K/V；由于布局以 layer 开头，其数据分布在各层区域，不要错误地将“这个 ID 的所有层数据”当作单一相邻 memcpy 段。

连续布局保持原 `[sequence][layer][K_or_V][position][kv_width]`。仅在 host dispatch 选择 accessor，不要求两种布局共用错误的 stride 解释。

所有乘加在 host checked size 计算中验证 `size_t/PTRDIFF_MAX`，转换为 kernel/grid int 前再验证。不能依赖 device 索引溢出后由 sanitizer 才发现。

### 7.2 Device block table

```text
int32[S][ceil(Lmax/P)]
```

无映射为 -1；有效范围 `[0,N)`。

首版 P=16 固定支持，不开放 page-size autotune。S4/L2048 时 table 为 `4*128*4=2048 bytes`。可以每次 mappings 变化时上传整表，先保持简单；no mapping change 时无需传。记录 `page_table_h2d_bytes`，不可将新增流量隐藏在旧 metadata 固定计数里。

Slab 和 table 是两个 owning allocation，或 table 并入既有 workspace arena；选择其中一种并精确计入 plan，不能重复计量。正式性能区间无新的自有 cudaMalloc/cudaFree。

### 7.3 Kernel memory safety

- 先验证 sequence/position 和 logical block 范围，再读取 table。
- page==-1 或 page>=N 时设置 existing device error/status，不进行 KV load/store。
- QK 必须先做 causal 检查；future position 不读 KV，而不是读出 NaN 后乘零。
- PV 只遍历 query 可见的 `[0,position]`。
- GQA 保持 `kv_head = query_head / (query_heads/kv_heads)`。
- `store_kv` 只写本批已经分派且归该 sequence 的位置。
- 页重用后未写区域可能有旧值；正确性只依赖长度/mask，不依赖每次将整页清零。测试必须用 NaN/非零值污染尾部验证。
- Device owner 检查可用于 debug 测试；正常 hot loop 不引入全局哈希/锁。

### 7.4 不变量

在 ready、无 prefix share 情况下：

```text
assigned_pages + free_pages = N
每个 assigned page 只对应一个 sequence 的一个 logical block
每个有效逻辑块都有且只有一个有效 physical page
committed length 对应 ceil(length/P) 个 logical block
active table 和 device mirror 在下一次执行前一致
```

prepared 时 journal 不能让同一 ID 分派两次；executing 时已取出的新页计入 pending/隔离集合；poisoned 时不得继续报告“所有页可用”。

前述 equality 的各 phase 口径必须写入测试，不能用一个计数同时表示 ready assigned、pending 和 free。

### 7.5 保守 admission 足以保障本阶段的页容量

关闭 prefix，Engine 与 page allocator 使用相同 P。令请求 i 的完整承诺为：
```text
r_i = ceil((prompt_tokens_i + max_output_tokens_i) / P)
```

任意已执行时刻的长度 `length_i <= prompt_i + max_output_i`，故：
```text
sum(ceil(length_i/P)) <= sum(r_i) <= N
```

因此正常已接纳集合不会在 decode 中途因为受管理页池容量不足而全部停滞。这里未承诺外部 GPU memory 分配永不失败，也未承诺无限持续过载下所有到达请求都被接纳。

等待队列沿用已有策略。测试包含“请求承诺超过池容量→有请求等待→已有请求完成释放信用/页→后续请求进展”。不新增 watermark、增量 admission 或 preemption。

## 8. API Changes / Integration

### 8.1 Config

向 CudaRuntimeConfig / StorageLimits 末尾增加默认字段，以免破坏既有 aggregate 初始化：

```cpp
enum class CudaKvLayout { contiguous, paged };

CudaKvLayout kv_layout = CudaKvLayout::contiguous;
std::size_t kv_capacity_tokens = 0;
std::size_t page_tokens = 16;
```

具体声明可复用已有无 CUDA header 的 config 类型，不能让 llmserve_core 因读 enum 而必须链接 CUDA。

语义：
- contiguous：0 仍按 S×Lmax；如果显式 capacity 不等于该值，拒绝，不偷偷修改 slot layout。
- paged：明确正 capacity，且为 16 的整数倍；`physical_pages=capacity/16`。
- S/B/Lmax 维持现有上限 4/128/2048。
- prefix entries/tokens=0。
- 保持 `EngineConfig::validate()` 的 max_model_len<=context_tokens 等约束。
- paged Serving 的 physical capacity 必须等于 Engine context_tokens，block_size必须=P。
- 页池 capacity 不大于可寻址的 S×ceil(Lmax/P)×P；超过不能形成有效容量，不应静默接受冗余。
- --context 在 contiguous 仍是旧信用语义，在 paged 是信用和物理池的明确共同上限；README 必须区分。

新增一个 `--kv-layout contiguous|paged`，不新增另一个 backend 名称或 server。已有 device budget 继续约束总 allocation，包含 table/对齐。精度固定 `f32-pedantic`；本 SPEC 不携带未通过数值门槛的 F16 candidate。

### 8.2 Runtime / Layer

- 为 BatchState 增加 read-only pending lengths view。
- CudaStorage 根据 layout 构建 memory plan、slab 和可选 table。
- LayerExecutor 从 storage 获取相应非 owning KV view，选择对应 store/attention 实例。
- CudaRuntime 将 page transaction 包含在 forward 的同一完成/错误边界。
- `clear_sequence` 管理 host pages 和长度，正常保持旧复用语义。
- `copy_sequence` 仍不支持，capability.prefix_copy=false。

### 8.3 Resources / Metrics

复用已有 RunnerResources。必要增量只表达：
```text
layout
capacity_tokens
live_tokens（已提交逻辑长度）
live_kv_pages（paged 的实际分派页数；contiguous 为 null）
resident_kv_payload_bytes（slab allocation，不是 live payload）
page_table_bytes
owned_device_bytes
state_valid / reusable
```

额外 `free_pages/assigned_payload/tail_slack` 可由已发布标量在分析时计算，不必为每个派生值加一套 Runtime counter。
`page_table_h2d_bytes` 用于 diagnostics/benchmark，与其他 transfer 字段分开。

状态采集仍由模型线程 publish，HTTP 只读副本；noexcept resource API 不复制 vector、不调用 cudaMemGetInfo。Poisoned 时 live counters 为 null 或显式 invalid，resident 不伪造为0。

旧 JSON field 含义不变。增加明确 layout/capacity 支持；若 schema 必须升级，只增加一个版本且保留旧 reader，不修改历史 raw。需要未知/unsupported 的值继续是 null，不能填估算值冒充实测。

### 8.4 Files To Modify

Existing：
```text
CMakeLists.txt
include/minillm/cuda/runtime.h
src/minillm/cuda/runtime.cpp
src/minillm/cuda/storage.h
src/minillm/cuda/storage.cpp
src/minillm/cuda/batch_state.h
src/minillm/cuda/layer.h
src/minillm/cuda/layer.cpp
src/minillm/cuda/attention.h
src/minillm/cuda/attention.cu
include/llmserve/config.h
include/llmserve/telemetry.h
src/config.cpp
src/mini_cuda_runner.cpp
apps/server_main.cpp
apps/cuda_main.cpp
apps/cuda_runtime_bench.cpp
apps/cuda_kernel_bench.cpp
apps/telemetry_output.h
src/http_server.cpp（只序列化，保留已接纳的 drain 修复）
tests/cuda_layer_tests.cpp
tests/cuda_runtime_tests.cpp
tests/cuda_serving_tests.cpp
相关既有 benchmark/telemetry fixture
scripts/dev.sh
已有 Start/Benchmark/Analyze 脚本（只 layout 参数和结果解析）
```

如果目标仓库的实际测试文件名改变，以读取后的等价现有入口为准，在第一份 commit 说明中记录，不自行复制一套测试。
`src/engine.cpp` 只允许必要的启动能力验证或 resource 字段接入；不改 `admit`/`schedule_batch` 算法。
CPU `paged_kv.cpp`、SIMD、模型权重路径均不改。

New 主要文件仅为 §6.1 的 host page state/access policy/test。另允许一个固定新容量 trace、一份小 protocol 与一个研究说明；不新增证据框架。

## 9. Implementation Steps / Commit Plan

### Commit 1 — Host page state 与合同

```text
feat(cuda-kv): add bounded exclusive page-table state
```

内容：
- 增加 config 默认字段、checked limits 与 host page state。
- BatchState 提供 pending lengths 借用，不改变现有连续路径行为。
- CPU-only page-state unit/property tests；非法参数、capacity exhaustion、事务撤销、重复分派、clear/reuse。
- 尚未接通的 paged model 入口必须 fail closed，不冒充可执行能力。

门禁：现有 CPU/own-CUDA 默认构建通过；固定种子状态测试通过；无 GPU CI 可执行 host 逻辑。可单独提交。

### Commit 2 — Device slab / table 与 KV write

```text
feat(cuda-kv): add physical slab and device-visible mappings
```

内容：
- memory plan 依据全池 capacity，真实分配 slab/table。
- 地址政策与 store 路径；表变化 H2D；只在初始化分配。
- 有界 readback 用于 tests，验证不同层、K/V、slot、page/tail，而不是作为产品执行。
- 注意 table staging lifetime；计算总量溢出与启动 budget。

门禁：伪随机非连续 page IDs、末尾页、NaN未使用区、device memcheck通过。完整模型 paged 功能仍不宣称已完成。

### Commit 3 — Shared math + direct paged attention + Runtime transaction

```text
feat(cuda-kv): execute Qwen3 directly from paged KV
```

内容：
- 让 store/QK/PV 使用两种 access policy；softmax数学不变。
- host选择实例，保留 contiguous 原始对照，不重写两套 attention。
- forward 的 prepare/start/commit/poison、clear 正确串接。
- 完整层与模型数值对照。

门禁：实际随机 page layout 下产生正确 attention 与完整模型；no gather/no host KV roundtrip；preflight与postlaunch失败符合 §6.3。

### Commit 4 — Existing Serving integration

```text
feat(serving): expose bounded paged KV capacity and resources
```

内容：
- 既有 `mini-cuda` backend 新 layout 参数、物理/信用同容量检查。
- MiniCudaRunner resources 与旧 Engine publish/HTTP。
- deterministic mixed、异长slot、clear/reuse、poisoned、同池压力与finite progress。
- 保留已有HTTP停服修复；不重写服务层。

门禁：S1/S4模型与HTTP、一个受控postlaunch错误、相关memcheck、CPU regression通过。尚无速度结果时不写“更快”。

### Commit 5 — Bounded experiment / decision / stop

```text
bench(cuda-kv): document capacity-latency tradeoff
```

内容：
- 按 §11 的有限协议采集。
- 只做一次代表性 NSys；必要时一份定向 NCU，不形成新研究分支。
- Git保留小摘要/协议/定位入口，canonical raw外置。
- 明确 product eligible / research only / negative / inconclusive。
- 达到完成定义立即停止，不追加 prefix/fusion/precision。

以上五组是最大主要逻辑划分，不需要再写二十个规划任务。实际修复可以有小commit，但不借此突破scope或实验预算。

## 10. Correctness Gates

### 10.1 Unit：host state 与地址

- P=16、S=1/4、小页池与边界；table index、page ID、字节大小和溢出。
- 正常append到15/16/17及31/32/33等边界。
- 多个sequence同batch分配时没有重复ID。
- 容量不足在device执行前拒绝，active table/free set/lengths不变。
- clear两次、invalid sequence、clear后不同prompt复用。
- prepare后其它host检查失败能discard；未启动工作不poison。
- executing后错误不rollback到可复用；poisoned不允许继续prepare/clear。
- active/free/pending集合守恒，测试比较集合而非只比较总数。

### 10.2 Property / State

固定 seed，运行 10,000 个轻量 host 操作：
```text
append one/many
multi-sequence batch
discard prepared
clear
capacity failure
poison and recreate
```
对照简单 vector/set oracle，不需要引入 property-testing 平台。debug owner只用于发现重复和错归属，不成为GPU热路径新feature。

GPU fixture 把空闲/尾部填 NaN，物理 pages 随机置换，验证正确mask。保持必要CUDA错误状态，不用结果有限作为唯一内存安全检查。

### 10.3 Kernel / Layer

- 同一已存储 FP16 K/V 数值、query、position、GQA设置，两种布局逐元素对照。
- 关键维度使用目标 D=128、Hq=16、Hkv=8；toy fixture可用已支持更小尺寸。
- 有效长度包含1/15/16/17/127/128/129/1536/2048；future tokens填NaN不影响输出。
- 同batch包含prefill和decode、物理页顺序与logical页顺序不同。
- 两条路径使用相同归约/FMA/softmax代码，bitwise一致为首要预期。若仅地址模板导致编译器变化，必须定位并记录；不能静默放宽。
- 对原始scalar/FP64 reference沿用现有算子阈值，不以新的宽松模型阈值掩盖索引错误。

### 10.4 Model

限定代表集：
1. 稳定短golden与32-token续写。
2. prefill 128，chunk16/128，对应page边界。
3. prompt1536、后续decode，保留长上下文对照。
4. S4独立序列与mixed、clear/reuse。
5. Lmax2048边界，超界在预检失败。

完整模型对照使用串行构造/销毁的实例或独立进程，保存固定输入与必要 logits；不要同时常驻两个 S4 Runtime 使 8 GiB 显卡因对照程序自身而耗尽。

同GPU/math/相同token顺序的layout对照应不引入新的近似。短golden/token序列严格相同；full logits默认要求bitwise一致。发现非位级差异时先定位shared math/编译路径，作为独立评审项处理，不能直接改测试阈值。不同batch/chunk形状的既有数值契约仍按原合同运行，不承诺跨任意GEMM shape全位级一致。

原CPU/上游matched-effective-weight合同、RMSE/max-abs/cosine门槛均保留；本阶段不变precision，不能套用“FP16候选可容忍误差”放宽它们。

### 10.5 Serving / Regression

复用CUDA serving和HTTP suite：
- 每请求输出与独立Runtime reference对应。
- GatedRunner用于确定性共存，不把sleep竞争当作batch正确性证据。
- slot reuse、queue、cancel、timeout、disconnect、slow consumer、shutdowndrain保留。
- 4 active + queued 的受控postlaunch错误：本批无token、每请求一个终态、逻辑信用回收、resident隔离、后续拒绝。
- capacity不足但尚未被admit的请求继续等待或按原规则拒绝；不能触发全Engine错误。
- admitted请求若物理页不足，是容量不变量bug，不当作正常pagefault重试。
- representative CUDA memcheck覆盖随机页、tail、reuse、postlaunchfailure；不重复整个旧sanitizer笛卡尔积。
- 相关CPU、Windows/Linux核心/产品CI与候选自身CI通过；无GPU CI不能替代本机GPU验证。

## 11. Performance Gates / Protocol

### 11.1 比较边界与指标

三层独立：
- micro：地址/attention路径代价，含table维护成本单列；
- model：完整forward，包含tableH2D，不包含模型加载与prefix setup；
- serving：client壁钟，包括队列、组批、执行与SSE。

**Primary：给定GPU KV子系统预算下，异长请求集合的可接纳/共存能力及实际分配量。**
**Secondary：同容量的模型/Serving latency、吞吐、TTFT、meanTPOT、ITL尾部、table/copy成本。**

不是以throughput必须上升为完成标准。容量与速度分别出结论。

### 11.2 固定数据与运行条件

- checkpoint/source/effective-weight identity沿用固定manifest；F32device矩阵、F16KV。
- 单stream、同步execute、mixedpolicy、B128、chunk32、P16。
- 非比较项完全相同，尤其HTTP修复与compiler配置。
- 每进程只创建一种layout。reference在独立正确性运行，不与performance共驻。
- micro/model 每case两次预热，三次测量重复，三个独立paired trial；Serving 保留既有 `Hello` / 8 token 预热，不把整条正式trace重复算作inner repeat。
- AB/BA交替顺序，三对中首/第三AB、第二BA；下一种workload反转起始，保存温度/频率可得信息。
- inner repeats不计独立trial。启动/初始化/预热/prefixsetup与steady state分开。
- 记录源码SHA/dirty快照、二进制、模型、trace hash、layout/容量、实际memory plan和时钟边界。

### 11.3 实验一：同容量，观察分页代价

```text
contiguous:
    S=4, Lmax=2048, physical capacity=8192
paged:
    S=4, Lmax=2048, pool tokens=8192, P=16
```

这里二者KV payload都是896MiB；paged还增加table/少量metadata。不能宣称分页在该配置节省896MiB。

Micro最多6类：
```text
M=1 / M=4 / M=32
有效context=128 / 1536
```
令上表 context 为追加后最大可见长度 L：M1 使用一个 query、position=L−1；M4 使用四个独立 sequence，各一个 query、position=L−1；M32 使用一个 sequence 的最后32个 query，positions=L−32…L−1。M32 各行的 causal 长度不同，不能全部设成 L。KV/query 数值使用固定seed和同一既有recipe，先入库后采集。页边界的更多尺寸在正确性tests，不扩性能sweep。

Model最多4类：
```text
prefill-128
long-prefill-1536-chunk128
decode-prefix1536（追加后有效1537）
decode-batch4-prefix256
```

Serving复用已有mixed-length trace，24requests×32output，固定mixedpolicy；F32/prefix0/credits8192等配置不变，两layout各3trial，共6进程。

指标包括相同成功/请求集合、精确token对照、memory/slack、clientTTFT/meanTPOT/requestmaxITL、throughput、goodput。沿用已冻结SLO，不为新的结果移动阈值。

### 11.4 实验二：同预算，不偷换成同最大并发

**KV子系统device预算：288 MiB**，包含KV payload、device table、必要的KV专属对齐；不是整个GPU预算。

- baseline固定slot：每slot Lmax2048消耗224MiB，预算只允许S=1。不能把S=4的896MiBbaseline放在预算外再叫公平比较。
- paged：S=4、Lmax2048，pool=2560tokens=160pages=280MiB，加2KiB table和实际对齐，在预算内。
- 非KV weights/workspace尽可能相同；S引起的小metadata差异仍逐项计入总量，报告实际owneddevicebytes。device budget的完整启动校验照常。
- baseline仍允许每请求最长2048，不能为了短请求预知地缩小所有slot以掩盖长请求无法服务。
- 一个请求组的prompt长度 `[1536,128,128,128]`，各max_output32。
- 按P16计算最坏预留：
  `[1568,160,160,160]`，合计2048tokens≤2560。
- 用确定性功能测试建立四请求同时admitted/live的状态；baseline在同预算只支持一个slot并逐个完成。不得预设“性能正好4倍”。

正式容量trace可由六个上述组构成24requests，统一arrival_s=0，队列64，ignore_eos=true。全部输入/tokenID在采集前冻结；这是有界异长队列工作负载，不是代表所有在线流量。两个layout各3trial，共6进程。

另外只做功能压力反例：两个1536prompt加两个128prompt，其总承诺3456tokens>2560，应出现waiting，已有请求完成后进展。压力反例不混进all-success性能分析器，不需要新增一套允许任意失败的benchmark平台。

**诚实边界：**
- paged可能比baseline的S1实际分配更多KV，但仍在相同预算内；收益是更灵活的可用组合，不是这一组里allocation更低。
- 相比原S4的896MiB，280MiB只对受控制请求集合保留能力，不保留四条2048并发保证。
- 一个预知长度的非均匀连续slot allocator也可能有效。本实验比较的是项目现有等长静态slot，不证明分页优于所有可能的连续分配器。
- S=4上限不变；不宣称已经支持更多槽或更长单请求。

### 11.5 Profiler / measurement budget

最大正式performance进程：
```text
Micro：两layout各3trial，共6
Model：两layout各3trial，每进程覆盖4case，共6
Serving：两trace × 两layout × 3trial，共12
合计24
```

最多一次新增NSys，使用真实pagedServing路径：
- 确认GPUkernel直接读取device table；
- table传输很小且单列；
- 无全KVgather、无逐层host中间值往返；
- 无steady-state自有deviceallocation；
- batch/slot/page分派与clear在明确完成点关联。
必要时一份针对新页访问kernel的NCU诊断；无问题就不采。

复用已有profiler/parser；名称变化只增加必要映射，不新增“验证验证器”。
该预算是上限，不是必须跑满。正确性阻塞或已回答研究问题时提前停止。

### 11.6 采用与停止门槛

**Mandatory correctness/data-path gate：** §10全部相关检查通过；真实pool/table被完整Runtime/Serving使用。

**Capacity gate：**
- 预算计算和actual allocations一致；
- 异长请求组合在预算内按设计可共存且完成；
- tail waste与freepool分别报告；
- 反例等待能够有限推进；
- 不靠增加显存、降低数值门槛或缩减输入来通过。

**性能护栏：** 采集前使用10%作为工程review band，而不是已测统计显著性或噪声：
- 同容量modelcase的三对trial中位数相对退化≤10%，且不出现至少两轮退化>10%；
- 同容量Serving吞吐配对中位数降低不超过10%，TTFT P95、meanTPOT P95、request-max-ITL P95完整保留；
- 较小样本的尾部变化只描述，不声称普遍P99改善。

护栏超限后最多一次局部address/table维护改进，不允许加fusion/Graph/precision。如果仍超限，保留contiguous默认，将paged标为研究/实验能力，不宣称默认替换已成功。

允许 `no_performance_gain`、`measurement_inconclusive`、`capacity_gain_with_latency_cost`。不能重新挑primary metric将不利结果改成无条件成功。

## 12. Regression Gates

- CPU Runtime、SIMD、CPU PagedKV/share/COW、PrefixIndex功能不变。
- 原F32 CUDA连续路径保持源码可选择、数值与状态行为不变。
- source/device/activation/KV dtype及cuBLAScompute mode不变。
- 既有cancel/timeout/backpressure/poisoned与整批结果发布不变。
- `57268f9`的HTTPdrain行为保留，并对最终候选重新跑对应测试；不继承旧SHA的CI绿灯。
- ModelRunner的prefix能力仍false，单stream同步完成仍true。
- 隐式CPUfallback、pinning改造、Graph、多stream或F16candidate均不属于“修回归”。
- 当前历史raw、输入、oracle阈值、SLO不修改。
- 只为新layout增加必要fixture，不要求把已有统计/证据体系再写一遍。

## 13. Abort Conditions

出现以下任何情况应停止产品晋升并作出明确决定：

1. 不能证明page/free/pending ownership或失败后隔离，不进行性能采集。
2. 数学结果偏离且无法解释为可接受的既有对照差异；不能用原FP16研究的更大误差开脱索引问题。
3. 正常保守admission下发生物理池中途耗尽，先修一致性，不增加retry/preemption来遮掩。
4. 为使实验获胜，必须同时改并发上限、scheduler、precision、prefix或fusion。
5. 性能护栏失败且一次局部修订没有足够收益。
6. 为本功能增加通用allocator/新的benchmark框架比新增真实数据路径还复杂。
7. 预算实验没有新增解释力，或用户有限投入不再支持：当前系统已经可用于portfolio，允许停止并冻结。

FP16已失败不意味着必须把分页做成“成功补偿”。本规范允许负结果，也不再自动生成下一条重大fallback。

## 14. Completion Definition / Portfolio Evidence

### 14.1 Research Done

- [ ] 明确基点、已纳入的独立HTTP修复和实际采集身份。
- [ ] Page state、device table、真实GPUstore/attention完整连接。
- [ ] 同容量与同预算两个问题分开回答。
- [ ] 页尾浪费、空闲pool、residentallocation和logicalcredit没有混淆。
- [ ] 数值、状态、内存安全、Servingregression有对应证据。
- [ ] 包含不利样本与固定threshold下的失败/不确定，未补trial直到显著。
- [ ] 一份小报告解释收益、成本、适用范围与停止决定。
- [ ] 一个canonicalbundle可获取，Git只存小摘要/协议/命令/索引。
- [ ] 无新增prefix/quantization/fusion/async等scope。
- [ ] 完成后进入Plan的Portfolio Freeze，不继续列热门feature清单。

### 14.2 Product Eligible（与 Research Done 分开）

只有correctness、memory safety、failure semantics全部通过，capacity合同成立，并对性能护栏作出可接受的采用判断，paged才可成为正式可选择路径。

第一版默认仍保留contiguous。研究完成并不自动授予“默认更快”“生产级”或“与vLLM等价”的描述。

### 14.3 最终可以证明的能力

理想成果是：
```text
识别静态GPU KV容量绑定
→ 建立真实预算与请求长度模型
→ 设计host页事务和device映射
→ 用既有数学直接读取物理GPU页
→ 保持同步生命周期与保守准入不变量
→ 分别测容量和latency代价
→ 给出采用或停止决定
```

未来简历数字只能填实测。可描述自有page table、physicalKV、directpagedattention及验证范围；不能填写“节省85%”“吞吐4倍”，除非对应严格实验真的得到这些结果。

### 14.4 参考定位

固定基点：
- `src/minillm/cuda/storage.cpp`：memory plan / allocation。
- `src/minillm/cuda/batch_state.h/.cpp`：逻辑长度和phase。
- `src/minillm/cuda/runtime.cpp`：forward/commit/poison/diagnostics。
- `src/minillm/cuda/layer.cpp`：完整层中的store/attention。
- `src/minillm/cuda/attention.cu`：当前QK/softmax/PV。
- `src/mini_cuda_runner.cpp`、`src/engine.cpp`：adapter、credit、清理、发布。
- `benchmarks/results/cuda-serving-001/`：既有真实负载与资源证据。
- `tests/cuda_layer_tests.cpp`、`tests/cuda_runtime_tests.cpp`、`tests/cuda_serving_tests.cpp`：现有验证入口。

本规范为未来实现设计；本轮只创建文档，没有执行代码修改、GPU实验或发布操作。
