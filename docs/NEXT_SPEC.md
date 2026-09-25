# NEXT_SPEC — Own CUDA Resident Qwen3 Vertical Slice

## 文档身份

- Spec ID：`CUDA-VS-001`
- 对应 milestone：`PROJECT_PLAN_V2.md / V2-M1`；Step 1 含 V2-M0 的必要证据补丁。
- Audit HEAD：`68ac275913207975a88e2090c6617467e351301c`
- Audit Date：2026-09-23，Asia/Tokyo。
- 状态：执行中，Step 1–7 已验收；完整 CudaRuntime 与 token CLI 可用，接口见 `CUDA_RUNTIME.md`，阶段门禁见 `EXECUTION_STATUS.md`。全量数值语料、性能对照、Profiler 和完整性能包仍待 Step 8–9；当前不代表 V2-M1 整体完成。
- 推荐主分支：`feat/own-cuda-vertical-slice`；证据补丁可先独立 `fix/evidence-bundle-completeness`。
- 目标平台：本机 WSL2 Ubuntu，RTX 4070 Laptop，CUDA 12.8；现有 Windows/MSVC 与 Linux CPU 构建必须保留。
- 本阶段终点是完整模型产生真实 token 的 CLI/model path；GPU HTTP/LLMServe 接入属于紧随其后的 V2-M2，不得在本阶段提前宣称已完成。

## 1. Problem Statement

当前已有自有 CPU Qwen3 forward、SIMD、线程池、FP16 physical PagedKV、Serving 与数值参照。`Runtime::forward()` 已有阶段 profile，`ModelRunner::execute_profiled()` 与 `resources()` 已有扩展点，不能重新从零实现这些工具。

现有 evidence：
- `benchmarks/results/wsl-runtime-profile/README.md`：36 份报告、558 次测量；8 线程 prefill-128 的投影约占 96.16%，LM head 约 0.54%。
- `benchmarks/results/wsl-runtime-profile/context/forward-stages.json`：decode-1536 的 attention 中位 15.276339 ms，约占 profiled forward 26.00%；不是纯页表或 allocator 时间。
- `benchmarks/results/wsl-batch-telemetry/eng-008-analysis.md`：已测 scheduler 自身占比极小，特定长 token 停顿与连续 prefill 批次关联。
- `scripts/cuda_smoke.cu` 与 `benchmarks/results/wsl-environment/`：CUDA 编译、内存检查、NSight 工具已有环境证据。
- `CMakeLists.txt` 的 `LLMSERVE_CUDA` 仍只控制上游 ggml；`src/llama_runner.cpp` 调用 `llama_decode()`；`src/mini_runner.cpp` 仍执行自有 CPU Runtime。

因此下一阶段的结构性问题是：尚没有由项目控制 device weight、workspace、KV、执行和 token 输出的完整 CUDA 模型路径。继续搭建 CPU 观测平台的边际收益已经下降。

另一个必须先处理的交付缺口：至少 `wsl-runtime-profile/context/manifest.json` 引用的 source ZIP 不在当前仓库对应路径，而 `Analyze-Runtime.ps1` 要求它存在。新实验不得继承这个缺口。

上述数值属于各自历史采集 cohort。不得把 dirty source-state 记录重标为 Audit HEAD 的 clean build 实测。

## 2. Goal

实现并验证：

```text
GGUF host model
  -> validated immutable model binding
  -> device-resident effective weights
  -> preallocated device workspace
  -> cuBLAS matrix operations + own CUDA ops
  -> complete Qwen3 transformer layer
  -> contiguous GPU KV
  -> complete 28-layer Qwen3 forward
  -> own greedy argmax
  -> actual token IDs / decoded text
```

机械目标：
1. `LLMSERVE_CUDA=OFF` 时，自有 CUDA CLI 仍能运行完整目标模型。
2. 稳态不逐层传输权重或 hidden states，不调用上游模型 forward，不隐式回退 CPU。
3. 单序列与最多 4 个 sequence 的限定批处理正确。
4. 有 unit/state/model 证据和 CPU/HTTP 非回归证据。
5. 有 kernel、model 两层 GPU 测量；Serving 端到端 GPU 性能不在本阶段完成声明中。
6. 当前 GPU baseline 即使不快，也保留完整数据与解释；“正确模型路径完成”和“取得加速”分别验收。

## 3. Non-Goals

不做 GPU PagedAttention、GPU prefix alias/COW、multi-stream、CUDA Graph、async Serving、自定义 Q8/Q4 GEMM、activation quantization、multi-GPU、MoE、更多 architecture、随机采样、完整 API、通用 graph compiler。

不把现有 CPU PagedKV 改成 GPU 通用 allocator；不重写 scheduler、HTTP、Trie 或线程池。
不要求 custom GEMM 超过 cuBLAS。
不把 CPU profiler 的 elapsed 字段改称 GPU device time，也不放松全部旧验证器来兼容 GPU。

## 4. Existing Architecture

当前真实调用链：

```text
llmserve::Engine::Impl::iteration()
  -> ModelRunner::execute() / execute_profiled()
  -> MiniRunner::execute_impl()
  -> minillm::Runtime::forward()
  -> Runtime::Impl::forward()
      -> multiply() / normalize()
      -> ParallelExecutor::run()
      -> PagedKV::append/store/key/value
```

`Runtime::Impl` 同时承担：
- GGUF binding、ModelDimensions、Layer TensorView 与 norm weights；
- vocabulary-only llama model 的 tokenizer ownership；
- CPU executor、CPU KV、临时 activation 和数学执行。

CPU 模型层面复用的是 GGUF parser/tokenizer；不调用 `llama_decode()`。替代 `LlamaRunner::execute()` 才使用上游模型计算。

当前 `ModelRunner::resources()` 的两个物理字段是 CPU paged 语义。连续 GPU KV 不得伪造 `live_kv_pages`；本阶段使用独立 GPU diagnostics。以后 M2 采用 layout-aware schema 或明确 null，而不是修改字段含义。

## 5. Proposed Design

### 5.1 构建与能力边界

新增默认 OFF 的选项：

```cmake
option(MINILLM_ENABLE_CUDA "Build the project-owned CUDA runtime" OFF)
```

它与已有 `LLMSERVE_CUDA` 独立。验收组合至少包括：

```text
A. LLMSERVE_WITH_LLAMA=OFF, MINILLM_ENABLE_CUDA=OFF
   -> 现有独立核心测试

B. LLMSERVE_WITH_LLAMA=ON, LLMSERVE_CUDA=OFF, MINILLM_ENABLE_CUDA=OFF
   -> 现有完整 CPU 产品

C. LLMSERVE_WITH_LLAMA=ON, LLMSERVE_CUDA=OFF, MINILLM_ENABLE_CUDA=ON
   -> 自有 CUDA 模型；上游只用于 host parser/tokenizer/reference dependency

D. 既有 LLMSERVE_CUDA=ON 构建
   -> 仍保持上游 CUDA 参照；与 C 分开归因
```

自有 CUDA 开启时 `enable_language(CUDA)`、`find_package(CUDAToolkit 12.8 REQUIRED)`；项目 CUDA/C++ targets 使用 C++20，初期本机明确传 `CMAKE_CUDA_ARCHITECTURES=89`。现有上游 C++17 隔离不变。C 组合需要 parser/tokenizer 的 llama 依赖；WITH_LLAMA=OFF 与 own-CUDA model 同时开启时给出清楚配置错误，不静默关闭模型 target。

新增 targets：
- `minillm_model`：host immutable model/tokenizer，不包含 CUDA；
- `minillm_cuda`：自有 CUDA runtime；
- `mini-cuda-llm`：模型到 token 的 CLI；
- `mini-cuda-runtime-bench`：同一可执行文件中的 CPU/CUDA 可选择对照；
- `minillm-cuda-unit-tests`、`minillm-cuda-model-tests`。

上述 targets 随真实源文件就绪逐步加入，不用空 forward 或恒定 token 占位。

现有 CPU executable 不因选项 OFF 而依赖 CUDA DLL/so。不得将 NVCC flags 或 ISA flags 全局传播给 CPU/第三方代码。

### 5.2 最小 host model 提取

拟新增 `Qwen3Model` 和 `Tokenizer`：
- Qwen3Model 独占 GgufModel/mmap，并持有 immutable TensorViews 与 norm vectors。
- 从现有 `Runtime::Impl::Layer` 和 load_dimensions/matrix/norm_weight 提取 binding/validation，CPU 数学循环、顺序和 kernel dispatch 不改。
- Tokenizer 包装已有 vocabulary-only llama_model，提供 tokenize/token_piece/is_eog；不自行实现 tokenizer。
- 同名 `ModelDimensions/InputToken/Logits` 可移入 `model_types.h`，原 `runtime.h` 包含它以保留 source compatibility。
- CPU Runtime 和 CUDA Runtime 各自持有 host model owner；不要只为取 tokenizer 而构造一整套 CPU Runtime/thread pool/KV。

提取前后使用相同 CPU 编译/运行配置，分别记录两个二进制身份；固定输入的 logits/greedy、profile shape 与资源行为必须保持已有契约。

### 5.3 所有权类型

以下为拟新增接口示意，具体头文件实现须包含所需标准库/CUDA 声明：

```cpp
enum class CudaRuntimeState { ready, poisoned };
enum class CudaOutputMode { greedy, debug_logits };

struct CudaRuntimeConfig {
    std::string model_path;
    int device = 0;
    std::size_t max_sequences = 4;
    std::size_t max_model_len = 2048;
    std::size_t batch_tokens = 128;
    std::size_t device_budget_bytes = 0; // 0 表示按预算协议推导
};

struct CudaMemoryPlan {
    std::size_t weights_bytes = 0;
    std::size_t kv_bytes = 0;
    std::size_t activations_bytes = 0;
    std::size_t attention_scratch_bytes = 0;
    std::size_t logits_bytes = 0;
    std::size_t library_workspace_bytes = 0;
    std::size_t total_owned_bytes = 0;
};

struct CudaSample {
    std::int32_t sequence;
    std::size_t input_index;
    std::int32_t token;
};

struct CudaDiagnostics {
    std::size_t live_sequences = 0;
    std::size_t live_kv_tokens = 0;
    std::size_t kv_capacity_tokens = 0;
    CudaMemoryPlan resident;
    std::uint64_t weight_h2d_bytes = 0;
    std::uint64_t metadata_h2d_bytes = 0;
    std::uint64_t token_d2h_bytes = 0;
    std::uint64_t debug_d2h_bytes = 0;
    std::uint64_t owned_device_allocations = 0;
    CudaRuntimeState state = CudaRuntimeState::ready;
};

class CudaRuntime {
public:
    explicit CudaRuntime(CudaRuntimeConfig);
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;

    CudaForwardResult forward(std::span<const InputToken>,
                              CudaOutputMode = CudaOutputMode::greedy);
    void clear_sequence(std::int32_t);
    CudaDiagnostics diagnostics() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

`CudaForwardResult` 至少包含按 input_index 排序的 samples；debug 模式另带 logits，不强迫 greedy 路径下载全词表。设备计时用独立可选字段，不塞入 CPU `ForwardProfile::wall_ns`。

DeviceBuffer<T> 为 move-only owner，处理零长度、乘法溢出、分配失败与 move 后空对象；DeviceTensorView 仅存 device pointer、shape、stride 和 dtype，不拥有内存，host 不可解引用。

### 5.4 数值模式与权重 residency

第一版数值契约固定为：
- source weights：manifest 固定的 Qwen3-0.6B Q8_0 GGUF，包含其中实际各 tensor dtype；
- device weights：初始化时由现有 decode_row 解量化/转换得到的 FP32 有效权重；
- activations、QK/PV 与 GEMM accumulation：FP32；
- K/V physical storage：FP16；
- softmax 指数值为 FP32，分母累加先保留现有 CPU 的 FP64 语义；
- matrix provider：cuBLAS，`CUBLAS_COMPUTE_32F_PEDANTIC`，不使用 FAST_TF32/FAST_16F、不启用 `--use_fast_math`。

报告必须写 `source_weight_dtype=Q8_0` 与 `device_weight_dtype=F32`，不得写成“Q8 CUDA kernel”。转换不恢复量化前模型，也不新增训练/量化算法。

CudaWeights 使用一个 owning device arena，按 alignment 分配 TensorViews。明确 tied output/embedding 的别名，不能重复分配后又误宣称共享，更不能重复释放。

使用有上限的 host staging，例如 8 MiB；逐块转换、完成上传后再复用 staging。首版允许初始化阶段同步 copy，禁止对仍参与异步 H2D 的 host buffer 提前覆盖。所有权重在第一次模型 forward 前常驻，初始化结束记录每 tensor 名称、shape、source/storage dtype、device offset、bytes 和有效权重摘要。

### 5.5 Matrix layout

统一数学接口：

```text
X: row-major [M,K]
W: row-major [N,K]
Y: row-major [M,N]
Y = X * transpose(W)
```

采用 column-major cuBLAS 表达时：

```text
opA=T, opB=N
m=N, n=M, k=K
A=W, lda=K
B=X, ldb=K
C=Y, ldc=N
alpha=1, beta=0
```

尺寸和 leading dimension 转为 int 前必须检查。用非方阵、非对称值与已知小例子验证 transpose/stride，不能只测单位矩阵或方阵。

真实主 shape 从模型 metadata 与当前 profile 读取；目标模型典型 N/K：
- Q：2048/1024；
- K、V：1024/1024；
- attention output：1024/2048；
- gate/up：3072/1024；
- down：1024/3072；
- LM head：vocabulary/1024，vocabulary 必须读取，不能手写。

M 是 token 数，不是 sequence 数。为需要 logits 的 hidden rows 做 gather，再允许批量 LM head；CPU baseline 仍保留原行为，不在本 SPEC 顺手优化它。

### 5.6 Workspace 与显存预算

CudaWorkspace 预先分配 hidden/normalized/Q/K/V/attention/projected/gate/up/down、score/probability、logits、sample 与 status buffer。层间复用，不能每层或每 token cudaMalloc/cudaFree。

连续 KV 第一版布局建议：

```text
[sequence][layer][K_or_V][position][kv_head * head_dim]
```

max_sequences 与 max_model_len 是独立上限，不能把 CPU 的全池 context_tokens 直接解释成每序列 context。

目标模型 KV payload：
`2 * 28 * 8 * 128 * 2 = 114688 bytes/token`。
S=4、Lmax=2048 的静态 KV reservation 是 896 MiB；这是推导的 payload，不是实测 RSS，也不是 paged allocation 利用率。

内存计划必须使用实际 tensor 大小求和，包含：
- unique FP32 weights；
- 全部连续 KV slots；
- activation scratch；
- attention scores，最坏约 `B * Hq * Lmax * sizeof(float)`；
- 最坏 logits row buffer；
- 显式 cuBLAS workspace；
- metadata/status/output 与对齐成本。

创建 CUDA context/cuBLAS 后读取 free memory；在预分配前检查 total plan 不超过用户上限，也不超过此时 free memory 的 80%，并保留至少 512 MiB 安全余量。两项约束取更严格者。该余量是本 SPEC 的保守策略，不是显卡硬件保证；外部进程仍可能抢占显存，实际 allocation 错误必须处理。

不足时在模型执行前报错并输出 memory plan；不得静默缩短 context、回退 CPU、使用 Unified Memory oversubscription 或加载半个模型。开发诊断可显式降低 S/L/B，报告必须保留变更；正式验收不能隐藏未通过的目标配置。

### 5.7 自有模型执行

执行顺序：

```text
validate descriptors / prepare pending lengths
  -> copy token IDs + metadata
  -> embedding gather
  -> for each layer:
       RMSNorm
       Q/K/V projections
       Q/K RMSNorm
       NeoX RoPE
       FP16 KV write
       causal GQA attention
       attention output projection
       residual add
       FFN RMSNorm
       gate/up projections
       SwiGLU
       down projection
       residual add
  -> final RMSNorm
  -> selected-row LM head
  -> finite-check + deterministic argmax
  -> copy token IDs/status
  -> checked stream completion
  -> commit lengths and return outputs
```

RoPE 允许在初始化阶段按现有 CPU 公式预计算配置范围内的 cos/sin 并上传一次；旋转在 GPU 执行。不得每层/每 step 回 CPU 计算中间状态。显式 head_dim=128，不能由 hidden/heads 猜测。

Attention 第一版是可解释的 QK → softmax → PV 路径，不宣称 FlashAttention：
- score/probability buffer 按最大配置预分配、层间复用；
- 对每个 query 使用自己的 `position+1`，不是该 sequence 的最终 pending length；
- 同批未来位置的 KV 可以已写入，但不能被早期 query 读取；
- 无效位置必须先 predicate，不能先读 NaN/uninitialized KV 后乘零；
- GQA 映射 `kv_head = query_head / (query_heads / kv_heads)`；
- FP16 使用明确的 round-to-nearest-even 转换；
- 不用 CPU PagedKV 作为实际 GPU attention 的数据提供者。

Greedy 使用项目自己的 CUDA argmax；相等时选最小 token ID；任何 nonfinite logits 设置错误状态。只有 tests/reference 允许调用 llama_decode，产品 CUDA 路径禁止。

## 6. Ownership / Lifetime

| 对象 | Owner | 创建与复用 | 释放/失败 |
|---|---|---|---|
| GGUF mapping/views | Qwen3Model | 模型装载；immutable | views 全部失效后释放 mapping |
| Tokenizer | Tokenizer adapter | 模型初始化，vocab-only | 不持有 GPU/KV；按既有上游 deleter 释放 |
| Stream/handle | CudaContext | Runtime 初始化，handle 绑定单 stream | Runtime 先完成/终结在途工作，再释放依赖资源 |
| Device weight arena | CudaWeights | 初始化一次上传 | 最后一次 GPU 使用结束后释放 |
| Workspace | CudaWorkspace | 初始化/明确准备阶段分配；各层和 step 复用 | 不在 forward 内扩容或释放 |
| ContiguousKV storage | ContiguousKV | 按 S/Lmax 预留；clear 只重置逻辑 length | 清理请求不释放整个 pool；Runtime 结束再释放 |
| Host staging/output | Runtime/CLI | 预留；H2D/D2H 完成后复用 | 不允许异步传输访问已销毁 host buffer |
| Profile events | 可选诊断 owner | 在测量前创建；仅开启时记录 | 完成后读取/复用，关闭时不做逐阶段计时 |
| Debug logits | 调试结果 owner | 明确 debug 模式 | 不进入正式 greedy performance 数据 |

CudaRuntime 单调用者、不可重入。同一 stream 顺序执行所有 ops 和 cuBLAS；结果返回前执行一次 checked completion，不在每层同步。

析构不抛异常。析构前的正常完成检查与错误报告必须显式进行；不能依赖某次 cudaFree 的隐式同步作为协议。异常销毁尽力释放，记录失败，不承诺被 fatal device error 破坏的 context 可继续复用。

## 7. Invariants 与错误状态

1. `MINILLM_ENABLE_CUDA=OFF` 不改变 CPU 语义或引入 CUDA runtime 依赖。
2. 数值 oracle 的 checkpoint/effective weights 相同；禁止以文件名近似匹配。
3. 每个 device allocation 恰有一个 owner，views 不释放内存。
4. sequence、token、position、batch/context/byte size 的所有可预先检查条件，在首个写 kernel 前验证。
5. position 必须与该 sequence 当前长度连续；同批多 token 逐项检查。
6. 正常返回前，GPU 已完成本 batch，所有输出与逻辑长度一起发布。
7. 无效输入在 launch 前失败：状态与 KV logical lengths 不变，Runtime 仍可用。
8. launch 后失败：可能已有 device writes；Runtime 进入 `poisoned`，不发布本 batch 输出，不提交 pending lengths，不再接受 forward 或复用 KV。
9. 第 8 条是 fail-stop，不是物理 rollback。不得声称失败前 GPU 内存逐字节恢复。
10. clear_sequence 只允许在明确完成点操作；不能清除 poisoned 标记来继续生成。
11. 不可从仍 in-flight 的 page/buffer/sequence 读取或回收；首版通过同步契约解决，不引入并发回收。
12. steady state 不分配自有 device buffer，不搬运逐层权重或 hidden。
13. profiling disabled 不改变结果；可选 profiling 不在 inner loop 中动态分配。
14. GPU event time、host elapsed、线程 CPU time、logical bytes 和硬件计数器各自独立。
15. 只在本阶段真实实现的层次上声明能力；无 GPU Serving/PagedAttention 的完成声明。

故障注入分为：
- 可恢复的 preflight/初始化分配错误；
- 模拟 post-launch 失败，验证 poisoned/无输出；
- sanitizer 独立进程中的错误诊断。
不能把真正 illegal memory access 后的同一 context 当作可靠恢复试验。

## 8. Instrumentation

基础字段：
- run_id、source SHA/dirty/source-state、binary/model/input SHA；
- GPU UUID/设备名称/compute capability、driver、runtime、cuBLAS、nvcc、host compiler；
- source/device/activation/KV/softmax-accumulation dtype、math mode；
- S/Lmax/B、单 stream、batch tokens、logits rows、context sum/max；
- host_forward_to_token_ns：从 descriptors 已在 host 内存开始，到 token/status 可用并完成同步，包含必要 metadata copy 和 argmax；
- device_elapsed_ms：CUDA events 的 device 时间；关闭时 null；
- model_load_ns、weight_decode_upload_ns：初始化，单列；
- memory plan 与实分配 bytes/call count；
- weight/intermediate/metadata H2D，token/status/debug D2H 分类计数。

只有分析模式增加阶段 events/NVTX，events 预创建，最终完成后统一读取，不逐层 synchronize。
正式 GPU benchmark 在无 profiler 下运行。Nsight Systems/Compute 的采集另开进程。

operation count 例如 `2*M*N*K` 明确为 logical FLOPs；bytes/time 是 effective logical bandwidth，不能标为 DRAM 实际带宽。NCU 的 DRAM/SM/occupancy 等才是硬件采集项，不可用时为 null 并说明原因。

不要求为本 SPEC 建立新通用 observability 平台。CPU `ForwardProfile` 与现有 schema 保留；GPU schema 独立 version=1，明确 own forward / cuBLAS / upstream parser/tokenizer 的归属。

## 9. Benchmark Protocol

### 9.1 固定配置

- DUT checkpoint：`models/manifest.json` 当前固定 Q8_0，实际校验 SHA。
- 数值 reference：同 Q8 effective weights 解量化 F32，依据 reference-manifest 校验。
- GPU：device 0，记录实际 UUID；单 stream；S=4、Lmax=2048、Bmax=128。
- CPU 对照：同有效权重与 FP16 KV，分别完整报告 auto SIMD 下 8 与 16 线程，不混合两者最好值为一个配置。
- GPU KV：contiguous，page size=`not_applicable`；CPU 对照 page size=16。
- reference model 不在 performance 进程中常驻。
- 同一 benchmark executable 可选择 CPU 或 CUDA，但每个独立进程只构造所选 Runtime；初始化另计。
- 首版不使用旧 runtime-bench 的共享 prefix reset 来伪装 GPU alias 支持。

### 9.2 Workloads

| 层次 | 用例 |
|---|---|
| Matrix micro | 目标模型 Q/K/V/output/FFN/LM head 真实 N/K；M=1/2/4/8/16/18/32/64/128 的适用点 |
| Ops micro | RMSNorm/RoPE/softmax/attention，真实宽度与边界 |
| Model prefill | 16、128 tokens；额外 256/1536 通过 chunked prefill 做正确性与总 prefill time，不假装一次 B=1536 |
| Model fixed-context decode | KV=16/256/1536；每次测量输入相同，并明确计入当前 input 后的有效 length |
| Model small batch | 2/4 sequences，各自真实独立的 KV；不是共享 prefix 的特殊优惠 |
| Model mixed | 16 prefill tokens + 2 decode sequences，M=18；logits row 数明确 |
| Model natural generation | prompt16/128，生成32tokens；初始prefill与后续31次decode分开计时；长context32步记录真实增长区间 |

固定 context 的重复实验，在每轮计时前 clear 并重新构建对应 prefix，setup 不计入 forward；CPU/CUDA 使用相同 setup 语义。不为测量私自新增产品 truncate/rollback 接口。

### 9.3 顺序与重复

5 个独立 trial；每 case 2 次 untimed warmup、3 次 measured repetition。
比较时采用平衡顺序，例如 CPU8/CUDA/CUDA/CPU8，并在下一 block 反转，CPU16 另成对照；warmup 和 prefix setup 顺序也保留。
固定输入、编译与数值配置；同实现优化尽量同二进制选 variant。

记录温度、频率、功耗与背景负载中可获取的项目；无法锁频不冒充已控制。A/A 与异构对照分别运行，保留离群值。独立 trial 是统计单位，inner repeats 不是独立 trial。

已有历史 Windows/WSL 数据只用于背景，不直接算新 GPU 加速比。

### 9.4 原始产物

```text
benchmarks/results/<run_id>/
    manifest.json
    source-state.json
    source-snapshot.zip
    availability.json
    input.json
    weight-plan.json
    memory-plan.json
    validation-summary.json
    cpu8-*.json
    cpu16-*.json
    cuda-*.json
    copy-allocation-summary.json
    summary.json
    analysis.md
    profiler/
        artifact-manifest.json
        nsys-summary.json
        ncu-selected-kernel.json
```

大型 profiler 原始文件可存外部实验包；artifact-manifest 必须给出可获取位置、hash、工具版本和命令。仅指向作者 `.run` 绝对路径时，明确标为 local-only，不算完整可交付证据。

## 10. Correctness Tests

### 10.1 Unit

- DeviceBuffer：零长度、overflow、move、释放次数、分配失败清理。
- Matrix：非方阵与真实 N/K；不对称值验证 transpose、lda/ldb/ldc；输入输出 bounds。
- RMSNorm：epsilon、in-place 非支持时显式拒绝；零/小值/有限极值、真实宽度。
- RoPE：position=0/1/15/16/17/1535；NeoX 两半旋转；显式 D。
- FP16 KV conversion：RN-even 与现有转换 fixture 对照。
- Attention：GQA mapping、长度1/非整warp长度、多head、causal future masking。
- Argmax：相等取最低ID；NaN/Inf触发失败；不静默输出 token。
- 数学 fixture 的初始预注册 `atol=2e-4, rtol=2e-4`，使用已约束范围的输入与 CPU/FP64参考；具体边界 fixture 单列契约，不通过改统一阈值掩盖错误。

### 10.2 Property / State

固定 seed 生成 append/clear/interleaved-batch 序列，对照 CPU 状态模型：
- 每次 position/length 一致；
- full slot/context 边界前失败、不产生 partial logical commit；
- clear 后重新使用槽位与新实例计算一致；
- 用 NaN 污染未使用 KV，正确 mask 不受影响；
- 同批多 sequence 与按 sequence 单独执行数值一致；
- post-launch 注入故障后 poisoned，不输出、不继续；
- 有效运行与可测试异常销毁路径的 Compute Sanitizer memcheck/leak-check 通过。

GPU 没有 prefix alias/COW，因此本阶段不伪造 GPU COW 测试；现有 CPU COW 测试仍作为 host-model 提取的回归。

### 10.3 Model-level

用已有 CPU Runtime 与 matched-weight F32 llama reference 交叉验证，reference 运行与性能进程分离。

保留已有三个短英文 prompt 的8-token greedy金标准，必须一致。新增固定 token corpus：中文、英文、重复 token、特殊 token 输入与长度16/33/128/256/1536；特殊 token 的文本/类型语义独立记录，不修改下载模型。

teacher-forced 输入固定，不随两端自由生成的不同 token 漂移。比较多个位置的 logits，并保存 first divergence、RMSE、max absolute、cosine 与 argmax margin。

GPU 模型初始阈值：
- RMSE < 0.05；
- max absolute < 0.5；
- cosine >= 0.9999；
- 全部数值 finite；
- 原有稳定短样例 greedy 序列严格相同。

新 corpus 的 argmax：
- 当 reference top1-top2 gap > 2*observed_max_abs 时要求 argmax 相等；
- 其他位置标为 near-tie，记录差异，不把它宣传为严格 greedy 等价；
- 不能在结果出来后把任意 mismatch 改名 near-tie，判定公式和语料先入库。

GPU chunk/batch 对照使用独立的上述数值契约并记录生成差异；CPU 原有 same-runtime `1e-6/1e-5` gate 不变。不能强迫跨不同 GEMM shape/reduction 的 CPU/GPU 全 logits 位级相等，也不能删除同实现 off/on 位级检查。

### 10.4 Serving-level

本阶段 GPU Serving 尚未交付；必须运行现有 CPU `llmserve-http-tests`、核心生命周期与 deterministic mixed 测试，证明 host model/CMake 修改未破坏已有服务。

GPU HTTP、cancel、timeout、SSE 与压力正式 gate 在 V2-M2执行。本阶段最终报告必须明确这一边界，而不是填“8/8 HTTP”暗示GPU已经通过。

## 11. Performance Acceptance

以下在第一条性能实现提交前冻结。任何修改需独立修改 SPEC、说明原因并重新建立基线。

### Primary Metric

每个预注册 model case 的无 profiler `host_forward_to_token_ns`，以独立 trial 的中位数为统计样本。首个 GPU baseline 不设“必须快于 CPU/llama.cpp”的虚构承诺。

### Mandatory Data-path Gates

- 完整 forward 不调用 llama_decode/上游 graph compute；
- 稳态 weight H2D=0；
- 稳态 layer hidden H2D/D2H=0；
- 自有稳态 device allocation/free=0；
- 不下载全词表 logits 作为 greedy 正式路径；
- S/L/B 不被静默改变；memory plan与实际分配相符；
- CUDA 错误和 correctness 失败均为0。

### Secondary Metrics

prefill token/s、decode ms/token、每 batch latency、initialization/upload、peak owned bytes、设备时间、metadata/token transfer、launch count、selected kernel metrics。
不将 CLI 的模型时间称为 HTTP TTFT，也不把 device event 时间称为 client latency。

### Noise Threshold

先做 A/A。定义百分比噪声带：
`N = max(5%, abs(median(d_AA)) + 2 * MAD(d_AA))`，其中 d_AA 为独立 A/A paired relative differences。
使用配对的独立 trial 分析；置信区间与原始样本同时保存。仅5个trial时区间稳定性有限，落入噪声带或区间跨零的差异标记 inconclusive，不能挑最快轮次宣称加速。若 N 超过10%，本轮性能验收标为 measurement_inconclusive，不能据此宣称“没有退化”；正确性与数据路径结果仍单独保留。

### Allowed Regression

- 不涉及计算变化的 host-model/CMake 提取：旧 CPU 各代表 case 退化不应超过 N；超出先定位或回退提取。
- M1 GPU 首版：无旧自研 GPU performance baseline；以 mandatory gates 和正确性验收，不发明 allowed speed regression。
- 后续 GPU 优化：主 model case 收益需超过 N，且关键护栏 case 不得出现超过 N 的不解释退化；否则不晋升默认路径。

profiler on/off 差异另外报告；profiling 本身改变执行或开销明显时，仅用于诊断，不能代替无 profiler baseline。

## 12. Stop Conditions

1. 完整正确模型、resident data path、基线和证据包完成后，结束本 SPEC；下一步是 M2，而非顺手增加所有 kernel 优化。
2. custom GEMM 没有预先证据支持，不进入本阶段；cuBLAS允许长期保留。
3. GPU性能负结果保留并可作为完整 baseline 结论；数学/生命周期失败则为Not Done。
4. 内存预算不足时先报告计划与实际free memory，不能以隐藏offload绕过；缩小配置只能作为显式诊断。
5. 历史source ZIP无法找回时标记归档不完整，以新完整包继续；不无限追索。
6. 不以 standalone norm/softmax 或一层成功关闭完整模型任务。
7. 不为 CPU/GPU 统一而重写 graph、allocator、scheduler；小范围 binding 提取若造成大面积行为变化，应回退并缩小边界。

## 13. Files To Modify

### Existing

- `CMakeLists.txt`：独立own-CUDA选项/targets，保留上游开关与CPU-only。
- `include/minillm/runtime.h`、`src/minillm/runtime.cpp`：仅host binding/tokenizer/types的小范围提取；CPU执行顺序不改。
- `scripts/Benchmark-Common.ps1`：artifact availability/export可复用帮助函数。
- `scripts/Analyze-Runtime.ps1`、`scripts/Analyze-Benchmarks.ps1`：先验证输入/依赖，再原子发布新摘要；失败不破坏旧输出。
- `tests/model_tests.cpp`：可参数化的targeted validation入口；旧门槛保留。
- `tests/benchmark_validation_tests.ps1`、`tests/runtime_benchmark_tests.ps1`：缺件和非破坏性验证fixture。
- `scripts/dev.sh`：新增明确的 `own-cuda` 前缀或等价入口，不能改变原 `cuda`=上游语义。
- `docs/BENCHMARKS.md`、`docs/VALIDATION.md`、`THIRD_PARTY.md`、`docs/ENGINEERING_LOG.md`：实际能力、证据与失败边界。
- `.github/workflows/ci.yml`：保留CPU jobs；完整测试job明确检查依赖与预期套件，不用缺工具造成的skip冒充通过。

### New

```text
include/minillm/model_types.h
include/minillm/qwen3_model.h
include/minillm/tokenizer.h
src/minillm/qwen3_model.cpp
src/minillm/tokenizer.cpp

include/minillm/cuda/error.h
include/minillm/cuda/device_buffer.h
include/minillm/cuda/context.h
include/minillm/cuda/runtime.h
src/minillm/cuda/context.cpp
src/minillm/cuda/weights.cpp
src/minillm/cuda/workspace.cpp
src/minillm/cuda/contiguous_kv.cpp
src/minillm/cuda/matrix.cpp
src/minillm/cuda/kernels.cu
src/minillm/cuda/attention.cu
src/minillm/cuda/runtime.cpp
```

CUDA internal weights/workspace/KV declarations放在同目录internal header即可，不要求都成为public API。

```text
apps/cuda_main.cpp
apps/cuda_runtime_bench.cpp
tests/cuda_unit_tests.cu
tests/cuda_model_tests.cpp
tests/host_model_tests.cpp
tests/data/qwen3_validation_cases.json
tests/cuda_benchmark_validation_tests.py
scripts/Test-EvidenceAvailability.ps1
scripts/Export-BenchmarkBundle.ps1
scripts/Benchmark-CudaRuntime.ps1
scripts/Analyze-CudaRuntime.ps1
benchmarks/runtime-inputs/qwen3-cuda-v0.json
docs/CUDA_RUNTIME.md
```

不得在本阶段修改 `scheduler.cpp`、`prefix_index.cpp` 或把CPU PagedKV整体模板化。M2的MiniCudaRunner/HTTP接入文件不提前塞入本次范围。

## 14. Implementation Steps

### Step 1 — 证据 availability 与验证契约（INFRA）

实现非破坏性检查，输出exists/hash/mandatory/locator/status；fixture覆盖缺ZIP、损坏状态文件、路径迁移。分析器先preflight，向临时文件写新摘要，成功后原子替换。导出一个新baseline完整包并复验。

同时冻结数值语料recipe、math dtype/阈值、性能protocol；不要在结果出来后再选阈值。

### Step 2 — 独立 CUDA target 和 RAII（INFRA）

加入own-CUDA开关、CudaContext/DeviceBuffer/error路径，配置单stream与cuBLAS handle。用目标层面的单元测试验证move/overflow/异常清理和FP32 matrix小例子，复用现有smoke环境，不重装工具。

### Step 3 — Immutable host model（INFRA）

提取binding/types/tokenizer；旧CPU使用同一校验与同一计算流程。跑旧CPU numerical、profile、state、HTTP；固定输入logits与提取前对照，不能把差异混入GPU实现。

### Step 4 — Resident weights、workspace、matrix（INFRA + PRODUCT）

实现weight arena、受限staging、唯一tensor布局、memory plan、预分配workspace、真实shape GEMM。
一次性上传的有效权重与host decode对照，数据尺寸/leading dimension和tiedweight正确。

### Step 5 — 基础算子与层内数据流（PRODUCT）

实现gather、RMSNorm、Q/K norm、RoPE、residual、SwiGLU、argmax/finitecheck。各算子unit通过；此时尚不能声明完整层或模型完成。

### Step 6 — 连续 KV、causal attention 与完整层（PRODUCT）

实现per-seq slot/length、FP16store、QK-softmax-PV、因果/GQA、interleaved token语义。
通过共享输入的单层对照、随机state与mask污染测试；追加错误采用preflight/poisoned两类语义。

### Step 7 — 完整模型与 token CLI（PRODUCT）

串接28层、final norm、selected-row LMhead、greedy输出。
CLI支持明确model/limit参数及report。GGML_CUDA关闭组合完成真实生成，source=Q8_0/device=F32标记准确。

### Step 8 — 数值与性能对照（MEASUREMENT）

运行全部targeted corpus、crossbackend teacherforced、chunk/batch、32token生成；CPU8/16和GPU完整报告。
按照冻结protocol建立micro/model、A/A、copy/allocation和memory统计。现有CPU HTTP回归单列。

### Step 9 — Profiler 与交付（MEASUREMENT）

对完整模型而非smoke采NSys；选择一个kernel用NCU检查，不把replay时间作正式性能。
导出完整包、独立目录复验；写analysis/limitations/stop-or-next决定及第三方归属。

## 15. Commit Plan

| Commit | 推荐提交说明 | 必须附带的验证 |
|---|---|---|
| 1 | `fix(evidence): validate bundle availability before replacing summaries` | missing/tamper/relocation fixture；旧文件hash不变；新包复验 |
| 2 | `feat(cuda): add isolated runtime target and resource owners` | OFF/ON构建；RAII/unit；基础memcheck |
| 3 | `refactor(model): extract immutable Qwen3 binding and tokenizer` | CPU logits/profile/state/HTTP非回归 |
| 4 | `feat(cuda): upload effective weights and add resident matrix path` | weight/memoryplan；真实shape/GEMM layout |
| 5 | `feat(cuda): implement Qwen3 normalization rotary and pointwise ops` | unit/tails/finite/argmax |
| 6 | `feat(cuda): add contiguous KV and causal GQA layer execution` | layer/state/mask/poisoned/memcheck |
| 7 | `feat(cuda): execute full Qwen3 and generate greedy tokens` | 关闭上游CUDA的完整模型CLI/数值对照 |
| 8 | `test(cuda): add cross-backend model and benchmark contracts` | 完整targeted corpus、统计fixture、冻结protocol |
| 9 | `docs(cuda): publish validated model baseline and profiler evidence` | 可获取完整bundle、复验结果、负结果/限制 |

较大的实现commit可以再拆分，但不把specscope扩展到M2/M4。实验报告可能属于实现commit后的dirty文档工作区，必须记录实际采集身份。

## 16. Definition of Done

以下全部满足为 Done：

- [ ] Audit HEAD到开发HEAD的差异已复核；未执行无关scope扩展。
- [ ] 现有core/CPU构建与必需CTest/HTTP/模型回归通过。
- [ ] source-state、binary、model、input与工具配置完整可获取；缺artifact的旧归档不再冒充完整。
- [ ] independent own-CUDA flag生效；GGML_CUDA关闭组合实际生成token。
- [ ] Qwen3 host validation/tokenizer ownership明确，CPU行为保持原契约。
- [ ] device weights为一次性转换上传的FP32有效权重，source/device dtype均正确报告。
- [ ] 完整模型在GPU执行，连续FP16 KV、causal/GQA/RoPE与argmax正确。
- [ ] S=1与S=4目标配置有验证；B/context上界与失败行为明确。
- [ ] preflight错误不污染状态；post-launch错误进入poisoned，不发布本次结果。
- [ ] 正常与受控异常资源路径通过对应sanitizer/owner测试。
- [ ] kernel与model数值门槛通过；稳定短golden序列严格一致；新语料near-tie按预注册规则披露。
- [ ] 正式steady-state没有逐层weight/hidden传输、没有自有device allocation、没有全logits下载作为greedy路径。
- [ ] 全部预注册性能case有结果或明确失败；没有选择性删除慢case/失败case。
- [ ] NSys证据来自完整模型，时钟、logical bytes与硬件metrics分开解释。
- [ ] 完整bundle在独立目录复验；大artifact可获取性与hash清楚。
- [ ] 交付文档明确：GPU Serving、GPU paging、custom PagedAttention本阶段仍未交付。

正性能结果不是强制项；真实完整路径、correctness和资源/证据门禁是强制项。只做完DeviceBuffer、RMSNorm或单层时，结论必须是 Not Done，而不是“CUDA Runtime已完成”。

## 17. 官方参考与归属

实施时阅读并固定本次采用的版本/页面：
- CUDA 12.8 cuBLAS：data layout、GemmEx compute type、stream/workspace与错误语义。
- llama.cpp/ggml：只借鉴backend buffer/lifetime、矩阵执行边界；本项目GGUF/tokenizer依赖保持pin。
- vLLM：未来M5的allocation与pressure recovery；本阶段不移植其scheduler。
- SGLang：未来共享/eviction的引用保护思想，不在本阶段重写Trie。
- TensorRT-LLM：GPU KV预算与调度估计/物理复用边界；不复制分布式生产架构。

任何vendor库、参考转换与模型结构均按THIRD_PARTY记录。cuBLAS矩阵计算不是自研GEMM，但不妨碍本项目拥有自研CUDA模型执行、KV、数据流与生命周期。
