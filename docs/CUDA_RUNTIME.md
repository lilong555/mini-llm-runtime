# 自有 CUDA Runtime

`MINILLM_ENABLE_CUDA` 默认关闭，与控制上游 ggml 的 `LLMSERVE_CUDA` 独立。`CudaRuntime` 在单 stream 执行完整 Qwen3 forward，由项目控制常驻 FP32 有效权重、workspace、连续 FP16 KV、因果 GQA 和 greedy 输出，矩阵由 cuBLAS 提供。GPU Serving、prefix sharing 和 PagedAttention 尚未提供。短语料和 S=1/S=4 通过验证；全量长语料与正式性能基线仍在独立门禁内，见 [执行状态](EXECUTION_STATUS.md)。

## 构建与验收

WSL2、CUDA Toolkit >= 12.8、C++20 工具链和固定版本 llama.cpp 为构建前提。本机 RTX 4070 Laptop 使用架构 `89`：

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda generate --prompt "The capital of France is" --tokens 8
bash scripts/dev.sh own-cuda memcheck
bash scripts/dev.sh own-cuda storage-check
bash scripts/dev.sh own-cuda storage-memcheck
bash scripts/dev.sh own-cuda layer-check
bash scripts/dev.sh own-cuda layer-memcheck
python3 scripts/models.py --reference --converter build/wsl-own-cuda/bin/mini-llm
bash scripts/dev.sh own-cuda model-check
bash scripts/dev.sh own-cuda model-memcheck
```

构建目录为 `build/wsl-own-cuda`，上游 CUDA 关闭。可通过 `CUDA_ARCHITECTURES` 指定目标架构。`minillm_cuda` 链接 CUDA Runtime 和 cuBLAS，CPU 产品不链接该 target；`.cu` 测试使用 CUDA C++20。关闭自有 CUDA 的构建不需要 CUDA Toolkit。自有 CUDA 开启、`LLMSERVE_WITH_LLAMA=OFF` 的组合在配置阶段明确拒绝。

现有 `bash scripts/dev.sh cuda ...` 仍选择上游 llama.cpp CUDA 参照后端。自有 CUDA 的 `generate` 调用 `mini-cuda-llm`，没有 `serve` 入口。`storage-*`、`layer-*`、`model-*` 验证命令需要固定模型，`model-*` 还需要 matched-weight F32 参照；默认使用带时间和进程号的新报告目录，也可指定尚不存在的目录，不覆盖已有报告。

## 模型接口

公开头文件为 `include/minillm/cuda/runtime.h`，C++20 调用方链接 `minillm_cuda`：

- `CudaRuntimeConfig` 指定 model、device、S、Lmax、B 和可选设备预算；S 为 1–4，B 为 1–128，Lmax 不超过训练范围。默认 S=4、Lmax=2048、B=128，不自动缩小配置。
- `forward(span<InputToken>, mode, device_timing)` 同步返回 `CudaForwardResult`。`InputToken` 的 token、position、sequence 与 CPU 接口相同，`logits` 选择需要输出的行；支持同一序列多 token 与交错序列。
- `samples` 按原 batch 的 `input_index` 排序。默认 `greedy` 仅下载 token/status；显式 `debug_logits` 返回与 samples 对齐的全词表 `Logits`。没有输出行时仍完成 KV 写入与状态检查。
- `clear_sequence` 只在 ready 完成点重置逻辑长度；`diagnostics` 返回已提交长度、常驻计划、计数与 ready/poisoned 状态。
- `tokenize`、`token_piece`、`is_eog` 使用独立 vocab-only tokenizer；`weight_manifest` 提供唯一权重及别名的形状、源 dtype、设备偏移和有效权重摘要。

执行顺序为 embedding gather → 全部 28 层 → final norm → selected-row gather → LM head → finite/argmax → token/status 下载 → checked completion → 提交长度。完整路径不构造 CPU Runtime，不调用 `llama_decode()`，不回退 CPU。

单实例要求单调用者、不可重入。preflight 或输出容器准备失败发生在设备执行前，长度不变、实例仍可用。设备执行开始后，任何 CUDA/cuBLAS 错误或 nonfinite status 都自动使实例进入 poisoned，不返回本 batch 的结果、不提交长度，不能通过 clear 复用。异常路径在局部 debug 下载目标销毁前终结在途工作；这不是物理 KV 回滚，也不承诺 fatal device error 后的 context 可恢复。

## CLI 与报告

```bash
build/wsl-own-cuda/bin/mini-cuda-llm \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --tokens 8 \
  --context 2048 --sequences 4 --batch 128 --chunk 128 \
  --device 0 --device-time --output .run/cuda-generate.json
```

`--output` 必须是新文件；报告也写入 stdout。CLI 只生成 sequence 0，支持 chunked prefill 和逐 token decode，默认遵守 EOG，`--ignore-eos` 可固定输出数量。实际 KV 上限检查为 `prompt_tokens + completion_tokens - 1 <= Lmax`，最后一个输出无需再作为输入写入 KV。多序列通过 C++ 接口使用。

报告包含模型 SHA-256、实际 source/device dtype、设备和版本、S/L/B、输入与输出 token、文本、每次 forward 和清理前后的 diagnostics。`host_forward_to_token_ns` 包含预检、必要 copy、GPU 执行、argmax 与完成检查，不含模型加载；可选 `device_elapsed_ms` 使用预创建 CUDA events，关闭时为 null。`model_load_ns`、`storage_initialization_ns` 与其内部的 `weight_decode_upload_ns` 单列，后两者不是可相加的独立阶段。

显式 copy 的累计计数区分权重、RoPE、metadata、token、status 和 debug logits。每个成功 batch 的 metadata 为 `4*(3*M+R)` 字节，token 为 `4*R` 字节，status 为 8 字节；M 是输入行数，R 是输出行数。debug 另下载 `4*R*vocabulary` 字节。这些计数不包含 kernel 参数或 NVIDIA 库内部传输。Runtime 不搬运中间激活，稳态不再上传权重或 RoPE。

`owned_device_allocations/bytes` 表示当前 owner 的项目存储；实际热路径 allocation/free 调用另由 `allocation_stats()` 采样检查。CLI 计时不是 HTTP TTFT 或正式性能基线，不能据此得出加速比。

## 所有权与完成协议

`CudaStorage` 使用 [Qwen3 Host Model](HOST_MODEL.md) 的只读绑定，不创建 CPU Runtime、执行线程池或 CPU KV。其内部声明位于 `src/minillm/cuda/storage.h`，不构成模型 forward API。设备权重不依赖 host mapping 的后续存活。

- `DeviceBuffer<T>` 为不可复制、可移动的设备内存 owner；空 buffer 不分配，数量和字节范围在分配前检查。移动后源对象为空，清理不抛异常。
- `CudaContext` 为单调用者、不可重入的 owner，持有 nonblocking stream、cuBLAS handle 和默认 4 MiB 的显式 workspace。先设置 stream，再绑定 workspace，避免 `cublasSetStream` 重置绑定。
- 每个对象记录设备编号。设备操作临时选择该设备，随后恢复调用线程先前的设备。
- 入队成功不等于 GPU 执行完成。调用方在读取结果或释放、移动覆盖外部 buffer 前，必须调用并检查 `CudaContext::synchronize()`。context 析构只做尽力清理，不能代替显式错误检查，也不拥有调用方的 buffer。
- `DeviceTensorView<T>` 只保存设备指针、行列数、stride、capacity 和设备编号；dtype 由 `T` 决定。view 的有效期受 owner 约束，host 不能解引用设备指针。
- `CudaStorage` 持有一个权重 arena、一个 workspace arena 和一份连续 FP16 KV 预留。`output.weight` 在 tied 模型中直接别名到 embedding，不重复上传或释放。析构和构造异常路径先完成在途工作，再释放存储，context 最后销毁。
- `allocation_stats()` 是进程内项目分配器的累计计数，不含 CUDA/cuBLAS 内部资源，也不是显存占用率。归属到某个 owner 的测量区间不能有其他线程分配或释放。

## 权重与显存

每个唯一 tensor 由既有 `decode_row` 转成 FP32；8 MiB host staging 按完整行分块，每块完成上传后才复用。非有限有效权重导致初始化失败。源模型的 Q8_0 量化并未恢复到量化前权重，也没有 Q8 CUDA GEMM。每个 tensor 记录源 dtype、形状、256-byte 对齐偏移、字节数、别名和有效权重 SHA-256。

workspace 包含 hidden、normalized、Q/K/V、attention、投影、gate/up/down、选中行、scores/probabilities、logits、RoPE 表和整数 metadata/status/sample 区域。各层复用固定区域，类型和行数在返回 view 前检查。KV 布局为 `[sequence][layer][K_or_V][position][kv_head * head_dim]`；未使用区域初始化为 FP16 NaN。RoPE 表按 CPU 原公式初始化一次，有独立上传计数，不在层执行中上传。

目标模型在 S=4、Lmax=2048、B=128 下的计划与实际项目分配一致：

| 项目 | 字节 |
| --- | ---: |
| 唯一 FP32 权重 | 2,384,199,680 |
| workspace，含 RoPE 表与对齐 | 121,311,232 |
| 连续 FP16 KV 预留 | 939,524,096 |
| 显式 cuBLAS workspace | 4,194,304 |
| 自有分配总量 | 3,449,229,312 |

context/cuBLAS 建立后读取 free memory。预算取用户上限、free 的 80% 和 `free - 512 MiB` 中的最小值；用户上限为 0 时只应用后两项。当前 context 已分配的显式 cuBLAS workspace 仍计入完整计划比较，属于额外保守预留。预算不够时在分配三个 arena 前拒绝；不会缩短配置或回退 CPU。外部显存竞争仍可能使后续实际分配失败，错误保留 CUDA 诊断并清理已创建的资源。

## 矩阵契约

`matrix_multiply` 接受 row-major 的 `X[M,K]`、`W[N,K]`、`Y[M,N]`，计算 `Y = X * transpose(W)`。stride 和 capacity 以元素计。调用前检查形状、`INT_MAX` 范围、容量、设备、地址对齐和输出重叠；view 必须来自有效的设备分配，描述符本身不提供分配来源认证。

矩阵通过 `cublasGemmEx` 执行，参数为 `opA=T/opB=N/m=N/n=M/k=K`。输入、输出和计算模式固定为 FP32、`CUBLAS_COMPUTE_32F_PEDANTIC`，不启用 FAST_TF32、FAST_16F 或 `--use_fast_math`。接口只入队，允许调用方按同一 stream 组织依赖。

11 项设备单测覆盖移动和释放计数、分配故障、构造异常清理、cuBLAS 模式、kernel 到 GEMM 的 stream 顺序、非方阵、带 padding 的 NaN/guard、非 warp 对齐维度、Q 投影形状及非法描述符。原始 CTest、memcheck、构建边界和 CPU 回归见 [CUDA 基础验收](../benchmarks/results/validation/cuda-infra/README.md)。这些是基础层正确性证据，不构成模型性能结论。

[权重与存储验收](../benchmarks/results/validation/cuda-storage/README.md) 提供 310 个唯一 tensor 的全量逐字节回读、tied/untied 及 F32/F16 fixture、预算与初始化失败、88 组真实权重矩阵检查。M=1/2/4/8/16/18/32/64/128 使用稀疏四点输入，M=1/2 另有稠密输入，全部输出对照 CPU FP64。重复 GEMM 阶段项目分配/释放调用为零；这不是完整 forward 的稳态验收。实模型验证和资源测试在 Compute Sanitizer 下为 0 错误、0 泄漏。

## 基础算子

内部接口位于 `src/minillm/cuda/ops.h`，所有操作只向现有 context 的 stream 入队，不分配设备内存或在算子内部同步。调用方必须保证 view 指向有效分配，并在完成检查前保持其 owner 存活。

- `gather_rows` 支持 embedding 和选中 hidden 行的采集；索引可重复，非法设备索引先屏蔽读取，再写入 NaN 与错误标记。
- `rms_norm` 按 weight 宽度分组，覆盖 hidden norm 和 Q/K head norm；支持完全相同布局的原地输出。输入最大绝对值与 `sqrt(epsilon)` 共同决定缩放因子，归约和激活仍为 FP32，避免极值平方或方差加法溢出。
- `rope` 原地执行 NeoX 两半旋转。设备系数表为 `[L,D]`，前半 cosine、后半 sine，`D` 是显式 head_dim；初始化方提供系数表，算子不逐步调用 host 三角函数。
- `residual_add` 和 `swiglu` 支持有 stride 的逐元素原地操作，部分重叠在 launch 前拒绝。
- `argmax` 融合全部 logits 的 finite 检查；相等时选择最小 token ID，任意 NaN/Inf 使对应行返回 `-1`。
- `status[1,2]` 保存错误位集合与首个错误算子输入行。每次执行开始调用 `reset_status`，之后各算子累计标记；选中行的 LM head/argmax 行号不一定等于原 batch 下标。预检失败不改写设备内容，设备错误标记不能视为有效生成结果。

11 项算子测试覆盖实际宽度、151936 词表、非整 warp 尾部、stride/padding、重复与越界索引、原地与部分重叠、FP64 对照及单 stream 组合执行。普通 CTest 与 memcheck、racecheck、synccheck 均通过；完整证据见 [基础算子验收](../benchmarks/results/validation/cuda-ops/README.md)。`check_finite` 可独立累计任意 activation 的非有限值错误。

## 连续 KV 与层

内部 `BatchState` 仅管理逻辑长度，`CudaStorage` 拥有实际设备 KV。sequence ID 直接对应固定 slot，每序列有独立 Lmax；不支持分页、alias 或 COW。状态按 `ready → prepared → executing → ready` 运行：

- `prepare` 在首个设备写入前校验所有 token、sequence、连续 position 和 batch/context 上限，只准备 pending lengths。
- 预检失败不修改已提交长度，仍可继续使用；未启动的准备状态可以丢弃。
- 调用方在 checked completion 和设备 status 均通过后才能 `commit`。执行后失败必须 `poison`，不能 clear 或再次准备；这是 fail-stop，不是物理回滚。
- clear 仅在 ready 完成点重置逻辑长度；重用 slot 时从 position=0 开始覆盖，旧数据不得被新 query 读取。

`store_kv` 使用 `__float2half_rn` 写入 K/V，非法 metadata 在访问前屏蔽，转换后非有限值累计错误。`causal_attention` 使用显式 QK → softmax → PV：GQA 映射为 `query_head / (Hq/Hkv)`，每个 query 只读自己的 `position+1`，max_context 仅约束 launch 上界。QK/PV 为 FP32，softmax 指数为 FP32、分母为 FP64；不宣称 FlashAttention。

`LayerExecutor` 缓存同一 storage 的权重 views，在单 stream 串接 norm、Q/K/V、Q/K norm、RoPE、KV store、attention、output/residual、FFN 和 finite-check。调用方准备 hidden/metadata 并重置 status；层本身不 reset status、不同步、不提交长度、不分配设备内存。

[连续 KV 与层验收](../benchmarks/results/validation/cuda-layer/README.md) 包含 7 项状态/数学测试、六组首层与末层真实权重对照、三种 sanitizer，以及四种构建共 741 次用例执行。基础算子与共享 Q/K/V 边界对照保持 `2e-4` 混合容差；独立真实整层使用固定模型门槛，FP16 舍入跨界的原始失败与较大误差保留在 `ENG-039` 和报告中。该证据不包含完整 28 层或实际生成 token。

## 完整模型验证

[完整模型与 CLI 验收](../benchmarks/results/validation/cuda-model/README.md) 包含四种构建共 749 次用例执行、8 项 Runtime 检查、128 组 logits 对照、S=1/S=4 的六组 8-token 金标准和三个独立 CLI 生成。对照分别使用自有 Q8_0 CPU Runtime 与 matched-weight F32 llama.cpp CPU；两种参考与 GPU 顺序构造。最大 RMSE 为 `0.005696512`、最大绝对误差为 `0.024068833`，没有 argmax 差异。

Runtime 测试覆盖无输出行、选中行顺序、事件计时开关位级一致、预检恢复、clear 后重用、预算拒绝、非有限输出和受控完成检查失败。普通完整模型和 memcheck 均通过，memcheck 为 0 错误/0 泄漏，Runtime racecheck/synccheck 均为零错误。受控同步失败由测试包装器在实际完成后注入，不是一次真实 illegal access 的恢复试验。

该组覆盖短金标准、四类 33-token 语料、128-token prefill、16+2 mixed 和四序列交错。长度 256/1536、全部 chunk/sequence 组合、32-token 自然生成、CPU8/16 对照、A/A、完整模型 Nsight 与正式归档仍属于 Step 8–9，不能以当前记录关闭 V2-M1。

## 第三方边界

设备资源封装、权重转换调度、存储布局、预算、基础算子、KV 状态、因果 attention、完整 forward 与 greedy 由本项目实现；内存及 stream 由 NVIDIA CUDA Runtime 提供，矩阵内核由 NVIDIA cuBLAS 提供，block/warp 归约复用 Toolkit 的 CUB。当前 CUDA 12.8 安装包含 CUB 2.7.0。GGUF、tokenizer 与 SHA-256 来自固定 llama.cpp 依赖，模型计算只在 tests/reference 路径调用上游 forward。接口契约依据 [cuBLAS 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html)、[CUDA Runtime 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html) 与 [CUB BlockReduce](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html)。
