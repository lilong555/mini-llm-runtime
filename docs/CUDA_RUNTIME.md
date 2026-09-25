# 自有 CUDA 运行基础

`MINILLM_ENABLE_CUDA` 默认关闭，与控制上游 ggml 的 `LLMSERVE_CUDA` 独立。当前提供设备资源所有权、单 stream、cuBLAS FP32 矩阵接口及设备单元测试。完整 Qwen3 GPU forward、GPU token CLI、GPU Serving 和 GPU PagedAttention 尚未提供。实施阶段见 [执行状态](EXECUTION_STATUS.md)。

## 构建与验收

WSL2、CUDA Toolkit >= 12.8、C++20 工具链和固定版本 llama.cpp 为构建前提。本机 RTX 4070 Laptop 使用架构 `89`：

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda memcheck
```

构建目录为 `build/wsl-own-cuda`，上游 CUDA 关闭。可通过 `CUDA_ARCHITECTURES` 指定目标架构。`minillm_cuda` 链接 CUDA Runtime 和 cuBLAS，CPU 产品不链接该 target；`.cu` 测试使用 CUDA C++20。关闭自有 CUDA 的构建不需要 CUDA Toolkit。自有 CUDA 开启、`LLMSERVE_WITH_LLAMA=OFF` 的组合在配置阶段明确拒绝。

现有 `bash scripts/dev.sh cuda ...` 仍选择上游 llama.cpp CUDA 参照后端。自有 CUDA 当前只支持 `build/test/memcheck`，没有可调用的模型或服务入口。

## 所有权与完成协议

- `DeviceBuffer<T>` 为不可复制、可移动的设备内存 owner；空 buffer 不分配，数量和字节范围在分配前检查。移动后源对象为空，清理不抛异常。
- `CudaContext` 为单调用者、不可重入的 owner，持有 nonblocking stream、cuBLAS handle 和默认 4 MiB 的显式 workspace。先设置 stream，再绑定 workspace，避免 `cublasSetStream` 重置绑定。
- 每个对象记录设备编号。设备操作临时选择该设备，随后恢复调用线程先前的设备。
- 入队成功不等于 GPU 执行完成。调用方在读取结果或释放、移动覆盖外部 buffer 前，必须调用并检查 `CudaContext::synchronize()`。context 析构只做尽力清理，不能代替显式错误检查，也不拥有调用方的 buffer。
- `DeviceTensorView<T>` 只保存设备指针、行列数、stride、capacity 和设备编号；dtype 由 `T` 决定。view 的有效期受 owner 约束，host 不能解引用设备指针。

## 矩阵契约

`matrix_multiply` 接受 row-major 的 `X[M,K]`、`W[N,K]`、`Y[M,N]`，计算 `Y = X * transpose(W)`。stride 和 capacity 以元素计。调用前检查形状、`INT_MAX` 范围、容量、设备、地址对齐和输出重叠；view 必须来自有效的设备分配，描述符本身不提供分配来源认证。

矩阵通过 `cublasGemmEx` 执行，参数为 `opA=T/opB=N/m=N/n=M/k=K`。输入、输出和计算模式固定为 FP32、`CUBLAS_COMPUTE_32F_PEDANTIC`，不启用 FAST_TF32、FAST_16F 或 `--use_fast_math`。接口只入队，允许调用方按同一 stream 组织依赖。

11 项设备单测覆盖移动和释放计数、分配故障、构造异常清理、cuBLAS 模式、kernel 到 GEMM 的 stream 顺序、非方阵、带 padding 的 NaN/guard、非 warp 对齐维度、Q 投影形状及非法描述符。原始 CTest、memcheck、构建边界和 CPU 回归见 [CUDA 基础验收](../benchmarks/results/validation/cuda-infra/README.md)。这些是基础层正确性证据，不构成模型性能结论。

## 第三方边界

设备资源封装、矩阵描述符、边界检查和测试由本项目实现；设备内存及 stream 由 NVIDIA CUDA Runtime 提供，矩阵内核由 NVIDIA cuBLAS 提供。上游 llama.cpp 的完整 GPU 模型执行归属不变。接口契约依据 [cuBLAS 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html) 和 [CUDA Runtime 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY.html)。
