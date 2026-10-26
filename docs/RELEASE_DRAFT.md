# MiniLLM / LLMServe v0.2.0

状态：发布草稿。正式标签和公开 Release 由用户确认后创建，
候选验收要求见[版本约定](https://github.com/lilong555/mini-llm-runtime/blob/main/docs/VERSION_CONTROL.md)。

## 支持范围

C++20 单模型 CPU/CUDA Runtime 与单 GPU 在线 Serving。自有 forward、
KV 管理、组批和生命周期；GGUF 元数据与 tokenizer 复用 llama.cpp，
GPU 矩阵使用 cuBLAS，HTTP 使用 cpp-httplib。

默认 CPU 路径与 F32/contiguous CUDA 路径可用。GPU 分页为显式 opt-in
容量研究模式，最终 B；FP16 matrix 因数值门禁失败，Serving 使用 F32。
CUDA 范围为指定 Qwen3-0.6B、greedy、单 stream、同步 execute、
P16、S≤4、Lmax≤2048、B≤128，不支持 GPU prefix sharing。

## 环境与复现

本机功能验证环境为 WSL2、RTX 4070 Laptop、CUDA 架构 89。
构建要求 CMake≥3.24、C++20 编译器、Ninja、Python 3、PowerShell 7；
Own CUDA 另需 CUDA Toolkit≥12.8 和兼容驱动，其它 GPU 需选择对应架构。

从 clone 到 CPU/CUDA 生成、服务及 SSE 的入口见
[README](https://github.com/lilong555/mini-llm-runtime/blob/main/README.md)；
深入命令见 [WSL 开发](https://github.com/lilong555/mini-llm-runtime/blob/main/docs/WSL_DEVELOPMENT.md)。
功能验收与采集身份见
[项目状态](https://github.com/lilong555/mini-llm-runtime/blob/main/docs/EXECUTION_STATUS.md#候选验证)。

本机 CPU/CUDA CTest 为 16/16、23/23，HTTP 各 12/12，
短模型 128 次比较、6 个金标准与五组件 memcheck 通过。
CPU HTTP 保留一次停服建场失败和空闲复验通过的记录，见
[ENG-082](https://github.com/lilong555/mini-llm-runtime/blob/main/docs/ENGINEERING_LOG.md)。

## 历史结果

| 研究 | 结论 |
| --- | --- |
| CPU Q8 dot | hot-cache 算子微基准约 4.29 倍 |
| CUDA 模型 | 24 项比较中 14 项更快、10 项不确定 |
| CUDA Serving | mixed-length 吞吐中位数之比约提升 7.09%；burst 不确定 |
| GPU KV 同容量 | Serving 吞吐配对变化中位数为 -15.73% |
| GPU KV 同 288 MiB 预算 | 吞吐配对变化中位数为 +4.31%，TPOT/ITL 更高 |
| FP16 matrix | owned 显存减少 34.53%，模型 cosine 未通过，研究停止 |

各项工作负载、采集身份与限制见
[性能](https://github.com/lilong555/mini-llm-runtime/blob/main/docs/PERFORMANCE.md)。
各项数字沿用独立历史采集的源码、模型和环境身份。

## 证据与限制

- [GPU KV 完整证据索引](https://github.com/lilong555/mini-llm-runtime/blob/main/benchmarks/results/gpu-kv-001/README.md)
- [Serving 完整证据索引](https://github.com/lilong555/mini-llm-runtime/blob/main/benchmarks/results/cuda-serving-001/README.md)
- [精度负结果索引](https://github.com/lilong555/mini-llm-runtime/blob/main/benchmarks/results/cuda-precision-001/README.md)
- [模型基线](https://github.com/lilong555/mini-llm-runtime/blob/main/docs/CUDA_BENCHMARKS.md)

同容量两种布局的 KV payload 均为 896 MiB；同预算比较的对象为共享页池
与当前等长静态槽。适用范围以各份证据的模型、硬件和负载为准，
服务用于本机研究与演示。交付包含源码、文档、固定输入与公开证据索引；
模型通过 manifest 单独获取，构建输出、依赖 checkout 和服务状态留在本地。
