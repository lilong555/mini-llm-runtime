# MiniLLM / LLMServe v0.2.0

状态：本地发布草稿，未创建标签或公开 Release。最终 clean candidate SHA
和对应 CI 尚待核验；本页不是发布就绪声明。

## 支持范围

C++20 单模型 CPU/CUDA Runtime 与单 GPU 在线 Serving。自有 forward、
KV 管理、组批和生命周期；GGUF 元数据与 tokenizer 复用 llama.cpp，
GPU 矩阵使用 cuBLAS，HTTP 使用 cpp-httplib。

默认 CPU 路径与 F32/contiguous CUDA 路径可用。GPU 分页为显式 opt-in
容量研究模式，最终 B；FP16 matrix 因数值门禁失败，禁止在 Serving 启用。
CUDA 范围为指定 Qwen3-0.6B、greedy、单 stream、同步 execute、
P16、S≤4、Lmax≤2048、B≤128，不支持 GPU prefix sharing。

## 环境与复现

本机功能验证环境为 WSL2、RTX 4070 Laptop、CUDA 架构 89。
构建要求 CMake≥3.24、C++20 编译器、Ninja、Python、PowerShell；
own CUDA 另需 CUDA Toolkit≥12.8 和兼容驱动。89 不是通用 GPU 架构。

从 clone 到 CPU/CUDA 生成、服务及 SSE 的入口见 [README](../README.md)；
深入命令见 [WSL 开发](WSL_DEVELOPMENT.md)。
F0 隔离候选已通过 CPU/CUDA CTest 16/23、两后端 HTTP 各12项、
短模型128次 logits 比较及五个组件 memcheck；不冒充最终 SHA 验证。
版本更新为0.2.0后，在该隔离目录重新配置、构建，CPU/CUDA CTest
仍为16/16与23/23，两后端8-token生成一致；HTTP沿用本轮F0检查，
未在版本变更后另行重跑。最终提交后仍须核验其代码身份与适用门禁。

## 历史结果

| 研究 | 结论 |
| --- | --- |
| CPU Q8 dot | hot-cache 微基准约4.29倍，不是端到端加速 |
| CUDA 模型 | 24项比较中14项更快、10项不确定，非全面胜出 |
| CUDA Serving | mixed-length 吞吐中位数之比约提升7.09%；burst 不确定 |
| GPU KV 同容量 | Serving 吞吐配对中位退化15.73% |
| GPU KV 同288MiB预算 | 吞吐提升4.31%，TPOT/ITL更高，不是低延迟胜利 |
| FP16 matrix | owned 显存减少34.53%，模型 cosine 未通过，停止推进 |

各项工作负载、采集身份与限制见 [性能](PERFORMANCE.md)。
这些是独立历史采集，不是 v0.2.0 重新测量。

## 证据与限制

- [GPU KV canonical bundle 索引](../benchmarks/results/gpu-kv-001/README.md)
- [Serving canonical bundle 索引](../benchmarks/results/cuda-serving-001/README.md)
- [精度负结果索引](../benchmarks/results/cuda-precision-001/README.md)
- [模型基线](CUDA_BENCHMARKS.md)

同容量分页不节省 KV payload；同预算比较的是共享页池与当前等长静态槽，
不是所有连续分配器。已有测试不构成生产级 SLO、任意故障恢复或所有设备保证。
不附模型、凭据、构建输出、依赖 checkout 或本地服务状态。
