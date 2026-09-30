# 系统架构

MiniLLM 是限定模型的 C++20 CPU/CUDA Runtime 与单 GPU 在线 Serving
研究原型，不是分布式或生产级推理平台。

## 执行与依赖边界

```text
HTTP / SSE（httplib）
    -> Engine（准入、信用、请求生命周期）
    -> schedule_batch（动态组批、chunked prefill、mixed）
    -> MiniRunner / MiniCudaRunner / LlamaRunner
    -> 自有 CPU / 自有 CUDA / 上游 llama.cpp
    -> 整批 sample 验证
    -> RequestHandle / SSE
```

GGUF 元数据和 tokenizer 复用上游。自有 HostModel 管理 mmap、张量视图、
形状和 bounds；CPU 与 CUDA forward 自行实现 Qwen3 执行顺序。
CUDA 矩阵运算使用 cuBLAS，归约使用 CUB；这些不宣称为自研 GEMM。
`LlamaRunner` 的上游 GPU 执行与 `MiniCudaRunner` 的自有 CUDA 路径分开。

## Ownership

- `CudaStorage` 拥有 context、权重 arena、workspace、KV slab 和设备块表。
- `LayerExecutor` 借用 typed views，不拥有模型或按请求重新分配权重。
- `BatchState` 是 sequence length 的唯一逻辑账本。
- `PageTableState` 拥有 host free IDs、active/pending mappings 与页事务。
- `CudaRuntime` 管理执行和完成点；`MiniCudaRunner` 负责服务契约转换。
- `Engine` 负责逻辑信用，不能把信用释放解释为 GPU 物理状态恢复。

## 精度与布局

源模型为固定 Q8_0 GGUF，初始化解量化为 resident F32 大矩阵。
矩阵输入、输出和主体 hidden 为 F32；KV 为 F16。
这不是 native Q8 CUDA，也不是默认 FP16 Tensor Core 执行。

CPU KV 使用物理分页、共享引用和 COW；GPU 默认连续槽位，
可选共享容量但每页独占的分页池。GPU 不支持 prefix sharing。

GPU paged 布局：

```text
[layer][K_or_V][physical_page][token_in_page][kv_head][head_dim]
block_table[sequence][ceil(Lmax / 16)]
page = block_table[sequence][position / 16]
offset = position % 16
```

store、QK、PV 直接消费设备表，不先 gather 全部历史 KV。
P16 常量寻址不改变 reduction、softmax 或 PV 的 FMA 顺序。
目标模型每 token 的完整 KV 载荷为 112 KiB；
S4/L2048 连续槽位与等容量页池都是 896 MiB，后者另有 2 KiB table。

## 完成点与故障

同步 `execute`、单 stream。Prepare 验证需求并形成 pending mapping；
设备执行后检查 token/status 与 stream 完成，再提交映射和长度。
设备执行前失败不破坏已提交状态；执行后失败进入 poisoned，
不能声称物理 rollback，也不能 clear 后继续复用相关页。

正常 clear 归还页 ID，resident slab 保留。Poisoned 资源的 live 状态为未知，
设备分配仍报告为 resident，直到 owning storage 销毁。
Engine 验证整批 samples 后才发布，避免半批输出提交。
HTTP 停服另有停止接纳与有界排空，模型结束不等于响应已送达。

## 范围

Own CUDA 限定单 GPU、Qwen3-0.6B、S≤4、L≤2048、B≤128、greedy。
没有异步执行、多 stream、Graph、抢占重算、GPU prefix、PD 或多 GPU。
分页最终为容量／研究模式，不承诺速度提升，见
[性能](PERFORMANCE.md)、[GPU KV 研究](GPU_KV_STUDY.md)与[验证](VALIDATION.md)。

## 支持分层

- 默认可用：CPU 快速路径，F32 矩阵／contiguous KV 的 Own CUDA 展示路径。
- 实验入口：GPU paged 容量研究；显式 opt-in，接受已披露的性能代价。
- 已停止研究：FP16 matrix 候选，Serving 拒绝启用。
- 内部验证：debug logits、fault hooks、profiler timing，不属于对外服务承诺。
  GPU `stages` 不提供虚构的逐阶段 GPU 计时。

`BlockPool` 是 Engine 的容量信用，不是 GPU allocator；`clear` 归还池内资源，
不等于 `cudaFree`。CUDA 的 preflight/commit/poison 合同不自动意味着
CPU forward 可回滚；整批 sample 校验也不是整个网络响应的事务。

## 技术追问

| 问题 | 解释重点与代码入口 |
| --- | --- |
| 同一个逻辑位置如何找到物理 KV？ | P16 的 block/offset、layer/K/V 次序、table bounds 与非法页屏蔽；[地址策略](../src/minillm/cuda/kv_access.cuh) |
| 为什么保守信用能避免正常 decode 中途缺页？ | 每请求完整承诺向上取整到页，实际长度不超过承诺，总承诺不超过池；[Engine](../src/engine.cpp) |
| 为什么长度和页映射要分开管理？ | `BatchState` 是唯一长度账本，pending mapping 不提前破坏 active；[页事务](../src/minillm/cuda/page_table.cpp)、[Runtime](../src/minillm/cuda/runtime.cpp) |
| 为什么 poisoned 后不能 clear 再复用？ | 设备执行后无物理 rollback 保证；逻辑信用可回收不代表 slab 内容可信；[CUDA adapter](../src/mini_cuda_runner.cpp) |
| 为什么 FP16 显存下降仍不能上线？ | 固定 logits 门槛和 token 一致性是不同合同，不能事后降低阈值；[精度负结果](PRECISION_STUDY.md) |
| Engine 停止为什么不等于 HTTP 完成？ | 结果发布与传输排空是两个完成点，等待有界，断连不保证收到终态；[HTTP 生命周期](../src/http_server.cpp) |
| 为什么指令少了仍不能称为成功优化？ | 静态 PTX、kernel 时间、host forward 和请求时间不同；未锁频基线漂移、同容量与同预算必须分开；[实验取舍](GPU_KV_STUDY.md) |
