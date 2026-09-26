# NEXT_SPEC_V2：最小 Own CUDA Serving

## 标识与范围

- Spec ID：`CUDA-SERVE-001`，对应 [PROJECT_PLAN_V3](PROJECT_PLAN_V3.md) / M3-1。
- 起点：`608daf148a17007f96387040fd66ce130fe8d640`，现成 Windows 修复已包含。
- 终点：真实请求经过现有 HTTP → Engine → MiniCudaRunner → CudaRuntime → SSE，
  具有正确生命周期与首份限定规模 Serving baseline。
- 不修改 M1 模型数学、权重、workspace、连续 KV 或单 stream 执行。
- 不新增 GPU paging、prefix sharing、quantization、custom GEMM、async、graphs、
  multi-stream、preemption、backend registry 或另一套 Serving/benchmark 框架。

## Adapter 与能力

`src/mini_cuda_runner.cpp` 的 `MiniCudaRunner final : ModelRunner` 独占一个 CudaRuntime。
factory 为 `make_mini_cuda_runner(const ModelConfig&, const EngineConfig&)`。
输入保留 token/position/sequence/logits，复用预留映射缓冲；
调用 `forward(tokens, CudaOutputMode::greedy, false)`，不下载 debug logits 或重算 argmax。

输出必须按 `CudaSample::input_index` 核对 logits 输入、sequence、token 范围、
数量、重复与顺序。Engine 每 sequence 每轮最多一个输出。任意契约错误走 backend fail-stop。
`copy_sequence` 明确不支持；正常 GPU 配置不会调用。Tokenizer 复用 host owner，
由既有 `tokenizer_mutex` 保护，不给 forward 添加全局锁；execution/KV 是单调用者状态。

新增小型 `BackendCapabilities` 与 `capabilities() noexcept`：

| 字段 | 含义 |
| --- | --- |
| `max_sequences`、`max_batch_tokens`、`max_model_len` | 可靠容量上限，0 表示未提供，不表示零容量 |
| `prefix_copy` | 默认保留旧 copy 契约；GPU 为 false |
| `runtime_stage_profile` | CPU stages 为 true；GPU 不提供阶段测量 |
| `synchronous_execute` | 初版必须为 true |

MiniCudaRunner 受 S≤4、B≤128、Lmax≤2048 约束，报告实际实例容量；
CPU MiniRunner 按实际配置报告，LlamaRunner 仅报告可靠已知容量。
`tests/gated_runner.h` 必须转发能力与资源；fake runner 明确能力。
`ModelConfig` 末尾添加默认 `device=0`、`device_budget_bytes=0`，保留旧 aggregate 初始化。

## 资源与快照

扩展既有 `RunnerResources`，不并存第二套账本：

| 字段 | 语义 |
| --- | --- |
| `live_kv_pages` | optional；GPU 与上游未知页数为 null |
| `resident_kv_payload_bytes` | 实际 KV allocation 载荷；clear 后可保持非零 |
| `layout` | unknown、paged、contiguous |
| `capacity_tokens` | optional；GPU 为 S×Lmax 的物理槽容量 |
| `live_tokens` | optional；ready 时为 committed sequence lengths 之和 |
| `owned_device_bytes` | optional；项目 device arena/workspace，不是整个 GPU 进程占用 |
| `state_valid`、`reusable` | poisoned 后均为 false |

CPU 继续报告真实 PagedKV pages/payload，未知 live tokens 不得用 pages×page_size 冒充。
poisoned 的 live tokens 为 null，resident 保持尚未销毁的 allocation，不宣称回滚或零占用。
Runtime 只加必要无分配 scalar getter，固定 memory plan 可在 adapter 初始化时缓存。
`resources() noexcept` 不得调用复制 vector 的 `diagnostics()` 或 `cudaMemGetInfo`。

资源仅由模型线程在 `Engine::Impl::publish()` 采集，复制到 mutex 保护的快照。
HTTP `/metrics` 只读快照；标明采集边界，最后一次完成状态不等于实时设备状态。
`Engine::stop()` 正常结束后 live tokens 为 0，但 resident 可保留至 runner 析构。

## 配置

CLI backend 为 `mini-cuda`，metrics backend 为 `minillm-cuda`。

```text
max_active=4              max_model_len=2048
context_tokens=8192       batch_tokens=128
prefill_chunk=32          block_size=16
queue_capacity=64         event_buffer=128
prefix_cache_entries=0    prefix_cache_tokens=0
```

只给未显式提供的参数应用 GPU 默认值。显式 8 slots、256 batch、非零 prefix、
scalar kernel 或非零 gpu_layers 必须在 CUDA allocation/监听前拒绝。
未编译 `MINILLM_ENABLE_CUDA` 时清楚失败，不静默 fallback。

Runtime 的 max_sequences/max_model_len/batch_tokens 分别来自 Engine 的
max_active/max_model_len/batch_tokens，不能把全池 context 当作每序列长度。
验证 `context_tokens <= S×Lmax` 并检查乘法溢出，保留 block alignment、
请求长度与 budget 门禁。保守 credit 预留 prompt+max_tokens，不优化 admission。
保留 CudaStorage 预算检查；不承诺外部进程改变可用显存时绝不 OOM。

## 生命周期与故障

正常 execute 返回表示 GPU batch 已完成。取消和超时在 forward 边界生效，不是 GPU preemption。
ModelRunner 的 clear/synchronize 为 noexcept，不能无条件转调 poisoned 时会抛的 CUDA clear。

1. ready 的合法槽正常 clear，保持长度与复用语义。
2. post-launch failure 后不再 clear，不恢复 poisoned，不再执行同一实例。
3. Engine 发出单一 backend_error，归还逻辑 credit/slot，停止接收请求。
4. device allocation 隔离到原 owner 析构；快照 invalid/non-reusable，live=null。
5. 复用 Runtime 既有异常收尾，不新增逐层同步；致命错误只作 best-effort teardown。
6. 意外清理失败必须标记不可复用并触发 Engine fail-stop，不能吞错后宣称健康。
7. Runtime preflight 可恢复不等于 Serving 请求可恢复。

故障测试复用受控 post-launch 错误或最小 test wrapper，不添加生产任意 fault injector。
SSE headers 已发出时 HTTP 可仍为 200，错误通过单一 error event 和既有 `[DONE]` 表达。

## 观测与兼容

复用 Engine admission/scheduler/prepare/runner elapsed、batch composition、
prefill/decode/logits tokens、SSE request_order/token_index/batch_id。
GPU `RunnerTelemetry.available=false` 是正确结果；stages 模式保留 Engine 计时，
runner 为 null，不填伪 CPU stages 或零值。

JSONL header/资源使用明确的新 schema，保留 v1 读取，不改历史 raw。
`/metrics` 增加 capability/resource，保留 `kv_credits`。
GPU utilization 使用标明周期与整设备含义的低频设备采样，不从 event gap 伪推导。

## 验证

一个新入口 `tests/cuda_serving_tests.cpp`，其余扩展既有 suite。

- 无 GPU：容量/prefix/kernel/offload 预检、decorator、nullable 资源、清理状态机、
  单终态、拒绝新任务、CPU/upstream 序列化及旧 v1。
- 真 GPU：`GGML_CUDA=OFF`；S=1/4 对照同一 CudaRuntime 的独立请求 reference；
  GatedRunner 确定性 dynamic/mixed；样本映射；不同 prompt 复用 slot；容量边界。
- 稳定短 golden 要求 token 一致；更广语料只能使用已冻结 numerical/near-tie 契约，
  不以 tolerant logits 掩盖错序。
- 受控 post-launch fault：无新 sample、全部有限终态、credit 归还、拒绝新请求、
  poisoned 资源不复用，析构归还 owner allocation。
- HTTP：复用 streaming/non-streaming、token/usage、UTF-8、并发、显式 cancel、
  timeout、stream/nonstream disconnect；补慢消费者、shutdown、资源与能力。
- 慢 socket 只验证有界资源、最终清理和协议允许结果，不要求所有路径同一错误码。
- CPU 核心/实模型/HTTP 回归，代表性 GPU mixed/clear/failure memcheck；
  最终候选自己的跨平台 CI。不得重复完整 M1 sanitizer 笛卡尔积。

## 基线与完成门禁

固定配置、两条 trace、最多三次 pilot、12 个正式进程、一次 NSys，遵循 V3 的测量预算。
记录 source/candidate SHA、binary/model/trace hash、完整配置、设备/驱动/math mode、
warmup/trial/失败。初始化及 weight upload 单列，不计 steady-state。
SLO 采集前冻结，失败保留在 goodput 分母；尾分位数只作描述，不承诺生产 P99。

timeline 验证自有 kernel/cuBLAS、真实 dynamic batch、无逐层 weight/hidden 往返、
greedy 小 token/status D2H、无逐请求 Runtime 重建、单 stream 和同步完成边界。
只按真实 timeline 解释 host 等待与 GPU gap，不累加 API/device 重叠时间。

- [ ] 原 server 的 own-CUDA HTTP/SSE 在上游 GPU 关闭时通过。
- [ ] 启动前能力与参数校验、sample 映射、S=1/4、dynamic/mixed/reuse 通过。
- [ ] 正常 clear 的 live/resident 分离，poisoned invalid/non-reusable 与单终态通过。
- [ ] cancel/timeout/disconnect/backpressure/shutdown 有对应证据。
- [ ] 模型线程快照、noexcept 无分配、旧 CPU/upstream 与代表性 memcheck 通过。
- [ ] 最终候选自身 CI 通过，完整 Serving raw+summary+validation+命令可获取。
- [ ] 12 进程内的正式基线和一次 timeline 完整，所有不利结果保留。

达到门禁即停止，不因性能不确定继续 trial，不在本 SPEC 顺手实现 M3-2 优化。
