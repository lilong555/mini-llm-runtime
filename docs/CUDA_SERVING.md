# 自有 CUDA Serving

`llmserve --backend mini-cuda` 使用现有 HTTP/SSE、Engine 和 Scheduler，模型调用链为
`MiniCudaRunner → CudaRuntime::forward(greedy, false)`。不运行 CLI 子进程，
不调用 `llama_decode`，不构造 CPU Runtime。CLI 名称为 `mini-cuda`，
`/metrics` 的 backend 为 `minillm-cuda`。

## 启动

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda serve --port 8000
```

```powershell
.\scripts\Start-LLMServe.ps1 -Backend mini-cuda -Port 8000
```

PowerShell 示例用于 Windows 或持续的交互会话；WSL 临时执行会话使用前台 `serve`
或同一会话内完成启动/检查/停服的 `check-http`，不将短命 shell 当作服务守护进程。

WSL 产品目录为 `build/wsl-own-cuda/bin`。`MINILLM_ENABLE_CUDA=ON` 与上游
`LLMSERVE_CUDA` 独立，标准 own-cuda 构建使用 `GGML_CUDA=OFF`。
仅监听 loopback，没有认证，不应直接暴露到公网。

| 默认配置 | 值 |
| --- | ---: |
| `max_active` | 4 |
| `max_model_len` | 2048 |
| `context_tokens` | 8192 |
| `batch_tokens` | 128 |
| `prefill_chunk` | 32 |
| `block_size` / CLI `--page-size` | 16，仅信用粒度 |
| `queue_capacity` | 64 |
| `event_buffer` | 128 |
| prefix entries / tokens | 0 / 0 |

S≤4、B≤128、Lmax≤2048，且 credits≤S×Lmax；保留通用的对齐、batch 和请求长度限制。
只对未提供的参数应用默认值。显式超限配置、非零 prefix、scalar kernel 或非零
gpu_layers 均在设备分配与监听前拒绝。S=1 时需同时设置适配的全池信用：

```bash
bash scripts/dev.sh own-cuda serve --max-active 1 --context 2048 --port 8001
```

`--device 0` 选择 GPU，`--device-budget-bytes 0` 使用 Runtime 的可用显存预算；
正值是额外上限，不是预分配请求。PowerShell 对应 `Device`、`DeviceBudgetBytes`。
这两个参数只供 own-CUDA 使用；上游 GPU offload 仍属于 `--backend llama --gpu-layers ...`。

## 执行与资源

source Q8_0 在初始化时解量化为 FP32 device weights，激活为 FP32、KV 为连续 FP16。
单 GPU、单 stream、同步 execute；没有 GPU prefix copy/share、分页、PagedAttention、
native Q8 GEMM、async、CUDA Graph 或 preemption。

`/metrics.capabilities` 报告实例容量、`prefix_copy=false`、
`runtime_stage_profile=false`、`synchronous_execute=true`。
`/metrics.resources` 由模型线程在 publish 边界采集，HTTP 只读受 mutex 保护的副本；
`snapshot_boundary=model_thread_publish`、`batch_id` 标识采集边界，不是实时设备遥测。

- `layout=contiguous`，`live_kv_pages=null`。
- `capacity_tokens=S×Lmax` 是已静态预留的物理槽容量。
- `live_tokens` 为 ready 状态下 committed sequence lengths 之和。
- `resident_kv_payload_bytes` 为实际 KV allocation；clear 后不会下降。
- `owned_device_bytes` 是项目 arena/workspace，不是整个进程或整张 GPU 的显存占用。
- `kv_credits` 保留独立的保守信用口径，按 prompt+max_tokens 预留，不代表物理页分配。

正常停服后 live tokens 为 0，resident 可保持到 Engine/runner 析构。poisoned 后
`state_valid=false`、`reusable=false`、`live_tokens=null`，resident 仍报告隔离中的 allocation；
逻辑信用归还不表示设备写入回滚或显存已释放。

`/metrics.initialization` 提供 Runtime 已有的 model load、storage initialization 和
weight decode/upload 纳秒计时。weight decode/upload 包含在 storage initialization 内，
两者不能相加；初始化不进入客户端 steady-state 计时。未知后端的该对象为 null。

## 生命周期

execute 返回表示本 batch 已完成。取消、超时和断连在 forward 边界生效，不中断在途 kernel。
post-launch failure 后不发布本批 sample、不再 clear/reuse 同一 Runtime；Engine 给请求
一个 backend_error 终态并拒绝新任务，设备存储隔离到原 owner 析构。
SSE headers 已发送时 HTTP 状态可仍为 200，错误由 error event 与 `[DONE]` 表达。

Tokenizer 由已有 `tokenizer_mutex` 保护，可与 forward 并行；该锁不覆盖 GPU forward。
Runtime 的单调用者要求针对 execution/KV 与状态读取。

## 验证与观测

```bash
bash scripts/dev.sh own-cuda serving-check .run/cuda-serving-model.json
bash scripts/dev.sh own-cuda check-http 8015
bash scripts/dev.sh own-cuda serving-memcheck .run/cuda-serving-memcheck.json
```

报告路径须不存在。`cuda-serving` CTest 使用真实 GPU 小模型；`serving-check` 使用固定
Qwen3 和短金标准，覆盖 S=1/4、GatedRunner 动态 mixed、输出映射、槽复用和并发 tokenizer。
HTTP 复用同一套客户端检查；完整生命周期及性能验收状态见 [执行状态](EXECUTION_STATUS.md)。

观测默认关闭。`--telemetry batches|stages --telemetry-output NEW.jsonl` 使用 schema v2，
包含 capability 和 nullable resources；历史 v1 继续可读。GPU 在 stages 模式下仍只有
Engine 计时，runner 为 null，不伪报 GPU 阶段。详见 [在线观测](BATCH_TELEMETRY.md)。

Serving 测量使用 [既有策略回放](BENCHMARKS.md)，遵循 `CUDA-SERVE-001` 的两条 trace、
最多三个 pilot、12 个正式进程和一次 NSys 预算。模型 token/s、CPU SIMD 微基准和
上游 GPU 执行不能代替自有 GPU HTTP 性能。

[首份 Serving 基线](../benchmarks/results/cuda-serving-001/README.md) 已有 12 个独立进程、
288 个成功请求和一次完整时间线。mixed-length 的 mixed 吞吐中位数为
130.68 token/s，prefill_first 为 122.03；burst-reuse 的约 0.28% 差异保持测量不确定。
固定 SLO 的未达标请求和长 token 停顿均保留，结果不外推为生产级承诺。
