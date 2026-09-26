# PROJECT_PLAN_V3：自有 CUDA Serving

## 基线与主线

- 规范：`CUDA-SERVE-001`，见 [NEXT_SPEC_V2](NEXT_SPEC_V2.md)。
- 审计日期：2026-09-26。
- `main`：`68ac275913207975a88e2090c6617467e351301c`。
- CUDA 模型基线：`5a4508d6ace7777e39460675de9ed770fcb66d43`。
- 开发基点：`608daf148a17007f96387040fd66ce130fe8d640`，包含
  `a0a62146ec0cb26ef8c548ad7104884892adbed6` 的 Windows 修复。
- 基点自己的 CI：run `36233429810`，五个任务全部通过。
- 实施分支：`feat/own-cuda-serving`。
- 本轮交付：复用 HTTP、Engine、Scheduler、RequestHandle 的 MiniCudaRunner；
  本机生命周期与限定测量证据见 [M3-1 基线](../benchmarks/results/cuda-serving-001/README.md)。
  12 个正式进程与一次 NSys 已完成；M3-2/M3-3 尚未启动。
- M1 数值、375-case 微基准、70-process 模型协议与 Profiler 冻结。
  24 项模型比较中的 14 项 `faster`、10 项 `measurement_inconclusive` 保持原判定。
  模型基线不是 HTTP 性能证据。

## 架构边界

保留 Q8_0 source、初始化解量化的 FP32 device weights、FP32 activation/accumulation、
连续 FP16 KV、单 GPU、单 stream、同步 execute。项目拥有 execution、KV、workspace
与生命周期；矩阵使用 cuBLAS，归约使用 CUB。上游仅提供既有模型解析与 tokenizer 能力，
不得用 `llama_decode` 或 CLI 子进程实现自有 GPU Serving。

首版关闭 prefix cache。BlockPool 是保守容量信用，不是 GPU allocator。
CPU 新优化、native Q8 GEMM、GPU paging、PagedAttention、async、CUDA Graph、
multi-stream、增量 admission 与 preemption 均不属于 M3-1。

## 阶段

| 阶段 | 目标与进入条件 | 门禁与停止线 |
| --- | --- | --- |
| M3-0 | 纳入现成兼容修复，冻结 M1，明确证据政策 | 基点 CI 通过、政策明确即停止；Release 暂不可用不阻塞 adapter |
| M3-1 | 现有 HTTP → Engine → MiniCudaRunner → CudaRuntime → SSE | 生命周期、资源语义、CPU 回归、最终候选 CI、限定 Serving 基线及一次 timeline 通过 |
| M3-2 | 仅从 M3-1 证据选择一个主要瓶颈 | 预注册指标与护栏，最多两个 candidate；无收益或不稳定则记录并停止 |
| M3-3 | 有连续 GPU Serving、资源与负载证据后才考虑分页 | kernel 直接消费 device block table；保留连续 reference；容量与性能分别验收 |

M3-2 可选 F16 权重/矩阵、workspace、host launch、attention 访问或简单 batch cost 策略，
不能同时改变 dtype、KV layout 与 scheduler。F16 必须核对 cuBLAS 的 A/B/compute dtype
组合、activation 转换和新增舍入误差；Q8 block scale 不能视作任意 int8 GEMM 输入。
没有明确限制或两种设计均无收益时结束研究，不增加试验轮数来追求正结果。

M3-3 需要容量浪费、共享需求或明确的 memory-systems 学习目标。先单 stream，
最多 pool/table/lease 三类核心职责；不同时引入 GPU prefix tree、抢占或分布式协议。
CPU gather 全量 KV 不能冒充 GPU paged attention。

## M3-1 交付组

1. Adapter、capability/resource、预检、poisoned/noexcept 清理桥接与最小测试。
2. CMake/CLI/脚本、模型线程资源快照、metrics/telemetry、真实 mixed 与 slot reuse。
3. 既有 HTTP suite、取消/超时/断连/背压/shutdown、post-launch fault、代表性 memcheck。
4. 两条冻结 trace、12 个正式服务进程、一次 NSys、一个 canonical bundle 与小摘要。

主要实现提交不超过四组。只新增一种 adapter、一个 GPU Serving 测试入口；
不另建 server、registry、通用 benchmark、统计、封包或验证器框架。
新测试应覆盖独有故障，如错序样本、poisoned 复用、重复终态、非法内存、缺样本或错单位。

## 测量预算

- 固定 S=4、Lmax=2048、credits=8192、B=128、chunk=32、credit granularity=16、
  queue=64、event buffer=128；source Q8_0 / device F32 / KV F16。
- 两条主 trace 各 24 请求、32 输出 token、`ignore_eos=true`：
  mixed-length 使用 16/128/512 prompt；burst/reuse 使用短长交错并发生槽复用。
- 从既有 generator 生成一次并冻结字节。最多三个 pilot，只选择到达缩放，
  独立标记且不计正式结果；采集前冻结 SLO。
- 每条 trace 两策略 `mixed`/`prefill_first`，各三次独立 trial，共 12 个正式进程；
  按轮次反转策略顺序。不同 trace 不合成统一加速比。
- 正式基线使用 telemetry `off`；`batches` 用于关联。GPU stage profile 不可用时为 null。
- 主指标为成功输出 token/s、固定 SLO goodput；保留失败分母及所有不利结果。
  TTFT P50/P95、请求 mean TPOT、单 token ITL、最大停顿、拒绝/超时比例作为护栏。
- 过载协议检查与 all-success 时间线分析分开；不放松分析器来接受不完整采集。
- 只采一次代表性 Serving NSys；不追加 NCU、不重跑 M1 矩阵。
  正式性能不在 Profiler 下测，API 与 device 重叠时间不相加。
- 三轮结果不稳定即标不确定；没有必须超过 CPU/llama.cpp 的门槛。

## 证据与验收

遵循 [产物政策](ARTIFACT_POLICY.md)。Git 保留源码、固定输入、复现命令、小索引、
摘要及必要代表样本；完整 raw/log/source snapshot/Profiler 使用一个外部 canonical bundle。
先验证可获取性、摘要与已有验证器，再迁移重复副本；不得删除唯一负结果或改写共享历史。

M3-1 完成必须包含真实 GPU HTTP、dynamic/mixed、同步完成边界、slot reuse、
取消/超时/两类断连/背压/shutdown、poisoned 后拒绝新任务和不复用、
contiguous KV 的 capacity/live/resident，以及最终候选自身的跨平台 CI。
完成不表示 native Q8 CUDA GEMM、PagedAttention、async 或生产级承诺。
