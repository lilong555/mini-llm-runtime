# FP16 矩阵边界研究

## 当前状态

研究合同为 [CUDA-PREC-001](NEXT_OPT_SPEC.md)，历史计划为
[精度版 PROJECT_PLAN_V4](history/plans/PROJECT_PLAN_V4.md)；本文 M4 编号仅属于该历史计划。
GPU KV 研究也已结束，项目当前处于 [Portfolio Freeze](finalization/FINALIZATION_SPEC.md)。
精度研究的决策、已有时间线拆解和实验预注册已完成。第一组配置入口已通过本机验收；
F16 存储、矩阵边界与完整模型研究路径可执行，项目 owned 显存实测减少 34.53%。
完整模型数值门禁因长续写 cosine 失败，结论为 `blocked_correctness`，停止性能推进。
Serving 不开放候选，模型/Serving 正式实验未执行；不作端到端速度或产品晋升声明。
[单一原始包及复核入口](../benchmarks/results/cuda-precision-001/README.md) 已公开发布，
公开下载、摘要、独立目录微基准复核及失败 logits 重算均通过。

| 模式 | 当前执行能力 | 默认 |
| --- | --- | --- |
| `f32-pedantic` | 原自有 CUDA 模型与 Serving；矩阵输入、权重、累加、输出均为 F32 | 是 |
| `f16-matrix-f32acc` | storage/cast/GEMM、Runtime、模型 CLI 可供研究；数值门禁失败，Serving 拒绝 | 否 |

模式枚举不依赖 CUDA 头文件。Serving、CUDA CLI 和两个 CUDA benchmark 均识别
`--cuda-precision`；CPU 与上游后端显式使用该参数时失败。
`ModelInfo` 与 `/metrics` 的 `precision_mode` 单独标识精度，非自有 CUDA 后端为 null。
旧 F32 算术合同、有效权重摘要、模型输入和验证门槛保持不变。
新实验 JSON 已接入矩阵微基准的显式子协议；旧输入仍采用原 375-case F32 合同。

## 冻结合同

[qwen3-precision-v1.json](../benchmarks/runtime-inputs/qwen3-precision-v1.json)
是唯一的机器可读输入合同，SHA-256：

```text
3b9ec80ee5e9cc83865378f21c46d5dedf4975530e7686a1dfb61d5f4af992b8
```

用户提供的规范原件与仓库文档字节一致：

| 文件 | SHA-256 |
| --- | --- |
| `NEXT_OPT_SPEC.md` | `f53cdba33808cc7de48f9b51e7a3fbaa785eed838f54b8b629907364a204647e` |
| `PROJECT_PLAN_V4.md` | `05c956098cc569d8f8d6a0d5fc1cd26044a1d1e7fe10308fe5a95945bb120761` |

- 审计提交：`6ca7d2dfeccb38b11b7596a552046651afcb693e`，进入时无后续代码 delta。
  `origin/main` 与 merge base 均为 `68ac275913207975a88e2090c6617467e351301c`；
  本地旧 `main` 未移动。实施分支为 `perf/cuda-f16-matrix-path`。
- 当前实施基点包含独立 HTTP 停服修复 `57268f9`：响应排空与停服测试的接纳确认，
  见 `ENG-064`。它不改变 CUDA 模型、KV、调度算法或精度；后续 A/B 两臂均包含该修复。
- 唯一候选：常驻 F16 大矩阵及 embedding、F16 矩阵输入、F32 累加与输出；
  norm 权重、非矩阵 activation、attention、KV、scheduler 和请求生命周期不变。
- 同层 A/B 必须使用同一新二进制、同一模型和相同输入；每进程只驻留一种模式。
  旧日期的 F32 数字不作为新配对基线。
- 主指标：`prefill-128 / host_forward_to_token_ns`，三个配对 trial 的
  `median(1-T_f16/T_f32) >= 10%`，每个 trial 均至少改善 5%。
- 显存门槛：S4/L2048/B128 下项目 owned bytes 减少至少 30%。
  冻结预测为约 34.53% 的减少、2,258,046,976 bytes 的总量；实测见微基准结果。
- 其余五个模型 workload 的配对中位数退化不超过 5%，不能有至少两轮退化超过 5%；
  mixed-length 的吞吐和 goodput 配对中位数退化不超过 5%。
- 两条 Serving trace 的 `ttft_ms.p95`、`mean_tpot_ms.p95`、
  `request_max_itl_ms.p50` 的配对中位数退化不超过 5%。最后一项先对每请求最大 ITL
  求 P50，再比较 trial；每请求原值与整轮最大 ITL 另行完整报告，不用均值隐去极值。
- 不新增失败、拒绝或超时；两条固定 trace 的跨精度输出必须完全一致才可晋升产品。
  所有分叉保留首个位置、teacher-forced logits 和 margin。
- 5% 是工程决策带，不是已经测得的 A/A 噪声。结果不确定时不追加 trial。
- 数值门槛：RMSE `<0.05`、最大绝对误差 `<0.5`、cosine `>=0.9999`、全部 finite，
  稳定短 golden 完全一致。矩阵实现使用相同舍入后 operands 的 FP64 oracle，
  与模型相对原 F32 的偏差分开验收。

### 工作负载与预算

| 层次 | 固定范围 | 正式进程上限 |
| --- | --- | ---: |
| 微基准 | Q、gate、down、LM head，各 M=1/4/32/128；分别计 GEMM-only 与 cast-inclusive | 6 |
| 模型 | prefill-128、prefill-512-chunk32、decode-prefix16、decode-prefix1536、decode-batch4-prefix256、mixed16+2 | 6 |
| Serving | 两条已有 trace，仅 mixed；两模式各 3 trial；固定到达间隔、SLO、chunk、容量 | 12 |

模型验证为四类语料、长度 16/128/1536、chunk 16/128、S1/S4，共 48 个配置。
采样位置固定为合同中的位置集合与 chunk 边界、最后位置的去重并集；
另外保留 2048 边界、clear/reuse 与四个固定的 32-token 续写。
微基准保留原真实 shape 的输入 seed；20 次 inner iteration 不算独立 trial。

总预算为 24 个正式性能进程，新增 NSys 至多一次、必要 NCU 至多一次会话。
当前已使用 **6/24、0/1、0/1**，六个微基准进程全部完成，不追加 trial。
已有时间线的离线读取、构建与正确性测试不计作性能 trial。
只有一个主设计和一次有证据的小修订，所有最终 raw 使用一个 canonical bundle。

## 已有时间线

数据来自 [M3-1 canonical bundle](../benchmarks/results/cuda-serving-001/evidence.json)，
采集身份为 `b1ced89e98eef3bb2b6f21ac8f0f72c38067bfc2` 加冻结 dirty snapshot，
不是当前分支或 `6ca7d2d` 的新测量。既有 `analyze_serving()` 重算结果与保存摘要完全相等。
没有新增 GPU capture，初始化前的库事件不计入 forward，batch 1..8 为预热，9..288 为测量。

| 原始文件，位于包内 `profiler/` | SHA-256 |
| --- | --- |
| `nsys.sqlite` | `da7615f4c2cd0fae3ef422ac46a780f7d183903e70e6eb2fbd2eb475473c893b` |
| `batches.jsonl` | `53f6df7d98f3e34c353aadba1480c8aa604a7f5385dae2313df868e122316c83` |
| `manifest.json` | `9ab70699cf8f149e33b9452ffbba314a5276dbce84dcb48b107f67131bae4d93` |
| `nsys-summary.json` | `16352f777a9f8bdd086bfc2f2a23c2dba0a73994e338378fba14a98078563388` |

### Kernel 分组

矩阵组包含 cuBLAS GEMM/GEMV 及其 `cublasLt::splitKreduce_kernel`，不是仅按
`gemm` 名称匹配的子集；所有 vendor kernel 名称均已归类。时间为插桩记录的 kernel
经过时间总和，不是硬件忙碌率或正式无 profiler latency。

| 分组 | batch 数 | kernel 合计 ms | 矩阵 ms | QK ms | softmax ms | PV ms | 其他 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 预热 | 8 | 91.918 | 83.154 | 0.543 | 0.652 | 0.522 | 7.047 |
| 测量 | 280 | 5210.417 | 3366.973 | 1088.703 | 99.153 | 402.052 | 253.535 |

| 测量 batch 类型 | batch 数 | 矩阵占 kernel 时间 | QK+softmax+PV 占 kernel 时间 |
| --- | ---: | ---: | ---: |
| prefill-only | 5 | 80.29% | 13.25% |
| decode-only | 118 | 82.16% | 11.96% |
| mixed | 157 | 56.84% | 38.75% |
| 全部测量 batch | 280 | 64.62% | 30.51% |

整个时间线的含预热 GEMM/GEMV 名称子集占比 63.44% 与这里的 64.62% 定义不同：
后者排除预热并包含矩阵 split-K 归约。两者都不能解释任意单独 decode 的因果瓶颈。

### M、R 与追加边界

M 为本 batch 输入行数，R 为选中的 logits 行数。下面两列 context 分别是
每个 batch 的最大 `context_before`、最大 `context_after` 在该组内的最小值至最大值；
不是所有 sequence 都具有该长度，也不是该 batch 的 attention 总工作量。
原 JSONL 保留逐 slice 的长度和请求/slot 关联，完整 288 次调用见既有摘要 `calls`。

| 类型 | M | R | batch 数 | 最大追加前 context 范围 | 最大追加后 context 范围 |
| --- | ---: | ---: | ---: | --- | --- |
| decode | 1 | 1 | 57 | 16..542 | 17..543 |
| decode | 2 | 2 | 5 | 521..525 | 522..526 |
| decode | 3 | 3 | 5 | 517..521 | 518..522 |
| decode | 4 | 4 | 51 | 512..542 | 513..543 |
| mixed | 19 | 4 | 5 | 513..541 | 514..542 |
| mixed | 33 | 1 | 3 | 137..139 | 138..140 |
| mixed | 35 | 3 | 129 | 256..540 | 288..541 |
| mixed | 35 | 4 | 14 | 480..541 | 512..542 |
| mixed | 49 | 2 | 1 | 140 | 141 |
| mixed | 50 | 3 | 1 | 531 | 532 |
| mixed | 66 | 2 | 3 | 141..192 | 160..224 |
| mixed | 66 | 3 | 1 | 224 | 256 |
| prefill | 16 | 1 | 1 | 0 | 16 |
| prefill | 32 | 0 | 3 | 0..64 | 32..96 |
| prefill | 32 | 1 | 1 | 96 | 128 |

相同 M/R 不保证相同 context 或 kernel signature，不能据上表宣称 CUDA Graph 可复用率。
Q/K/V 的同一输入有三个矩阵消费者，gate/up 有两个；候选每层四个转换边界，
28 层加有 logits 的 LM head 最多增加 113 个 cast launch，不能宣称 launch 数减少。

### 关联拷贝

通过 `correlationId` 将每次 device copy 与唯一 `cudaMemcpyAsync` Runtime API 配对，
按 reset kernel 定义的 forward 窗口归属。每个 batch 的传输字节与既有摘要完全相等。

| 范围 | 方向 | 调用数 | 字节 | host API 合计 ms | device copy 合计 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 测量 | H2D | 1117 | 74976 | 7.432058 | 0.453672 |
| 测量 | D2H | 557 | 5312 | 4139.231382 | 0.614859 |
| 预热 | H2D | 32 | 128 | 0.198897 | 0.014272 |
| 预热 | D2H | 16 | 96 | 58.851811 | 0.018016 |

4.2 秒 API elapsed 主要属于 D2H 调用；device copy 本身很短。
该关联仍不能区分等待既有 GPU 工作、driver staging 和插桩开销的各自份额，
也不证明 pinning 能消除这段时间。API、kernel、copy 和 host runner 时间不相加。

### 决策与限制

选择 F16 的确定依据是 F32 权重成本；时间分解支持进行有限 cast-inclusive 探针，
不保证收益。按 Amdahl 模型，若矩阵部分比例为 f、矩阵加速比为 s、新增转换相对成本为 c，
同一测量边界下的理想剩余比例为 `1-f+f/s+c`。取测量 kernel 总和的 f=0.6462，
假设 s=2、c=0，仅得到 kernel 总和减少约 32.31% 的思想实验，不能当作模型或 Serving 预测。

该 Serving capture 没有 M=128 的 batch，因此不能据它验证预注册主指标 `prefill-128`。
M=1/4/32/128 的 16 个真实 shape 探针仍是后续进入模型测量的必要证据。
mixed 中 attention 份额较高并不是单独 decode 的归因证据，也不构成启动融合路线的依据。
没有 Tensor Core 使用证据，当前记为 `unverified`。

## 离线复现

以下只读命令使用已有分析器与 SQLite，不启动模型、不写回原始包。
`profiler/` 可来自 canonical bundle，也可使用本机保存的目录。

```bash
python3 scripts/analyze_cuda_profiler.py --serving --directory .run/cuda-serving-001/profiler
```

分组查询的窗口由项目 `reset_kernel` 确定；用标准库执行下列 SQL 可得到预热/测量角色耗时。
同一窗口序号与 `nsys-summary.json` 的 `calls`、`batches.jsonl` 的 `batch_id` 对应。

```sql
WITH events AS (
  SELECT k.start, k.end, n.value AS name
  FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds n ON n.id=k.demangledName
), tagged AS (
  SELECT *, SUM(CASE WHEN name LIKE '%reset_kernel%' THEN 1 ELSE 0 END)
    OVER (ORDER BY start,end ROWS UNBOUNDED PRECEDING) AS batch_id
  FROM events
)
SELECT CASE WHEN batch_id<=8 THEN 'warmup' ELSE 'measured' END AS phase,
  CASE WHEN name NOT LIKE '%minillm::cuda::%' THEN 'matrix'
       WHEN name LIKE '%qk_kernel%' THEN 'qk'
       WHEN name LIKE '%softmax_kernel%' THEN 'softmax'
       WHEN name LIKE '%pv_kernel%' THEN 'pv'
       ELSE 'other' END AS role,
  COUNT(*) AS calls, SUM(end-start) AS elapsed_ns
FROM tagged WHERE batch_id>0 GROUP BY phase,role;
```

关联拷贝 SQL 同样排除初始化，只计有唯一 Runtime API 关联的显式拷贝：

```sql
WITH markers AS (
  SELECT k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds n ON n.id=k.demangledName WHERE n.value LIKE '%reset_kernel%'
), windows AS (
  SELECT start, LEAD(start) OVER (ORDER BY start) AS next_start,
    ROW_NUMBER() OVER (ORDER BY start) AS batch_id FROM markers
)
SELECT CASE WHEN w.batch_id<=8 THEN 'warmup' ELSE 'measured' END AS phase,
  e.label AS direction, COUNT(*) AS calls, SUM(m.bytes) AS bytes,
  SUM(a.end-a.start) AS host_api_ns, SUM(m.end-m.start) AS device_copy_ns
FROM windows w JOIN CUPTI_ACTIVITY_KIND_MEMCPY m
  ON m.start>=w.start AND (w.next_start IS NULL OR m.start<w.next_start)
JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
JOIN CUPTI_ACTIVITY_KIND_RUNTIME a ON a.correlationId=m.correlationId
JOIN StringIds n ON n.id=a.nameId
WHERE n.value LIKE 'cudaMemcpyAsync%' GROUP BY phase,direction;
```

## 验收与后续

第一组已通过本机验收，原始验证输出保存于 `.run/cuda-precision-001/contract/`。
该目录不是新的独立 canonical bundle，后续与整个研究 raw 一同归档。
没有将历史 CI 视作本次改动的 CI，也没有继承旧二进制的数值或性能身份。

初次配置回归中暴露的 HTTP 停服失败原件保留，独立修复的 52 套 CTest 和三后端各
12 项 HTTP 结果位于 `.run/http-shutdown-drain/`；它不是精度实验或新的性能样本。
包含该修复的合同最终回归位于 `.run/cuda-precision-001/contract/final/`。

| 当前合同回归 | 结果 |
| --- | --- |
| 自有 CUDA / CPU / 上游 CUDA / 独立核心 / ASan+UBSan 构建 | 五种均完成 |
| CTest | 74/74 套，1470 次用例执行，无遗漏计数 |
| CPU 真实模型 | 13/13，含三组短 golden、KV 与 Serving 状态 |
| CUDA 短模型验证 | S1/S4 共 2/2；128 次 logits 对照、6 组短 golden |
| CPU / 自有 CUDA / 上游 CUDA HTTP | 各 12/12，包含 precision metadata 与停服单终态 |

上述 CUDA 验证不是 48 配置候选数值验证，也不是新性能样本。
验证身份为 `57268f93aecc7c947146d60a3d61a03d84f46a1b` 加
`final/source-state.json`、`final/source-snapshot.zip`；`final/verification.json`
记录源码、二进制、模型、冻结输入与原始报告摘要，不把 dirty 构建重标为 clean build。
独立修复自身的 CI run `36406587106` 已通过；精度合同 `5a934a1` 自身的
CI run `36409279196` 五个任务均通过。这些 CI 不包含 GPU 上板执行。

## 底层矩阵边界

`CudaStorage` 仍独占 weight/workspace/KV arena。候选大矩阵采用 F16、norm 为 F32，
同一 tied embedding/head 共用物理权重。`effective_sha256` 保留解码 F32 含义，
`device_payload_sha256` 单独校验实际设备载荷；旧 `weight()` 拒绝把 F16 地址解释为 F32。
每个完整行 staging 块的 F32 与 F16 总量受 8 MiB 限制；上传异常先同步，再释放 staging。

`matrix_input(M,K)` 返回实际行数、列数及 stride=K，容量来自最大 scratch。
`cast_matrix_input()` 使用 device RN-even 并将非有限值与溢出写入既有 status；
half gather 先屏蔽非法索引，再解码至 F32。cuBLAS 候选使用 F16 operands、F32
累加和输出，并禁止 reduced-precision reduction；context 与 operand 类型不匹配时拒绝。

底层原始验证位于 `.run/cuda-precision-001/matrix-boundary/`，不是独立 canonical bundle。

| 验证 | 结果与范围 |
| --- | --- |
| 最终 CTest | 22/22 套、376 次用例执行 |
| Matrix / ops / storage | 13/13、14/14、15/15 |
| RN-even | 全部有限 half 位模式、相邻有限值中点及两侧 float 邻域、溢出边界 |
| GEMM | 非方阵、padding、guard、非对齐 shape，舍入后 operands 的 FP64 对照 |
| Scratch | 同 stream 的四组输入、Q/K/V 和 gate/up 复用；执行期间无新设备分配或释放 |
| 内存检查 | 上述三套 memcheck 均为 0 错误、0 泄漏 |
| F32 实模型回归 | S1/S4 共 2/2；128 次 GPU logits 摘要与合同验收一致，6 组短 golden 通过 |
| F32 自有 CUDA HTTP | 12/12，精度字段、资源与停服单终态通过 |

本组身份为 `5a934a1958965729436fc45d83197c5cc984bd24` 加该目录的 source
snapshot，`verification.json` 固定源码、二进制、设备、输入与所有原始报告摘要。
本机 CUDA Runtime 为 12080，cuBLAS 为 120805，设备为 RTX 4070 Laptop、SM 8.9。

底层用例本身不代表 F16 完整模型或 Serving 验收。真实载荷和 shape 结果见下节；
完整模型门禁见模型数值边界一节，Tensor Core 使用仍为 `unverified`。

## 微基准入口

```bash
pwsh -NoProfile -File scripts/Benchmark-CudaMicro.ps1 \
  -PrecisionStudy -OutputDirectory .run/cuda-precision-001/micro
```

`-PreflightOnly` 只校验来源、构建、输入及六进程计划，不启动性能采样。
完整采样固定 trial 顺序为 F32/F16、F16/F32、F32/F16，奇数 trial 反转 shape 顺序。
每个 shape 有五次重复，每次依次执行 GEMM-only、cast-inclusive；前两次为预热。
两种边界都包含 20 次 GEMM，候选 cast-inclusive 另含 20 次转换，F32 两种边界不转换。
host 区间从 event 提交前持续到 checked completion；CUDA events 包围同 stream 的调用序列，
其中可能包含提交空隙。两者都不解释为单 kernel duration。

每进程初始化后逐字节回读 310 个唯一权重 tensor 并检查实际载荷摘要，不将回读计入稳态。
FP64 oracle 检查所有输出，候选使用相同舍入后的 W/X；计算中位数前不删除失败或慢样本。
每个 shape 的准备阶段使用已有八线程 CPU executor 计算 FP64 dot 与绝对乘积和，
该 executor 在 GPU 计时前析构。原始报告保存准备、重置、验证用时和分阶段传输，
以及每个样本的最坏绝对误差、最坏误差界比例、RMSE 与全输出摘要。
CPU oracle 造成的采样间隔和本机未锁频是微基准限制，不能把该执行节奏当作模型稳态。

当前入口通过自有 CUDA/CPU 的 37/37 套 CTest、679 次用例执行；
含真实 executable 的 13 项微基准反例覆盖漏计转换、漏样本、类型/载荷错误、
跨轮不确定输出、负收益保留和目录迁移。验证位于 `.run/cuda-precision-001/micro-validation/`。

## 微基准结果

六个正式进程使用 clean 提交 `03492ca9e3d7a4765ecaa659d109ff1757c8514f` 的同一二进制。
其 CI run `36416066381` 五个任务通过；GPU 结果来自本机，不能由 CI 替代。
原始记录位于 `.run/cuda-precision-001/micro/`，包含一个 manifest、六份报告及各进程环境，
完整 raw 由 [canonical bundle](../benchmarks/results/cuda-precision-001/evidence.json) 定位。
独立目录的离线合同复核通过，不是第二次 GPU 执行。

| 身份 | SHA-256 |
| --- | --- |
| `mini-cuda-kernel-bench` | `7cd5c4960422ad39422b0f9fd2349b6551dda08fedb35a4366f3a75e1144d256` |
| `manifest.json` | `93262e3fbb8fa8b5cd939e3275af7ec0ea175dbc6e7b5213e49484bd1e0c3261` |
| `source-state.json` | `d165bbec9931e58c3c0f9a85e3622d9b9dc6cf5bae56cb0555772ddefb7e8e89` |
| `summary.json` | `268e268304daab6f1da823990e96c0f79d8eeb3f85e729e04df9e5608a00d603` |
| `collection-status.json` | `97fd4991fdd8d27f9f741325a1b3f2efc4d4adfecd9a8eff88777eb9e3c9ec8e` |

全部 960 个样本保留，其中正式测量 576 个。每进程逐字节验证 310 个唯一 tensor，
并对 26,083,200 个输出元素建立 FP64 oracle；相同精度的全部重复和跨 trial 输出摘要一致。
最大实现误差为 `3.386521711945534e-7`，最大固定误差界比例为 `0.0004331271549129347`。
这些是同 operands 的矩阵实现检查，不是 F16 相对 F32 的全模型误差。

| 项目，S4/L2048/B128 | F32 | F16 矩阵 |
| --- | ---: | ---: |
| 设备权重载荷 bytes | 2,384,199,680 | 1,192,230,912 |
| 最大转换 scratch bytes | 0 | 786,432 |
| 项目 owned bytes | 3,449,229,312 | 2,258,046,976 |
| 权重上传 staging 峰值 bytes | 8,388,608 | 8,386,560 |

owned 减少 **34.5347%**，符合静态预测及 30% 的内存门槛。它不是整卡显存占用，
不包含库内部资源，也不意味着请求并发上限变化。

下表为 host enqueue-to-completion 的配对改善百分比 `100*(1-T_f16/T_f32)`：
先取每进程三次正式样本中位数，再逐 trial 配对，最后取三轮改善中位数。
所有负值保留；不以三轮结果声明普适统计显著性。

| 矩阵 | M | GEMM-only 改善中位数 % | 含转换改善中位数 % | 含转换 trial 0 / 1 / 2 % |
| --- | ---: | ---: | ---: | --- |
| Q | 1 | 69.55 | 55.82 | 55.82 / -51.42 / 56.83 |
| Q | 4 | 32.11 | 2.78 | 2.78 / -7.27 / 6.94 |
| Q | 32 | 56.72 | 38.26 | 38.26 / 32.12 / 38.97 |
| Q | 128 | 67.48 | 61.49 | 62.51 / 58.47 / 61.49 |
| gate | 1 | 47.10 | 38.73 | 60.06 / -35.50 / 38.73 |
| gate | 4 | 38.04 | 3.61 | 1.61 / 6.41 / 3.61 |
| gate | 32 | 65.07 | 50.81 | 54.41 / 42.75 / 50.81 |
| gate | 128 | 57.78 | 53.61 | 55.08 / 53.61 / 53.13 |
| down | 1 | 52.09 | 48.84 | 48.84 / 64.18 / 43.52 |
| down | 4 | 31.12 | 5.06 | -0.29 / 5.06 / 10.30 |
| down | 32 | 49.14 | 38.72 | 39.12 / 30.79 / 38.72 |
| down | 128 | 69.14 | 61.50 | 62.06 / 61.14 / 61.50 |
| LM head | 1 | 49.93 | 49.04 | 49.04 / 50.20 / 45.79 |
| LM head | 4 | 50.94 | 51.08 | 51.08 / 54.08 / 46.99 |
| LM head | 32 | 48.26 | 47.60 | 47.60 / 52.88 / 46.44 |
| LM head | 128 | 52.52 | 50.32 | 51.76 / 50.32 / 50.10 |

M=4 的 Q/gate/down 在计入转换后，收益明显收窄。Q/M=1 的 F32 含转换边界
三轮中位数为 `38.095/10.787/40.148 us`，与正序/逆序分组对应；候选对应
`16.829/16.334/17.333 us`。仅两次预热和计时顺序不能排除 cache、库选择、时钟及
提交间隙影响，`ENG-048` 的归因仍未完成，不筛掉逆序负结果。
LM head 各 M 的收益不能直接按 body 的 M 外推，模型中 LM head 行数实际为选中 logits 的 R。

**微基准入口决定：支持进入 Primary 的模型接入与冻结数值验证。** M=128 的三类 body shape
在每一轮计入转换后仍有明显改善，有限探针没有触发速度方向的停止线。
它们不证明 `prefill-128` 的主指标已达标，也不形成 `performance_success` 或
`memory_only_success`。完整模型数值结果如下；不再采微基准，不自动启动 attention 备选。

## 模型数值边界

同一 `CudaRuntime` 按 precision mode 使用 typed matrix views。每层 Q/K/V 共用一次输入转换，
O 一次、gate/up 共用一次、down 一次；最终先选择 logits 行，再转换 LM head 输入。
每次 forward 核对 `4*layers + (有 logits ? 1 : 0)` 次转换，F32 为零。
hidden、residual、norm、attention、FP16 contiguous KV 与执行状态合同保持不变。

```bash
build/wsl-own-cuda/bin/minillm-cuda-model-tests \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --contract tests/data/qwen3_validation_cases.json \
  --precision-study benchmarks/runtime-inputs/qwen3-precision-v1.json \
  --output .run/precision-numerical
```

输出目录必须尚不存在；该入口按冻结 SHA 校验输入，不能与旧 `--full` 或
`--reference-model` 混用。每个 S 下先执行 F32、保留选中 logits，再销毁 GPU 实例并执行
F16，不同时驻留两套权重。续写分叉后只在固定 F32 token 轨迹上比较数值。

原始记录位于 `.run/cuda-precision-001/model-validation/`，`execution-identity.json`
绑定运行源码快照、二进制、模型与输入。采集身份为 `a8a56de` 加
`source-state.json` / `source-snapshot.zip`，不是 clean HEAD 采集。
完整研究使用同一个 canonical bundle，本目录不是独立发布包。

| 检查 | 结果 |
| --- | --- |
| 固定 teacher forcing | 48/48 配置、840 行通过 |
| 短 golden | F32/F16、S1/S4 共 12/12 组完全一致 |
| 长度边界与状态 | 两模式的 slot 3 / 2048、越界 preflight、clear/reuse、mixed 和计时开关检查通过 |
| 四组 32-token 自由续写 | 全部 token 一致，不代表数值合同通过 |
| 全部 logits 检查 | 1099 次比较，2 次失败对应同一输出位置的两种检查标签 |
| 全部比较最坏指标 | RMSE 0.045953874、max absolute 0.206753254、cosine 0.999885866 |

原始边界记录中的 `reuse_bitwise_equal` 只执行了有限 float 逐值相等检查，
不提供正负零等位模式一致性保证；当前入口使用 `reuse_values_equal` 字段。
计时开关检查确实使用 `memcmp`，与槽复用的判定不同，见 `ENG-068`。

唯一失败位置为 `generation-repeated`：1536-token prompt、step=19（第 20 个输出，
输入最后位置 1554）。RMSE 为 0.044546700、max absolute 为 0.183897972，
cosine 为 **0.9998858663580449 < 0.9999**。所有值有限，两个 argmax 均为 330；
near-tie 规则不豁免 cosine 门槛。自然轨迹未分叉，因此它与 F32 teacher-forced 检查
使用同一行，不能报告成两个独立失败样本。

`first-numeric-failure-logits.json` 保存两个完整词表向量。独立 CPU `math.fsum`
重算 cosine 为 0.9998858663580924，确认失败不是摘要舍入或原归约顺序造成。
当前证据不能把偏差唯一归因于权重舍入、activation cast 或跨层传播中的某一项，
不据矩阵单测通过宣称全模型无精度损失。

**决定：`blocked_correctness`。** 按规范停止 Primary 的模型/Serving 性能采集，
不放宽阈值、不更换 golden、不引入 BF16/TF32/自动 fallback。显存门槛虽通过，
仍不符合 `memory_only_success` 或产品资格。正式预算保持 micro 6/6、model 0/6、
Serving 0/12，新增 NSys 0/1、NCU 0/1。研究原始证据已归档，不再恢复 Primary 性能采集。
唯一后续实现规范为 [GPU-KV-001](NEXT_SPEC_V3.md)，研究共享容量与直接分页访问，
保持 F32 数学；不启动精度版 V4 的融合 attention 备选。

| 回归与内存检查 | 结果 |
| --- | --- |
| 自有 CUDA / CPU CTest | 22/22、15/15 套，共 683 次用例执行 |
| 原 F32 短模型 | S1/S4 2/2，128 次 logits 比较、6 组短 golden |
| 原 F32 HTTP | 12/12，包括停服单终态 |
| Runtime memcheck | 11/11 用例，0 错误、0 泄漏；包含候选转换溢出的 fail-stop |
| 真实 F16 模型 CLI memcheck | 8-token golden 一致，0 错误、0 泄漏；不是完整数值通过或性能证据 |

`precision/validation-summary.json` 的 SHA-256 为
`7485476ee97c5083589253620cee9bc7dc2333d8161002a220e9b82d52cd065e`；
`precision/first-numeric-failure-logits.json` 为
`cbd0777a39cad561e891d02e6095a6c03b578a3faafcaee4bbef84382239635d`。
`verification.json` 索引原始报告及最终回归身份；原数值失败不被后续 CTest 成功覆盖。

## 研究交付

Release `cuda-prec-001-20260928` 绑定实现 `103070a`，自身 CI run `36426140443`
五任务成功；它是数值负结果的研究预发布，不是候选晋升。单包为 6,684,838 bytes、
257 个文件，SHA-256 为
`eaa77db15c7a7fefd5f145a5a6565abcbb11d13350fe3283e98bd0b53da23aa8`。
全部 256 个列入 `SHA256SUMS` 的文件、独立目录微基准和公开下载均已复核。
包中保存原始失败，而不是用最后一次成功回归覆盖失败。

停止线使原计划的六个模型性能进程、十二个 Serving 对照进程和新增 Profiler 不执行。
这些是有明确原因的未测项，不能填写为零退化、无收益或已验证性能。
研究结论、边界、输入、身份与原始证据完整；产品数值资格仍然失败。
