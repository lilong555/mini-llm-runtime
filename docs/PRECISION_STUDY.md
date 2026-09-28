# FP16 矩阵边界研究

## 当前状态

规范为 [CUDA-PREC-001](NEXT_OPT_SPEC.md)，路线为 [PROJECT_PLAN_V4](PROJECT_PLAN_V4.md)。
M4-0 的决策、已有时间线拆解和实验预注册已完成。第一组配置入口已通过本机验收；
F16 存储、转换、矩阵执行与三层性能结果尚未提供，不作显存或速度改善声明。

| 模式 | 当前执行能力 | 默认 |
| --- | --- | --- |
| `f32-pedantic` | 原自有 CUDA 模型与 Serving；矩阵输入、权重、累加、输出均为 F32 | 是 |
| `f16-matrix-f32acc` | 预留名称；模型、存储和 CLI 在执行前明确拒绝 | 否 |

模式枚举不依赖 CUDA 头文件。Serving、CUDA CLI 和两个 CUDA benchmark 均识别
`--cuda-precision`；CPU 与上游后端显式使用该参数时失败。
`ModelInfo` 与 `/metrics` 的 `precision_mode` 单独标识精度，非自有 CUDA 后端为 null。
旧 F32 算术合同、有效权重摘要、模型输入和验证门槛保持不变。
新实验 JSON 尚未接入执行器，不能交给旧基准入口运行并当作新实验。

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
  约 34.53% 的减少、2,258,046,976 bytes 的总量均是静态预测，不是实测。
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
当前已使用 **0/24、0/1、0/1**。已有时间线的离线读取、构建与正确性测试不计作性能 trial。
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
mixed 中 attention 份额较高不自动开启备选：Primary 尚未停止，
且备选还需独立目标 workload 的可复验基线。没有 Tensor Core 使用证据，当前记为 `unverified`。

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
独立修复自身的 CI run `36406587106` 已通过；精度合同的 CI 单独核对。

下一项为同一 owner 下的 F16 typed arena、有限 staging、RN-even 转换、half gather 和
cuBLAS 矩阵边界。完整模型只在这些门禁通过后接入，遵守一个主设计与一次小修订的停止线。
