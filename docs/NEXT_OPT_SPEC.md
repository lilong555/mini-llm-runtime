# NEXT_OPT_SPEC — 有界 FP16 矩阵路径实验

## 0. Spec ID / Audit HEAD / 执行身份

- Spec ID：`CUDA-PREC-001`。
- 对应计划：`PROJECT_PLAN_V4.md / M4-1`。
- Audit Date：2026-09-26，Asia/Tokyo。
- Audit HEAD：`6ca7d2dfeccb38b11b7596a552046651afcb693e`。
- Audit Branch：`feat/own-cuda-serving`。
- 推荐新分支：`perf/cuda-f16-matrix-path`，从已包含当前 Serving 的候选创建。
- 当前 F32 路径为 reference，默认不变。以下 precision mode、F16 view/cast 与新实验合同均为拟新增。
- 只实施一种优化：**F16 常驻大矩阵 + F16 GEMM operands + F32 accumulation/output**。
- 不是“全模型 FP16”，不是 native Q8，不是自研 GEMM。
- 第一次性能实现前提交本文的协议与门槛。不得采样后改 primary metric、输入、精度门槛或失败分母。
- 开始前只检查 Audit HEAD 后的新 delta；已完成的 CUDA Runtime/Serving 不重新规划。

## 1. Problem Statement

当前 `src/minillm/cuda/matrix.cpp::matrix_multiply()` 使用：
- W、X、Y：F32；
- `CUBLAS_COMPUTE_32F_PEDANTIC`；
- Context 初始化为 `CUBLAS_PEDANTIC_MATH`。

`CudaStorage` 把源 Q8_0 有效权重在初始化时展开为 F32。当前模型：
- 权重 2,384,199,680 bytes；
- 项目设备分配 3,449,229,312 bytes；
- 连续 F16 KV 939,524,096 bytes；
- S=4，Lmax=2048，Bmax=128。

这不是错误：它降低了第一版模型调试复杂度。但是现在完整 GPU Serving 已成立，下一研究问题是：

> 能否在不更改 KV、attention 算法、scheduler 或请求生命周期的条件下，减少权重存储，并用受控精度的矩阵执行改善模型性能；其收益有多少能传导到已有 Serving？

当前还没有证明矩阵是所有 Serving batch 的主要耗时，也没有 FP16 性能结果。不能预先宣称2倍/4倍，不能把存储减少等同于速度提升。

## 2. Observed Evidence

固定于 Audit HEAD 的证据：

1. `src/minillm/cuda/storage.cpp/.h`：单个 weight arena、显式 workspace、连续KV；weight()仅返回F32 view。
2. `matrix.cpp`、`context.cpp`：当前F32/PEDANTIC组合。
3. `layer.cpp`：Q/K/V共享normalized输入，gate/up共享FFN normalized输入，可按共享输入组复用一次转换。
4. `runtime.cpp`：GPU embedding、完整层、selected-row LM head、greedy；同步返回前提交状态。
5. `src/mini_cuda_runner.cpp`：实际GPU Serving执行，CPU/其他后端不是待重新接入的任务。
6. `benchmarks/results/cuda-serving-001/analysis.md`：mixed-length、burst-reuse、12进程与单次NSys。
7. 同目录summary/validation/evidence：逐轮结果、明确采集source-state、14,130,560-byte canonical raw bundle。
8. `tests/cuda_model_tests.cpp`、`cuda_full_validation.cpp`、`cuda_serving_tests.cpp`：现有数学和生命周期门禁。

正式性能采集来自`b1ced89... + dirty snapshot`，发布候选为6ca7d2d。后续A/B两臂必须来自新的同一实现候选和同一可执行文件，不能把旧不同日期的F32数据直接当作paired baseline。

## 3. Hypothesis

H1 — Memory：
把197个大矩阵（包含tied embedding）用F16存储、113个norm tensor保持F32，预计减少约1,191,968,768 bytes权重；加一个最大786,432-byte cast buffer，总owned预计从3.449GB到约2.258GB，约减少34.53%。

H2 — Matrix / Model：
对于M=32/128等真实shape，F16输入与权重、F32累加/输出可能减少带宽负担并允许更高效的硬件执行，超过新增cast开销。M=1不保证使用Tensor Core，也不保证相同比例收益。

H3 — Serving：
模型执行改善可能降低TTFT/TPOT和排队，但开放到达trace的输出token/s可能受请求到达率限制。不能因burst吞吐接近不变就宣布kernel优化无效；也不能只因micro变快就宣布Serving加速。

H4 — Numerics：
保留F32 residual/norm/RoPE/QK/PV/LM output和F16 KV，可能使新增误差维持在既有模型门槛内。这是需要验证的假设；FP16溢出、near-tie与生成分叉必须保留。

## 4. Why This / Why Not Alternatives

选择理由：
- F32权重成本确定，收益上限可算；减少量大于当前完整KV reservation。
- 直接补充precision/storage/compute/numerical contract的新能力，不复制已有CPU分页经历。
- 可以复用cuBLAS与完整Runtime，范围小于GPU pool+table+admission的联合改造。
- 相同Serving与KV可保留，较容易分离因果。

本次不做：
- PagedAttention：没有容量拒绝证据；固定pool与hard S=4不会自动因分页提高并发。
- Native Q8：会引入block-scale kernel及prefill/decode双路径，验证与维护成本更高。
- Scheduler：mixed已是现成改善，不能以旧prefill_first的停顿为新精度优化唯一对照。
- CUDA Graph：未建立shape/signature可复用率或关键路径收益；kernel数量不够构成理由。
- Host pinning：API elapsed不是纯copy；不能把4.2s直接当可消除成本。
- Fused attention：有价值但属于唯一条件Backup；不与本SPEC并行。

## 5. Current Data Path / Target Data Path

### Current

```text
source Q8_0/F32 tensors
  -> host decode_row()
  -> F32 weight arena
  -> F32 hidden/normalized activation
  -> cuBLAS F32 PEDANTIC
  -> F32 outputs / attention / residual
  -> F16 KV
  -> F32 LM head logits -> argmax -> token/status
```

### Target

```text
same source tensors
  -> host decode_row() to effective F32
  -> dense weights: round once to F16, upload once
     norm weights: F32
  -> F16 embedding gather -> F32 hidden
  -> F32 norm / residual / RoPE / attention
  -> grouped F32-to-F16 matrix-input cast
  -> cuBLAS(F16 W, F16 X, F32 accumulate, F32 Y)
  -> F32 outputs / attention
  -> same F16 KV, same sequence state, same Serving
```

不能把源Q8_0直接reinterpret成F16；不能下载另一个未匹配的F16 checkpoint替代同一有效权重。

## 6. Design / Numerical Mode

新增一个共享precision enum，示意：

```cpp
enum class PrecisionMode {
    f32_pedantic,
    f16_matrix_f32acc
};
```

建议放在`include/minillm/cuda/precision.h`，仅提供枚举和名称转换，不依赖CUDA头文件。也可在已有合适无CUDA依赖头中定义，禁止重复定义两份枚举。

`CudaRuntimeConfig`末尾增加mode，默认`f32_pedantic`；StorageLimits/MemoryPlan传递同一mode。
`ModelConfig`可在末尾增加精度字符串或无CUDA依赖枚举，CPU-only构建不应因配置类型引入CUDA库。

CLI统一新增：
```text
--cuda-precision f32-pedantic|f16-matrix-f32acc
```
适用于own-CUDA CLI/model bench/Serving；其他backend显式传该参数时拒绝，不能静默忽略。

公开报告新增：
- source checkpoint hash；
- source effective F32 hash；
- device weight dtype：mixed(F16 matrices/F32 norms)；
- matrix operand dtype、accumulation、output dtype；
- KV dtype、非矩阵activation dtype；
- cuBLAS compute/math mode；
- precision_mode。
不能把backend名称换成一个新框架；仍为`minillm-cuda`，精度单独标识。

## 7. Data Structures / Typed Layout

### WeightRecord

保留source_type，不将它改作device dtype。扩展记录：
- device_storage_type；
- device payload bytes；
- `effective_sha256`仍指原始有效F32序列，保持旧语义；
- `device_payload_sha256`指实际上传字节；
- tied alias仍指同一allocation，不重复存储。

现有StorageType可以扩展f16；不能仅根据sizeof(T)==2判断它是“量化权重”或“KV”。
内部提供两种显式view：
- norm_weight：`DeviceTensorView<const float>`；
- matrix_weight：F32或F16的tagged view/`std::variant`，只允许这两种类型。
不要建立任意rank、任意device、任意dtype的通用Tensor框架。

现有`weight(name)`的F32调用不能在F16模式悄悄返回半精度地址reinterpret为float。调用方必须选择正确typed入口；非法组合在launch前拒绝。

### Workspace

保留现有F32激活workspace，增加一个F16 matrix-input scratch，最大：
```text
Bmax * max K * 2 = 128 * 3072 * 2 = 786432 bytes
```

每次返回的view必须是本次实际[M,K]，stride=K，capacity来自最大buffer。不能返回固定columns=3072然后对1024列矩阵误用lda/ldb。

转换scratch与weights、KV、输出不重叠。所有乘法、加法、alignment、lda范围继续checked。新增半精度区域至少满足16-byte对齐，并保持当前arena对齐策略。

## 8. Memory Ownership / Lifetime

Owner沿用：
- Qwen3Model：host mapping与原权重view。
- CudaStorage：唯一device weight arena、workspace、KV、context。
- LayerExecutor：借用prebound typed views，不拥有权重。
- CudaRuntime：state、metadata、workspace调度。
- MiniCudaRunner / Engine：不新增一份模型，也不按请求重建。

初始化转换：
1. 调用既有`decode_row`得到相同effective F32；
2. 使用既有`minillm::float_to_half`的RN-even契约，增加与CUDA转换的边界对照；
3. 对F16结果检查finite；
4. 分块上传，完成后复用host staging；
5. 不保存整套GPU F32镜像，不依赖运行时重新展开整层权重。

F16模式下F32 staging与half staging总量仍限制在8MiB以内，按完整行分块；不能在原8MiB buffer外再无说明追加4MiB或整模型拷贝。norm直接保留F32。
若CPU转换与device RN-even在支持范围不一致，先修验证或停止，不能在报告里掩盖。

forward内使用同一stream：
```text
cast X -> scratch
GEMM consumers of X
下一组cast覆盖scratch
```
同stream顺序保证最后一个consumer结束后才写下一组；不加逐GEMM host synchronize。
没有新device allocation/free、weight upload、hidden D2H或全词表greedy下载。

析构与poisoned规则沿用当前实现。不同precision benchmark分别独立进程运行，不同时在8GiB显卡上常驻两份模型来测单模式内存。

## 9. CUDA Execution Model / Matrix / Kernel Design

### cuBLAS组合

F32模式完全保留：
```text
A/B/C = CUDA_R_32F
compute = CUBLAS_COMPUTE_32F_PEDANTIC
math = CUBLAS_PEDANTIC_MATH
```

F16候选：
```text
A = weights, CUDA_R_16F
B = cast activation, CUDA_R_16F
C = output, CUDA_R_32F
compute = CUBLAS_COMPUTE_32F
alpha/beta = float
algorithm = CUBLAS_GEMM_DEFAULT
```

Context按mode在初始化时设定math policy；候选使用支持上述运算的默认math，并显式避免reduced-precision reduction（按CUDA12.8官方允许的math flag组合设置和验证）。
保留atomics策略、单stream和显式library workspace。stream先绑定，再绑定workspace；不能因为修改模式重置已绑定workspace。

仍计算：
```text
Y[M,N] = X[M,K] * transpose(W[N,K])
cuBLAS column-major view: opA=T, opB=N, m=N,n=M,k=K
lda=W.stride, ldb=X16.stride, ldc=Y.stride
```
F16/F32stride都是元素数，不是byte数；原arena的byte offset不能直接作为leading dimension。

仅配置上述类型不等于已使用Tensor Core。记录实际kernel，并在必要NCU采集中检查所支持的tensor-pipeline指标或指令。M=1可以选择SIMT；如未验证，不写“所有矩阵都在Tensor Core上”。

### 自有kernel仅两类小改动

1. **F32 → F16 cast**
   - 一维线性/二维stride-aware，coalesced写；
   - `__float2half_rn`；
   - bounds predicate；
   - 转换前后finite检查，复用现有device status；
   - 允许正常underflow到subnormal/zero；overflow到Inf是错误，不能silent clamp；
   - 不在本阶段做cast+norm融合。

2. **F16 embedding gather → F32 hidden**
   - 复用现有index和status语义；
   - 先验证index再读，F16解码后写F32；
   - tied embedding/LM head共用同一F16物理权重；
   - 不先把整个embedding转回F32device buffer。

### 按输入组复用cast

每层四个转换边界：
1. attention norm结果：一次cast供Q/K/V三次GEMM；
2. attention output：一次cast供O projection；
3. FFN norm结果：一次cast供gate/up两次GEMM；
4. SwiGLU结果：一次cast供down projection。

final norm选中行：一次cast供LM head。

最多新增约4*28+1次cast launch/forward（无logits时少一次），不是声称launch数减少。真实开销必须计入model/Serving，不能只测预先half化的GEMM。

不把if(mode)散落成两套完整forward。只在初始化binding和上述matrix group边界分支，共享全部attention、residual和state代码。

## 10. Failure Semantics / Invariants

1. 数值模式在Runtime生命周期内固定，不允许同一请求中途切换。
2. 初始化预算与F16转换失败发生在服务监听前，已创建owner正确释放。
3. 不足显存时不降S/L/B、不fallback CPU、不加载双份权重。
4. preflight非法view、dtype、stride、shape继续在launch前失败。
5. runtime cast产生nonfinite或cuBLAS错误，复用当前post-launch poisoned/fail-stop。
6. 失败batch不发布token，不提交logical length；不声称device bytes已物理rollback。
7. Engine credit可以归还，但poisoned resident隔离至owner析构，保持当前资源schema。
8. 正常clear只重置sequence长度，resident不因clear而降为0。
9. F32模式原kernel、运算顺序、KV语义和输出契约不变。
10. profiling关闭时无额外device事件；debug logits明确独立模式。
11. 性能模式不常驻CPU/GPU reference；数值模式也尽量逐个GPU实例顺序比较。
12. 原始失败、SLO未达标、输出分叉都保留。
13. 不为了让旧validator通过而伪报device F32或旧memory总量。
14. artifact中的source/hash/config对应实际执行，而不是本SPEC的审计SHA。

## 11. Numerical Contract：实现正确性与模型偏差分开

### A. 转换本身

使用可枚举边界、正负零、subnormal、normal、half midpoint和overflow附近值，对比CPU `float_to_half`与device `__float2half_rn`。
正常finite支持范围要求bit一致；NaN/Inf按明确拒绝/错误策略测试，不以NaN payload逐bit相同作为产品要求。

### B. 矩阵实现正确性

Reference使用**相同已经舍入的F16 W/X**，解码后在CPU FP64计算。
这样检查的是layout、转换与GEMM执行，而不是把应有的FP16量化误差误判为实现bug。

小型bounded fixture沿用`atol=2e-4, rtol=2e-4`；真实shape检查除上述误差外记录condition-sensitive FP32 dot accumulation bound。
对输出元素以u=2^-24、gamma_K=K*u/(1-K*u)计算FP32累加误差尺度，允许界限事先固定为：
```text
abs(error) <= 2e-4 + 2e-4*abs(reference) + 4*gamma_K*sum(abs(x16*w16))
```
该上界仅用于定位GEMM实现，不代替严格的模型质量gate；不得采集后继续调大系数。
保留误差原值与最大比率，覆盖非方阵、stride padding、输出guard和全量元素。

### C. 模型偏差

Teacher-forced输入固定，比较：
- 原F32自有CUDA baseline；
- 新F16矩阵路径；
- 必要时既有matched-weight F32非融合CPU参考，沿用已冻结配置。

不能为了过关改变checkpoint、参考attention模式或门槛。
模型门槛保留：
```text
RMSE < 0.05
max_absolute < 0.5
cosine >= 0.9999
all_finite = true
```

原稳定短golden要求token序列严格相同。
更广语料在argmax near-tie时按现有预注册margin公式记录，不通过“near-tie”覆盖任何数值失败。reference top1-top2 gap > 2*max_absolute时要求argmax相同；其余位置只允许明确披露不确定/分叉，不宣传全模型位级等价。

模型级验证预算：
- 复用已有四类固定语料；
- 长度16/128/1536；
- chunk16/128；
- S=1/4，共48个配置；
- 采样位置规则在运行前固定，包含chunk边界、首尾与长context点；
- 另测2048边界与clear/reuse、少量32-token续写。
结果聚合到少量文件，不逐配置创建一整套bundle。

### D. Serving质量与可比性

同一precision的三轮固定输入输出必须一致。
两条已冻结Serving trace的输出与F32进行完整对照；任何分叉保留first-divergence、对应teacher-forced logits与margin。

本SPEC的“行为保持的Serving优化”晋升要求两条冻结trace的token输出相同。若不相同：
- 不篡改旧golden；
- 不放松所有backend的deterministic validator；
- 可以完成研究报告，但候选不获得“同语义Serving加速”结论；
- free-generation分叉后的timing不作为完全相同执行轨迹的paired模型对照。

固定语料通过也不等于一般文本质量不变。未做独立质量评估，不写“无精度损失”。

## 12. Tests

### Unit
扩展既有CUDA unit/storage/ops测试：
- mixed dtype weight arena和normF32；
- tied/untied alias与正确bytes/hash；
- cast RN-even、bounds、overflow/nonfinite；
- F16 gather index/stride/tails；
- GEMM非方阵、leading dimension、actual pointer alignment；
- scratch跨Q/K/V和gate/up复用，同stream不会被过早覆盖；
- candidate mode与原F32math mode互不污染；
- allocation failure部分构造清理。

### Property / State
复用当前BatchState与Runtime tests：
- 两模式各自append/clear/reuse/interleaved；
- profiling off/on同模式一致；
- preflight失败不提交；
- cast fault/执行错误进入poisoned，无fault-batch输出；
- 全部scratch owner在异步工作完成前不销毁；
- 输出顺序、每sequence单sample、没有隐藏fallback。

### Model
执行第11节固定验证集合。F32自身保留原契约；新F16不改变旧历史门槛。
参考与DUT的源有效权重一致；device half hash另记。

### Serving
复用`tests/cuda_serving_tests.cpp`与`tests/http_tests.cpp`：
- 实际S1/4、确定性mixed、slot reuse；
- concurrent tokenize；
- SSE/nonstream、cancel/timeout/disconnect/backpressure/shutdown；
- precision metadata准确；
- candidate故障复用原4active+2queued测试机制；
- CPU与upstream backend仍工作。

代表性mixed+clear+fault跑一次memcheck；不重复240-case×全部sanitizer的巨大笛卡尔积。

## 13. Microbenchmark

不重跑375-case M1。使用：
```text
Q projection: N=2048,K=1024
FFN gate:     N=3072,K=1024
FFN down:     N=1024,K=3072
LM head:      N=151936,K=1024
M in {1,4,32,128}
```
共16个shape。原K/V/O/up通过unit/模型覆盖，不必全部加入正式微基准矩阵。

每个shape记录两种时间：
1. 原生F32 GEMM与已half输入的F16 GEMM；
2. F32输入cast + F16 GEMM的完整矩阵边界时间。

第一种用于硬件kernel研究；第二种才是Runtime替换成本。权重初始化不计入稳态，但单列。
Q/K/V共同输入组至少有一个组合检查，避免把一次cast与三次cast的收益口径混淆。

独立进程A/B配对3轮；每进程2个warmup、3个测量重复。短kernel可以固定inner_iterations=20放大计时，除以次数，但不能把20次当20个独立trial。
原F32和新F16来自同一benchmark binary，进程只构造所选模式。
所有shape结果写入每进程一个报告，不为每个shape建目录。

## 14. Model Benchmark

固定S=4、Lmax=2048、Bmax=128、contiguous F16 KV。
六个workload：

| 名称 | 输入与输出 |
|---|---|
| prefill-128 | 空KV，M=128，最后一行logits，R=1；**预注册primary** |
| prefill-512-chunk32 | 空KV，总512，chunk32，仅最后一行logits；计全部prefill |
| decode-prefix16 | 先重建16-token prefix，再追加1；有效KV长度17 |
| decode-prefix1536 | 先重建1536，再追加1；有效KV长度1537 |
| decode-batch4-prefix256 | 四个独立sequence各重建256，再各追加1；M=R=4 |
| mixed16+2 | 一条空sequence预填16，两条prefix256各decode1，M=18；三条需要logits |

每轮独立重建prefix，不采用只在某一模式存在的prefix alias。
setup、clear、初始化、JSON编码不计主forward；同样流程用于两模式。
设备events诊断与host forward-to-token不同，不用events替换主指标。
3个独立配对trial；每进程2warmup/3repetition。只报告进程中位数的配对关系，不把内部重复冒充独立样本。

若需复用现有benchmark冻结协议，新增明确的`precision-experiment-v1`输入子集/分支；保留原input hash和旧schema的语义，不改旧文件令其“接受所有新参数”。

## 15. Serving Benchmark

复用两条现有trace：
```text
benchmarks/traces/cuda-serving-mixed-s20260926.jsonl
benchmarks/traces/cuda-serving-burst-s20260927.jsonl
```
从仓库读取并校验真实路径/hash；不重新生成更有利负载。

固定：
```text
policy=mixed
S=4, Lmax=2048, credits=8192
B=128, chunk=32, credit_granularity=16
queue=64, event_buffer=128, prefix=0
telemetry=off（正式性能）
每请求32token，ignore_eos=true
TTFT SLO<=1000ms，request mean TPOT SLO<=100ms
```

每trace F32/F16各3trial，总12个正式服务进程。
同一server binary切mode；顺序按trial反转，两trace起始顺序错开。
不拿prefill_first作为新优化唯一baseline，不同时调chunk、policy或arrival scale。
保留所有错误、SLO未达标、trial、温度/频率可用信息。
不锁频就明确“不锁频”，设备采样不是进程独占占用。

指标：
- output token/s、goodput、TTFT P50/P95；
- request mean TPOT、ITL分布与request max ITL；
- failure/rejection/timeout；
- owned/resident/live/capacity；
- batch composition。
开放到达导致burst吞吐变化很小可以是正确结果；不能把原70token/s当显卡饱和性能。

## 16. Profiler Plan

### 第一步：零新增采集的现有数据检查
读取M3-1的canonical bundle/SQLite，在已有工具基础上导出：
- measured batch中M、R、context；
- matrix kernel、QK、softmax、PV、其他kernel的分组耗时；
- 分清warmup/setup/measured；
- cudaMemcpyAsync按H2D/D2H和correlation拆分，避免把等待当传输成本；
- 不用全部201216kernel的总和直接解释某个单独decode。
只需一次性分析结果，不建立新的Profiler Framework。

### 新采集预算
- 最多一次NSys：candidate的完整Serving，固定trace与mode，证明新增cast进入正确stream、无权重往返/动态allocation，确认实际矩阵kernel与batch对应。
- 必要时一次NCU会话：选择一个代表性M32或128矩阵，最多比较baseline/candidate两个kernel，核查TC/tensor-pipeline、内存与duration。使用本机工具实际支持的metric，不猜metric名称。
- NCU replay、缓存/时钟变化与NSys软件插桩不是正式A/B时间。
- 如果权限/工具不能取得TC证据，记录`tensor_core_usage=unverified`，可报告F16混合精度实现，不能写“已验证Tensor Core加速”。

不是每个算子都需要NCU，也不补采M1遗留不确定项。

## 17. Primary Metric / Guardrails / Performance Gate

### 预注册主指标
Model `prefill-128` 的无profiler `host_forward_to_token_ns`。
对每个trial取进程内3次测量中位数，计算：
```text
gain_i = 1 - T_f16_i / T_f32_i
```
目标为3个paired trial的median gain≥10%，且三轮都至少改善5%。
这是项目工程门槛，不宣称3轮提供普适统计显著性。

### Memory Gate
在S4/L2048/B128、相同模型下：
```text
owned_device_bytes reduction >=30%
```
预期约34.53%，以实际arena与alignment为准。
不能把正常cudaMemGetInfo波动代替own allocation，也不能把双模型常驻时的差值当单模式结果。

### Guardrails
- 所有correctness gate通过；
- 原F32模式行为不回归；
- model另外五项：配对中位数退化≤5%，不能有两轮以上都退化超过5%；
- Serving mixed-length的goodput和output throughput配对中位数退化不超过5%；
- 两trace的TTFT P95、request mean TPOT P95、request max ITL中位数退化不超过5%，并保留绝对值；
- 不能新增失败/拒绝/超时，不能减少成功输出的计数；
- 稳态项目device allocation/free=0，weight/hidden往返=0。

5%是预注册的决策无差异带，不冒称已测出的A/A硬件噪声。结果方向反复或跨轮波动无法支持上述结论时，标记measurement_inconclusive，不追加trial直到通过。

可报告单独的memory_only_success：memory gate和correctness通过、性能护栏通过，但主速度目标没有通过。不得因此改primary metric或宣传整体加速。

## 18. Acceptance / Stop Conditions

### 研究完成与产品晋升分离
研究完成：
- 实现/未实施的原因明确；
- 输入、身份、数值、内存、三层结果和限制完整；
- 决定accept、memory-only、reject或inconclusive；
- 保留不利结果。

产品晋升：
- 新精度模式可选择，不改变原F32默认；
- 数值/生命周期/Serving可比性通过；
- 按第17节取得performance_success或符合memory-only护栏；
- 不能只凭micro更快进入默认路径。

### 具体停止线
1. 转换或模型数值不能满足冻结门槛：停止，不扩BF16/TF32/自动fallback来“救”结果。
2. 真实shape的cast+GEMM已显示目标模型不可能达到10%速度目标：不继续大范围调GEMM；可以完成有限memory-only验证并收尾。
3. 最多一个主设计与一次基于证据的小修订；不扫十几个库算法。
4. 已有目标workspace/ownership足够时不重构通用Tensor。
5. model有收益而Serving不明显：完整报告该结论，不能增加负载/arrival直到出现“胜利”。
6. 已到24个正式performance进程（6micro+6model+12Serving）、1NSys/必要1NCU上限：停止采集；因工具失败重试须保留失败并不能用新样本择优。
7. Primary形成成功或memory-only成果后feature freeze，不自动写分页。
8. 只有Primary被停止、且attention在独立目标workload份额≥25%时，才由PROJECT_PLAN_V4允许另立唯一Backup；本SPEC不实现它。
9. 没有TC证据不强行补所有形状；保留unverified表述。
10. 负结果是有效结论，但不能把未解决memory safety或错误token叫做可交付产品。

## 19. Files To Modify / New Files

### 产品
- `include/minillm/cuda/runtime.h`、`src/minillm/cuda/runtime.cpp`：固定precision mode，typed embedding/head binding，沿用forward/state。
- `include/minillm/cuda/context.h`、`src/minillm/cuda/context.cpp`：初始化math policy，默认F32不变。
- `include/minillm/cuda/matrix.h`、`src/minillm/cuda/matrix.cpp`：F16 operands/F32 output入口与preflight。
- `src/minillm/cuda/storage.h/.cpp`：explicit device dtype、arena、hash与cast scratch。
- `src/minillm/cuda/ops.h/.cu`：checked cast、half embedding gather。
- `src/minillm/cuda/layer.h/.cpp`：typed矩阵views、按四组复用cast；不复制完整层代码。
- `include/llmserve/config.h`、`src/config.cpp`、`src/mini_cuda_runner.cpp`：传递精度与参数验证，不改scheduler。
- `apps/server_main.cpp`、`apps/cuda_main.cpp`：选择mode。
- `apps/cuda_reports.h`及既有ModelInfo/HTTP序列化的必要字段：准确精度与资源元数据。

### 测试/实验
- `tests/cuda_storage_tests.cpp`、`cuda_ops_tests.cu`、`cuda_runtime_tests.cpp`：新增上述独有failure检查。
- `tests/cuda_model_tests.cpp`、`cuda_full_validation.cpp`、`cuda_serving_tests.cpp`：按precision复用现有入口。
- `apps/cuda_kernel_bench.cpp`、`cuda_runtime_bench.cpp`：小型precision子协议，不改旧默认protocol。
- 既有`Benchmark-Policies.ps1`、`Benchmark-Common.ps1`、`Analyze-Benchmarks.ps1`以及CUDA分析脚本：增加精度对照的显式合同，保留旧同模式检查。
- `apps/bench_main.cpp`若仅需记录precision metadata可最小修改；不改计时分母。
- `scripts/dev.sh`：透传已存在后端的precision参数。
- 相关Python/PowerShell旧fixture仅增加必要新mode案例，不新建验证层。

### 最多新增
- `include/minillm/cuda/precision.h`，仅当无合适现有头；
- `benchmarks/runtime-inputs/qwen3-precision-v1.json`，一个冻结子协议；
- `docs/PRECISION_STUDY.md`，一份研究报告。
- 独立测试文件仅在既有测试过大时允许一个，不能新增多个runner/framework。

不修改`attention.cu`数学、`paged_kv.cpp`、`prefix_index.cpp`、`scheduler.cpp`。
如果需要修无关bug，单独记录并保持与精度实验的身份/因果分离。

## 20. Comparison / Artifact Contract

现有若有硬编码：
```text
device_weight_dtype=F32
fixed owned bytes
full workload matrix
exact effective-weight hash interpretation
```
则以新`precision-experiment-v1`模式显式版本化，而不是删检查或伪造旧字段。

跨模式允许变化：精度、对应device payload/hash、新增cast、预测/实际memory、时延。
必须相同：原checkpoint/effective-source、token输入、KV dtype/layout、S/L/B、policy、arrival、SLO、成功/失败统计口径。
Serving相同请求到达不保证相同batch执行轨迹；这属于实测结果。相同M/context的因果对照由model benchmark提供。

同precision跨trial仍要求确定性；跨precision遵守第11节的明确数学与生成门槛。
保留源码SHA或必要dirty snapshot、二进制/model/inputhash、raw samples、validation、summary和复现命令，即达到可信交付要求。

全部raw聚合成**一个canonical bundle**；Git只保存协议、小摘要、报告与locator/hash。不新建archive框架，不把每个trial/shape的临时解包、源码副本和中间CSV重复提交。

## 21. Commit Plan / Implementation Steps

### Commit 1 — 冻结合同与模式入口
`feat(cuda): define explicit precision experiment contract`

检查最新delta、已有NSys分组与真实shape；提交precision enum、mode metadata、协议与门槛。
配置尚未实现的mode可显式拒绝；不得用常量输出/CPUfallback占位伪装可运行。
原F32构建与测试通过。

### Commit 2 — F16存储、转换和矩阵边界
`feat(cuda): add resident half weights and mixed-precision matrix path`

实现arena/typedviews/hash/预算、checkedcast、half gather、cuBLAS入口。
运行conversion/storage/layout/unit与有限16shape探针。
若cast+matrix结果不支持继续，按停止条件结束或只完成memory研究；不跳过负结果。

### Commit 3 — 接入原模型与Serving并验证数值
`feat(runtime): integrate bounded half matrix groups with fp32 state`

按四组cast复用连接同一LayerExecutor、finalhead、MiniCudaRunner；不改attention/KV/scheduler。
完成48个模型配置、短golden、clear/fault/mixed、HTTP和代表性memcheck。
任何精度失败不得靠修改golden或换checkpoint解决。

### Commit 4 — 三层结果与最终决策
`bench(cuda): publish precision tradeoff and bounded end-to-end results`

按预算运行model/Serving与必要profiler，提交报告、结果状态、canonical索引。
根据门槛决定可选产品模式或负结果归档。
更新README实际能力与限制，不把未通过项写成TensorCore/Serving加速。

四组可以为易review的小补丁拆分，但不新增第五星期式“完善整个框架”阶段，不追加第二条优化。

## 22. Definition of Done

### Research Done
- [ ] Actual HEAD/source/binary/model/input/config身份明确。
- [ ] 原F32基线保持，未重复实现CUDA/Serving。
- [ ] 两模式的dtype、weight alias、scratch ownership与错误状态明确。
- [ ] 数值实现与模型偏差分开检验；失败未被放宽门槛掩盖。
- [ ] micro的GEMM-only与cast-inclusive分开。
- [ ] model包含prefill/decode/mixed；setup与计时边界一致。
- [ ] 两条Serving trace仅改precision；所有失败和输出分叉保留。
- [ ] memory实分配与理论计划对应，未把整设备显存误当owned。
- [ ] TC使用有证据或明确unverified；未用dtype名称代替硬件证明。
- [ ] 结果判为performance_success/memory_only_success/negative/inconclusive之一，决定后续停止。
- [ ] 无新统计/封包/验证验证器框架；证据不重复膨胀。

### Product Eligible（仅拟晋升时要求）
- [ ] 现有F32及CPU/上游回归通过。
- [ ] 模型门槛与稳定golden通过，冻结Serving输出可比性通过。
- [ ] fault/清理/poisoned/owner回收与代表性memcheck通过。
- [ ] owned减少≥30%，性能护栏通过。
- [ ] 速度声明仅覆盖实际通过的层与workload。
- [ ] 默认F32保留；新mode明确可选且配置生效。

Research Done不要求正速度结果；Product Eligible不能用负结果豁免正确性。

## 23. Resume Evidence Target

未来只在实际通过后填写：
> 在自研C++/CUDA Qwen3 Runtime中实现F16常驻矩阵与F32累加/输出边界，复用共享输入转换并保持FP32 attention/状态路径；在固定模型与S/L/B下将owned显存从[实测A]降至[实测B]，prefill延迟降低[实测X%]，并报告decode与两条Serving trace的收益/退化；使用数值oracle、内存检查与Nsight核对精度和实际kernel。

没有速度收益时改为：
> 验证显存节省与数值边界，并量化转换/小batch或到达率约束如何限制端到端收益。

不要填写预测值作为实测，不写“无精度损失”“生产级”“所有模型更快”。TensorCore只能在实际验证使用的shape上描述。

## 24. 官方技术参照

实施时核对并记录本机实际库版本：
```text
CUDA 12.8 cuBLAS（GemmEx datatype / computeType / math policy / layout）
https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html

CUDA 12.8 API同步语义（Async不保证调用方绝不等待）
https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html
```

本SPEC是项目的设计选择；官方库支持某种组合不等于承诺当前shape使用TensorCore或取得某个速度。
