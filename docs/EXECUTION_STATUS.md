# 项目执行状态

决策入口为 [PROJECT_PLAN_V3](PROJECT_PLAN_V3.md)，当前实施规范为
[CUDA-SERVE-001](NEXT_SPEC_V2.md)。[CUDA-VS-001](NEXT_SPEC.md) 是冻结的 M1 模型规范；
V1/V2 计划保留为历史参考。证据存放遵循 [产物政策](ARTIFACT_POLICY.md)。

## 基点与依赖

- 审计基点：`68ac275913207975a88e2090c6617467e351301c`。
- 本地原起点：`235c5c4`；与审计基点的 Git tree 相同，审计基点包含主分支合并记录。
- 开发基点：`608daf148a17007f96387040fd66ce130fe8d640`，包含现成 Windows 修复；
  对应 CI run `36233429810` 的五个任务全部通过。
- 实施分支：`feat/own-cuda-serving`。
- 唯一主线：M3-1 自有 CUDA Serving；M3-2 单项优化和 M3-3 GPU 分页须满足 V3 进入条件。
- M3-0 已完成：兼容修复与基点 CI 已确认，M1 冻结，产物政策已明确。
- M3-1 实施中：adapter、清理契约、backend/脚本、资源快照与 schema v2 已接通。
  自有 CUDA、CPU、上游 CUDA、独立核心、ASan/UBSan 五种构建共 74/74 套 CTest 通过；
  真实 Qwen3 的 S=1/S=4 共 80 个输出与独立 Runtime 和冻结短金标准一致，
  S=4 有 3 个确定性 mixed batch，槽复用与并发 tokenize 通过。
  自有 CUDA（上游 `GGML_CUDA=OFF`）、CPU、上游 CUDA 的 HTTP 均为 12/12，
  包含 UTF-8、取消、超时、两类断连、慢 socket 和活动/排队请求停服。
  真实模型 memcheck 覆盖 mixed/reuse 与 4 active + 2 queued 的 post-launch fault，
  结果为 0 错误、0 泄漏；故障批次没有 token 发布，六个请求各有一个 backend_error，
  逻辑信用归还，poisoned resident 保留到 owner 析构。
  CPU 实模型 13/13 通过；证据位于 `.run/cuda-serving-001/validation/`。
  `f88886e` 的 CI run `36241363029` 为 4/5，sanitizers 的 unit 超时；
  停机谓词锁修正后的最终候选 CI 与限定 Serving 基线仍待完成，见 `ENG-062`。

## 阶段门禁

| 阶段 | 状态 | 验收范围 |
| --- | --- | --- |
| V2-M0 / Step 1 | 已验收 | 非破坏性归档检查、严格分析、导出与独立目录复验；固定 CUDA 数值语料和性能协议 |
| V2-M1 / Step 2 | 已验收 | 独立 CUDA target、资源所有权、单 stream/cuBLAS 与矩阵单测；memcheck 0 错误、0 泄漏 |
| V2-M1 / Step 3 | 已验收 | 独立只读模型绑定与词表 owner；CPU logits/profile/KV/HTTP 及配对性能门禁通过 |
| V2-M1 / Step 4 | 已验收 | 唯一 FP32 weight arena、8 MiB staging、workspace 与显存预算、310 个唯一 tensor 及 88 组真实矩阵检查 |
| V2-M1 / Step 5 | 已验收 | gather、分组 RMSNorm、NeoX RoPE、residual、SwiGLU、finite/argmax；11 项算子检查与三类 sanitizer |
| V2-M1 / Step 6 | 已验收 | 连续 KV 状态、FP16 store、causal GQA attention、完整层、preflight/poisoned 状态 |
| V2-M1 / Step 7 | 已验收 | 完整 28 层、selected-row LM head、greedy、真实 CLI；S=1/S=4、128 组 logits、六组短金标准 |
| V2-M1 / Step 8 数值 | 已验收 | 240 个组合、11760 次固定输入与 768 次生成比较；长语料、独立 KV、32-token 续写、计时开关与容量边界 |
| V2-M1 / Step 8 性能 | 已验收，测量不确定项保留 | 375 项微基准与正式 70 进程模型基线；正确性、数据路径和跨进程输出一致性通过，24 项比较中 10 项为 `measurement_inconclusive` |
| V2-M1 / Step 9 | 已验收 | 完整模型 NSys、选定 kernel 的 NCU、五组件完整包及独立目录复验通过；工具回归、CPU 模型/HTTP 通过 |
| M3-1 | 实施中 | GPU Serving 接入及 HTTP/SSE、生命周期、限定基线验收 |
| M3-2/M3-3 | 尚未进入 | 根据 Serving 证据选择单项优化或满足 GPU 分页进入条件 |

当前自有模型具有 CPU 与完整 CUDA Runtime/CLI，并通过现有 HTTP/Engine 提供 [CUDA Serving](CUDA_SERVING.md)。[Host Model](HOST_MODEL.md) 提供独立只读绑定与词表 owner，[CUDA Runtime](CUDA_RUNTIME.md) 提供常驻权重、连续 FP16 KV、完整 28 层与 greedy。[全量数值验证](CUDA_NUMERICS.md)、[真实形状微基准](CUDA_MICROBENCHMARKS.md)、[模型性能对照](CUDA_BENCHMARKS.md) 与 [完整模型 Profiler](CUDA_PROFILING.md) 均已冻结。M1 的正确性、资源和数据路径成立，24 项模型比较中 10 项保持测量不确定。GPU Serving 的限定性能基线仍待采集，GPU PagedAttention 尚未提供；不将旧模型基线作为 HTTP 证据。

## 可复核证据

[Windows/Linux 兼容性验收](../benchmarks/results/validation/windows-ci/README.md) 关联源码提交 `a0a6214`。远端 CI 五任务全部通过，五份 JUnit 共 63 套、1185 次用例执行；本地四种构建共 62 套 CTest、1034 次用例执行，另有 `cp1252` 环境的三套、37 次检查。CPU 实模型 13/13、HTTP 8/8，临时服务已回收。原生 Windows 的符号链接权限限制和原始失败记录单列，见 `ENG-057` 至 `ENG-059`。本组没有重跑完整 CUDA 数值、性能或 Profiler，旧模型证据仍绑定各自的源码和二进制；M2 尚未实施。

[CUDA-VS-001 完整证据包](../benchmarks/results/cuda-vs-001/README.md) 包含模型基线、微基准、数值验收、Profiler 和工具回归五个独立组件，ZIP 为 56872322 字节、1050 个文件。独立目录复验通过，缺源码、缺组件、同后端输出变化、NCU 单位错误及派生摘要伪造五项反例均被拒绝，原 ZIP 不变。包内保留各阶段的真实采集身份，不含模型权重、编译产物或依赖 checkout。

[完整模型 Profiler](../benchmarks/results/cuda-model-profiler/README.md) 包含五个独立诊断进程，每个进程均执行 585 次 forward。NSys 记录 368610 次 kernel，验证每次 forward 的 28 层、单项目 stream 和显式传输；NCU 选定 forward 55、第 27 层 PV，原始符号与 NSys 精确一致。42 个必需原始产物及三个原始 Profiler 文件可获取，迁移和三项反例通过。legacy software-instrumented trace、未验收的 Unified Memory 跟踪和诊断开销明确保留，不替代正式无 Profiler 基线。

[Profiler 与证据工具验收](../benchmarks/results/validation/cuda-profiler/README.md) 包含自有 CUDA 21/21、CPU 15/15、独立核心 11/11、上游 CUDA 15/15 CTest，共 1034 次用例执行；CPU 模型 13/13、HTTP 8/8。12528 次完整数值比较按相同 Runtime 源码与二进制继承，本组没有重跑全量数值或 sanitizer。236 个登记产物可独立复核，237 个既有文件在迁移检查中不变，三项反例被拒绝。CPU HTTP 不代表 GPU Serving。

[正式 CUDA 模型基线](../benchmarks/results/cuda-model-baseline/README.md) 包含 70 个独立进程、40950 次 forward 和 2520 次 measured repetition；全部进程正常退出，同后端及跨后端 token 一致，稳态数据路径门禁通过。24 项比较中 14 项为 `faster`、10 项因噪声超过 10% 为 `measurement_inconclusive`；不作统一加速或无退化声明。289 个必需原始产物可独立获取，当前复核入口为 `revalidate.py`，295 个既有文件在迁移与五项反例检查中保持不变；见 `ENG-049`、`ENG-051`。

[CUDA 微基准与数值验收](../benchmarks/results/validation/cuda-micro/README.md) 包含自有 CUDA 18/18、CPU 12/12、独立核心 8/8、上游 CUDA 12/12 CTest，共 926 次用例执行；CPU 模型 13/13、HTTP 8/8 和 CUDA 层级三种 sanitizer 通过。该归档所记二进制的 12528 次完整数值比较通过，最大 RMSE `0.019126342`、最小 cosine `0.999986580`，无 argmax 分歧或 near-tie；415 个产物在独立目录复核，六项归档反例被拒绝。模型基准预检绑定该次采集的源码与数值测试二进制，没有继承旧编译身份。

[CUDA 真实形状基线](../benchmarks/results/cuda-micro-baseline/README.md) 包含 375 个用例、五个独立 trial、9375 个原始样本与 5625 个测量样本；跨进程输出一致，计时区间无项目设备分配或显式传输。27 个必需原始产物可迁移复核，六项缺件或语义反例被拒绝。Q projection M=1 的正序/逆序差异与 48 个相对 MAD 超过 10% 的用例完整保留，见 `ENG-048`；没有模型加速、无退化或 A/A 噪声结论。

[CUDA 模型基准工具验收](../benchmarks/results/validation/cuda-benchmark/README.md) 包含自有 CUDA 16/16、CPU 10/10、独立核心 7/7、上游 CUDA 10/10 CTest，共 875 次用例执行；CPU 模型 13/13、HTTP 8/8、CUDA 短模型通过。四个真实基准进程各覆盖 12 项 workload，共 2340 次 forward、144 次 measured repetition，输出一致，copy/allocation 门禁通过。完整数值门禁同时约束源码与模型测试二进制身份。100 个产物在独立目录复核通过，五项缺件/语义反例被拒绝；单进程计时完整保留，不代表正式 A/A 性能基线。

[CUDA 全量数值验收](../benchmarks/results/validation/cuda-full/README.md) 包含自有 CUDA 14/14、CPU 8/8、独立核心 6/6、上游 CUDA 8/8 CTest，共 807 次用例执行；CPU 模型 13/13、HTTP 8/8 和默认短模式通过。240 个组合及 12 组 32-token 续写共 12528 次比较通过，最大 RMSE `0.019126342`、最大绝对误差 `0.072307349`、最小 cosine `0.999986580`，无 argmax 差异或 near-tie。S=4 的 1536-token 计时开关与 2048-token 边界通过；783 个产物独立目录复验通过，12 项归档反例被拒绝。上游 CPU 融合 attention 的两处 cosine 超限与固定输入诊断完整保留，参照配置见 `ENG-042`。本组没有性能或 GPU Serving 结论。

[CUDA 完整模型与 CLI 验收](../benchmarks/results/validation/cuda-model/README.md) 包含自有 CUDA 12/12、CPU 7/7、独立核心 5/5、上游 CUDA 7/7 CTest，共 749 次用例执行；8 项 Runtime 检查、128 组 CPU/F32 参照 logits、六组 S=1/S=4 短金标准和三个 CLI 通过。最大 RMSE `0.005696512`、最大绝对误差 `0.024068833`，无 argmax 差异；完整模型 memcheck 为 0 错误/0 泄漏，Runtime racecheck/synccheck 均通过。CPU 模型 13/13、HTTP 8/8 单列，不代表 GPU Serving。稳态无权重/hidden 传输或项目设备分配，greedy 不下载全词表。该组不包含全量长语料或正式性能验收。

[CUDA 连续 KV 与层验收](../benchmarks/results/validation/cuda-layer/README.md) 包含自有 CUDA 11/11、CPU 7/7、独立核心 5/5、上游 CUDA 7/7 CTest，共 741 次用例执行；7 项层级用例、六组首层/末层真实权重对照、88 组矩阵回归、CPU 模型 13/13 和 HTTP 8/8 通过。设备与真实层 memcheck 为 0 错误/0 泄漏，racecheck 为 0 hazards，synccheck 为 0 错误。独立整层最大绝对误差 `0.12060546875`、FP16 舍入边界诊断和失败源码均保留；不能将共享 Q/K/V 对照冒充独立整层等价。尚无完整 GPU 模型或性能结论。

[CUDA 基础算子验收](../benchmarks/results/validation/cuda-ops/README.md) 包含自有 CUDA 10/10、CPU 7/7、独立核心 5/5、上游 CUDA 7/7 CTest，共 733 次用例执行；11 项基础算子、真实权重矩阵回归 88 组、CPU 模型 13/13 和 HTTP 8/8 通过。算子 memcheck 为 0 错误/0 泄漏，racecheck 为 0 hazards，synccheck 为 0 错误。没有完整 GPU 层或模型性能结论。

[CUDA 权重与存储验收](../benchmarks/results/validation/cuda-storage/README.md) 包含自有 CUDA 9/9、CPU 7/7、独立核心 5/5、上游 CUDA 7/7 CTest，共 722 次用例执行；CPU 模型 13/13、HTTP 8/8。S=4、Lmax=2048、B=128 的项目分配为 3,448,180,736 字节，310 个唯一 tensor 全量回读与 host 解码一致；88 组矩阵检查通过，重复 GEMM 阶段没有项目设备分配或释放。Compute Sanitizer 的设备单测和实模型存储验证均为 0 错误、0 泄漏。该证据不包含完整 GPU 模型、性能或 GPU HTTP 验收。

[Host Model 验收](../benchmarks/results/validation/host-model/README.md) 包含 CPU 7/7、自有 CUDA 8/8、独立核心 5/5、上游 CUDA 7/7 CTest，共 712 次用例执行；独立 host-model 实模型 8/8，CPU/上游 CUDA 参照模型各 13/13、HTTP 各 8/8。44 个 Runtime 进程、792 次测量的 logits 摘要、greedy 与 KV 状态一致，36 对 profile 样本的阶段与形状契约一致。前后配对中位差异为 −6.05%～+5.80%，各案例退化均在预定 A/A 阈值内；噪声带内或置信区间跨零的差异为 `inconclusive`，没有加速结论。

[CUDA 基础验收](../benchmarks/results/validation/cuda-infra/README.md) 包含自有 CUDA 7/7、CPU 6/6、独立核心 5/5、上游 CUDA 6/6 CTest；设备单测 11/11、Compute Sanitizer 0 错误/0 泄漏、CPU 模型 13/13 与 HTTP 8/8。CPU 动态依赖和独立选项组合有实际检查。该记录不包含 GPU 模型或性能验收。

[M0 验证记录](../benchmarks/results/validation/evidence-m0/README.md) 包含 CPU 6/6 与独立核心 5/5 CTest、CPU 13/13 模型检查、8/8 HTTP 检查，以及缺件/篡改/迁移/发布回滚反例。CI 配置要求 Python 3 和 PowerShell；缺少工具时配置失败。本次未运行 Windows 或远程 CI。

[CPU 归档基线](../benchmarks/results/evidence-m0/README.md) 包含 6 个独立进程、36 次测量、源码快照及可直接解压复验的完整归档包。它验证归档交付链路，不是 CUDA 性能基线或加速结论。归档复验不需要原采集绝对路径；模型和二进制重跑依赖另行取得。

`wsl-runtime-profile/context` 的历史原始快照与原 manifest 摘要相符；缺件反例和原件复验有独立证据。其他历史归档的可获取性须逐一检查。基准和验证各自保留实际源码状态，不合并为同一次 clean build 测量。

## 固定验证输入

- `tests/data/qwen3_validation_cases.json`：Q8_0/F32 参照 SHA-256、源 tensor dtype、数值模式、四类固定 token 语料、长度 16/33/128/256/1536、采样位置、near-tie 公式与三个短样例的 8-token 金标准。
- `benchmarks/runtime-inputs/qwen3-cuda-v0.json`：S=4、Lmax=2048、B=128、CPU 8/16 线程、5 个独立 trial、A/A 噪声规则及稳态数据路径门禁。
- `benchmarks/runtime-inputs/qwen3-cuda-micro-v0.json`：375 个真实形状、五个独立 trial、每样本 32 次 API 调用、FP64 对照与计时边界。
- 全量数值、微基准、完整模型 A/A/异构基线、完整模型 Profiler 和完整证据包均已有独立验收且冻结；测量不确定项保留。当前主线为 M3-1。
