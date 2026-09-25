# 项目执行状态

决策入口为 [PROJECT_PLAN_V2](PROJECT_PLAN_V2.md)，立即实施规范为 [CUDA-VS-001](NEXT_SPEC.md)。旧 `PROJECT_PLAN.md` 保留为历史参考。

## 基点与依赖

- 审计基点：`68ac275913207975a88e2090c6617467e351301c`。
- 本地原起点：`235c5c4`；与审计基点的 Git tree 相同，审计基点包含主分支合并记录。
- 实施分支：`feat/own-cuda-vertical-slice`；V2-M0 提交为 `e2fcb6d`。
- 顺序：V2-M0 → V2-M1 → V2-M2；CPU 有界研究、GPU 分页和 Serving 策略按 V2 依赖与进入条件推进。

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
| V2-M1 / Step 8–9 | 待实施 | 全量长语料与 chunk/batch、32-token 生成、CPU8/16/A/A、完整模型 Profiler 与性能交付 |
| V2-M2 | 等待完整模型 gate | GPU Serving、HTTP/SSE 和生命周期验收 |
| V2-M3/M4/M5 | 条件进入 | 依照 V2 计划的研究预算、连续 GPU baseline 和压力证据 |

当前自有模型具有 CPU 与完整 CUDA Runtime/CLI 两条路径。[Host Model](HOST_MODEL.md) 提供独立只读绑定与词表 owner，[CUDA Runtime](CUDA_RUNTIME.md) 提供常驻权重、连续 FP16 KV、完整 28 层与 greedy。GPU Serving 与 GPU PagedAttention 尚未提供。下一项为 Step 8 全量数值与性能对照，随后执行 Step 9 Profiler/完整包门禁；Step 7 不代表 V2-M1 整体完成。

## 可复核证据

[CUDA 完整模型与 CLI 验收](../benchmarks/results/validation/cuda-model/README.md) 包含自有 CUDA 12/12、CPU 7/7、独立核心 5/5、上游 CUDA 7/7 CTest，共 749 次用例执行；8 项 Runtime 检查、128 组 CPU/F32 参照 logits、六组 S=1/S=4 短金标准和三个 CLI 通过。最大 RMSE `0.005696512`、最大绝对误差 `0.024068833`，无 argmax 差异；完整模型 memcheck 为 0 错误/0 泄漏，Runtime racecheck/synccheck 均通过。CPU 模型 13/13、HTTP 8/8 单列，不代表 GPU Serving。稳态无权重/hidden 传输或项目设备分配，greedy 不下载全词表。全量长语料和性能门禁仍未完成。

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
- 语料与性能协议的固定不表示 GPU 数值或性能已验收；GPU oracle 和测量由后续模型路径执行。
