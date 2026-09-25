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
| V2-M1 / Step 3 | 待实施 | Immutable Qwen3 host model/tokenizer，保留 CPU 数学与行为契约 |
| V2-M1 / Step 4–9 | 未实施 | 常驻权重、完整 GPU forward、真实 token、数值/性能/Profiler 证据 |
| V2-M2 | 等待完整模型 gate | GPU Serving、HTTP/SSE 和生命周期验收 |
| V2-M3/M4/M5 | 条件进入 | 依照 V2 计划的研究预算、连续 GPU baseline 和压力证据 |

当前自有模型执行仍是 CPU。[CUDA 基础层](CUDA_RUNTIME.md) 提供设备资源所有权和 cuBLAS FP32 矩阵接口；完整 GPU 模型、GPU Serving 与 GPU PagedAttention 尚未提供。

## 可复核证据

[CUDA 基础验收](../benchmarks/results/validation/cuda-infra/README.md) 包含自有 CUDA 7/7、CPU 6/6、独立核心 5/5、上游 CUDA 6/6 CTest；设备单测 11/11、Compute Sanitizer 0 错误/0 泄漏、CPU 模型 13/13 与 HTTP 8/8。CPU 动态依赖和独立选项组合有实际检查。该记录不包含 GPU 模型或性能验收。

[M0 验证记录](../benchmarks/results/validation/evidence-m0/README.md) 包含 CPU 6/6 与独立核心 5/5 CTest、CPU 13/13 模型检查、8/8 HTTP 检查，以及缺件/篡改/迁移/发布回滚反例。CI 配置要求 Python 3 和 PowerShell；缺少工具时配置失败。本次未运行 Windows 或远程 CI。

[CPU 归档基线](../benchmarks/results/evidence-m0/README.md) 包含 6 个独立进程、36 次测量、源码快照及可直接解压复验的完整归档包。它验证归档交付链路，不是 CUDA 性能基线或加速结论。归档复验不需要原采集绝对路径；模型和二进制重跑依赖另行取得。

`wsl-runtime-profile/context` 的历史原始快照与原 manifest 摘要相符；缺件反例和原件复验有独立证据。其他历史归档的可获取性须逐一检查。基准和验证各自保留实际源码状态，不合并为同一次 clean build 测量。

## 固定验证输入

- `tests/data/qwen3_validation_cases.json`：Q8_0/F32 参照 SHA-256、源 tensor dtype、数值模式、四类固定 token 语料、长度 16/33/128/256/1536、采样位置、near-tie 公式与三个短样例的 8-token 金标准。
- `benchmarks/runtime-inputs/qwen3-cuda-v0.json`：S=4、Lmax=2048、B=128、CPU 8/16 线程、5 个独立 trial、A/A 噪声规则及稳态数据路径门禁。
- 语料与性能协议的固定不表示 GPU 数值或性能已验收；GPU oracle 和测量由后续模型路径执行。
