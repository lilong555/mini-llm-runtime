# CUDA 完整模型与 CLI 验收

范围为 `CUDA-VS-001 / Step 7`。自有 `CudaRuntime` 与 `mini-cuda-llm` 在 `GGML_CUDA=OFF` 下执行完整 28 层 Qwen3 并输出真实 greedy token。矩阵由 cuBLAS 提供，源权重为 Q8_0、设备权重为 F32。该组不是 V2-M1 全量语料或性能基线，不包含 GPU Serving/PagedAttention。

## 回归与数值

| 配置 | CTest 套件 | 用例执行 |
| --- | --- | ---: |
| 自有 CUDA ON、上游 CUDA OFF | 12/12 | 226 |
| CPU | 7/7 | 178 |
| 独立核心，无 llama | 5/5 | 167 |
| 上游 CUDA ON、自有 CUDA OFF | 7/7 | 178 |

共 31 次套件、749 次用例执行，含重复执行。CPU 模型 13/13、HTTP 8/8；临时服务已退出。

8 项 Runtime 检查覆盖 tokenizer/manifest、完整 fixture 与 CPU 对照、selected-row 顺序、无 logits 行、计时开关位级一致、无设备热分配、copy 字节实测、preflight 恢复、context 上界与 clear 后重用、预算失败、真实 nonfinite 输出及受控完成检查失败。B=129 在模型加载前拒绝，参见 `ENG-040`。

完整模型使用两类参照，各有 64 组 logits 比较：

| 参照 | 最大 RMSE | 最大绝对误差 | 最小 cosine |
| --- | ---: | ---: | ---: |
| 自有 Q8_0 CPU Runtime | 0.002589873 | 0.011019230 | 0.9999997762 |
| matched-weight F32 llama.cpp CPU | 0.005696512 | 0.024068833 | 0.9999986486 |

128 组均满足冻结门槛：RMSE < 0.05、max absolute < 0.5、cosine >= 0.9999、全部 finite。没有 near-tie 或 argmax 差异。输入覆盖中文、英文、重复、特殊 token 的长度 33，长度 128 prefill、16+2 mixed，以及四序列交错。`input.json` 保存确切 batch 描述；每个比较保留 FP32 输出摘要、位置、margin 和 near-tie 判定。

S=1 与 S=4 各自通过三个稳定短样例，共六组 8-token 金标准；三个独立 CLI 也逐 token 相同。两个参考与 GPU 顺序构造，不同时驻留。参考与性能测试不是同一种进程契约。

## 数据路径与生命周期

S=4、Lmax=2048、B=128 的项目常驻分配为 3,449,229,312 字节。完整 forward 不再上传权重或 RoPE，不传输中间 hidden，项目 allocation/free 调用均为零。greedy 不下载全词表；每个 CLI 的 8 个输出仅下载 32 字节 token 和 64 字节 status。debug logits 仅出现在显式数值检查。

S=1/S=4 完整模型及全部设备用例的 memcheck 均为 0 错误、0 泄漏；Runtime racecheck 为 0 hazards，synccheck 为 0 错误。preflight 不提交逻辑长度；执行后失败自动 poisoned，不发布输出、不提交 pending lengths，不能 clear 后继续。受控 CUDA 完成错误通过包装器注入，不宣称从真实 illegal access 恢复。

`build-boundaries.json` 记录独立开关、编译命令和动态依赖。CPU 产品没有 CUDA 动态依赖；CUDA CLI 没有自有 CPU Runtime 符号。源码与实际 GPU 路径、单 stream、FP32 PEDANTIC 共同界定归属，不把依赖库中存在的符号当作调用证据。

## 复核与边界

```bash
bash scripts/dev.sh own-cuda build
bash scripts/dev.sh own-cuda test
bash scripts/dev.sh own-cuda model-check
bash scripts/dev.sh own-cuda model-memcheck
bash scripts/dev.sh own-cuda generate --prompt "The capital of France is" --tokens 8
pwsh -NoProfile -File benchmarks/results/validation/cuda-model/verify.ps1
```

`evidence.json` 固定实际源码、模型、二进制、工具环境和产物摘要。归档带源码 ZIP 与包内校验入口；独立目录复核、缺 ZIP、篡改、已有报告保护、错误 checkpoint 和预算拒绝分别记录。模型权重、二进制、依赖 checkout 与工具链不在包中；归档复核不执行模型，不等于另一台机器重跑。

长度 256/1536 的完整语料、全部 chunk/sequence 组合、32-token 自然生成、CPU8/16 与 A/A、完整模型 Nsight 和正式性能包仍为 Step 8–9。CLI 的初始化、设备事件与 host 计时仅为诊断数据，没有加速比或 GPU HTTP TTFT 结论。
