# V2-M0 验证记录

本目录记录 V2-M0 的本机 WSL2 验收。归档门禁、模型检查、HTTP 检查与固定验证契约均属于本次工作区；模型和二进制路径只作为采集身份，不随归档包提交。

## 结果

- CPU 产品 CTest：6 个套件、172 个用例全部通过。
- 无 llama 依赖核心 CTest：5 个套件、167 个用例全部通过。
- Qwen3-0.6B Q8_0 对 matched-weight F32 参照：13/13 通过。
- MiniLLM CPU HTTP：8/8 通过。
- 固定数值契约：4/4 通过；包含中文、英文、重复 token、特殊 token，长度 16/33/128/256/1536 和三个 8-token greedy 金标准。
- 归档反例：缺失 ZIP、源码状态损坏、ZIP 条目篡改、非法路径、目录迁移、缺测试工具和原子发布回滚均通过拒绝检查。

## 归档复验

`historical-revalidation.json` 记录审计基点 `68ac275913207975a88e2090c6617467e351301c` 下历史 Runtime 归档的缺件拒绝和本机原 ZIP 复验。缺 ZIP 时既有 16 个文件保持原字节；补入与历史 manifest 一致的原件后，独立目录和导出包复验通过。

`independent-revalidation.json` 与 `telemetry-revalidation.json` 分别记录 Runtime 和在线观测导出包的包内脚本复验、逐文件保留检查、压缩的 `strace` 文件访问跟踪和结果数量。`original_model_build_results_path_accesses=0` 表示包内复验没有读取采集机的模型或构建目录。

两个可交付包位于 [M0 归档基线](../../evidence-m0/README.md)：

- `baseline-bundle.zip`：6 个 Runtime 报告、36 次测量。
- `telemetry-bundle.zip`：历史在线观测组的 2 份服务报告和 2 份 JSONL。

包内 `verification/` 保存导出时的验证入口，`bundle-manifest.json` 和同名 `.sha256` 固定包内容。归档用途是 `archive_revalidation`，不代表在另一台机器重新执行了模型。

## 边界

本记录不证明自有 CUDA Runtime、GPU 完整模型、GPU Serving 或 GPU PagedAttention 已实现。当前 CUDA 工具链仅作为后续 V2-M1 的本机环境前提；Windows、远程 CI 和 GPU 模型验收未在本次记录中执行。
