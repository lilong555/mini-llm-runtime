# Windows/Linux 兼容性验收

本目录验证 C++ 产品构建、测试进程编码和现有 CPU 产品，关联源码提交 `a0a62146ec0cb26ef8c548ad7104884892adbed6`。

## 结果

| 检查 | 结果 |
| --- | --- |
| 自有 CUDA CTest | 21/21 |
| CPU CTest | 15/15 |
| 独立核心 CTest | 11/11 |
| 上游 CUDA CTest | 15/15 |
| 四构建用例执行 | 1034/1034 |
| WSL `cp1252` 编码反例 | 三套 CTest、37/37 次用例执行 |
| CPU 实模型 | 13/13 |
| CPU HTTP | 8/8；临时服务已退出 |
| 原生 Windows 数值报告 | 默认编码与 `cp1252` 均为 13/13 |
| 原生 Windows 微基准 | 默认编码与 `cp1252` 均为 9/9 |
| 原生 Windows 模型基准编码断言 | 两种环境各 1/1；完整套件受符号链接权限限制 |

原生 Windows 使用 Python 3.10.11，默认区域编码为 `cp936`。完整模型基准套件的两次运行均因 `WinError 1314` 中止，保留非零退出码，不计为通过；定向编码断言不是完整套件的替代。

[远端 CI](../../ci/a0a6214/README.md) 的五个任务全部通过，Windows 和 Ubuntu 的完整套件没有跳过符号链接检查。远端检查不改变本机权限，参见 `ENG-057` 至 `ENG-059`。

## 原始证据

- `ctest-*.xml`、`ctest-*.txt`：四构建回归与受控编码检查；五份 JUnit 合计 65 套、1071 次用例执行。
- `cpu-model.json`、`cpu-http.json`：实际 CPU 数值、生成、KV、HTTP/SSE、断连及超时检查。
- `native-windows/`：六次完整测试进程和两次定向进程的命令、源码摘要、输出与退出码。
- `diagnostics/`：源码 `5a4508d` 的 Windows CI 失败日志、运行身份及 WSL 编码复现；不是通过结果。
- `commands.json`、`environment.json`：执行命令、模型与二进制身份，不包含模型权重、编译产物或本地服务控制状态。
- `source-state.json`、`source-snapshot.zip`、`source-identity.json`：144 个源码文件及其校验。

本地采集保留 `5a4508d` dirty worktree 的实际身份；源码快照的全部 144 个文件与提交 `a0a6214` 相同，不把提交后的 Git 状态倒写进原始记录。

## 复核

在本目录执行字节校验，并使用源码快照内的既有 CTest 验收器：

```bash
sha256sum -c SHA256SUMS
unzip source-snapshot.zip -d /path/to/new-source
pwsh -NoProfile -File /path/to/new-source/scripts/Test-CtestEvidence.ps1 -Directory .
```

`evidence.json` 登记产物摘要、验证范围和各类结果；`SHA256SUMS` 同时覆盖该索引。复核不需要模型、CUDA、Windows 或原采集绝对路径，也不等于重跑这些进程。

本组没有重跑完整 CUDA 数值、模型性能或 Profiler，不将旧基线归属于新二进制；不提供 GPU Serving、Windows CUDA 实模型或端到端加速结论。
