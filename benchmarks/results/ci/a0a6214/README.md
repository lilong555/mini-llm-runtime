# Windows/Ubuntu 远端 CI

GitHub Actions [36232341864](https://github.com/lilong555/mini-llm-runtime/actions/runs/36232341864) 对源码 `a0a62146ec0cb26ef8c548ad7104884892adbed6` 的五个任务全部通过。

| 任务 | CTest |
| --- | --- |
| `unit (ubuntu-24.04)` | 11/11 |
| `unit (windows-2025)` | 11/11 |
| `cpu-product (ubuntu-24.04)` | 15/15 |
| `cpu-product (windows-2025)` | 15/15 |
| `sanitizers` | 11/11 |

五份 JUnit 合计 63 套测试、1185 次用例执行。`run.json` 保留任务、步骤和源码身份，`artifacts.json` 保留 GitHub artifact ID、尺寸与服务端 SHA-256。

目录中的五个 ZIP 是 GitHub 原始下载，摘要与服务端记录逐一匹配；解压后的 `test-results.xml` 与 ZIP 成员逐字节一致。`SHA256SUMS` 覆盖本目录交付文件，复核：

```bash
sha256sum -c SHA256SUMS
pwsh -NoProfile -File ../../../../scripts/Test-CtestEvidence.ps1 -Directory .
```

验收器也可从[本地兼容性证据](../../validation/windows-ci/README.md)的源码快照取得。此前未通过的 Windows 运行与本机权限限制在该目录的诊断记录中单列，不计入本表。

CI 只下载固定版本 C++ 依赖，不下载模型权重。ASan/UBSan 覆盖独立核心，不覆盖 GGUF、HTTP transport 或真实模型；本目录不提供 Windows CUDA、GPU Serving、模型精度或性能验收。
