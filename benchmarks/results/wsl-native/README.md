# WSL2 原生验证

运行位置：Ubuntu `/home/li/code/mini-llm-runtime`，Linux 文件系统。GCC 11.4.0、CMake 4.4.0、Ninja 1.13.0，`RelWithDebInfo`，CPU-only。`llmserve` 为 Linux x86-64 ELF，包含调试符号。

源码基点为 `bf6c491af486`，运行工作区包含 PLAN-001 未提交开发内容和 WSL 开发入口。此处记录功能验证，不用于性能对照或宣称精确提交的性能结果。依赖固定为 `911f6cdc8ab8a530b2bee09ee61471a6f3178eeb`；Q8_0 与 F32 参照均已通过各自 manifest 的大小和 SHA-256 校验。

| 验证 | 结果 | 原始报告 |
| --- | --- | --- |
| Linux CPU 产品构建 | 114 个构建步骤成功 | 本地 `build/wsl-cpu` |
| CTest | unit、gguf，2/2 通过 | `ctest.xml` |
| 真实 HTTP | 8/8 通过 | `http.json` |
| 真实模型 | 九项数值与 KV 检查完成，三组生成一致；混合批断言失败 | `model.json` |

模型套件失败属于待解决的 `ENG-017`，不能视为完整模型验收通过。未安装 Linux PowerShell，因此本次未运行 PowerShell benchmark-validation fixture。未配置 Linux CUDA Toolkit，未进行 GPU 原生编译或性能测试。

复现：`bash scripts/dev.sh build`、`bash scripts/dev.sh test`、`bash scripts/dev.sh validate`、`bash scripts/dev.sh check-http 8015`。测试日志和临时服务状态保留在 Linux 工作区的 `build/wsl-cpu` 与 `.run`，不提交模型、构建目录或依赖 checkout。
