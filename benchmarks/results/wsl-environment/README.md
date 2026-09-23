# WSL 工具链验收

验收日期：2026-09-22。源码基点为 `a1fe5d327adb3f806c0dfa7042565e42fbaff0d5`，分支为 `build/wsl-native`，工作区包含未提交代码。源码差异摘要、额外源码文件摘要、二进制、模型、工具版本及本地采集文件身份见 [environment.json](environment.json)。

本目录记录环境能力和功能验证，不是精确提交的性能基线，不用于声明优化收益。CPU/CUDA 使用 `RelWithDebInfo`，Clang 产品构建使用 `Release`。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| GCC CPU 产品 CTest | 3/3 套件通过 | [cpu-ctest.xml](cpu-ctest.xml) |
| CUDA 产品 CTest | 3/3 套件通过 | [cuda-ctest.xml](cuda-ctest.xml) |
| Clang 19 CPU 产品 CTest | 3/3 套件通过 | [clang-ctest.xml](clang-ctest.xml) |
| MiniLLM CPU HTTP | 8/8 通过；测试后在途请求为 0 | [cpu-http.json](cpu-http.json) |
| llama.cpp CUDA HTTP | 8/8 通过；`gpu=true`，测试后在途请求为 0 | [cuda-http.json](cuda-http.json) |
| CPU 数值参照的完整模型套件 | 失败；九项数值与 KV 检查完成，`ENG-017` | [cpu-model.json](cpu-model.json) |
| CUDA 数值参照的完整模型套件 | 失败；九项数值与 KV 检查完成，`ENG-017` | [cuda-model.json](cuda-model.json) |
| CUDA 冒烟与内存检查 | 1048593 个结果正确，0 错误、0 泄漏字节 | [cuda-sanitizer.txt](cuda-sanitizer.txt) |
| Nsight Systems GPU 时间线 | 64 次 kernel、3 次内存传输 | [nsys-stats.txt](nsys-stats.txt)、[nsys-validation.json](nsys-validation.json) |
| Nsight Compute | 基础采集通过；kernel 耗时、SM/DRAM 周期和占用率可读 | [ncu-validation.json](ncu-validation.json)、[ncu-basic.txt](ncu-basic.txt) |
| perf 用户态事件与符号 | 事件可用；457 个采样，丢失 0，包含 MiniLLM 符号 | [perf-stat.txt](perf-stat.txt)、[perf-report-summary.txt](perf-report-summary.txt) |
| clangd 编译数据库与解析 | `kernels.cpp` 检查通过；`runtime.cpp` 的 LSP 诊断为空 | [clangd-kernels.txt](clangd-kernels.txt)、[clangd-lsp.json](clangd-lsp.json) |

各 CTest 报告包含 `unit`、`benchmark-validation` 和 `gguf`。模型验证的失败不能被 CTest 或 HTTP 通过替代；两种参照配置都在 `stats.mixed_batches > 0` 处失败。

本目录保留上述环境检查的原始结果。确定性混合批与完整模型的当前验证见 [WSL 正确性验证](../validation/wsl-deterministic/README.md)。三个旧基准套件只提供完成标记，没有单项用例数量；归档检查将数量明确标为未报告。

Nsight Systems 2024.6.2 的不利证据保留在 [nsys-2024-diagnostics.json](nsys-2024-diagnostics.json)：驱动接口不受支持，报告没有 kernel 表。2026.1.3 的 kernel 与内存传输记录可用，但仍有 Unified Memory 跟踪限制。

Nsight Compute 以 Windows 主机授权为前提，普通 WSL 用户可完成 `LaunchStats` 和 `basic` 采集。原始指标见 [ncu-metrics.csv](ncu-metrics.csv)，命令日志见 [ncu-launch-capture.txt](ncu-launch-capture.txt) 和 [ncu-basic-capture.txt](ncu-basic-capture.txt)；未授权时的 `ERR_NVGPUCTRPERM` 诊断保留在 [ncu.txt](ncu.txt)。基础采集只包含 1 个 kernel，经 8 次 profiler replay，不作为模型性能结论。

`perf` 只验证了进程级用户态事件和 DWARF 调用栈；报告中的 `tips.txt` 提示不代表缺失采样数据，但帮助资源未验收。clangd 的 LSP 检查关闭后台索引，只验收目标文件诊断和正常退出，不代表索引或全部重构功能已验证。

复现入口见 [WSL 开发指南](../../../docs/WSL_DEVELOPMENT.md)。大型报告保留在元数据列出的本地 `.run` 路径，不入库。CUDA 服务计算属于 llama.cpp；向量冒烟程序仅验证工具链，不属于 MiniLLM GPU forward 或自研 PagedAttention。
