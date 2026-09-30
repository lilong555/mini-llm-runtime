# CUDA Profiler 与证据工具验收

本目录验证模型输出一致性、完整模型 Profiler、可迁移证据包及现有 CPU 产品。Runtime、算子和模型基准二进制与正式模型基线相同；工具验证和模型性能统计分开记录。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 自有 CUDA CTest | 21/21 |
| CPU CTest | 15/15 |
| 独立核心 CTest | 11/11 |
| 上游 CUDA CTest | 15/15 |
| 当前 CTest 用例执行 | 1034/1034 |
| CPU 实模型 | 13/13 |
| CPU HTTP | 8/8 |
| 完整数值继承 | 12528 次比较，源码与二进制相同 |

模型基准检查覆盖全部同后端 A/A、跨 trial 和对照组输出；两类真实报告反例被拒绝，原始 70 进程报告通过。Profiler 检查覆盖层顺序、单 stream、传输、原始符号、硬件单位、进程返回值和环境白名单。封包检查覆盖文件闭合、Git 字节保护元数据、派生摘要重算、独立工具来源与不可覆盖发布。

`numerical-inheritance.json` 明确 `rerun_full_numeric=false`。数值测试二进制仍为 `fb9dc71c05a44f19eec764ada02d9a9a6918c89327dd65228ce268f23b213f34`，模型基准仍为 `d7eb569942f6ffa6621f346c351e16d531dad70041afc76c0cc25dbac0f53180`；完整数值原件位于 [cuda-micro](../cuda-micro/README.md)。CPU 回归保留相同、未变 C++ 二进制的实际命令与报告，不将复制或离线复核计为额外模型执行。

## 身份与诊断

`source-state.json`、`source-snapshot.zip`、`environment.json` 固定当前工具身份，源码基点为 `0754722511b98a8d834167e3befb2c3307e81fa2`，实际为 dirty worktree。完整包的复核工具绑定本目录的验收源码；[Profiler](../../cuda-model-profiler/README.md) 保留自己的采集源码身份。

`diagnostics/pre-bundle-tools/` 保留前一组工具验证，其中 `diagnostics/initial-tools/` 包含初次 CTest 与进程返回值故障，`diagnostics/ncu-symbol-profiler/` 包含首轮五进程诊断、原始 profiler ZIP、名称简化失败和符号重导出。导出参数及文件白名单的拒绝结果也保留在 `diagnostics/`。这些记录不累加到上表当前 CTest 数量，见 `ENG-049` 至 `ENG-056`。

## 独立复核

需要 PowerShell 7 与 Python 3.10+，不需要模型、二进制、CUDA 或 Nsight：

```bash
pwsh -NoProfile -File verify.ps1 -Directory .
```

`evidence.json` 登记 236 个产物。237 个既有文件在独立目录复核中保持不变，缺少源码 ZIP、伪造 CTest 通过、数值编译身份不符三项反例均被拒绝，见 `revalidation.json`。本组没有重跑全量数值或 sanitizer。

HTTP 检查为 CPU `minillm` 后端，不是 GPU Serving；Windows 和远程 CI 未在本机复测。
