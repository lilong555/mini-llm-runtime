# CUDA Profiler 与证据工具验收

本目录验证模型输出一致性、完整模型 Profiler、可迁移证据包及现有 CPU 产品。当前 C++ Runtime、算子和模型基准二进制与正式模型基线相同；本组不重新建立模型性能统计。

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

模型基准检查覆盖全部同后端 A/A、跨 trial 和对照组输出；真实报告副本中的两类不一致被拒绝，原始 70 进程报告通过。Profiler 检查覆盖层顺序、单 stream、传输、事件缺失、原始符号、硬件单位、进程返回值与环境白名单。完整包检查覆盖文件闭合、派生摘要重算、来源身份与不可覆盖发布。

`numerical-inheritance.json` 明确 `rerun_full_numeric=false`。数值测试二进制 SHA-256 仍为 `fb9dc71c05a44f19eec764ada02d9a9a6918c89327dd65228ce268f23b213f34`，模型基准仍为 `d7eb569942f6ffa6621f346c351e16d531dad70041afc76c0cc25dbac0f53180`；完整数值原件位于 [cuda-micro](../cuda-micro/README.md)。本次没有重跑全量数值或 sanitizer，不将归档复核计作新的 GPU 执行。

## 身份与诊断

`source-state.json`、`source-snapshot.zip` 和 `environment.json` 固定当前工具与验证身份，源码基点为 `0754722511b98a8d834167e3befb2c3307e81fa2`，实际为 dirty worktree。源码基点不是所有当前修改已经提交的声明。

`diagnostics/initial-tools/` 保留前一组工具验证、初次 CTest 与 PowerShell 返回值故障；`diagnostics/ncu-symbol-profiler/` 保留五进程首轮诊断、原始 profiler ZIP、名称简化导致的分析失败和同一原件的符号重导出。它们不替代 [正式 Profiler](../../cuda-model-profiler/README.md)，也不累加到上表当前 CTest 数量。问题见 `ENG-049`、`ENG-050`、`ENG-053`、`ENG-054`。

## 独立复核

需要 PowerShell 7 与 Python 3.10+，不需要模型、二进制、CUDA 或 Nsight：

```bash
pwsh -NoProfile -File verify.ps1 -Directory .
```

`evidence.json` 登记 180 个产物。181 个既有文件在独立目录复核中保持不变，缺少源码 ZIP、伪造 CTest 通过、数值编译身份不符三项反例均被拒绝，见 `revalidation.json`。

HTTP 检查为 CPU `minillm` 后端，不是 GPU Serving；Windows 和远程 CI 未在本机复测。
