# CUDA 微基准与数值验收

本目录验证真实形状微基准工具、独立 softmax 入口及当前编译产物的完整模型数值。五进程性能原始数据位于 [微基准基线](../../cuda-micro-baseline/README.md)，不计入本目录的功能或 CTest 数量。

## 验证范围

| 检查 | 结果 |
| --- | --- |
| 自有 CUDA CTest | 18/18 |
| CPU CTest | 12/12 |
| 独立核心 CTest | 8/8 |
| 上游 CUDA CTest | 12/12 |
| CTest 用例执行 | 926/926 |
| CPU 实模型 | 13/13 |
| CPU HTTP | 8/8 |
| CUDA 层级 memcheck | 9/9，0 错误、0 泄漏 |
| CUDA 层级 racecheck / synccheck | 0 hazards / 0 错误 |
| 完整数值比较 | 12528/12528 |

微基准计划检查覆盖 375 个真实形状、metadata、抽样与稳定 seed；报告检查覆盖缺样本、错误形状、数值伪造、传输边界、分配与释放、统计单位、目录迁移及发布回滚。独立 softmax 使用原 attention 内核，设备用例覆盖真实宽度、因果 NaN 尾部、padding、输入保持与非法 metadata。

`diagnostics/functional-report.json` 是无 manifest 的早期单进程功能检查，含 375 个用例、1875 个样本；其执行时二进制身份未单独绑定，不作为正式独立 trial 或编译身份证明。正式微基准包另有逐进程 manifest 绑定。

## 当前数值证据

固定模型与原容差不变。240 个 teacher-forcing 组合包含 11760 次比较，12 组自然生成另有 768 次比较；三个短 prompt 在 S=1/S=4 下的六组金标准、计时开关、2048-token 边界与 clear 复用均通过。

- 最大 RMSE：`0.01912634175393243`。
- 最大绝对误差：`0.07230734825134277`。
- 最小 cosine：`0.9999865801437234`。
- 数值失败、argmax 分歧、near-tie 均为 0。

数值测试二进制 SHA-256 为 `fb9dc71c05a44f19eec764ada02d9a9a6918c89327dd65228ce268f23b213f34`。`numerical-source-identity.json` 记录执行前后的源码与二进制一致；`environment.json`、`source-state.json`、`source-snapshot.zip` 固定构建及源文件。源码基点为 `dcb07b7c179d9f9710b606fbf11b5ce33e9b40e9`，采集时为 dirty worktree，不将基点 SHA 冒充全部实际源码。

`model-preflight/` 已通过当前完整数值身份的继承检查，计划为 70 个独立进程，但没有执行模型性能采样。四项拒绝检查验证已有目录保护、不完整模型基准拒绝和错误数值编译身份拒绝，已有文件保持不变。

## 独立复核

需要 PowerShell 7 与 Python 3.10+，不需要 CUDA 设备、模型权重、依赖 checkout 或原二进制：

```powershell
pwsh -NoProfile -File ./verify.ps1
```

入口从已校验的源码快照提取复核工具，核对原始文件、完整数值、CTest、sanitizer、CPU HTTP、编译身份与模型预检来源。旧 [全量数值归档](../cuda-full/README.md) 保持独立身份。

415 个产物在独立目录复核通过。`revalidation.json` 保留六项缺件或语义篡改的拒绝结果，包括数值编译身份、API 调用次数与数值归档定位；原始文件保持不变。

本目录不是 CPU8/16 与 CUDA 的模型性能基线，不证明 GPU HTTP、GPU paging 或自有 PagedAttention。正式模型 A/A/异构采集与 Step 9 Profiler/完整性能包仍有独立门禁；Windows 与远程 CI 未在本机复测。
