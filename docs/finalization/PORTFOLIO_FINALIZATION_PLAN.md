# PORTFOLIO_FINALIZATION_PLAN

## 目标与基线

基线：`main@9a571a5d7a6570c6dfeeac9062adefca1840534e`，2026-09-30。

目标不是扩大产品，而是让陌生读者能迅速理解、复现和追问已有 CPU/CUDA Runtime → Serving 链路。

当前事实：PR #2/#3 已合并，open PR 为 0；main 自身 CI run `36709597589` 成功。不得重新规划已经完成的合并。WSL 实机、clean clone 和 Release 字节级复核未由本次审计执行。

执行前只刷新一次仓库身份并记录 delta。若 main 改变，针对变更文件修订本计划，不自动触发新一轮全项目规划。

## F0 — 修复公开事实与复现入口

T01/P0：同步当前事实。修复 THIRD_PARTY 的“没有自有 GPU Serving”、GPU_KV_STUDY 开头的待采用/待合并、CUDA_SERVING 的尚未采集；历史段落加时间和身份，不改历史数据。AGENTS 加入简短 Feature Freeze 约束。

T02/P0：整理 Golden Path。README 不使用作者绝对路径；列出 CPU/CUDA 前置工具；明确 PowerShell/Python 是脚本及完整测试依赖，而不是 C++ runtime 的运行依赖；GPU 架构默认 89 不泛化到所有机器。最多为 dev.sh 添加短前置检查，不创建新开发平台。

T03/P0：在可用的实际环境验证候选。记录 WSL 工作树、未提交改动、依赖和模型身份；使用隔离工作树或不覆盖原产物的新目录运行已有构建/CTest、短生成、Serving/HTTP 与相关 GPU 检查。没有 GPU 或接口不可用就记 BLOCKED，不借用旧 passed。

F0 出口：当前事实无矛盾；CPU 和 Own CUDA 入口清楚；本机验证实际完成或明确阻塞。历史 benchmark 不重跑、不重标。

## F1 — 收束展示与历史内容

T04/P1：README 收束到 What / Why / What I Built / Architecture / Results / Quick Start / Deep Dive，最多四个卖点。API 全表、历史实验过程、重复平台命令移到已有深层文档；不是删功能。

T05/P1：PERFORMANCE 重排为 CPU micro、CUDA model、CUDA Serving、GPU KV 取舍、FP16 停止。公开表最多六行，逐行写 workload、baseline、metric、limitation 和证据入口。

T06/P1：旧计划归档到 docs/history/plans，并检查所有引用。活跃阅读入口维持五份；实现细节文档不为目录美观全部迁移。三份本次审计文件不永久占据 README 主导航。

T07/P1：支持与 API 分层。区分 CPU 默认快速入口和 GPU 作品展示入口；stable/default、experimental、internal-only 分开。只补注释/文档，不重构 ModelRunner、状态机或内存管理。

T08/P1：证据整理。优先核查已跟踪的 cuda-vs-001.zip；替代发布可下载、hash 一致、离线复核通过后才能删除工作树副本。Release 不可用则保留原件，记例外；不重写历史，不重新封装所有旧实验。

F1 出口：主阅读路径不再被历史计划和失败研究主导；关键正结果与限制同屏；文档迁移不破坏源码、测试或复现入口。

## F2 — 发布与求职交付

T09/P2：分支引用整理。九个已检查分支均 ahead=0；实施时重新比较并排除用户本地未推送工作。保留 main 与研究标签，已整合分支只在确认后删除。不是发布阻塞项。

T10/P1：准备一个作品总览 Release。建议下一个 minor 版本 `v0.2.0`，标题注明 Portfolio baseline；先检查版本未占用，再使 CMake/tag 一致。不创建暗示生产成熟度的 v1.0，不移动 v0.1.0 或研究标签。Release 汇总 supported scope、环境、主结果、负结果和复现；链接既有包，不重复上传所有 raw。

T11/P1：求职与面试交付。四条简历、一条主故事、两条副故事、十个必答问题；人工按 30 秒/5 分钟/30 分钟阅读检查。演示复用现有 serve/check-http/benchmark，不新增 Web UI，不为演示建立新平台。

T12/P2：低风险最后整理。更新 repository About 和代码导航；可为内部 MappedFile 显式禁用复制，但没有已确认复制调用时不把它当 bug 修复宣传。无必要收益则跳过。

F2 出口：候选有自己的成功 CI 和实际本机 smoke；Release 内容与提交一致；README 和简历不夸大。完成后停止本项目的主要开发，将增量精力转向有实际需求的上游工作。

## Adjustment Matrix

| Problem | Evidence | Impact | Proposed change | Effort | Portfolio ROI | Priority |
|---|---|---|---|---|---|---|
| 自研 GPU Serving 被旧文档否认 | THIRD_PARTY 与 mini_cuda_runner 相冲突 | 可信度受损 | T01 同步当前事实 | 小 | Very High | P0 |
| 当前/历史研究状态混写 | GPU_KV_STUDY 首尾矛盾 | 看起来仍未完成 | T01 分层并置顶最终决定 | 小 | Very High | P0 |
| clone/run 入口隐含本机条件 | README、dev.sh、CMake | 陌生人容易失败 | T02 + T03 | 中 | Very High | P0 |
| 性能页只突出 paging | PERFORMANCE 与已有 Serving analysis | 主价值不突出 | T05 六行分层结果 | 小 | High | P1 |
| README 说明书化 | 多平台命令与 API/实验长段落 | 筛选成本高 | T04 精简与导航 | 小 | High | P1 |
| 历史计划抢占入口 | docs 目录及研究页链接 | 方向混乱 | T06 归档并保链接 | 中 | High | P1 |
| 默认/研究/API 边界不够集中 | 各配置与不同文档 | 错误理解 CUDA 能力 | T07 明示，不改语义 | 小 | High | P1 |
| Git 中有历史大包 | 56,872,322-byte ZIP | 工作树和阅读噪声 | T08 条件迁移 | 中 | Medium | P1 |
| 分支仍很多 | 九分支 ahead=0 | 首页像施工现场 | T09 条件清理引用 | 小 | Medium | P2 |
| 缺少当前系统总览 Release | 研究标签与既有 v0.1.0 | 交付入口分散 | T10 新版本汇总 | 中 | High | P1 |
| 技术深度没有转成面试故事 | 代码深、资料散 | 贡献无法被解释 | T11 求职交付 | 中 | High | P1 |
| About/所有权表达小瑕疵 | repo description / MappedFile | 轻微阅读风险 | T12 可选整理 | 小 | Low | P2 |

Effort 为相对改动复杂度，不是承诺工时。

## 改动预算与停止条件

产品数学、scheduler、HTTP lifecycle、CUDA 执行合同：零计划性改动。

脚本新增逻辑原则上只在现有入口补前置检查，总量以 80 行左右为上限；超预算先删除低收益要求，而不是放宽预算。除确认断链的最小回归外，不新增测试框架或新收集平台。

不新增正式性能 trial，不重跑 70 进程历史模型基线或 30 进程 paging 研究。最终候选的功能 smoke 是交付验收，不算追加优化实验，不生成新性能结论。

归档后的历史文本可保留，活跃文档总阅读量应下降。P2 未做不阻塞求职；全部 P0 完成且 P1 核心展示就位后停止。本计划最多十二项任务，不扩展 P3 功能路线。
