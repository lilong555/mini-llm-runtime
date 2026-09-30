# 产物政策

## 公开发布

仓库与对外分享的 canonical bundle 采用公开发布策略；GitHub 仓库保持 `PUBLIC`。
源码、测试、文档、小摘要及符合本政策的 Release 证据可公开交付。
上传前检查文件清单和内容，凭据、访问令牌、模型权重、依赖 checkout、构建产物、
本地服务状态及辅助 Python 实验不得随源码或证据包发布。
冻结的历史源码快照保留原始内容，不作为当前发布规则；当前规则以仓库根目录
`AGENTS.md` 和 [版本控制约定](VERSION_CONTROL.md) 为准。

## 存放边界

2026-09-30 收尾保留例外：`benchmarks/results/cuda-vs-001/cuda-vs-001.zip`
（56,872,322 bytes，SHA-256
`a0cecd6ae37413705613daeb20e1e67c3ec1bbc45ce5a1737f72f40b845c9a0c`）
仍为 KEEP。本轮没有通过匿名下载、完整性及既有复核器验证的外部替代来源，
不删除 Git 工作树中的原件，不重写历史；以后迁移仍须先通过上述门禁。

| 位置 | 内容 |
| --- | --- |
| Git | 产品源码、测试、固定输入/seed/schema、复现命令、小型证据索引、主要结论与限制、必要代表性原始样本 |
| 外部 canonical bundle | 完整数值/性能 raw、日志、source snapshots、NSys/NCU/SQLite、诊断和失败记录；优先 GitHub Release asset |
| 本地可再生目录 | 重复解包、roundtrip、副本、临时 stdout/stderr、可由 canonical raw 重算的中间表 |

每个实验只有一个 canonical raw bundle，索引保存 URL/路径、SHA-256、尺寸、来源
和复现入口。模型权重、依赖 checkout、build output、凭据、本地服务状态与辅助 Python
实验不进 Git。最终候选的 CI 必须绑定它自己的 SHA。

## 迁移规则

先验证目标包可下载、摘要一致、可解包并通过已有验证器，再删除已确认重复的工作树副本。
没有可获取替代品时保留原件。不得删除唯一证据、抹掉失败或不利结果、改写共享历史或移动
已发布标签。删除当前文件不等于删除历史 blob，也不承诺缩小已有 clone。
Release 暂不可用不阻塞产品开发。

## 冻结基线

- 实验：`CUDA-VS-001`，模型/微基准/数值/Profiler/验证五组件。
- 当前 locator：`benchmarks/results/cuda-vs-001/cuda-vs-001.zip`。
- 尺寸：56872322 字节。
- SHA-256：`a0cecd6ae37413705613daeb20e1e67c3ec1bbc45ce5a1737f72f40b845c9a0c`。
- 既有复验入口：`scripts/cuda_evidence_bundle.py`。
- 原组件和历史失败保留；M1 不再扩展 sweep、trial 或嵌套封包。

## 证据停止线

source SHA 或必要 dirty snapshot、binary/model/input hash、配置、raw、summary、
validation、复现命令齐全且可获取即可。允许检查错 token、重复终态、非法内存、
缺样本、错单位、错源码；不新增验证器的验证器，不以产物数量作为完成指标。

## CUDA Serving

`CUDA-SERVE-001` 使用 [单一证据索引](../benchmarks/results/cuda-serving-001/evidence.json)
定位 Release asset、SHA-256、尺寸、采集源码与既有复核命令。
完整 raw/日志/源码快照/NSys/SQLite 不进入 Git；Git 中只保留固定输入与小摘要。
实验的 source SHA/dirty snapshot 与最终发布候选分开，后者的 SHA 及自身 CI run
由 Release 元数据绑定，不通过反复重封包更新候选身份。

## GPU KV 研究

`GPU-KV-001` 的[单一公开索引](../benchmarks/results/gpu-kv-001/evidence.json)
定位最终 B 结论、446 文件的 canonical bundle、摘要及离线命令。
30 个正式性能进程、唯一 NSys、不同阶段的必要源码快照和失败诊断
保留在同一个外部包；Git 不存放 raw/profiler 或重复组件包。
