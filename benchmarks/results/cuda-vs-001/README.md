# CUDA-VS-001 完整证据包

本目录交付 V2-M1 的完整模型、数据路径、数值、性能测量和 Profiler 证据。自有 CUDA Runtime 与真实 token CLI 已验收；GPU Serving、GPU paging 和自有 PagedAttention 尚未提供，下一阶段为 V2-M2。

## 内容与结果

| 组件 | 验收范围 |
| --- | --- |
| 模型基线 | 70/70 进程、40950 次 forward、2520 次测量；同后端及跨后端输出一致 |
| 真实形状微基准 | 375 个用例、五个独立 trial、9375 个原始样本 |
| 数值与回归 | 12528 次完整数值比较通过；保留数值、状态与 sanitizer 原始证据 |
| 完整模型 Profiler | 五个完整模型进程；NSys 585 次 forward、368610 次 kernel；NCU 选定第 27 层 PV |
| 工具验收 | 四种构建共 62 套 CTest、1034 次用例执行；CPU 模型 13/13、HTTP 8/8 |

模型基线的 24 项比较中，14 项为 `faster`、10 项为 `measurement_inconclusive`，整体仍为 `measurement_inconclusive`。微基准顺序差异、慢样本、Profiler 开销及跟踪限制全部保留，不作统一加速、无退化或 GPU HTTP 性能声明。

各组件保留独立的采集源码、二进制和工具身份；模型基线为 clean source，Profiler 和工具验收为各自的 dirty source。工具验收按相同 Runtime 源码和二进制继承完整数值结果，没有重跑全量数值或 sanitizer，也不将 CPU 报告的复制或离线复核计为新执行。

## 获取与复核

- 完整包：[cuda-vs-001.zip](cuda-vs-001.zip)，56872322 字节、1050 个文件。
- SHA-256：`a0cecd6ae37413705613daeb20e1e67c3ec1bbc45ce5a1737f72f40b845c9a0c`。
- 独立摘要：[cuda-vs-001.zip.sha256](cuda-vs-001.zip.sha256)。
- 导出及复核结果：[export.json](export.json)、[revalidation.json](revalidation.json)。

需要 Python 3.10+、PowerShell 7 和 ZIP 解压工具，不需要模型、编译产物、CUDA、Nsight 或原采集绝对路径。在本目录校验并解压到新的目录：

```bash
sha256sum -c cuda-vs-001.zip.sha256
unzip cuda-vs-001.zip -d /path/to/new-directory
python3 -B /path/to/new-directory/verify.py --directory /path/to/new-directory
```

包内根目录的复核器检查五个组件及其关联、全部文件摘要、派生统计和工具来源。Profiler 子归档的 `profiler-raw.zip` 实际包含 NSys 原始报告、SQLite 和 NCU 原始报告，各成员都有摘要与尺寸。

独立目录迁移复验通过；缺源码、缺组件、同后端 A/A token 变化、NCU 单位错误、派生摘要伪造五项反例，即使同步更新外层摘要也均被拒绝。原始 ZIP 保持不变。

模型权重、可执行文件、第三方依赖 checkout 和本地服务状态不包含在归档中；模型与构建来源另有固定元数据。离线复核不是原二进制重跑或可信执行证明。矩阵计算归 NVIDIA cuBLAS，自有 Runtime 负责模型执行、连续 GPU KV、数据流与生命周期。
