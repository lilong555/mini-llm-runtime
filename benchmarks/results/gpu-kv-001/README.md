# GPU-KV-001

**B：容量／研究模式。** 默认 contiguous，paged 显式 opt-in；
功能与性能研究结束，不再新增主要功能或优化 kernel。

[公开证据 ZIP](https://github.com/lilong555/mini-llm-runtime/releases/download/gpu-kv-001-20260930/gpu-kv-001-20260930.zip)
是唯一 canonical bundle：17,667,946 bytes、446 个文件。
SHA-256：
`123480f3295ae42d20ea0b5793bb2c53397907069f7594dc903dd57f03190807`。

包包含原始 micro、两组 model、两组 Serving、唯一 NSys、源码身份、
数值和生命周期验证、失败诊断及现有分析器。包内 README 提供离线复核命令，
不需要下载模型或重新运行 GPU。机器可读入口为 [evidence.json](evidence.json)。

2026-09-30 已从无凭据的公开 URL 下载，核对 ZIP SHA-256 与解包后的
`SHA256SUMS`，并使用包内现有分析器复验 micro、两组 model 和两组 Serving。
五项复验均完成；模型护栏超限和 Serving 性能负结果未被改写为通过。

## 结论

- 同容量 Serving 吞吐配对退化中位数 15.73%，未通过 10% 护栏。
- 同 288 MiB KV 子预算吞吐提升中位数 4.31%，但 TPOT/ITL 更高。
- 288 请求全部成功、9216 输出 token、逐请求输出一致，无失败。
- 唯一修订后的长 prefill/decode 仍未通过模型护栏。
- micro 6、初始 model 6、确认 model 6、Serving 12，共 30 个正式进程。

容量灵活性不等于四倍吞吐或任意负载低延迟。
详细取舍、逐轮数字及边界见 [GPU KV 研究](../../../docs/GPU_KV_STUDY.md)。
Release 绑定 `368d274` 及其自身成功 CI；原始采集使用不同阶段的固定
dirty snapshot，不冒充 Release clean build 重测。
