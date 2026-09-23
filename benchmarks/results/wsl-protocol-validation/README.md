# 原生回放终态验证

本目录验证真实 HTTP/SSE 失败路径，不用于性能结论。两类输入各运行 mixed 和 prefill-first 一次，使用独立的 MiniLLM CPU 服务、关闭预热和前缀缓存、`max_active=1`、`queue_capacity=1`。

| 输入 | 实际结果 | 严格验收 |
| --- | --- | --- |
| `validation-timeout.jsonl` | 每轮一个 HTTP 200 SSE 超时，`error=timeout`，单个终态和 `[DONE]` | 两个失败请求均保留；成功输出比较数为 0 |
| `validation-queue-full.jsonl` | 每轮一个成功请求和一个 HTTP 429，`error=queue_full`；拒绝响应没有 SSE 终态 | 两次成功输出一致，两个拒绝请求均保留 |

允许的失败在 manifest 中显式声明，不计入成功、吞吐分子或 goodput。超时场景通过的是合法终态与报告结构验收，不是模型输出正确性验收。

原始报告、源码快照、输入和验收记录分别位于 [timeout](timeout/manifest.json) 和 [queue-full](queue-full/manifest.json)；命令输出为 `collection.txt`。每轮服务均已退出。可在原生 `pwsh` 中复现：

```powershell
$settings = @{
    Trials = 1; Port = 8033; MaxActive = 1; QueueCapacity = 1
    Context = 128; MaxModelLen = 64; BatchTokens = 16; PrefillChunk = 4
    PrefixEntries = 0; PrefixTokens = 0; NoWarmup = $true
}
./scripts/Benchmark-Policies.ps1 @settings `
    -Trace benchmarks/traces/validation-timeout.jsonl `
    -AllowedRequestOutcomes success,timeout -OutputDirectory .run/protocol-timeout
./scripts/Benchmark-Policies.ps1 @settings `
    -Trace benchmarks/traces/validation-queue-full.jsonl `
    -AllowedRequestOutcomes success,queue_full -OutputDirectory .run/protocol-queue-full
```
