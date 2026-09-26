# 自有 CUDA Serving 测量

## 身份与协议

- 采集源码：`b1ced89e98eef3bb2b6f21ac8f0f72c38067bfc2` 加冻结的工作树快照。
  三个采集目录的源码清单摘要均为
  `ccd8d3f1a4b830a24249ee70d4067633c8236590171e0f96e5d588e7a65afe12`；
  不将 dirty 采集标成该提交的 clean build。
- 服务端 SHA-256：`b9608fbdf1b50798e113b196a5501a7fdf04a8c402de34aa5738f1e5c5edacb2`。
  客户端 SHA-256：`2544b16f1bb7540939f21bacafd76c4457f9ebd0d9c1e19ccf3122e33372186c`。
- RTX 4070 Laptop 8 GiB，WSL2，驱动 `591.74`，CUDA 12.8；
  GNU 11.4 / Ninja / RelWithDebInfo，`-O2 -g -DNDEBUG`，架构 89。
  自有 CUDA 开启，上游 GPU 关闭；模型与两个 trace 的完整摘要见 [协议](protocol.json)。
- source Q8_0 → device F32，activation F32、KV F16、单 stream、同步 execute。
  模型数学沿用冻结的 CUDA Runtime，不含 native Q8 GEMM。
- S=4、Lmax=2048、credits=8192、B=128、chunk=32、信用粒度 16、
  queue=64、event buffer=128、prefix=0；正式测量 telemetry=off。
- 两条 trace 各 24 请求、每请求 32 输出 token；每策略三轮、轮次反转顺序。
  共 12 个正式服务进程，无 pilot；所有进程使用相同源码、二进制和模型。
  固定预热为 `Hello` / 8 token。SLO 为 TTFT≤1000 ms 且每请求 mean TPOT≤100 ms。

## 正式结果

下表为每轮指标的三轮中位数，不是把不同进程的 token 混合后重新计算分位数。
完整逐轮指标、ITL、每请求最大停顿、初始化和设备采样见 [summary.json](summary.json)。

| Trace | 策略 | 输出 token/s | Goodput 请求/s | TTFT P50 ms | TTFT P95 ms | mean TPOT P95 ms | SLO 请求 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed-length | mixed | 130.68 | 4.084 | 402.38 | 706.31 | 22.79 | 72/72 |
| mixed-length | prefill_first | 122.03 | 3.496 | 707.10 | 1050.65 | 32.90 | 66/72 |
| burst-reuse | mixed | 70.02 | 2.188 | 277.43 | 539.43 | 23.62 | 72/72 |
| burst-reuse | prefill_first | 70.22 | 2.194 | 236.58 | 459.31 | 30.43 | 72/72 |

288 个正式请求全部成功，输出 9216 token；每条 trace 的六轮输出逐请求一致。
拒绝、超时和请求失败比例均为 0；未达 SLO 的六个请求仍保留在完整集合和 goodput 分母。
这不是过载容量实验，不能据此宣称任意负载下不会失败。

- mixed-length：三轮中 mixed 吞吐均较高，中位数约高 7.09%，goodput 约高 16.82%。
  mixed 的 batch 数为 280–282，每轮 157 个 mixed batch；
  prefill_first 为 366 个 batch、0 个 mixed batch。两种策略都保留原始结果。
- burst-reuse：吞吐中位数相差约 0.28%，配对方向随 trial 变化，判为
  `measurement_inconclusive`，不增加 trial。两策略每轮均为 288 个 batch，
  mixed 每轮有 96 个 mixed batch。固定突发间隔也计入完整 trace 吞吐分母。
- prefill_first 的 TTFT 较低或常见 ITL 较短，不代表所有停顿更小。
  mixed-length 的最大单 token 间隔为 mixed 31.14 ms、prefill_first 340.46 ms；
  burst-reuse 分别为 47.77 ms、466.71 ms。
  后者仍可满足 mean TPOT SLO，说明均值与单 token 长停顿必须分别报告。
- 未新增 A/A、bootstrap 或显著性框架；上述三轮差异是本机固定负载的描述，
  不是跨 CPU/llama.cpp 的加速结论，也不是生产 P99 保证。

## 资源与初始化

每个进程的 KV resident 为 939524096 字节，项目 owned device 为 3449229312 字节，
物理容量为 8192 token；请求结束后 live tokens=0、逻辑信用归零，
resident 保留至 runner 析构。`live_kv_pages=null`，不把信用块当作 GPU 页。

单次 NSys 中的峰值 live tokens 为 1255，峰值保留信用为 81 块；
这是该时间线的观测值，不是整个正式基线的峰值。
四个槽分别服务 7、5、6、6 个测量请求，存在动态加入、等待和复用。

| 初始化中位数 | mixed-length 六进程 | burst-reuse 六进程 |
| --- | ---: | ---: |
| model load ms | 252.08 | 248.26 |
| storage initialization ms | 10415.90 | 10096.32 |
| weight decode/upload ms | 9950.48 | 9748.94 |

weight decode/upload 包含在 storage initialization 中，不可相加。
服务进程重启不代表操作系统文件缓存冷启动；这些初始化时间不计入正式客户端吞吐。

`nvidia-smi` 请求周期为 1000 ms，共 126 个整设备采样，
窗口从 ready 后至停服，包含预热和客户端调度等待。GPU 利用率样本为 1%–99%，
最高整设备显存 5311 MiB、温度 71°C；它们不代表项目独占显存或稳态 SM 利用率。
采样原始时间戳使用主机本地时间 Asia/Shanghai；未锁频或固定 CPU affinity。

## 单次 NSys

只采集 mixed-length / mixed 一次。完整 288 次 forward 中包含 8 次预热与
280 个测量 batch，记录 201216 次 kernel；每次 forward 的 28 层、项目 kernel 顺序、
cuBLAS 矩阵 kernel、单 stream 与 batch ID 对应均通过。
157 个测量 batch 同时含 prefill/decode，768 个输出与正式参照一致。

- 测量部分 H2D 共 74976 字节，仅输入/位置/sequence/logits 索引；
  D2H 共 5312 字节，每 batch 最多 24 字节，为 token/status。
  没有逐层 weight/hidden 往返或全词表 logits 下载。
- 从首个 forward 到最后设备完成的 Runtime API 范围，没有 cudaMalloc/cudaFree
  或非零 API 返回码；结合固定 resident 与完整传输检查，没有逐请求 Runtime/weight 重建。
- 测量部分 host runner 合计 5729.53 ms，设备 span 合计 5702.04 ms，
  device busy 并集合计 5211.49 ms，span 内 gap 合计 490.56 ms。
  这些边界不同且重叠，不能相加，也不能把 event busy/gap 当作 GPU 利用率。
- 含预热的 API 范围中，cudaMemcpyAsync 的 host 经过时间合计 4205.71 ms，
  cudaLaunchKernel 1321.28 ms，cudaStreamSynchronize 仅 1.74 ms。
  异步复制 API 可能发生 host 等待，不能据最后同步很短声称没有等待。
- 使用 legacy software-instrumented trace；保留 Unified Memory 无法跟踪的诊断。
  本次仅验收显式传输和该执行路径，不声称硬件 tracing 或 Unified Memory 已验收。
  没有 NCU，也没有用这次诊断耗时代替正式性能。

## 验收与边界

本机生命周期五构建 74/74 套 CTest、基准扩展 own-CUDA/CPU 37/37 套通过。
CPU 实模型 13/13；三后端 HTTP 各 12/12；真实 Qwen3 的 S=1/4、mixed/reuse、
并发 tokenize 及 post-launch fault 在 memcheck 下为 0 错误、0 泄漏。
故障时 4 active + 2 queued 各有一个 backend_error，故障 batch 不发 token，
信用归还但 poisoned resident 隔离至 owner 析构。

`b1ced89` 的 CI run `36242754913` 为 5/5。旧 `f88886e` 的 4/5、
sanitizer 超时及短命启动会话连接失败均在证据包中保留；
最终发布候选的 SHA 和自身 CI run 由 Release 元数据绑定，见 [证据索引](evidence.json)。

M3-1 的实验预算已用完并结束。没有 GPU paging、PagedAttention、prefix sharing、
async、multi-stream、native Q8 或生产级承诺；M3-2/M3-3 尚未启动。
