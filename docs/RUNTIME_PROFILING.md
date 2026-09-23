# Runtime 计时与模型基准

`mini-runtime-bench` 直接调用 MiniLLM CPU Runtime，不经过 HTTP、队列或调度器。`Benchmark-Runtime.ps1` 管理独立进程、实验身份及交替轮次，`Analyze-Runtime.ps1` 验收并汇总归档。上游 llama.cpp 仅提供 GGUF/tokenizer 基础设施；该入口不执行上游模型 forward，也不常驻数值参照模型。

## 运行

WSL2 使用原生 PowerShell 和 `RelWithDebInfo` 产品：

```bash
bash scripts/dev.sh build
bash scripts/dev.sh runtime-benchmark \
  -Trials 3 -Repeats 3 \
  -OutputDirectory benchmarks/results/runtime-run

pwsh -NoProfile -File scripts/Analyze-Runtime.ps1 \
  -Directory benchmarks/results/runtime-run
```

默认比较 1、2、4、8、16 线程，16 线程包含 SMT。自定义数组在 PowerShell 中传入，例如 `-Threads @(1, 8)`；从 Bash 运行单线程配置可使用 `-Threads 8`。`Trials` 是独立进程轮数，`Repeats` 是每个进程中每种输入的测量次数，`Warmup` 是测量前的同输入预热次数。每轮反转线程顺序，并交替执行 `none / stages` 与 `stages / none`。频率、温度和后台负载没有被固定，环境字段明确记录这一限制。

`InputFile`、`Model`、`ModelManifest` 和 `BinaryDirectory` 可指定输入、模型及构建目录。模型的文件名、大小和 SHA-256 必须与来源元数据一致；输出目录必须为空。采集前增量构建，并在每个进程前后校验源码、依赖、模型、二进制、输入和 manifest。

直接运行单个配置不构成完整的身份验收：

```bash
build/wsl-cpu/bin/mini-runtime-bench \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --input benchmarks/runtime-inputs/qwen3-cpu.json \
  --threads 8 --profiler stages --warmup 1 --repeats 3 \
  --output .run/runtime-stages.json
```

## 固定输入

`benchmarks/runtime-inputs/qwen3-cpu.json` 定义 Runtime 容量、固定 token 池和完整 workload 集合，不进行随机生成或根据输出更改测量输入。

| 输入 | Prefill token | Decode 序列 | 每个 decode 的初始 KV 长度 |
| --- | ---: | ---: | ---: |
| `prefill-16` | 16 | 0 | 0 |
| `prefill-128` | 128 | 0 | 0 |
| `decode-16` | 0 | 1 | 16 |
| `decode-256` | 0 | 1 | 256 |
| `decode-1536` | 0 | 1 | 1536 |
| `mixed-16-2` | 16 | 2 | 256 |

`qwen3-scaling.json` 提供不含 `decode-1536` 的五项线程扩展配方；完整长上下文实验可单独指定 `-Threads 8` 运行 `qwen3-cpu.json`。两份输入分别生成 manifest 和结果，不混合轮次或跨输入计算 profiler 开销。

位置 `p` 的前缀和 prefill token 为 `token_ids[p % token_ids.size()]`。保留序列 0 存放不变的 KV 前缀；prefill 使用序列 1，decode 按顺序使用剩余序列，其输入为 `token_ids[(kv_tokens + sequence) % token_ids.size()]`。只有 prefill 最后一个 token 及每个 decode token 要求 logits。原始报告包含完整展开后的前缀 token 和实际 batch，包括 position、sequence、logits 和阶段分类。

每种 workload 先准备一次序列 0，分块 forward 不请求 logits。每次预热和测量前清空活动序列，再从序列 0 分享相同长度的前缀；因此 decode 的 KV 长度不会随重复次数增长。默认长度按页对齐；自定义非对齐长度会触发实际尾页写时复制，其成本包含在 forward 中。前缀构建和分享不在测量边界内。

这是共享固定前缀、已预热的模型级实验，不是冷启动、独立前缀布局对照、自由生成或真实到达过程。不同 workload 之间释放全部 KV 引用，但物理页分配的存储可复用。每份报告记录页面使用和已分配的 KV 载荷字节；`resident_bytes` 包含无引用但仍保留存储的页，不是进程 RSS，也不包含 allocator 元数据。

## 计时边界

外层 `sample.wall_ns` 覆盖一次完整 `Runtime::forward` 调用。模型加载、前缀构建、每次序列重置、argmax、有限值检查、完整 logits 摘要及 JSON 序列化不在其中；加载、构建、重置及 argmax 有独立字段。

`ForwardProfile::wall_ns` 从 profiler 状态准备后开始，到 forward 局部对象清理后结束；预留记录的内存不计入该内层时间。`unaccounted_ns` 是没有归入阶段的真实剩余时间，包括临时对象管理、记录开销和阶段间代码。每个成功 forward 满足：

```text
sum(stages.wall_ns) + unaccounted_ns == profile.wall_ns <= sample.wall_ns
```

阶段互不嵌套：KV 准备、embedding、RoPE 表、attention norm、Q/K/V projection、Q/K norm + RoPE + KV store、attention、output projection、残差、FFN norm、gate/up projection、SwiGLU、down projection、残差、final norm、LM head。attention 保留原有融合循环，包含 QK、softmax 和 PV，不声称已分离 page lookup 或三者成本。

矩阵字段定义为输入 `[M, K]` 乘权重 `[N, K]` 的转置。普通投影的 `M` 是完整 batch token 数；当前每个 logits token 单独执行一次 `M=1` 的 LM head。非矩阵阶段的 M/N/K 为 0。混合 batch 的共享阶段只计一次，不能同时完整归入 prefill 和 decode。

线程池嵌入各阶段，包含 `count`、`grain`、实际消费的 chunk 数、参与线程数及下列重叠时间：

| 字段 | 含义 |
| --- | --- |
| `parallel.wall_ns` | 一次任务发布、消费及等待完成的经过时间，不含最终统计归并 |
| `dispatch_ns` | 调用线程发布任务并通知 worker 的经过时间 |
| `caller_work_ns` | 调用线程处于消费循环的经过时间 |
| `caller_wait_ns` | 调用线程退出消费后等待其他 worker 完成的经过时间 |
| `worker_work_sum_ns`、`worker_work_max_ns` | worker 消费循环经过时间的和与最大值 |
| `worker_start_delay_max_ns` | 从调用开始到最迟 worker 进入消费循环的时间 |

消费循环包含任务分发、原子索引争用和被操作系统抢占的时间，不是纯算术 CPU 时间。没有取得 chunk 的 worker 也会付出唤醒及退出成本。worker 时间相互重叠，也与调用线程工作/等待重叠，不能与阶段 wall time 相加；调用线程等待不代表其他线程空闲。当前不提供内核调度跟踪或硬件周期计数。

## Runtime 接口

```cpp
auto profile = runtime.make_profile();
profile.batch_id = 1;
auto logits = runtime.forward(tokens, &profile);
```

省略第二个参数时默认关闭计时，不读取阶段或线程池时钟。阶段使用固定枚举，worker 槽位在创建线程池时分配，阶段记录可复用；dot 内没有新增计时、日志或全局原子操作。报告序列化属于 benchmark，不在 Runtime 中。

传入的 `ForwardProfile` 每次被重置，保留调用方的 `batch_id` 和已预留存储；`completed` 仅在 forward 成功后置为真。失败保留已执行阶段及前后 KV 页数，不能当作完整性能样本。预分配失败时 forward 尚未开始，profile 保持未完成、无计时状态，不沿用上次成功的字段。Runtime 和 profile 均由同一个调用线程独占，不支持并发调用同一实例。空 batch、非法位置、容量失败和随后恢复均有真实模型检查。

## 归档与验收

| 文件 | 内容 |
| --- | --- |
| `manifest.json`、`source-state.json`、`source-snapshot.zip` | 源码、编译选项、依赖、模型来源、二进制、环境及实验协议 |
| `input.json` | 输入配方的原始字节 |
| `threads-T-MODE-N.json` | 独立进程的加载、setup、测量、逐层逐阶段时间和输出身份 |
| `validation-summary.json` | 完整性、身份、输出及计时验收；原始报告 SHA-256 |
| `runtime-prefill.json`、`runtime-decode.json`、`runtime-mixed.json` | 无 profiler 的逐样本时间、中位数、范围和模型级输入吞吐 |
| `forward-stages.json` | 按 workload、线程数和阶段汇总的 profile 时间与线程池指标 |
| `profiler-overhead.json` | off/on 中位数、每轮配对及百分比差异 |
| `collection-status.json` | 采集终态及完成进程数 |

验收按输入配方重建精确 token、序列和位置集合，核对全部线程/模式/轮次/重复数以及每层阶段顺序和 matrix shape。它检查页面占用、驻留字节、计时和未归类时间的等式，拒绝缺失、重复、身份变化、错误类型、非法采样结果或不完整 profile。

所有测量的完整 logits 以 `sequence:i32le + vector_length:u32le + logits:f32le` 逐向量连接后计算 SHA-256，同时保留实际贪心 token。相同输入在所有线程、模式及轮次之间必须具有一致摘要和 token；真实模型测试另做逐字节 logits 比较、贪心续写及 KV 语义检查。这里不代替独立数值参照的容差验收。

目录移动后可离线重新验收，无需原模型和可执行文件的绝对路径仍有效。归档校验不是可信执行证明。失败会移除旧汇总、保留原始报告并写入失败状态。

`overhead_percent` 是独立进程中位数的相对差异；负值或小幅正值可能来自系统噪声，不能解释为 profiler 加速或绝对的插桩成本。实际噪声与结果范围须一同报告。在线调度和 batch 组成的扰动由 [在线观测入口](BATCH_TELEMETRY.md) 单独测量；本入口的固定 batch 不回答 Serving 策略优劣。
