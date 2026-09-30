# Qwen3 Host Model

`minillm_model` 提供不依赖 CUDA 的模型绑定与分词器。`Qwen3Model` 独占 GGUF 的只读映射，`Tokenizer` 独占 llama.cpp 的 vocab-only 模型；两者均不创建 CPU executor、KV cache 或 llama 执行上下文。CPU `Runtime` 另外持有自己的线程池与物理分页 KV。

## 模型与视图

```cpp
#include "minillm/qwen3_model.h"
#include "minillm/tokenizer.h"

minillm::Qwen3Model model("models/Qwen3-0.6B-Q8_0.gguf");
minillm::Tokenizer tokenizer("models/Qwen3-0.6B-Q8_0.gguf", model.dimensions().vocabulary);
const auto tokens = tokenizer.tokenize("The capital of France is");
const auto& query = model.layers().front().query;
```

`Qwen3Model` 在构造时读取并校验 dense Qwen3 metadata、各层矩阵和归一化权重。矩阵保留原始 F32/F16/Q8_0 `TensorView`；norm 转为只读的 FP32 vector。调用方通过 `source()` 读取原始 metadata 和 tensor 信息，通过 `dimensions()`、`embeddings()`、`output()`、`output_norm()`、`layers()` 获取绑定结果。

所有接口返回只读引用或只读数据视图。owner 不可复制、不可移动；借用的矩阵、层、norm 和 metadata 引用不能超出 owner 的生命周期。没有 `output.weight` 时，`output()` 与 `embeddings()` 指向同一段映射，`tied_output()` 为真，不复制权重。

`head_dim` 来自 `qwen3.attention.key_length`，并与 value length、可选 RoPE dimension 校验；不能用 embedding/head_count 推算。Qwen3-0.6B 的 embedding 为 1024、Q heads 为 16、KV heads 为 8、head_dim 为 128，因此 Q 投影宽度为 2048。模型只接受偶数 head_dim、可整除的 GQA 分组、有限正数的 epsilon/base，以及未缩放的 RoPE。

`ModelDimensions`、`InputToken`、`Logits` 位于 `model_types.h`，`runtime.h` 继续包含这些类型。CPU forward 接口、执行阶段顺序和物理 KV 契约见 [Runtime 计时与模型基准](RUNTIME_PROFILING.md)。

## 分词契约

`Tokenizer(path, expected_vocabulary)` 使用固定版本 llama.cpp，配置 `vocab_only=true`、`n_gpu_layers=0`，核对词表大小与 embedding 行数。它提供 `tokenize()`、`token_piece()`、`is_eog()` 和 `vocabulary_size()`；CPU Runtime 的对应接口使用同一个实现。

分词固定使用 `add_special=true`、`parse_special=true`。输入上限为 256 KiB；token piece 使用可扩展缓冲，控制 token 的可见性遵循上游 `llama_token_to_piece(..., special=false)`。不自动应用 chat template，不自行实现 tokenizer，也不调用上游模型 forward。

## 验证入口

```bash
bash scripts/dev.sh build
bash scripts/dev.sh test
build/wsl-cpu/bin/minillm-host-model-tests \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --contract tests/data/qwen3_validation_cases.json
bash scripts/dev.sh validate
bash scripts/dev.sh check-http 8057
```

CTest 的 `host-model` 使用微型 GGUF fixture 检查绑定、共享/独立 output、metadata 与逐项形状拒绝；实模型检查单独验证全部 28 层绑定、7 组固定分词输入及 151936 个 token 的 piece/EOG。CPU 数值、profile、KV、HTTP 及提取前后固定输入对照的证据位于 [Host Model 验收](../benchmarks/results/validation/host-model/README.md)。
