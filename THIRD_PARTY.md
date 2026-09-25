# Dependencies And Ownership

## Runtime Dependency

`ggml-org/llama.cpp` is pinned to
`911f6cdc8ab8a530b2bee09ee61471a6f3178eeb` and used without source modifications.

- License: MIT, retained at `third_party/llama.cpp/LICENSE`.
- MiniLLM uses its GGUF parser and tokenizer. File mapping, tensor views, shape
  checks, Q8/F16/F32 kernels, Qwen3 forward execution and CPU paged KV storage
  are implemented in this repository.
- `src/llama_runner.cpp` uses its complete model execution and CPU/CUDA kernels
  as an alternative backend. Those kernels and that backend's GPU KV layout
  are upstream work.
- Reference weight conversion uses `llama_model_quantize()`, not a custom
  quantization algorithm.
- The Qwen3 implementation follows the model's published architecture. The
  architecture, RoPE, RMSNorm, GQA and SwiGLU mechanisms are not new inventions.

The same pinned checkout vendors:

- cpp-httplib, MIT, used for HTTP transport. Its license remains at
  `third_party/llama.cpp/vendor/cpp-httplib/LICENSE`.
- nlohmann/json, MIT, used for structured JSON parsing and serialization. Its
  notice is in `third_party/llama.cpp/licenses/LICENSE-jsonhpp` and its headers.

Upstream is compiled as C++17; project code is C++20. No generated upstream
files are edited. CMake verifies the dependency commit.

## NVIDIA CUDA 与 cuBLAS

`MINILLM_ENABLE_CUDA=ON` 使用 CUDA Toolkit >= 12.8 的 CUDA Runtime 与 cuBLAS，遵循 NVIDIA 随 Toolkit 提供的许可条款。仓库不包含这些二进制库。

`minillm_cuda` 的资源所有权、设备视图、矩阵布局检查和测试为项目代码；内存/stream API 和 FP32 GEMM 内核由 NVIDIA 提供。当前自有 CUDA 能力为资源与矩阵基础层，没有完整 GPU 模型执行或 GPU PagedAttention。具体边界见 `docs/CUDA_RUNTIME.md`。`LLMSERVE_CUDA` 仍单独控制上游 ggml CUDA 后端。

## Model

- Repository: `Qwen/Qwen3-0.6B-GGUF`.
- Revision: `23749fefcc72300e3a2ad315e1317431b06b590a`.
- Artifact: `Qwen3-0.6B-Q8_0.gguf`.
- License: Apache-2.0, provided by the model repository.
- Download manifest and checksum: `models/manifest.json`.
- Derived F32 reference provenance: `models/reference-manifest.json`.

Weights are not included in source control. The F32 reference only
dequantizes Q8_0 values; it does not recover the original unquantized model.

## Serving Implementation

The request lifecycle, capacity credits, token Trie, LRU policy, iteration
scheduler, chunked prefill integration, streaming adapter, cancellation,
backpressure and benchmark tools are project code.

Continuous batching, paging, prefix sharing and priority aging are established
systems techniques. Their implementation in this repository is not a claim of
novel algorithms or of reproducing published performance results.

## Excluded Local Experiments

Any local `baselines/python/` experiment is excluded from this repository.
It is not imported, launched, or required by the C++ runtime. Its tests,
dependencies and performance evidence are not attributed to this project.
