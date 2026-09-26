#pragma once

#include "minillm/model_types.h"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace minillm::cuda {

enum class CudaRuntimeState { ready, poisoned };
enum class CudaOutputMode { greedy, debug_logits };

struct CudaRuntimeConfig {
    static constexpr std::size_t max_supported_sequences = 4, max_supported_batch_tokens = 128;
    std::string model_path;
    int device = 0;
    std::size_t max_sequences = max_supported_sequences;
    std::size_t max_model_len = 2048;
    std::size_t batch_tokens = max_supported_batch_tokens;
    std::size_t device_budget_bytes = 0;
};

struct CudaMemoryPlan {
    std::size_t weights_bytes = 0, workspace_bytes = 0, kv_bytes = 0, library_workspace_bytes = 0;
    std::size_t activations_bytes = 0, attention_scratch_bytes = 0, logits_bytes = 0;
    std::size_t metadata_bytes = 0, rope_bytes = 0, padding_bytes = 0, total_owned_bytes = 0;
};

struct CudaSample {
    std::int32_t sequence;
    std::size_t input_index;
    std::int32_t token;
};

struct CudaForwardResult {
    // 按 input_index 排序；debug logits 与 samples 一一对应，greedy 模式为空。
    std::vector<CudaSample> samples;
    std::vector<Logits> logits;
    std::uint64_t host_forward_to_token_ns = 0;
    std::optional<double> device_elapsed_ms;
};

struct CudaDeviceInfo {
    std::string name, uuid;
    int compute_major = 0, compute_minor = 0;
    int driver_version = 0, runtime_version = 0, cublas_version = 0;
};

struct CudaWeightInfo {
    std::string name, source_dtype, alias_of, effective_sha256;
    std::size_t rows, columns, offset, bytes;
};

struct CudaDiagnostics {
    CudaRuntimeState state = CudaRuntimeState::ready;
    std::vector<std::size_t> sequence_lengths;
    std::size_t live_sequences = 0, live_kv_tokens = 0, kv_capacity_tokens = 0;
    CudaMemoryPlan resident;
    std::uint64_t weight_h2d_bytes = 0, rope_h2d_bytes = 0, metadata_h2d_bytes = 0;
    std::uint64_t token_d2h_bytes = 0, status_d2h_bytes = 0, debug_d2h_bytes = 0;
    std::uint64_t intermediate_h2d_bytes = 0, intermediate_d2h_bytes = 0;
    std::size_t owned_device_allocations = 0, owned_device_bytes = 0;
    std::uint64_t completed_forwards = 0, post_launch_failures = 0;
    std::uint64_t model_load_ns = 0, storage_initialization_ns = 0, weight_decode_upload_ns = 0;
};

// execution/KV 状态单调用者、不可重入；tokenizer 可在外部词表锁下与 forward 并行。
// 返回前检查 stream 和 status，再同时发布结果与逻辑长度。
// preflight 失败可继续；设备执行开始后的失败进入 poisoned，必须销毁该实例。
class CudaRuntime {
public:
    explicit CudaRuntime(CudaRuntimeConfig config);
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;
    const CudaRuntimeConfig& config() const noexcept;
    const ModelDimensions& dimensions() const noexcept;
    const CudaDeviceInfo& device_info() const noexcept;
    CudaRuntimeState state() const noexcept;
    // 最后 committed 长度；poisoned 时不能作为有效设备状态对外发布。
    std::size_t live_kv_tokens() const noexcept;
    std::vector<CudaWeightInfo> weight_manifest() const;
    std::vector<std::int32_t> tokenize(std::string_view text) const;
    std::string token_piece(std::int32_t token) const;
    bool is_eog(std::int32_t token) const;
    CudaForwardResult forward(std::span<const InputToken> tokens,
                              CudaOutputMode mode = CudaOutputMode::greedy, bool device_timing = false);
    void clear_sequence(std::int32_t sequence);
    CudaDiagnostics diagnostics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
