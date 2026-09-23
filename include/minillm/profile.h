#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace minillm {

enum class ProfileStage {
    kv_prepare,
    embedding,
    rope_prepare,
    attention_norm,
    query_projection,
    key_projection,
    value_projection,
    qk_norm_rope_kv,
    attention,
    output_projection,
    attention_residual,
    ffn_norm,
    gate_projection,
    up_projection,
    swiglu,
    down_projection,
    ffn_residual,
    final_norm,
    lm_head
};

constexpr std::string_view profile_stage_name(ProfileStage stage) noexcept {
    switch (stage) {
    case ProfileStage::kv_prepare: return "kv_prepare";
    case ProfileStage::embedding: return "embedding";
    case ProfileStage::rope_prepare: return "rope_prepare";
    case ProfileStage::attention_norm: return "attention_norm";
    case ProfileStage::query_projection: return "query_projection";
    case ProfileStage::key_projection: return "key_projection";
    case ProfileStage::value_projection: return "value_projection";
    case ProfileStage::qk_norm_rope_kv: return "qk_norm_rope_kv";
    case ProfileStage::attention: return "attention";
    case ProfileStage::output_projection: return "output_projection";
    case ProfileStage::attention_residual: return "attention_residual";
    case ProfileStage::ffn_norm: return "ffn_norm";
    case ProfileStage::gate_projection: return "gate_projection";
    case ProfileStage::up_projection: return "up_projection";
    case ProfileStage::swiglu: return "swiglu";
    case ProfileStage::down_projection: return "down_projection";
    case ProfileStage::ffn_residual: return "ffn_residual";
    case ProfileStage::final_norm: return "final_norm";
    case ProfileStage::lm_head: return "lm_head";
    }
    return "unknown";
}

struct ParallelProfile {
    std::size_t count = 0;
    std::size_t grain = 0;
    std::size_t threads = 0;
    std::size_t chunks = 0;
    std::size_t participating_threads = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t dispatch_ns = 0;
    std::uint64_t caller_work_ns = 0;
    std::uint64_t caller_wait_ns = 0;
    std::uint64_t worker_work_sum_ns = 0;
    std::uint64_t worker_work_max_ns = 0;
    std::uint64_t worker_start_delay_max_ns = 0;
    bool completed = false;
};

struct StageRecord {
    ProfileStage stage;
    std::int32_t layer = -1;
    std::size_t input_tokens = 0;
    std::size_t logits_tokens = 0;
    // [M, K] 的输入乘以 [N, K] 权重的转置；非矩阵阶段均为零。
    std::size_t matrix_m = 0;
    std::size_t matrix_n = 0;
    std::size_t matrix_k = 0;
    std::uint64_t wall_ns = 0;
    ParallelProfile parallel{};
};

struct ForwardProfile {
    std::uint64_t batch_id = 0;
    bool completed = false;
    std::size_t input_tokens = 0;
    std::size_t logits_tokens = 0;
    std::size_t sequences = 0;
    std::size_t context_before_sum = 0;
    std::size_t context_before_max = 0;
    std::size_t context_after_sum = 0;
    std::size_t context_after_max = 0;
    std::size_t threads = 0;
    std::size_t kv_pages_before = 0;
    std::size_t kv_pages_after = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t unaccounted_ns = 0;
    std::vector<StageRecord> stages;

    void reset(std::size_t capacity) {
        const auto id = batch_id;
        auto storage = std::move(stages);
        *this = {};
        batch_id = id;
        stages = std::move(storage);
        stages.clear();
        stages.reserve(capacity);
    }
};

} // namespace minillm
