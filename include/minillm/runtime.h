#pragma once

#include "minillm/kernels.h"
#include "minillm/profile.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace minillm {

struct RuntimeConfig {
    std::string model_path;
    std::size_t context_tokens = 2048;
    std::size_t page_tokens = 16;
    std::size_t max_sequences = 16;
    std::size_t batch_tokens = 256;
    std::size_t threads = 8;
    KernelMode kernels = KernelMode::automatic;
};

struct ModelDimensions {
    std::size_t embedding;
    std::size_t layers;
    std::size_t heads;
    std::size_t kv_heads;
    std::size_t head_dim;
    std::size_t feed_forward;
    std::size_t vocabulary;
    std::size_t trained_context;
    float rms_epsilon;
    float rope_base;
};

struct InputToken {
    std::int32_t token;
    std::int32_t position;
    std::int32_t sequence;
    bool logits;
};

struct Logits {
    std::int32_t sequence;
    std::vector<float> values;
};

class Runtime {
public:
    explicit Runtime(RuntimeConfig config);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    const ModelDimensions& dimensions() const noexcept;
    const RuntimeConfig& config() const noexcept;
    std::vector<std::int32_t> tokenize(std::string_view text) const;
    std::string token_piece(std::int32_t token) const;
    bool is_eog(std::int32_t token) const;
    // 在测量前预留阶段记录；forward 会重用存储并保留调用方设置的 batch_id。
    ForwardProfile make_profile() const;
    std::vector<Logits> forward(std::span<const InputToken> tokens, ForwardProfile* profile = nullptr);
    void share_prefix(std::int32_t source, std::int32_t target, std::size_t tokens);
    void clear_sequence(std::int32_t sequence) noexcept;
    std::size_t used_kv_pages() const noexcept;
    std::size_t resident_kv_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace minillm
