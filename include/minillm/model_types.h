#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minillm {

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

} // namespace minillm
