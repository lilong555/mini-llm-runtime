#include "llmserve/config.h"

#include <limits>
#include <stdexcept>

namespace llmserve {

void EngineConfig::validate() const {
    if (telemetry_capacity == 0 || telemetry_capacity > 16384 ||
        (telemetry_mode != TelemetryMode::off && telemetry_mode != TelemetryMode::batches &&
         telemetry_mode != TelemetryMode::stages)) {
        throw std::invalid_argument("invalid telemetry mode or capacity (1..16384)");
    }
    if (block_size == 0 || context_tokens == 0 || context_tokens % block_size != 0) {
        throw std::invalid_argument("context_tokens must be a positive multiple of block_size");
    }
    if (max_model_len < 2 || max_model_len > context_tokens) {
        throw std::invalid_argument("max_model_len must be in [2, context_tokens]");
    }
    if (max_active == 0 || max_active > 128 || batch_tokens < max_active ||
        batch_tokens > context_tokens) {
        throw std::invalid_argument("require 1 <= max_active <= 128 and max_active <= batch_tokens <= context_tokens");
    }
    if (prefill_chunk == 0 || prefill_chunk > batch_tokens || queue_capacity == 0 ||
        queue_capacity > 65536 || event_buffer_size == 0 || event_buffer_size > 65536) {
        throw std::invalid_argument("invalid prefill, queue, or event buffer limit");
    }
    if (prefix_cache_entries > 128 || prefix_cache_tokens > context_tokens ||
        (prefix_cache_entries > 0 && prefix_cache_tokens < block_size)) {
        throw std::invalid_argument("invalid prefix cache capacity");
    }
    if (aging_ms <= 0 || admission_reserve_ms <= 0 ||
        context_tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid aging interval or context size");
    }
}

std::string policy_name(SchedulingPolicy policy) {
    return policy == SchedulingPolicy::mixed ? "mixed" : "prefill_first";
}

} // namespace llmserve
