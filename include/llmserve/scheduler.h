#pragma once

#include "llmserve/config.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace llmserve {

struct ScheduleItem {
    std::size_t key;
    std::size_t prefill_remaining;
    int priority;
    std::int64_t wait_ms;
    std::uint64_t order;
};

struct ScheduledSlice {
    std::size_t key;
    std::size_t tokens;
    bool prefill;
};

struct BatchPlan {
    std::vector<ScheduledSlice> slices;
    std::size_t prefill_tokens = 0;
    std::size_t decode_tokens = 0;
    std::size_t token_count() const { return prefill_tokens + decode_tokens; }
    bool mixed() const { return prefill_tokens > 0 && decode_tokens > 0; }
};

std::int64_t effective_priority(int priority, std::int64_t wait_ms, int aging_ms);
BatchPlan schedule_batch(std::span<const ScheduleItem> items, const EngineConfig& config);

} // namespace llmserve
