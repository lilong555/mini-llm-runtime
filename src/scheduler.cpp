#include "llmserve/scheduler.h"

#include <algorithm>

namespace llmserve {

std::int64_t effective_priority(int priority, std::int64_t wait_ms, int aging_ms) {
    return priority + std::max<std::int64_t>(0, wait_ms) / aging_ms;
}

BatchPlan schedule_batch(std::span<const ScheduleItem> items, const EngineConfig& config) {
    std::vector<ScheduleItem> prefill;
    std::vector<ScheduleItem> decode;
    for (const auto& item : items) {
        (item.prefill_remaining ? prefill : decode).push_back(item);
    }
    const auto before = [&](const ScheduleItem& left, const ScheduleItem& right) {
        const auto lp = effective_priority(left.priority, left.wait_ms, config.aging_ms);
        const auto rp = effective_priority(right.priority, right.wait_ms, config.aging_ms);
        return lp != rp ? lp > rp : left.order < right.order;
    };
    std::stable_sort(prefill.begin(), prefill.end(), before);
    std::stable_sort(decode.begin(), decode.end(), before);
    BatchPlan plan;
    auto remaining = config.batch_tokens;
    const auto add_decode = [&] {
        for (const auto& item : decode) {
            if (remaining == 0) {
                break;
            }
            plan.slices.push_back({item.key, 1, false});
            ++plan.decode_tokens;
            --remaining;
        }
    };
    if (config.policy == SchedulingPolicy::mixed || prefill.empty()) {
        add_decode();
    }
    for (const auto& item : prefill) {
        const auto count = std::min({remaining, item.prefill_remaining, config.prefill_chunk});
        if (count == 0) {
            break;
        }
        plan.slices.push_back({item.key, count, true});
        plan.prefill_tokens += count;
        remaining -= count;
    }
    return plan;
}

} // namespace llmserve
