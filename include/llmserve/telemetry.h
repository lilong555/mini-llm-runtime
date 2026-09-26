#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace llmserve {

enum class TelemetryMode { off, batches, stages };

constexpr std::string_view telemetry_mode_name(TelemetryMode mode) noexcept {
    switch (mode) {
    case TelemetryMode::off: return "off";
    case TelemetryMode::batches: return "batches";
    case TelemetryMode::stages: return "stages";
    }
    return "unknown";
}

enum class KvLayout { unknown, paged, contiguous };

constexpr std::string_view kv_layout_name(KvLayout layout) noexcept {
    switch (layout) {
    case KvLayout::unknown: return "unknown";
    case KvLayout::paged: return "paged";
    case KvLayout::contiguous: return "contiguous";
    }
    return "unknown";
}

// 仅表达后端直接提供的物理状态；容量信用、进程 RSS 和共享引用另有口径。
struct RunnerResources {
    std::optional<std::size_t> live_kv_pages = std::nullopt;
    std::size_t resident_kv_payload_bytes = 0;
    KvLayout layout = KvLayout::unknown;
    std::optional<std::size_t> capacity_tokens = std::nullopt;
    std::optional<std::size_t> live_tokens = std::nullopt;
    std::optional<std::size_t> owned_device_bytes = std::nullopt;
    bool state_valid = true;
    bool reusable = true;
};

struct RunnerStage {
    std::string_view name;
    std::size_t calls = 0;
    std::size_t matrix_m = 0;
    std::size_t matrix_n = 0;
    std::size_t matrix_k = 0;
    bool varying_shape = false;
    std::uint64_t wall_ns = 0;
    std::uint64_t parallel_wall_ns = 0;
    std::uint64_t caller_wait_ns = 0;
    std::uint64_t worker_work_sum_ns = 0;
};

// 跨层按固定阶段汇总；字段不依赖具体 Runtime 或上游计算图类型。
struct RunnerTelemetry {
    bool available = false;
    bool completed = false;
    std::uint64_t forward_ns = 0;
    std::uint64_t unaccounted_ns = 0;
    std::uint64_t sampling_ns = 0;
    std::array<RunnerStage, 32> stages{};
};

struct TokenTelemetry {
    std::uint64_t batch_id = 0;
    std::uint64_t request_order = 0;
    std::size_t token_index = 0;
    std::uint64_t engine_elapsed_ns = 0;
};

struct SliceTelemetry {
    std::array<char, 129> request_id{};
    std::uint64_t request_order = 0;
    std::int32_t sequence = 0;
    bool prefill = false;
    std::size_t tokens = 0;
    std::size_t context_before = 0;
    std::size_t logits_tokens = 0;
    std::size_t token_index = 0;
    std::optional<std::int32_t> sampled_token;
    bool emitted = false;
    std::uint64_t emitted_ns = 0;
};

struct BatchTelemetry {
    std::uint64_t batch_id = 0;
    bool completed = false;
    bool runner_completed = false;
    std::uint64_t start_ns = 0;
    std::uint64_t admission_ns = 0;
    std::uint64_t scheduler_ns = 0;
    std::uint64_t prepare_ns = 0;
    std::uint64_t runner_start_ns = 0;
    std::uint64_t runner_ns = 0;
    std::uint64_t finish_ns = 0;
    std::size_t waiting_requests = 0;
    std::size_t active_requests = 0;
    std::size_t prefill_tokens = 0;
    std::size_t decode_tokens = 0;
    std::size_t logits_tokens = 0;
    std::size_t sequences = 0;
    std::size_t context_before_sum = 0;
    std::size_t context_before_max = 0;
    std::size_t context_after_sum = 0;
    std::size_t context_after_max = 0;
    std::size_t reserved_unique_blocks = 0;
    std::optional<RunnerResources> resources_before;
    std::optional<RunnerResources> resources_after;
    RunnerTelemetry runner;
    // 启动时按 max_active 分配，仅前 sequences 个元素有效。
    std::vector<SliceTelemetry> slices;
};

struct TelemetryCapture {
    TelemetryMode mode = TelemetryMode::off;
    std::size_t recorded = 0;
    std::uint64_t dropped = 0;
    std::size_t storage_bytes = 0;
    std::optional<RunnerResources> resources_final;
    // 启动时分配，采集期间不扩容。只可在 Engine::stop() 后读取。
    std::vector<BatchTelemetry> batches;
};

} // namespace llmserve
