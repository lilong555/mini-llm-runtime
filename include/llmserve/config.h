#pragma once

#include "llmserve/telemetry.h"

#include <cstddef>
#include <string>

namespace llmserve {

enum class SchedulingPolicy { mixed, prefill_first };

struct EngineConfig {
    std::size_t context_tokens = 8192;
    std::size_t max_model_len = 2048;
    std::size_t max_active = 8;
    std::size_t queue_capacity = 64;
    std::size_t batch_tokens = 256;
    std::size_t prefill_chunk = 128;
    std::size_t block_size = 16;
    std::size_t prefix_cache_entries = 4;
    std::size_t prefix_cache_tokens = 2048;
    std::size_t event_buffer_size = 128;
    int aging_ms = 250;
    int admission_reserve_ms = 2000;
    SchedulingPolicy policy = SchedulingPolicy::mixed;
    TelemetryMode telemetry_mode = TelemetryMode::off;
    std::size_t telemetry_capacity = 1024;

    void validate() const;
};

struct ModelConfig {
    std::string path;
    int gpu_layers = 99;
    int threads = 8;
    bool flash_attention = true;
    bool scalar_kernels = false;
    int device = 0;
    std::size_t device_budget_bytes = 0;
};

std::string policy_name(SchedulingPolicy policy);
// 不加载模型或设备，用于 factory 与 CPU-only 启动预检。
void validate_mini_cuda_config(const ModelConfig& model, const EngineConfig& engine);

} // namespace llmserve
