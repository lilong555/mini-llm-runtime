#pragma once

#include "llmserve/engine.h"
#include <nlohmann/json.hpp>
#include <ostream>

inline void write_telemetry(std::ostream& output, const llmserve::Engine& engine) {
    using json = nlohmann::json;
    const auto& capture = engine.telemetry();
    const auto resources = [](const std::optional<llmserve::RunnerResources>& value) -> json {
        if (!value) { return nullptr; }
        return {{"live_kv_pages", value->live_kv_pages ? json(*value->live_kv_pages) : json(nullptr)},
                {"resident_kv_payload_bytes", value->resident_kv_payload_bytes}};
    };
    output << json{{"type", "header"}, {"schema_version", 1},
        {"clock", "engine_relative_steady_ns"}, {"mode", llmserve::telemetry_mode_name(capture.mode)},
        {"policy", llmserve::policy_name(engine.config().policy)},
        {"backend", engine.model_info().backend}, {"capacity", capture.batches.size()},
        {"storage_bytes", capture.storage_bytes}, {"runtime_replay_available", false}}.dump() << '\n';
    for (std::size_t index = 0; index < capture.recorded; ++index) {
        const auto& batch = capture.batches[index];
        json slices = json::array();
        for (std::size_t i = 0; i < batch.sequences; ++i) {
            const auto& slice = batch.slices[i];
            slices.push_back({{"request_id", slice.request_id.data()}, {"request_order", slice.request_order},
                {"sequence", slice.sequence}, {"prefill", slice.prefill}, {"tokens", slice.tokens},
                {"context_before", slice.context_before}, {"logits_tokens", slice.logits_tokens},
                {"token_index", slice.token_index},
                {"sampled_token", slice.sampled_token ? json(*slice.sampled_token) : json(nullptr)},
                {"emitted", slice.emitted}, {"emitted_ns", slice.emitted_ns}});
        }
        json runner = nullptr;
        if (batch.runner.available) {
            json stages = json::array();
            for (const auto& stage : batch.runner.stages) {
                if (stage.calls == 0) { continue; }
                stages.push_back({{"name", stage.name}, {"calls", stage.calls},
                    {"matrix_m", stage.matrix_m}, {"matrix_n", stage.matrix_n}, {"matrix_k", stage.matrix_k},
                    {"varying_shape", stage.varying_shape}, {"wall_ns", stage.wall_ns},
                    {"parallel_wall_ns", stage.parallel_wall_ns}, {"caller_wait_ns", stage.caller_wait_ns},
                    {"worker_work_sum_ns", stage.worker_work_sum_ns}});
            }
            runner = {{"completed", batch.runner.completed}, {"forward_ns", batch.runner.forward_ns},
                {"unaccounted_ns", batch.runner.unaccounted_ns}, {"sampling_ns", batch.runner.sampling_ns},
                {"stages", stages}};
        }
        output << json{{"type", "batch"}, {"batch_id", batch.batch_id}, {"completed", batch.completed},
            {"runner_completed", batch.runner_completed}, {"start_ns", batch.start_ns},
            {"admission_ns", batch.admission_ns}, {"scheduler_ns", batch.scheduler_ns},
            {"prepare_ns", batch.prepare_ns}, {"runner_start_ns", batch.runner_start_ns},
            {"runner_ns", batch.runner_ns}, {"finish_ns", batch.finish_ns},
            {"waiting_requests", batch.waiting_requests}, {"active_requests", batch.active_requests},
            {"prefill_tokens", batch.prefill_tokens}, {"decode_tokens", batch.decode_tokens},
            {"logits_tokens", batch.logits_tokens}, {"sequences", batch.sequences},
            {"context_before_sum", batch.context_before_sum}, {"context_before_max", batch.context_before_max},
            {"context_after_sum", batch.context_after_sum}, {"context_after_max", batch.context_after_max},
            {"reserved_unique_blocks", batch.reserved_unique_blocks},
            {"resources_before", resources(batch.resources_before)},
            {"resources_after", resources(batch.resources_after)}, {"runner", runner}, {"slices", slices}}.dump() << '\n';
    }
    output << json{{"type", "footer"}, {"recorded", capture.recorded}, {"dropped", capture.dropped},
        {"complete", capture.dropped == 0}, {"engine_error", engine.statistics().last_error},
        {"resources_final", resources(capture.resources_final)}}.dump() << '\n';
    output.flush();
    if (!output) { throw std::runtime_error("cannot write telemetry capture"); }
}
