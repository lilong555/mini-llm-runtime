#include "test_support.h"
#include "qwen3_fixture.h"
#include "gated_runner.h"
#include "llmserve/engine.h"
#include "minillm/cuda/runtime.h"
#include "minillm/cuda/device_buffer.h"
#include "llama.h"
#include "../apps/options.h"
#include "../apps/cuda_reports.h"

#include <array>
#include <atomic>
#include <future>
#include <thread>

using namespace llmserve;
using namespace std::chrono_literals;
using minillm::cuda::CudaRuntime;
using cuda_reports::json;

#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
namespace {
std::atomic<bool> fail_completion{false};
std::atomic<std::size_t> h2d_bytes{0}, d2h_bytes{0}, event_records{0};
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const auto result = __real_cudaStreamSynchronize(stream);
    if (result == cudaSuccess && fail_completion.exchange(false)) { return cudaErrorUnknown; }
    return result;
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* to, const void* from, std::size_t bytes,
                                             cudaMemcpyKind kind, cudaStream_t stream) {
    const auto result = __real_cudaMemcpyAsync(to, from, bytes, kind, stream);
    if (result == cudaSuccess) {
        if (kind == cudaMemcpyHostToDevice) { h2d_bytes += bytes; }
        if (kind == cudaMemcpyDeviceToHost) { d2h_bytes += bytes; }
    }
    return result;
}
extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    ++event_records;
    return __real_cudaEventRecord(event, stream);
}
#endif

namespace {
std::string model_path(Qwen3Fixture& fixture, float scale = 1.0f) {
    return fixture.write(true, [](gguf_context* info) {
        const char* tokens[]{"a","b","c","d","e","f","g","<|endoftext|>","<|im_end|>"};
        const char* merges[]{"a b"};
        const std::int32_t types[]{1,1,1,1,1,1,1,3,3};
        gguf_set_val_str(info, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(info, "tokenizer.ggml.pre", "qwen2");
        gguf_set_arr_str(info, "tokenizer.ggml.tokens", tokens, 9);
        gguf_set_arr_str(info, "tokenizer.ggml.merges", merges, 1);
        gguf_set_arr_data(info, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types, 9);
        gguf_set_val_u32(info, "tokenizer.ggml.bos_token_id", 7);
        gguf_set_val_u32(info, "tokenizer.ggml.eos_token_id", 8);
        gguf_set_val_bool(info, "tokenizer.ggml.add_bos_token", false);
        gguf_set_val_bool(info, "tokenizer.ggml.add_eos_token", false);
    }, {}, {}, GGML_TYPE_F32, false, std::numeric_limits<float>::infinity(), scale);
}
EngineConfig config_for(std::size_t slots = 4) {
    EngineConfig config;
    config.max_active = slots;
    config.max_model_len = 16;
    config.context_tokens = slots * config.max_model_len;
    config.batch_tokens = 8;
    config.prefill_chunk = 2;
    config.block_size = 1;
    config.prefix_cache_entries = config.prefix_cache_tokens = 0;
    return config;
}
std::unique_ptr<ModelRunner> make_runner(const std::string& path, const EngineConfig& config) {
    return make_mini_cuda_runner({path, 0, 1}, config);
}
RequestInput request(std::vector<Token> prompt, std::size_t output = 4) {
    RequestInput input;
    input.prompt = std::move(prompt);
    input.max_tokens = output;
    input.ignore_eos = true;
    return input;
}
struct Collected {
    std::vector<Token> tokens;
    Event terminal;
};
Collected collect(const std::shared_ptr<RequestHandle>& handle) {
    Collected result;
    const auto deadline = Clock::now() + 30s;
    while (Clock::now() < deadline) {
        const auto event = handle->next(20ms);
        if (!event) { continue; }
        if (event->token) { result.tokens.push_back(*event->token); }
        if (event->kind != Event::Kind::token) {
            result.terminal = *event;
            CHECK(!handle->next(1ms));
            return result;
        }
    }
    throw std::runtime_error("CUDA Serving 请求没有有限终态");
}

std::vector<Token> reference_tokens(CudaRuntime& runtime, const std::vector<Token>& prompt,
                                    std::size_t count, std::size_t chunk) {
    runtime.clear_sequence(0);
    std::vector<Token> result;
    minillm::cuda::CudaForwardResult output;
    for (std::size_t position = 0; position < prompt.size();) {
        std::vector<minillm::InputToken> input;
        const auto end = std::min(prompt.size(), position + chunk);
        for (; position < end; ++position) {
            input.push_back({prompt[position], static_cast<std::int32_t>(position), 0, position + 1 == prompt.size()});
        }
        output = runtime.forward(input);
    }
    while (result.size() < count) {
        CHECK(output.samples.size() == 1 && output.samples[0].sequence == 0);
        result.push_back(output.samples[0].token);
        if (result.size() == count) { break; }
        output = runtime.forward(std::array<minillm::InputToken,1>{{
            {result.back(), static_cast<std::int32_t>(prompt.size() + result.size() - 1), 0, true}}});
    }
    runtime.clear_sequence(0);
    return result;
}

json check_engine_outputs(const std::string& path, EngineConfig config,
                          const std::vector<std::vector<Token>>& prompts,
                          const std::vector<std::vector<Token>>& expected) {
    config.telemetry_mode = TelemetryMode::stages;
    config.telemetry_capacity = 128;
    auto gate = std::make_shared<test::RunnerGate>();
    auto runner = make_runner(path, config);
    const auto resident = *runner->resources();
    Engine engine(config, config.max_active == 1 ? std::move(runner) :
        std::make_unique<test::GatedRunner>(std::move(runner), gate));
    std::atomic<bool> tokenizer_bad{false};
    std::atomic<std::size_t> tokenizations{0};
    const auto text = "a";
    const auto tokenized = engine.tokenize(text);
    std::jthread tokenizer([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            try {
                if (engine.tokenize(text) != tokenized) { tokenizer_bad = true; }
            } catch (...) { tokenizer_bad = true; }
            ++tokenizations;
            std::this_thread::sleep_for(1ms);
        }
    });
    std::vector<Collected> outputs;
    if (config.max_active == 1) {
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            outputs.push_back(collect(engine.submit(request(prompts[i], expected[i].size()))));
        }
    } else {
        std::vector<std::shared_ptr<RequestHandle>> handles;
        try {
            handles.push_back(engine.submit(request(prompts[0], expected[0].size())));
            gate->wait_until_sampled();
            for (std::size_t i = 1; i < prompts.size(); ++i) {
                handles.push_back(engine.submit(request(prompts[i], expected[i].size())));
            }
        } catch (...) {
            gate->release();
            throw;
        }
        gate->release();
        for (const auto& handle : handles) { outputs.push_back(collect(handle)); }
        CHECK(engine.statistics().mixed_batches > 0);
        CHECK(engine.statistics().max_batch_sequences == 4);
    }
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        CHECK(outputs[i].terminal.status == 200 && outputs[i].terminal.usage.cached_tokens == 0);
        CHECK(outputs[i].tokens == expected[i]);
    }
    const auto reused = collect(engine.submit(request(prompts[1], expected[1].size())));
    CHECK(reused.terminal.status == 200 && reused.tokens == expected[1]);
    engine.stop();
    tokenizer.request_stop();
    tokenizer.join();
    CHECK(!tokenizer_bad && tokenizations > 0);
    const auto stats = engine.statistics();
    CHECK(stats.kv_used_blocks == 0 && stats.active_requests == 0 && stats.outstanding_requests == 0);
    CHECK(stats.resources && stats.resources->live_tokens == 0 && stats.resources->state_valid && stats.resources->reusable);
    CHECK(stats.resources->resident_kv_payload_bytes == resident.resident_kv_payload_bytes);
    CHECK(stats.resources->owned_device_bytes == resident.owned_device_bytes);
    const auto& capture = engine.telemetry();
    CHECK(capture.dropped == 0 && capture.recorded == stats.batches);
    std::size_t mapped_samples = 0;
    for (std::size_t i = 0; i < capture.recorded; ++i) {
        const auto& batch = capture.batches[i];
        CHECK(batch.completed && batch.runner_completed && !batch.runner.available);
        CHECK(batch.resources_before && batch.resources_after);
        CHECK(!batch.resources_before->live_kv_pages && !batch.resources_after->live_kv_pages);
        CHECK(*batch.resources_after->live_tokens == *batch.resources_before->live_tokens +
              batch.prefill_tokens + batch.decode_tokens);
        for (std::size_t j = 0; j < batch.sequences; ++j) {
            const auto& slice = batch.slices[j];
            CHECK(slice.sequence >= 0 && static_cast<std::size_t>(slice.sequence) < config.max_active);
            CHECK(slice.emitted == (slice.logits_tokens == 1));
            CHECK(slice.sampled_token.has_value() == slice.emitted);
            mapped_samples += slice.emitted ? 1 : 0;
        }
    }
    std::size_t output_count = reused.tokens.size();
    json generations = json::array();
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        output_count += outputs[i].tokens.size();
        generations.push_back({{"input_token_ids", prompts[i]}, {"token_ids", outputs[i].tokens}});
    }
    CHECK(mapped_samples == output_count && stats.generated_tokens == output_count);
    return {{"max_sequences", config.max_active}, {"batches", stats.batches}, {"mixed_batches", stats.mixed_batches},
        {"max_batch_sequences", stats.max_batch_sequences}, {"mapped_samples", mapped_samples},
        {"tokenizations", tokenizations.load()}, {"live_tokens_after_stop", stats.resources->live_tokens},
        {"capacity_tokens", resident.capacity_tokens}, {"resident_kv_payload_bytes", resident.resident_kv_payload_bytes},
        {"owned_device_bytes", resident.owned_device_bytes}, {"generations", generations},
        {"slot_reuse", true}, {"stage_profile_available", false}};
}
}

TEST(cuda_runner_mapping_compact_copy_and_ready_clear) {
    Qwen3Fixture fixture;
    const auto path = model_path(fixture);
    const auto config = config_for();
    CudaRuntime reference({path, 0, 4, 16, 8, 0});
    const std::vector<minillm::InputToken> input{{1,0,3,false},{2,0,0,true},{3,1,3,true},{4,1,0,false}};
    const auto expected = reference.forward(input);
    auto runner = make_runner(path, config);
    CHECK(runner->info().model_load_ns > 0 && runner->info().storage_initialization_ns > 0);
    CHECK(runner->info().weight_decode_upload_ns > 0 &&
          runner->info().weight_decode_upload_ns <= runner->info().storage_initialization_ns);
    const auto caps = runner->capabilities();
    CHECK(caps.max_sequences == 4 && caps.max_batch_tokens == 8 && caps.max_model_len == 16);
    CHECK(!caps.prefix_copy && !caps.runtime_stage_profile && caps.synchronous_execute);
    CHECK(runner->tokenize("a") == std::vector<Token>{0});
    CHECK(runner->token_piece(0) == "a" && runner->is_eog(8));
    const auto before = *runner->resources();
    CHECK(before.layout == KvLayout::contiguous && !before.live_kv_pages);
    CHECK(before.capacity_tokens == 64 && before.live_tokens == 0);
    CHECK(before.resident_kv_payload_bytes > 0 && before.owned_device_bytes > before.resident_kv_payload_bytes);
    const auto allocated = minillm::cuda::allocation_stats();
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
    h2d_bytes = d2h_bytes = event_records = 0;
#endif
    RunnerTelemetry profile;
    const std::vector<BatchToken> batch{{1,0,3,false},{2,0,0,true},{3,1,3,true},{4,1,0,false}};
    const auto actual = runner->execute_profiled(batch, profile);
    CHECK(!profile.available && actual.size() == expected.samples.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(actual[i].sequence == expected.samples[i].sequence && actual[i].token == expected.samples[i].token);
    }
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
    CHECK(h2d_bytes == 56 && d2h_bytes == 16 && event_records == 0);
#endif
    CHECK(runner->resources()->live_tokens == 4);
    runner->clear_sequence(0);
    runner->clear_sequence(3);
    CHECK(runner->resources()->live_tokens == 0 && runner->resources()->reusable);
    CHECK(runner->resources()->resident_kv_payload_bytes == before.resident_kv_payload_bytes);
    CHECK(runner->resources()->owned_device_bytes == before.owned_device_bytes);
    CHECK(minillm::cuda::allocation_stats().allocation_calls == allocated.allocation_calls);
    const auto reused = runner->execute(std::array<BatchToken,1>{{{5,0,3,true}}});
    reference.clear_sequence(3);
    const auto isolated = reference.forward(std::array<minillm::InputToken,1>{{{5,0,3,true}}});
    CHECK(reused[0].token == isolated.samples[0].token && reused[0].sequence == 3);
    test::throws<std::logic_error>([&] { runner->copy_sequence(3, 2, 1); });
}

TEST(cuda_runner_contract_errors_quarantine_without_device_work) {
    Qwen3Fixture fixture;
    const auto path = model_path(fixture);
    for (const auto& invalid : std::vector<std::vector<BatchToken>>{
        {}, {{1,0,4,true}}, {{1,0,-1,true}}, {{1,0,0,true},{2,1,0,true}},
        {{-1,0,0,true}}, {{9,0,0,true}}, {{1,1,0,true}}}) {
        auto runner = make_runner(path, config_for());
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
        h2d_bytes = d2h_bytes = event_records = 0;
#endif
        test::throws<std::invalid_argument>([&] { runner->execute(invalid); });
        runner->clear_sequence(0);
        runner->synchronize();
        const auto state = *runner->resources();
        CHECK(!state.state_valid && !state.reusable && !state.live_tokens && !state.live_kv_pages);
        CHECK(state.resident_kv_payload_bytes > 0 && state.owned_device_bytes > 0);
        test::throws<std::runtime_error>([&] { runner->execute(std::array<BatchToken,1>{{{1,0,0,true}}}); });
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
        CHECK(h2d_bytes == 0 && d2h_bytes == 0 && event_records == 0);
#endif
    }
}

TEST(cuda_runner_invalid_noexcept_clear_is_not_healthy) {
    Qwen3Fixture fixture;
    auto runner = make_runner(model_path(fixture), config_for(1));
    runner->clear_sequence(1);
    runner->synchronize();
    CHECK(!runner->resources()->reusable && !runner->resources()->state_valid);
    CHECK(!runner->resources()->live_tokens);
}

TEST(cuda_engine_mixed_join_reuse_and_concurrent_tokenizer) {
    Qwen3Fixture fixture;
    const auto path = model_path(fixture);
    const std::vector<std::vector<Token>> prompts{{1,2,3}, {2,4,5}, {6,1}, {3,2,4,1}};
    std::vector<std::vector<Token>> expected;
    {
        CudaRuntime reference({path, 0, 1, 16, 8, 0});
        for (const auto& prompt : prompts) { expected.push_back(reference_tokens(reference, prompt, 4, 2)); }
    }
    check_engine_outputs(path, config_for(1), prompts, expected);
    check_engine_outputs(path, config_for(4), prompts, expected);
}

TEST(cuda_engine_admission_boundary_and_clear) {
    Qwen3Fixture fixture;
    const auto config = config_for(1);
    Engine engine(config, make_runner(model_path(fixture), config));
    const auto boundary = collect(engine.submit(request(std::vector<Token>(15, 1), 1)));
    CHECK(boundary.tokens.size() == 1 && boundary.terminal.status == 200);
    test::throws<RequestError>([&] { engine.submit(request(std::vector<Token>(16, 1), 1)); });
    test::throws<RequestError>([&] { engine.submit(request({9}, 1)); });
    CHECK(collect(engine.submit(request({2}, 1))).terminal.status == 200);
    engine.stop();
    CHECK(engine.statistics().resources->live_tokens == 0 && engine.statistics().kv_used_blocks == 0);
}

TEST(cuda_runner_nonfinite_engine_failure_has_single_terminal) {
    Qwen3Fixture fixture;
    auto runner = make_runner(model_path(fixture, 1e30f), config_for());
    auto* observed = runner.get();
    Engine engine(config_for(), std::move(runner));
    const auto handle = engine.submit(request({1,2,3}));
    const auto output = collect(handle);
    CHECK(output.tokens.empty() && output.terminal.error_code == "backend_error");
    engine.stop();
    CHECK(engine.statistics().kv_used_blocks == 0 && !engine.statistics().ready);
    CHECK(!observed->resources()->state_valid && !observed->resources()->live_tokens);
    CHECK(observed->resources()->resident_kv_payload_bytes > 0);
    test::throws<RequestError>([&] { engine.submit(request({1})); });
}

#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
TEST(cuda_runner_post_launch_failure_retains_resident_until_owner_destruction) {
    Qwen3Fixture fixture;
    const auto before = minillm::cuda::allocation_stats();
    {
        auto runner = make_runner(model_path(fixture), config_for());
        runner->execute(std::array<BatchToken,1>{{{1,0,0,false}}});
        CHECK(runner->resources()->live_tokens == 1);
        const auto resident = runner->resources()->owned_device_bytes;
        fail_completion = true;
        test::throws<minillm::cuda::Error>([&] {
            runner->execute(std::array<BatchToken,1>{{{2,1,0,true}}});
        });
        CHECK(!fail_completion);
        for (SequenceId slot = 0; slot < 4; ++slot) { runner->clear_sequence(slot); }
        runner->synchronize();
        CHECK(!runner->resources()->state_valid && !runner->resources()->reusable);
        CHECK(!runner->resources()->live_tokens && runner->resources()->owned_device_bytes == resident);
        CHECK(minillm::cuda::allocation_stats().releases == before.releases);
    }
    const auto after = minillm::cuda::allocation_stats();
    CHECK(after.allocations - before.allocations == after.releases - before.releases);
}
#endif

int main(int argc, char** argv) {
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::cerr << text; }
    }, nullptr);
    json report{{"schema_version", 1}, {"spec_id", "CUDA-SERVE-001"}, {"status", "failed"},
                {"backend", "minillm-cuda"}, {"checks", json::array()}};
    std::filesystem::path output;
    try {
        Options options(argc, argv, {"--model", "--contract", "--output"});
        if (options.has("--help")) {
            std::cout << "llmserve-cuda-serving-tests [--model MODEL --contract JSON --output NEW_REPORT.json]\n";
            return 0;
        }
        if (!options.has("--model")) { return test::run(); }
        output = options.get("--output");
        if (output.empty() || std::filesystem::exists(output)) {
            output.clear();
            throw std::invalid_argument("真实 CUDA Serving 验证需要不存在的 --output 文件");
        }
        const auto contract_path = options.get("--contract");
        std::ifstream contract_file(contract_path);
        const auto contract = json::parse(contract_file);
        const auto path = options.get("--model");
        report["model_sha256"] = cuda_reports::file_hash(path);
        CHECK(report["model_sha256"] == contract.at("model").at("sha256"));
        report["contract_sha256"] = cuda_reports::file_hash(contract_path);
        report["binary_sha256"] = cuda_reports::file_hash(argv[0]);
        std::vector<std::vector<Token>> prompts, expected;
        {
            CudaRuntime reference({path, 0, 1, 2048, 128, 0});
            report["device"] = cuda_reports::device(reference.device_info());
            report["arithmetic"] = cuda_reports::arithmetic(reference);
            for (const auto& item : contract.at("stable_greedy")) {
                prompts.push_back(item.at("input_token_ids").get<std::vector<Token>>());
                CHECK(reference.tokenize(item.at("text").get<std::string>()) == prompts.back());
                expected.push_back(reference_tokens(reference, prompts.back(), 8, 2));
                CHECK(expected.back() == item.at("expected_token_ids").get<std::vector<Token>>());
            }
        }
        CHECK(prompts.size() == 3);
        prompts.push_back(prompts[0]);
        expected.push_back(expected[0]);
        for (const std::size_t slots : {1, 4}) {
            auto config = config_for(slots);
            config.max_model_len = 2048;
            config.context_tokens = slots * config.max_model_len;
            config.batch_tokens = 128;
            config.block_size = 16;
            report["checks"].push_back(check_engine_outputs(path, config, prompts, expected));
        }
        report["status"] = "passed";
        report["passed"] = report["checks"].size();
        cuda_reports::write(output, report);
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        if (!output.empty()) { cuda_reports::write(output, report); }
        std::cerr << report.dump(2) << '\n';
        return 1;
    }
}
