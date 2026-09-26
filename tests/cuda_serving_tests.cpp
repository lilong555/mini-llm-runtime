#include "test_support.h"
#include "qwen3_fixture.h"
#include "gated_runner.h"
#include "llmserve/engine.h"
#include "minillm/cuda/runtime.h"
#include "minillm/cuda/device_buffer.h"
#include "llama.h"

#include <array>
#include <atomic>
#include <future>
#include <thread>

using namespace llmserve;
using namespace std::chrono_literals;
using minillm::cuda::CudaRuntime;

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
}

TEST(cuda_runner_mapping_compact_copy_and_ready_clear) {
    Qwen3Fixture fixture;
    const auto path = model_path(fixture);
    const auto config = config_for();
    CudaRuntime reference({path, 0, 4, 16, 8, 0});
    const std::vector<minillm::InputToken> input{{1,0,3,false},{2,0,0,true},{3,1,3,true},{4,1,0,false}};
    const auto expected = reference.forward(input);
    auto runner = make_runner(path, config);
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

int main() {
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::cerr << text; }
    }, nullptr);
    return test::run();
}
