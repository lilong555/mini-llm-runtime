#include "test_support.h"
#include "qwen3_fixture.h"
#include "minillm/cuda/runtime.h"
#include "minillm/cuda/device_buffer.h"
#include "minillm/runtime.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cstring>

using namespace minillm;
using namespace minillm::cuda;

#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
namespace {
bool fail_completion = false;
std::size_t observed_h2d = 0, observed_d2h = 0, completions = 0;
}
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    ++completions;
    const auto result = __real_cudaStreamSynchronize(stream);
    if (result == cudaSuccess && fail_completion) {
        fail_completion = false;
        return cudaErrorUnknown;
    }
    return result;
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* to, const void* from, std::size_t bytes,
                                             cudaMemcpyKind kind, cudaStream_t stream) {
    const auto result = __real_cudaMemcpyAsync(to,from,bytes,kind,stream);
    if (result == cudaSuccess) {
        if (kind == cudaMemcpyHostToDevice) { observed_h2d += bytes; }
        if (kind == cudaMemcpyDeviceToHost) { observed_d2h += bytes; }
    }
    return result;
}
#endif

namespace {
std::string model_path(Qwen3Fixture& fixture, float scale = 1.0f) {
    return fixture.write(true,[](gguf_context* info) {
        const char* tokens[]{"a","b","c","d","e","f","g","<|endoftext|>","<|im_end|>"};
        const char* merges[]{"a b"};
        const std::int32_t types[]{1,1,1,1,1,1,1,3,3};
        gguf_set_val_str(info,"tokenizer.ggml.model","gpt2");
        gguf_set_val_str(info,"tokenizer.ggml.pre","qwen2");
        gguf_set_arr_str(info,"tokenizer.ggml.tokens",tokens,9);
        gguf_set_arr_str(info,"tokenizer.ggml.merges",merges,1);
        gguf_set_arr_data(info,"tokenizer.ggml.token_type",GGUF_TYPE_INT32,types,9);
        gguf_set_val_u32(info,"tokenizer.ggml.bos_token_id",7);
        gguf_set_val_u32(info,"tokenizer.ggml.eos_token_id",8);
        gguf_set_val_bool(info,"tokenizer.ggml.add_bos_token",false);
        gguf_set_val_bool(info,"tokenizer.ggml.add_eos_token",false);
    },{},{},GGML_TYPE_F32,false,std::numeric_limits<float>::infinity(),scale);
}
CudaRuntimeConfig config_for(const std::string& path) { return {path,0,4,16,8,0}; }
std::int32_t best(const std::vector<float>& values) {
    return static_cast<std::int32_t>(std::max_element(values.begin(),values.end())-values.begin());
}
void near(const std::vector<float>& actual, const std::vector<float>& expected) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        CHECK(std::abs(double(actual[i])-expected[i]) <= 2e-4 + 2e-4*std::abs(double(expected[i])));
    }
}
void unchanged_allocations(AllocationStats before) {
    const auto after = allocation_stats();
    CHECK(after.allocation_calls == before.allocation_calls && after.release_calls == before.release_calls);
}
}

TEST(runtime_manifest_tokenizer_and_memory) {
    Qwen3Fixture fixture;
    CudaRuntime runtime(config_for(model_path(fixture)));
    CHECK(runtime.dimensions().vocabulary == 9 && runtime.dimensions().layers == 2);
    CHECK(runtime.tokenize("a") == std::vector<std::int32_t>{0});
    CHECK(runtime.token_piece(0) == "a" && runtime.is_eog(8));
    test::throws<std::invalid_argument>([&] { runtime.token_piece(-1); });
    test::throws<std::invalid_argument>([&] { runtime.is_eog(9); });
    const auto manifest = runtime.weight_manifest();
    CHECK(manifest.size() == 25 && manifest[0].name == "token_embd.weight");
    CHECK(manifest[2].alias_of == manifest[0].name && manifest[2].offset == manifest[0].offset);
    CHECK(manifest[2].effective_sha256 == manifest[0].effective_sha256 && manifest[0].effective_sha256.size() == 64);
    CHECK(std::all_of(manifest.begin(),manifest.end(),[](const auto& w) { return w.source_dtype == "F32"; }));
    const auto d = runtime.diagnostics();
    CHECK(d.state == CudaRuntimeState::ready && d.sequence_lengths == std::vector<std::size_t>(4,0));
    CHECK(d.resident.total_owned_bytes == d.owned_device_bytes && d.owned_device_allocations == 4);
    CHECK(d.resident.weights_bytes+d.resident.workspace_bytes+d.resident.kv_bytes+
          d.resident.library_workspace_bytes == d.owned_device_bytes);
    CHECK(d.rope_h2d_bytes == 16*4*sizeof(float) && d.rope_h2d_bytes == d.resident.rope_bytes);
    CHECK(d.weight_h2d_bytes > 0 && d.metadata_h2d_bytes == 0 && d.token_d2h_bytes == 0 && d.debug_d2h_bytes == 0);
    CHECK(d.model_load_ns > 0 && d.storage_initialization_ns > 0);
    CHECK(!runtime.device_info().name.empty() && runtime.device_info().uuid.size() == 36);
    CHECK(runtime.device_info().compute_major > 0 && runtime.device_info().cublas_version > 0);
}

TEST(runtime_full_forward_matches_cpu_and_output_selection) {
    Qwen3Fixture fixture;
    const auto path = model_path(fixture);
    CudaRuntime runtime(config_for(path));
    Runtime cpu({path,16,1,4,8,1,KernelMode::scalar});
    const std::vector<InputToken> batch{{1,0,3,false},{2,0,0,true},{3,1,3,true},{4,1,0,false}};
    const auto expected = cpu.forward(batch);
    const auto before = runtime.diagnostics();
    const auto allocations = allocation_stats();
    const auto actual = runtime.forward(batch,CudaOutputMode::debug_logits);
    CHECK(actual.samples.size() == 2 && actual.logits.size() == 2 && actual.host_forward_to_token_ns > 0);
    CHECK(!actual.device_elapsed_ms);
    CHECK(actual.samples[0].input_index == 1 && actual.samples[0].sequence == 0);
    CHECK(actual.samples[1].input_index == 2 && actual.samples[1].sequence == 3);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(actual.logits[i].sequence == expected[i].sequence && actual.samples[i].token == best(expected[i].values));
        near(actual.logits[i].values,expected[i].values);
    }
    unchanged_allocations(allocations);
    const auto after = runtime.diagnostics();
    CHECK(after.sequence_lengths == (std::vector<std::size_t>{2,0,0,2}) && after.live_kv_tokens == 4);
    CHECK(after.weight_h2d_bytes == before.weight_h2d_bytes && after.rope_h2d_bytes == before.rope_h2d_bytes);
    CHECK(after.metadata_h2d_bytes-before.metadata_h2d_bytes == (3*4+2)*sizeof(std::int32_t));
    CHECK(after.token_d2h_bytes-before.token_d2h_bytes == 2*sizeof(std::int32_t));
    CHECK(after.status_d2h_bytes-before.status_d2h_bytes == 2*sizeof(std::int32_t));
    CHECK(after.debug_d2h_bytes-before.debug_d2h_bytes == 2*9*sizeof(float));
    CHECK(after.intermediate_h2d_bytes == 0 && after.intermediate_d2h_bytes == 0);
    runtime.clear_sequence(0); runtime.clear_sequence(3);
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
    observed_h2d = observed_d2h = completions = 0;
#endif
    const auto greedy = runtime.forward(batch);
    CHECK(greedy.logits.empty() && greedy.samples.size() == 2);
    for (std::size_t i = 0; i < greedy.samples.size(); ++i) { CHECK(greedy.samples[i].token == actual.samples[i].token); }
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
    CHECK(observed_h2d == 56 && observed_d2h == 16 && completions == 1);
#endif
    CHECK(runtime.diagnostics().debug_d2h_bytes == after.debug_d2h_bytes);
    unchanged_allocations(allocations);
}

TEST(runtime_no_logits_and_timing_equivalence) {
    Qwen3Fixture fixture;
    CudaRuntime runtime(config_for(model_path(fixture)));
    const std::array<InputToken,2> batch{{{1,0,0,false},{2,1,0,false}}};
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
    observed_h2d = observed_d2h = completions = 0;
#endif
    const auto empty = runtime.forward(batch);
    CHECK(empty.samples.empty() && empty.logits.empty() && runtime.diagnostics().live_kv_tokens == 2);
#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
    CHECK(observed_h2d == 24 && observed_d2h == 8 && completions == 1);
#endif
    const std::array<InputToken,1> last{{{3,2,0,true}}};
    const auto off = runtime.forward(last,CudaOutputMode::debug_logits);
    runtime.clear_sequence(0);
    CHECK(runtime.forward(batch).samples.empty());
    const auto on = runtime.forward(last,CudaOutputMode::debug_logits,true);
    CHECK(on.device_elapsed_ms && *on.device_elapsed_ms >= 0 && !off.device_elapsed_ms);
    CHECK(on.samples[0].token == off.samples[0].token);
    CHECK(std::memcmp(on.logits[0].values.data(),off.logits[0].values.data(),9*sizeof(float)) == 0);
}

TEST(runtime_preflight_is_recoverable) {
    Qwen3Fixture fixture;
    CudaRuntime runtime(config_for(model_path(fixture)));
    runtime.forward(std::array<InputToken,1>{{{1,0,0,true}}});
    const auto before = runtime.diagnostics();
    const auto allocations = allocation_stats();
    for (const auto& invalid : std::vector<std::vector<InputToken>>{
        {}, {{1,1,4,true}}, {{1,1,-1,true}}, {{-1,1,0,true}}, {{9,1,0,true}},
        {{1,16,0,true}}, {{1,-1,0,true}}, {{1,2,0,true}},
        {{1,1,0,false},{2,1,0,true}}, std::vector<InputToken>(9,{1,1,0,true})}) {
        test::throws<std::invalid_argument>([&] { runtime.forward(invalid); });
        const auto after = runtime.diagnostics();
        CHECK(after.state == CudaRuntimeState::ready && after.sequence_lengths == before.sequence_lengths);
        CHECK(after.metadata_h2d_bytes == before.metadata_h2d_bytes && after.status_d2h_bytes == before.status_d2h_bytes);
        CHECK(after.completed_forwards == before.completed_forwards && after.post_launch_failures == 0);
    }
    test::throws<std::invalid_argument>([&] {
        runtime.forward(std::array<InputToken,1>{{{1,1,0,true}}},static_cast<CudaOutputMode>(99));
    });
    test::throws<std::invalid_argument>([&] { runtime.clear_sequence(-1); });
    test::throws<std::invalid_argument>([&] { runtime.clear_sequence(4); });
    CHECK(runtime.forward(std::array<InputToken,1>{{{2,1,0,true}}}).samples.size() == 1);
    unchanged_allocations(allocations);
}

TEST(runtime_context_clear_reuse_and_owner) {
    Qwen3Fixture fixture;
    auto config = config_for(model_path(fixture));
    config.max_sequences = 1; config.max_model_len = 4; config.batch_tokens = 4;
    const auto before = allocation_stats();
    {
        CudaRuntime runtime(config);
        const std::array<InputToken,4> full{{{1,0,0,false},{2,1,0,false},{3,2,0,false},{4,3,0,true}}};
        runtime.forward(full);
        test::throws<std::invalid_argument>([&] { runtime.forward(std::array<InputToken,1>{{{1,4,0,true}}}); });
        CHECK(runtime.diagnostics().live_kv_tokens == 4 && runtime.diagnostics().state == CudaRuntimeState::ready);
        runtime.clear_sequence(0);
        CHECK(runtime.diagnostics().live_kv_tokens == 0 && runtime.diagnostics().live_sequences == 0);
        const auto reused = runtime.forward(std::array<InputToken,1>{{{5,0,0,true}}},CudaOutputMode::debug_logits);
        CudaRuntime fresh(config);
        const auto expected = fresh.forward(std::array<InputToken,1>{{{5,0,0,true}}},CudaOutputMode::debug_logits);
        CHECK(reused.logits[0].values == expected.logits[0].values);
    }
    const auto after = allocation_stats();
    CHECK(after.allocations-before.allocations == 8 && after.releases-before.releases == 8);
}

TEST(runtime_nonfinite_is_fail_stop) {
    Qwen3Fixture fixture;
    const auto before = allocation_stats();
    {
        CudaRuntime runtime(config_for(model_path(fixture,1e30f)));
        const std::array<InputToken,1> batch{{{1,0,0,true}}};
        test::throws<Error>([&] { runtime.forward(batch,CudaOutputMode::debug_logits); });
        auto d = runtime.diagnostics();
        CHECK(d.state == CudaRuntimeState::poisoned && d.live_kv_tokens == 0 && d.completed_forwards == 0);
        CHECK(d.post_launch_failures == 1);
        test::throws<Error>([&] { runtime.forward(batch); });
        test::throws<Error>([&] { runtime.clear_sequence(0); });
        CHECK(runtime.diagnostics().post_launch_failures == 1);
    }
    const auto after = allocation_stats();
    CHECK(after.allocations-before.allocations == after.releases-before.releases);
}

TEST(runtime_initialization_failure_releases_resources) {
    Qwen3Fixture fixture;
    auto config = config_for(model_path(fixture));
    config.device_budget_bytes = 1;
    const auto before = allocation_stats();
    test::throws<Error>([&] { CudaRuntime runtime(config); });
    const auto after = allocation_stats();
    CHECK(after.allocations-before.allocations == after.releases-before.releases);
    config.max_sequences = 5;
    test::throws<std::invalid_argument>([&] { CudaRuntime runtime(config); });
    config.max_sequences = 1; config.max_model_len = 0;
    test::throws<std::invalid_argument>([&] { CudaRuntime runtime(config); });
    auto oversized = config_for("不存在的模型文件");
    oversized.batch_tokens = 129;
    test::throws<std::invalid_argument>([&] { CudaRuntime runtime(oversized); });
}

#ifdef MINILLM_TEST_CUDA_COMPLETION_FAILURE
TEST(runtime_checked_completion_failure_is_fail_stop) {
    Qwen3Fixture fixture;
    CudaRuntime runtime(config_for(model_path(fixture)));
    runtime.forward(std::array<InputToken,1>{{{1,0,0,false}}});
    const auto allocations = allocation_stats();
    fail_completion = true;
    test::throws<Error>([&] { runtime.forward(std::array<InputToken,1>{{{2,1,0,true}}},CudaOutputMode::debug_logits,true); });
    CHECK(!fail_completion);
    const auto d = runtime.diagnostics();
    CHECK(d.state == CudaRuntimeState::poisoned && d.sequence_lengths[0] == 1);
    CHECK(d.completed_forwards == 1 && d.post_launch_failures == 1);
    test::throws<Error>([&] { runtime.clear_sequence(0); });
    test::throws<Error>([&] { runtime.forward(std::array<InputToken,1>{{{2,1,0,true}}}); });
    unchanged_allocations(allocations);
}
#endif

int main() {
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_ERROR) { std::cerr << text; }
    },nullptr);
    return test::run();
}
