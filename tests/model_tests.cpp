#include "test_support.h"
#include "../apps/options.h"

#include "llmserve/engine.h"
#include "minillm/runtime.h"
#include "ggml-backend.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>

using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

class Reference {
public:
    Reference(const std::string& path, int gpu_layers, int threads) {
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        model_.reset(llama_model_load_from_file(path.c_str(), mp));
        if (!model_) {
            throw std::runtime_error("reference model load failed");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_seq_max = 8;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.kv_unified = true;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_.reset(llama_init_from_model(model_.get(), cp));
        if (!context_) {
            throw std::runtime_error("reference context load failed");
        }
        batch_ = llama_batch_init(128, 0, 1);
        vocabulary_ = static_cast<std::size_t>(llama_vocab_n_tokens(llama_model_get_vocab(model_.get())));
    }
    ~Reference() {
        llama_batch_free(batch_);
        context_.reset();
        model_.reset();
        llama_backend_free();
    }
    void clear() {
        llama_synchronize(context_.get());
        llama_memory_clear(llama_get_memory(context_.get()), false);
    }
    std::vector<float> forward(const std::vector<std::int32_t>& tokens, std::int32_t start = 0) {
        CHECK(!tokens.empty() && tokens.size() <= 128);
        batch_.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            batch_.token[i] = tokens[i];
            batch_.pos[i] = start + static_cast<std::int32_t>(i);
            batch_.n_seq_id[i] = 1;
            batch_.seq_id[i][0] = 0;
            batch_.logits[i] = static_cast<std::int8_t>(i + 1 == tokens.size());
        }
        CHECK(llama_decode(context_.get(), batch_) == 0);
        const auto* values = llama_get_logits_ith(context_.get(), batch_.n_tokens - 1);
        CHECK(values);
        return {values, values + vocabulary_};
    }
private:
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
    llama_batch batch_{};
    std::size_t vocabulary_ = 0;
};

std::int32_t argmax(const std::vector<float>& values) {
    return static_cast<std::int32_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

json compare(const std::vector<float>& actual, const std::vector<float>& expected,
             double max_rmse, double max_absolute, double minimum_cosine) {
    CHECK(actual.size() == expected.size() && !actual.empty());
    double squared = 0, maximum = 0, dot = 0, an = 0, en = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        const auto error = static_cast<double>(actual[i]) - expected[i];
        squared += error * error;
        maximum = std::max(maximum, std::abs(error));
        dot += static_cast<double>(actual[i]) * expected[i];
        an += static_cast<double>(actual[i]) * actual[i];
        en += static_cast<double>(expected[i]) * expected[i];
    }
    const auto rmse = std::sqrt(squared / static_cast<double>(actual.size()));
    const auto cosine = dot / std::sqrt(an * en);
    std::cout << "logits rmse=" << rmse << " max_abs=" << maximum << " cosine=" << cosine << '\n';
    CHECK(rmse < max_rmse);
    CHECK(maximum < max_absolute);
    CHECK(cosine >= minimum_cosine);
    return {{"rmse", rmse}, {"max_abs", maximum}, {"cosine", cosine},
            {"argmax_equal", argmax(actual) == argmax(expected)}};
}

std::vector<float> prefill(minillm::Runtime& runtime, const std::vector<std::int32_t>& prompt,
                          std::int32_t sequence, std::size_t chunk) {
    std::vector<minillm::Logits> output;
    for (std::size_t i = 0; i < prompt.size();) {
        const auto end = std::min(prompt.size(), i + chunk);
        std::vector<minillm::InputToken> batch;
        for (; i < end; ++i) {
            batch.push_back({prompt[i], static_cast<std::int32_t>(i), sequence, i + 1 == prompt.size()});
        }
        output = runtime.forward(batch);
        CHECK(end == prompt.size() || output.empty());
    }
    CHECK(output.size() == 1);
    return output.front().values;
}

std::vector<std::int32_t> reference_generate(Reference& reference,
                                            const std::vector<std::int32_t>& prompt, int count) {
    reference.clear();
    auto logits = reference.forward(prompt);
    std::vector<std::int32_t> result;
    for (int i = 0; i < count; ++i) {
        const auto token = argmax(logits);
        result.push_back(token);
        if (i + 1 < count) {
            logits = reference.forward({token}, static_cast<std::int32_t>(prompt.size()) + i);
        }
    }
    return result;
}

std::vector<std::int32_t> collect(const std::shared_ptr<llmserve::RequestHandle>& request) {
    const auto deadline = llmserve::Clock::now() + 180s;
    std::vector<std::int32_t> tokens;
    while (llmserve::Clock::now() < deadline) {
        if (const auto event = request->next(100ms)) {
            if (event->token) {
                tokens.push_back(*event->token);
            }
            if (event->kind != llmserve::Event::Kind::token) {
                CHECK(event->kind == llmserve::Event::Kind::done);
                return tokens;
            }
        }
    }
    throw std::runtime_error("real model request timed out");
}

void write_report(const std::string& path, const json& report) {
    if (path.empty()) {
        return;
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot write model validation report");
    }
    stream << report.dump(2) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    json report{{"status", "failed"}, {"checks", json::array()},
        {"thresholds", {{"reference_rmse", 0.05}, {"reference_max_abs", 0.5},
                        {"reference_cosine", 0.9999}, {"same_runtime_max_abs", 0.00001}}}};
    std::string report_path;
    try {
        Options options(argc, argv, {"--model", "--reference-model", "--gpu-layers", "--threads", "--output"});
        report_path = options.get("--output");
        if (options.has("--help") || !options.has("--model")) {
            std::cout << "llmserve-model-tests --model MODEL.gguf --reference-model DEQUANTIZED_F32.gguf\n"
                         "                     [--gpu-layers 0] [--threads 8] [--output REPORT.json]\n";
            return options.has("--help") ? 0 : 1;
        }
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::cerr << text;
            }
        }, nullptr);
        const auto path = options.get("--model");
        const auto threads = static_cast<int>(options.integer("--threads", 8, 1, 256));
        const auto gpu_layers = static_cast<int>(options.integer("--gpu-layers", 0, 0, 10000));
        report["model_path"] = path;
        const auto reference_path = options.get("--reference-model", path);
        report["reference_model_path"] = reference_path;
        report["reference_gpu_layers"] = gpu_layers;
        report["kernel"] = minillm::kernel_name(minillm::KernelMode::automatic);
        Reference reference(reference_path, gpu_layers, threads);
        minillm::RuntimeConfig cfg{path, 512, 16, 8, 128, static_cast<std::size_t>(threads)};
        minillm::Runtime runtime(cfg);
        auto& checks = report["checks"];
        const std::vector<std::string> texts{
            "The capital of France is", "One plus one equals", "Memory is important because"};
        std::vector<std::vector<std::int32_t>> expected_generations;
        for (const auto& text : texts) {
            const auto prompt = runtime.tokenize(text);
            reference.clear();
            runtime.clear_sequence(0);
            const auto expected = reference.forward(prompt);
            const auto actual = prefill(runtime, prompt, 0, 128);
            const auto metrics = compare(actual, expected, 0.05, 0.5, 0.9999);
            CHECK(argmax(actual) == argmax(expected));
            checks.push_back({{"name", "teacher_forced_logits"}, {"prompt", text}, {"metrics", metrics}});
            expected_generations.push_back(reference_generate(reference, prompt, 8));
        }
        {
            auto scalar_cfg = cfg;
            scalar_cfg.kernels = minillm::KernelMode::scalar;
            minillm::Runtime scalar(scalar_cfg);
            const auto prompt = runtime.tokenize("Hello");
            runtime.clear_sequence(0);
            const auto metrics = compare(prefill(runtime, prompt, 0, 128),
                prefill(scalar, prompt, 0, 128), 0.02, 0.1, 0.99999);
            checks.push_back({{"name", "scalar_simd_logits"}, {"metrics", metrics}});
        }
        std::vector<std::int32_t> repeated;
        const auto unit = runtime.tokenize("Memory and scheduling use bounded resources. ");
        while (repeated.size() < 33) {
            repeated.insert(repeated.end(), unit.begin(), unit.end());
        }
        repeated.resize(33);
        runtime.clear_sequence(0);
        const auto full = prefill(runtime, repeated, 0, 128);
        for (const auto chunk : {1, 7, 16}) {
            runtime.clear_sequence(1);
            const auto metrics = compare(prefill(runtime, repeated, 1, chunk), full, 1e-6, 1e-5, 0.9999999);
            checks.push_back({{"name", "chunk_boundary_logits"}, {"chunk", chunk}, {"metrics", metrics}});
        }
        runtime.clear_sequence(1);
        runtime.share_prefix(0, 1, 17);
        const auto continuation = runtime.tokenize(" test").back();
        const minillm::InputToken branch{continuation, 17, 1, true};
        const auto shared = runtime.forward({&branch, 1}).front().values;
        std::vector<std::int32_t> branched(repeated.begin(), repeated.begin() + 17);
        branched.push_back(continuation);
        runtime.clear_sequence(2);
        const auto fresh = prefill(runtime, branched, 2, 128);
        checks.push_back({{"name", "paged_tail_copy_on_write_logits"},
            {"metrics", compare(shared, fresh, 1e-6, 1e-5, 0.9999999)}});
        for (int sequence = 0; sequence < 8; ++sequence) {
            runtime.clear_sequence(sequence);
        }
        CHECK(runtime.used_kv_pages() == 0);
        checks.push_back({{"name", "physical_kv_reclaimed"}, {"used_pages", runtime.used_kv_pages()}});
        llmserve::EngineConfig ec;
        ec.context_tokens = 512;
        ec.max_model_len = 128;
        ec.max_active = 4;
        ec.batch_tokens = 16;
        ec.prefill_chunk = 4;
        ec.block_size = 4;
        ec.prefix_cache_tokens = 128;
        llmserve::ModelConfig mc{path, 0, threads};
        llmserve::Engine engine(ec, llmserve::make_mini_runner(mc, ec));
        std::vector<std::shared_ptr<llmserve::RequestHandle>> handles;
        for (std::size_t i = 0; i < texts.size(); ++i) {
            llmserve::RequestInput input;
            input.prompt = texts[i];
            input.max_tokens = 8;
            input.ignore_eos = true;
            input.timeout_ms = 180000;
            handles.push_back(engine.submit(input));
        }
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const auto actual = collect(handles[i]);
            report["generations"].push_back({{"prompt", texts[i]}, {"actual", actual},
                                             {"reference", expected_generations[i]}});
            CHECK(actual == expected_generations[i]);
        }
        llmserve::RequestInput cached;
        cached.prompt = texts[0];
        cached.max_tokens = 8;
        cached.ignore_eos = true;
        cached.timeout_ms = 180000;
        CHECK(collect(engine.submit(cached)) == expected_generations[0]);
        for (int i = 0; i < 100 && engine.statistics().active_requests; ++i) {
            std::this_thread::sleep_for(10ms);
        }
        const auto stats = engine.statistics();
        CHECK(stats.mixed_batches > 0);
        CHECK(stats.max_batch_sequences > 1);
        CHECK(stats.cache_hits > 0);
        CHECK(stats.kv_active_unique_blocks == 0);
        checks.push_back({{"name", "real_continuous_batching_and_prefix_cache"},
            {"mixed_batches", stats.mixed_batches}, {"max_batch_sequences", stats.max_batch_sequences},
            {"cache_hits", stats.cache_hits}, {"greedy_generations_matched", handles.size() + 1}});
        engine.stop();
        report["status"] = "passed";
        report["passed"] = checks.size();
        write_report(report_path, report);
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        write_report(report_path, report);
        std::cerr << report.dump(2) << '\n';
        return 1;
    }
}
