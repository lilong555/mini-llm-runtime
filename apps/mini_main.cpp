#include "options.h"

#include "minillm/gguf_model.h"
#include "minillm/runtime.h"
#include "llmserve/utf8.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <filesystem>

using json = nlohmann::json;

int main(int argc, char** argv) {
    try {
        Options options(argc, argv, {"--model", "--prompt", "--tokens", "--threads", "--kernel",
                                    "--context", "--chunk", "--dequantize-ref"},
                        {"--help", "--inspect", "--ignore-eos"});
        if (options.has("--help") || !options.has("--model")) {
            std::cout << "mini-llm --model MODEL.gguf [--prompt TEXT] [--tokens 32]\n"
                         "         [--threads 8] [--kernel auto|scalar] [--context 2048]\n"
                         "         [--chunk 32] [--ignore-eos] [--inspect]\n"
                         "         [--dequantize-ref OUTPUT.gguf]\n";
            return options.has("--help") ? 0 : 1;
        }
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::cerr << text;
            }
        }, nullptr);
        if (options.has("--dequantize-ref")) {
            const auto output = options.get("--dequantize-ref");
            if (output.empty() || std::filesystem::exists(output)) {
                throw std::invalid_argument("reference output must be a new file");
            }
            auto params = llama_model_quantize_default_params();
            params.ftype = LLAMA_FTYPE_ALL_F32;
            params.output_tensor_type = GGML_TYPE_F32;
            params.token_embedding_type = GGML_TYPE_F32;
            params.allow_requantize = true;
            params.pure = true;
            params.nthread = static_cast<std::int32_t>(options.integer("--threads", 8, 1, 256));
            params.max_buf_size = 256 * 1024 * 1024;
            if (llama_model_quantize(options.get("--model").c_str(), output.c_str(), &params) != 0) {
                throw std::runtime_error("reference weight dequantization failed");
            }
            std::cout << json{{"source", options.get("--model")}, {"reference", output},
                {"weight_type", "F32"}, {"converter", "llama.cpp"}}.dump(2) << '\n';
            return 0;
        }
        if (options.has("--inspect")) {
            minillm::GgufModel model(options.get("--model"));
            json metadata;
            for (const auto* name : {"qwen3.embedding_length", "qwen3.block_count",
                     "qwen3.attention.head_count", "qwen3.attention.head_count_kv",
                     "qwen3.attention.key_length", "qwen3.attention.value_length",
                     "qwen3.feed_forward_length", "qwen3.context_length", "qwen3.rope.dimension_count"}) {
                if (model.contains(name)) {
                    metadata[name] = model.integer_value(name);
                }
            }
            std::cout << json{{"gguf_version", model.version()}, {"bytes", model.file_bytes()},
                {"tensors", model.tensor_count()}, {"architecture", model.string_value("general.architecture")},
                {"kernel", minillm::kernel_name(minillm::KernelMode::automatic)},
                {"dimensions", metadata}}.dump(2) << '\n';
            return 0;
        }
        minillm::RuntimeConfig config;
        config.model_path = options.get("--model");
        config.threads = static_cast<std::size_t>(options.integer("--threads", 8, 1, 256));
        config.context_tokens = static_cast<std::size_t>(options.integer("--context", 2048, 16, 32768));
        config.batch_tokens = std::min<std::size_t>(256, config.context_tokens);
        const auto mode = options.get("--kernel", "auto");
        if (mode != "auto" && mode != "scalar") {
            throw std::invalid_argument("--kernel must be auto or scalar");
        }
        config.kernels = mode == "scalar" ? minillm::KernelMode::scalar : minillm::KernelMode::automatic;
        const auto chunk = static_cast<std::size_t>(options.integer("--chunk", 32, 1,
            static_cast<std::int64_t>(config.batch_tokens)));
        const auto count = static_cast<std::size_t>(options.integer("--tokens", 32, 1, 4096));
        minillm::Runtime runtime(config);
        auto prompt = runtime.tokenize(options.get("--prompt", "The capital of France is"));
        if (prompt.empty() || prompt.size() + count > config.context_tokens) {
            throw std::invalid_argument("prompt and output exceed the context budget");
        }
        const auto start = std::chrono::steady_clock::now();
        std::vector<minillm::Logits> logits;
        for (std::size_t pos = 0; pos < prompt.size();) {
            std::vector<minillm::InputToken> batch;
            const auto end = std::min(prompt.size(), pos + chunk);
            for (; pos < end; ++pos) {
                batch.push_back({prompt[pos], static_cast<std::int32_t>(pos), 0, pos + 1 == prompt.size()});
            }
            logits = runtime.forward(batch);
        }
        llmserve::Utf8Buffer text;
        std::string output;
        std::vector<std::int32_t> generated;
        double ttft_ms = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& values = logits.at(0).values;
            const auto token = static_cast<std::int32_t>(std::max_element(values.begin(), values.end()) - values.begin());
            generated.push_back(token);
            if (i == 0) {
                ttft_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
            }
            const auto eog = runtime.is_eog(token) && !options.has("--ignore-eos");
            if (!eog) {
                output += text.append(runtime.token_piece(token));
            }
            if (eog || i + 1 == count) {
                break;
            }
            const minillm::InputToken next{token, static_cast<std::int32_t>(prompt.size() + i), 0, true};
            logits = runtime.forward({&next, 1});
        }
        output += text.finish();
        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const auto peak_pages = runtime.used_kv_pages();
        runtime.clear_sequence(0);
        std::cout << json{{"backend", "minillm"}, {"kernel", minillm::kernel_name(config.kernels)},
            {"text", output}, {"token_ids", generated}, {"prompt_tokens", prompt.size()},
            {"completion_tokens", generated.size()}, {"ttft_ms", ttft_ms},
            {"elapsed_s", elapsed}, {"output_tokens_per_second", generated.size() / elapsed},
            {"used_kv_pages_before_release", peak_pages},
            {"used_kv_pages_after_release", runtime.used_kv_pages()}}.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mini-llm: " << error.what() << '\n';
        return 1;
    }
}
