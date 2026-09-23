#include "llmserve/model_runner.h"

#include "ggml-backend.h"
#include "llama.h"

#include <array>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llmserve {
namespace {

struct BackendLifetime {
    BackendLifetime() {
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~BackendLifetime() { llama_backend_free(); }
};

class LlamaRunner final : public ModelRunner {
public:
    LlamaRunner(const ModelConfig& model_config, const EngineConfig& engine) {
        engine.validate();
        if (model_config.path.empty() || model_config.threads <= 0 || model_config.gpu_layers < 0) {
            throw std::invalid_argument("invalid model path, threads, or GPU layer count");
        }
        static BackendLifetime backend;
        info_.backend = "llama.cpp";
        info_.model = std::filesystem::path(model_config.path).stem().string();
        info_.device = "CPU";
        info_.threads = model_config.threads;
        info_.gpu_layers = model_config.gpu_layers;
        info_.kernel_mode = "upstream";
        for (std::size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto device = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                model_config.gpu_layers > 0) {
                info_.device = ggml_backend_dev_description(device);
                info_.gpu = true;
                break;
            }
        }
        if (model_config.gpu_layers > 0 && !info_.gpu) {
            throw std::runtime_error("GPU offload requested but no GPU backend is available; use --gpu-layers 0 for CPU");
        }
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = model_config.gpu_layers;
        model_.reset(llama_model_load_from_file(model_config.path.c_str(), model_params));
        if (!model_) {
            throw std::runtime_error("cannot load GGUF model");
        }
        std::array<char, 128> architecture{};
        llama_model_meta_val_str(model_.get(), "general.architecture", architecture.data(),
                                 architecture.size());
        info_.architecture = architecture.data();
        if (info_.architecture != "qwen3" && info_.architecture != "llama") {
            throw std::invalid_argument("only full-attention qwen3 and llama decoder models are supported");
        }
        if (engine.max_model_len > static_cast<std::size_t>(llama_model_n_ctx_train(model_.get()))) {
            throw std::invalid_argument("max_model_len exceeds the model's trained context");
        }
        vocab_ = llama_model_get_vocab(model_.get());
        info_.vocab_size = static_cast<std::size_t>(llama_vocab_n_tokens(vocab_));
        auto params = llama_context_default_params();
        params.n_ctx = static_cast<std::uint32_t>(engine.context_tokens);
        params.n_batch = static_cast<std::uint32_t>(engine.batch_tokens);
        params.n_ubatch = params.n_batch;
        params.n_seq_max = static_cast<std::uint32_t>(engine.max_active + engine.prefix_cache_entries);
        params.n_threads = model_config.threads;
        params.n_threads_batch = model_config.threads;
        params.kv_unified = true;
        params.swa_full = true;
        params.offload_kqv = info_.gpu;
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.flash_attn_type = model_config.flash_attention
            ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.no_perf = false;
        context_.reset(llama_init_from_model(model_.get(), params));
        if (!context_) {
            throw std::runtime_error("cannot allocate llama.cpp context");
        }
        info_.context_tokens = llama_n_ctx(context_.get());
        if (info_.context_tokens < engine.context_tokens) {
            throw std::runtime_error("backend context is smaller than the KV credit budget");
        }
        batch_ = llama_batch_init(static_cast<std::int32_t>(engine.batch_tokens), 0, 1);
        batch_capacity_ = engine.batch_tokens;
        batch_initialized_ = true;
        sampler_.reset(llama_sampler_init_greedy());
        if (!sampler_) {
            throw std::runtime_error("cannot create greedy sampler");
        }
    }

    ~LlamaRunner() override {
        synchronize();
        if (batch_initialized_) {
            llama_batch_free(batch_);
        }
    }

    const ModelInfo& info() const noexcept override { return info_; }

    std::vector<Token> tokenize(std::string_view text) const override {
        if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - 8)) {
            throw std::invalid_argument("prompt is too large");
        }
        std::vector<Token> tokens(text.size() + 8);
        auto count = llama_tokenize(vocab_, text.data(), static_cast<std::int32_t>(text.size()),
            tokens.data(), static_cast<std::int32_t>(tokens.size()), true, true);
        if (count < 0) {
            tokens.resize(static_cast<std::size_t>(-count));
            count = llama_tokenize(vocab_, text.data(), static_cast<std::int32_t>(text.size()),
                tokens.data(), static_cast<std::int32_t>(tokens.size()), true, true);
        }
        if (count < 0) {
            throw std::runtime_error("tokenization failed");
        }
        tokens.resize(static_cast<std::size_t>(count));
        return tokens;
    }

    std::string token_piece(Token token) const override {
        std::array<char, 256> buffer{};
        auto count = llama_token_to_piece(vocab_, token, buffer.data(),
                                         static_cast<std::int32_t>(buffer.size()), 0, false);
        if (count >= 0) {
            return {buffer.data(), static_cast<std::size_t>(count)};
        }
        std::string result(static_cast<std::size_t>(-count), '\0');
        count = llama_token_to_piece(vocab_, token, result.data(),
                                     static_cast<std::int32_t>(result.size()), 0, false);
        if (count < 0) {
            throw std::runtime_error("token detokenization failed");
        }
        result.resize(static_cast<std::size_t>(count));
        return result;
    }

    bool is_eog(Token token) const override { return llama_vocab_is_eog(vocab_, token); }

    std::vector<Sample> execute(std::span<const BatchToken> tokens) override {
        if (tokens.empty() || tokens.size() > batch_capacity_) {
            throw std::invalid_argument("invalid batch size");
        }
        batch_.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            batch_.token[i] = tokens[i].token;
            batch_.pos[i] = tokens[i].position;
            batch_.n_seq_id[i] = 1;
            batch_.seq_id[i][0] = tokens[i].sequence;
            batch_.logits[i] = static_cast<std::int8_t>(tokens[i].logits);
        }
        const auto status = llama_decode(context_.get(), batch_);
        if (status != 0) {
            synchronize();
            throw std::runtime_error("llama_decode failed with status " + std::to_string(status));
        }
        std::vector<Sample> samples;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].logits) {
                samples.push_back({tokens[i].sequence,
                    llama_sampler_sample(sampler_.get(), context_.get(), static_cast<std::int32_t>(i))});
            }
        }
        // Sequence aliases and resource reclamation are only changed after GPU work completes.
        synchronize();
        return samples;
    }

    void copy_sequence(SequenceId source, SequenceId target, std::size_t tokens) override {
        llama_memory_seq_cp(llama_get_memory(context_.get()), source, target, 0,
                            static_cast<llama_pos>(tokens));
    }

    void clear_sequence(SequenceId sequence) noexcept override {
        if (context_) {
            llama_memory_seq_rm(llama_get_memory(context_.get()), sequence, -1, -1);
        }
    }

    void synchronize() noexcept override {
        if (context_) {
            llama_synchronize(context_.get());
        }
    }

private:
    ModelInfo info_;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context_{nullptr, llama_free};
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_{nullptr, llama_sampler_free};
    const llama_vocab* vocab_ = nullptr;
    llama_batch batch_{};
    std::size_t batch_capacity_ = 0;
    bool batch_initialized_ = false;
};

} // namespace

std::unique_ptr<ModelRunner> make_llama_runner(const ModelConfig& model,
                                             const EngineConfig& engine) {
    return std::make_unique<LlamaRunner>(model, engine);
}

} // namespace llmserve
