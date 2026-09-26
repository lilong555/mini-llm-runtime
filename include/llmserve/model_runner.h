#pragma once

#include "llmserve/config.h"
#include "llmserve/prefix_index.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmserve {

struct BatchToken {
    Token token;
    std::int32_t position;
    SequenceId sequence;
    bool logits;
};

struct Sample {
    SequenceId sequence;
    Token token;
};

struct ModelInfo {
    std::string backend;
    std::string model;
    std::string architecture;
    std::string device;
    std::size_t context_tokens = 0;
    std::size_t vocab_size = 0;
    bool gpu = false;
    int threads = 0;
    int gpu_layers = 0;
    std::string kernel_mode;
    std::optional<std::uint64_t> model_load_ns = std::nullopt;
    std::optional<std::uint64_t> storage_initialization_ns = std::nullopt;
    std::optional<std::uint64_t> weight_decode_upload_ns = std::nullopt;
};

struct BackendCapabilities {
    // 0 表示未提供该上限，不表示零容量；sequence 容量包含 prefix 槽。
    std::size_t max_sequences = 0;
    std::size_t max_batch_tokens = 0;
    std::size_t max_model_len = 0;
    bool prefix_copy = true;
    bool runtime_stage_profile = false;
    bool synchronous_execute = true;
};

class ModelRunner {
public:
    virtual ~ModelRunner() = default;
    virtual const ModelInfo& info() const noexcept = 0;
    virtual BackendCapabilities capabilities() const noexcept { return {}; }
    virtual std::vector<Token> tokenize(std::string_view text) const = 0;
    virtual std::string token_piece(Token token) const = 0;
    virtual bool is_eog(Token token) const = 0;
    virtual std::vector<Sample> execute(std::span<const BatchToken> batch) = 0;
    virtual std::vector<Sample> execute_profiled(std::span<const BatchToken> batch,
                                                RunnerTelemetry& profile) {
        profile = {};
        return execute(batch);
    }
    virtual std::optional<RunnerResources> resources() const noexcept { return std::nullopt; }
    virtual void copy_sequence(SequenceId source, SequenceId target, std::size_t tokens) = 0;
    virtual void clear_sequence(SequenceId sequence) noexcept = 0;
    virtual void synchronize() noexcept = 0;
};

std::unique_ptr<ModelRunner> make_llama_runner(const ModelConfig& model,
                                             const EngineConfig& engine);
std::unique_ptr<ModelRunner> make_mini_runner(const ModelConfig& model,
                                            const EngineConfig& engine);
std::unique_ptr<ModelRunner> make_mini_cuda_runner(const ModelConfig& model,
                                                 const EngineConfig& engine);

} // namespace llmserve
