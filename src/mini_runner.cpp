#include "llmserve/model_runner.h"

#include "minillm/runtime.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace llmserve {
namespace {

class MiniRunner final : public ModelRunner {
public:
    MiniRunner(const ModelConfig& model, const EngineConfig& engine)
        : runtime_({model.path, engine.context_tokens, engine.block_size,
                    engine.max_active + engine.prefix_cache_entries, engine.batch_tokens,
                    static_cast<std::size_t>(model.threads),
                    model.scalar_kernels ? minillm::KernelMode::scalar : minillm::KernelMode::automatic}) {
        info_ = {"minillm", std::filesystem::path(model.path).stem().string(), "qwen3",
                 "CPU/" + std::string(minillm::kernel_name(runtime_.config().kernels)),
                 engine.context_tokens, runtime_.dimensions().vocabulary, false};
        if (model.gpu_layers != 0) {
            throw std::invalid_argument("MiniLLM is a CPU SIMD runtime; GPU offload belongs to --backend llama");
        }
        if (engine.max_model_len > runtime_.dimensions().trained_context) {
            throw std::invalid_argument("max_model_len exceeds the trained context");
        }
    }
    const ModelInfo& info() const noexcept override { return info_; }
    std::vector<Token> tokenize(std::string_view text) const override { return runtime_.tokenize(text); }
    std::string token_piece(Token token) const override { return runtime_.token_piece(token); }
    bool is_eog(Token token) const override { return runtime_.is_eog(token); }
    std::vector<Sample> execute(std::span<const BatchToken> batch) override {
        std::vector<minillm::InputToken> input;
        input.reserve(batch.size());
        for (const auto& token : batch) {
            input.push_back({token.token, token.position, token.sequence, token.logits});
        }
        auto logits = runtime_.forward(input);
        std::vector<Sample> result;
        for (const auto& output : logits) {
            if (output.values.empty() || !std::all_of(output.values.begin(), output.values.end(),
                                                     [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("MiniLLM produced invalid logits");
            }
            const auto best = std::max_element(output.values.begin(), output.values.end());
            result.push_back({output.sequence, static_cast<Token>(best - output.values.begin())});
        }
        return result;
    }
    void copy_sequence(SequenceId source, SequenceId target, std::size_t tokens) override {
        runtime_.share_prefix(source, target, tokens);
    }
    void clear_sequence(SequenceId sequence) noexcept override { runtime_.clear_sequence(sequence); }
    void synchronize() noexcept override {}
private:
    minillm::Runtime runtime_;
    ModelInfo info_;
};

} // namespace

std::unique_ptr<ModelRunner> make_mini_runner(const ModelConfig& model, const EngineConfig& engine) {
    engine.validate();
    if (model.gpu_layers != 0 || model.threads <= 0 || model.threads > 256 || model.path.empty()) {
        throw std::invalid_argument("MiniLLM requires a model path, CPU execution, and 1..256 threads");
    }
    return std::make_unique<MiniRunner>(model, engine);
}

} // namespace llmserve
