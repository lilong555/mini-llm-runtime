#include "llmserve/model_runner.h"

#include "minillm/runtime.h"

#include <algorithm>
#include <cmath>
#include <chrono>
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
                 engine.context_tokens, runtime_.dimensions().vocabulary, false, model.threads,
                 model.gpu_layers, model.scalar_kernels ? "scalar" : "auto"};
        if (model.gpu_layers != 0) {
            throw std::invalid_argument("MiniLLM is a CPU SIMD runtime; GPU offload belongs to --backend llama");
        }
        if (engine.max_model_len > runtime_.dimensions().trained_context) {
            throw std::invalid_argument("max_model_len exceeds the trained context");
        }
        if (engine.telemetry_mode == TelemetryMode::stages) { profile_ = runtime_.make_profile(); }
    }
    const ModelInfo& info() const noexcept override { return info_; }
    BackendCapabilities capabilities() const noexcept override {
        const auto& config = runtime_.config();
        return {config.max_sequences, config.batch_tokens,
                std::min(config.context_tokens, runtime_.dimensions().trained_context), true, true, true};
    }
    std::vector<Token> tokenize(std::string_view text) const override { return runtime_.tokenize(text); }
    std::string token_piece(Token token) const override { return runtime_.token_piece(token); }
    bool is_eog(Token token) const override { return runtime_.is_eog(token); }
    std::vector<Sample> execute(std::span<const BatchToken> batch) override {
        return execute_impl(batch, nullptr);
    }
    std::vector<Sample> execute_profiled(std::span<const BatchToken> batch,
                                         RunnerTelemetry& profile) override {
        profile = {};
        profile.available = true;
        return execute_impl(batch, &profile);
    }
    std::optional<RunnerResources> resources() const noexcept override {
        return RunnerResources{runtime_.used_kv_pages(), runtime_.resident_kv_bytes(),
                               KvLayout::paged, runtime_.config().context_tokens};
    }
    void copy_sequence(SequenceId source, SequenceId target, std::size_t tokens) override {
        runtime_.share_prefix(source, target, tokens);
    }
    void clear_sequence(SequenceId sequence) noexcept override { runtime_.clear_sequence(sequence); }
    void synchronize() noexcept override {}
private:
    std::vector<Sample> execute_impl(std::span<const BatchToken> batch, RunnerTelemetry* profile) {
        std::vector<minillm::InputToken> input;
        input.reserve(batch.size());
        for (const auto& token : batch) {
            input.push_back({token.token, token.position, token.sequence, token.logits});
        }
        auto logits = runtime_.forward(input, profile ? &profile_ : nullptr);
        const auto sampling_start = profile ? std::chrono::steady_clock::now() :
                                              std::chrono::steady_clock::time_point{};
        std::vector<Sample> result;
        for (const auto& output : logits) {
            if (output.values.empty() || !std::all_of(output.values.begin(), output.values.end(),
                                                     [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("MiniLLM produced invalid logits");
            }
            const auto best = std::max_element(output.values.begin(), output.values.end());
            result.push_back({output.sequence, static_cast<Token>(best - output.values.begin())});
        }
        if (profile) {
            profile->sampling_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - sampling_start).count());
            profile->forward_ns = profile_.wall_ns;
            profile->unaccounted_ns = profile_.unaccounted_ns;
            for (const auto& stage : profile_.stages) {
                auto& total = profile->stages[static_cast<std::size_t>(stage.stage)];
                total.name = minillm::profile_stage_name(stage.stage);
                if (total.calls > 0 && (total.matrix_m != stage.matrix_m ||
                    total.matrix_n != stage.matrix_n || total.matrix_k != stage.matrix_k)) {
                    total.varying_shape = true;
                }
                ++total.calls;
                total.matrix_m = stage.matrix_m;
                total.matrix_n = stage.matrix_n;
                total.matrix_k = stage.matrix_k;
                total.wall_ns += stage.wall_ns;
                total.parallel_wall_ns += stage.parallel.wall_ns;
                total.caller_wait_ns += stage.parallel.caller_wait_ns;
                total.worker_work_sum_ns += stage.parallel.worker_work_sum_ns;
            }
            profile->completed = profile_.completed;
        }
        return result;
    }
    minillm::Runtime runtime_;
    minillm::ForwardProfile profile_;
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
