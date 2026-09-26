#include "llmserve/model_runner.h"

#include "minillm/cuda/runtime.h"

#include <array>
#include <filesystem>
#include <stdexcept>

namespace llmserve {
namespace {

class MiniCudaRunner final : public ModelRunner {
public:
    MiniCudaRunner(const ModelConfig& model, const EngineConfig& engine)
        : runtime_({model.path, model.device, engine.max_active, engine.max_model_len,
                    engine.batch_tokens, model.device_budget_bytes}) {
        info_ = {"minillm-cuda", std::filesystem::path(model.path).stem().string(), "qwen3",
                 runtime_.device_info().name, engine.context_tokens, runtime_.dimensions().vocabulary,
                 true, model.threads, 0, "cuda-f32"};
        input_.reserve(engine.batch_tokens);
        samples_.reserve(engine.max_active);
        // 固定内存计划仅在初始化时读取；热路径资源查询不复制 diagnostics 中的 vector。
        const auto initial = runtime_.diagnostics();
        resident_ = {std::nullopt, initial.resident.kv_bytes, KvLayout::contiguous,
                     initial.kv_capacity_tokens, 0, initial.owned_device_bytes};
    }

    const ModelInfo& info() const noexcept override { return info_; }
    BackendCapabilities capabilities() const noexcept override {
        const auto& config = runtime_.config();
        return {config.max_sequences, config.batch_tokens, config.max_model_len, false, false, true};
    }
    std::vector<Token> tokenize(std::string_view text) const override { return runtime_.tokenize(text); }
    std::string token_piece(Token token) const override { return runtime_.token_piece(token); }
    bool is_eog(Token token) const override { return runtime_.is_eog(token); }

    std::vector<Sample> execute(std::span<const BatchToken> batch) override {
        try {
            if (!healthy()) { throw std::runtime_error("mini-cuda 实例不可复用"); }
            if (batch.empty() || batch.size() > runtime_.config().batch_tokens) {
                throw std::invalid_argument("mini-cuda batch token 数量无效");
            }
            std::array<bool, minillm::cuda::CudaRuntimeConfig::max_supported_sequences> selected{};
            std::array<std::size_t, minillm::cuda::CudaRuntimeConfig::max_supported_sequences> indices{};
            std::size_t outputs = 0;
            input_.clear();
            for (std::size_t i = 0; i < batch.size(); ++i) {
                const auto& token = batch[i];
                if (token.sequence < 0 || static_cast<std::size_t>(token.sequence) >= runtime_.config().max_sequences) {
                    throw std::invalid_argument("mini-cuda sequence 越界");
                }
                if (token.logits) {
                    if (selected[static_cast<std::size_t>(token.sequence)]) {
                        throw std::invalid_argument("mini-cuda 每个 sequence 每轮最多一个输出");
                    }
                    selected[static_cast<std::size_t>(token.sequence)] = true;
                    indices[outputs++] = i;
                }
                input_.push_back({token.token, token.position, token.sequence, token.logits});
            }
            const auto result = runtime_.forward(input_, minillm::cuda::CudaOutputMode::greedy, false);
            if (result.samples.size() != outputs || !result.logits.empty() || result.device_elapsed_ms) {
                throw std::runtime_error("mini-cuda greedy 输出数量或模式不符合契约");
            }
            samples_.clear();
            for (std::size_t i = 0; i < outputs; ++i) {
                const auto& sample = result.samples[i];
                if (sample.input_index != indices[i] || sample.sequence != batch[indices[i]].sequence ||
                    sample.token < 0 || static_cast<std::size_t>(sample.token) >= info_.vocab_size) {
                    throw std::runtime_error("mini-cuda sample 的 input_index、sequence 或 token 无效");
                }
                samples_.push_back({sample.sequence, sample.token});
            }
            return samples_;
        } catch (...) {
            // Serving 契约错误同样是 fail-stop；不继承原始 Runtime 的 preflight 可恢复性。
            reusable_ = false;
            throw;
        }
    }

    std::optional<RunnerResources> resources() const noexcept override {
        auto result = resident_;
        result.state_valid = result.reusable = healthy();
        result.live_tokens = result.state_valid ? std::optional<std::size_t>(runtime_.live_kv_tokens()) : std::nullopt;
        return result;
    }
    void copy_sequence(SequenceId, SequenceId, std::size_t) override {
        throw std::logic_error("mini-cuda 不支持 prefix copy/share");
    }
    void clear_sequence(SequenceId sequence) noexcept override {
        if (!healthy()) { return; }
        try {
            runtime_.clear_sequence(sequence);
        } catch (...) {
            // noexcept 通过资源状态通知 Engine；Engine 必须停止，隔离存储直至 owner 析构。
            reusable_ = false;
        }
    }
    void synchronize() noexcept override {
        // forward 的正常与异常路径均已有完成收尾；poisoned 实例只能等待 owner 析构。
    }

private:
    bool healthy() const noexcept {
        return reusable_ && runtime_.state() == minillm::cuda::CudaRuntimeState::ready;
    }
    minillm::cuda::CudaRuntime runtime_;
    ModelInfo info_;
    RunnerResources resident_;
    std::vector<minillm::InputToken> input_;
    std::vector<Sample> samples_;
    bool reusable_ = true;
};

} // namespace

std::unique_ptr<ModelRunner> make_mini_cuda_runner(const ModelConfig& model, const EngineConfig& engine) {
    validate_mini_cuda_config(model, engine);
    return std::make_unique<MiniCudaRunner>(model, engine);
}

} // namespace llmserve
