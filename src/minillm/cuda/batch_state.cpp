#include "batch_state.h"
#include "tensor_validation.h"

#include <algorithm>
#include <numeric>

namespace minillm::cuda {

BatchState::BatchState(StorageLimits limits, std::size_t vocabulary)
    : limits_(limits), vocabulary_(vocabulary) {
    detail::as_int(limits.max_sequences);
    detail::as_int(limits.max_model_len);
    detail::as_int(limits.max_batch_tokens);
    detail::as_int(vocabulary);
    lengths_.resize(limits.max_sequences, 0);
    pending_.resize(limits.max_sequences, 0);
}

BatchSummary BatchState::prepare(std::span<const InputToken> tokens) {
    if (phase_ != BatchPhase::ready) { throw Error("CUDA KV 状态不是 ready，不能准备执行"); }
    if (tokens.empty() || tokens.size() > limits_.max_batch_tokens) {
        throw std::invalid_argument("CUDA batch token 数量无效");
    }
    std::copy(lengths_.begin(), lengths_.end(), pending_.begin());
    BatchSummary summary{tokens.size(), 0, 0};
    for (const auto& token : tokens) {
        if (token.sequence < 0 || std::size_t(token.sequence) >= pending_.size() ||
            token.token < 0 || std::size_t(token.token) >= vocabulary_ ||
            token.position < 0 || std::size_t(token.position) >= limits_.max_model_len) {
            throw std::invalid_argument("CUDA token、sequence 或 position 越界");
        }
        auto& length = pending_[std::size_t(token.sequence)];
        if (std::size_t(token.position) != length) { throw std::invalid_argument("CUDA position 必须连续追加"); }
        ++length;
        summary.logits += token.logits ? 1 : 0;
        summary.max_context = std::max(summary.max_context, length);
    }
    phase_ = BatchPhase::prepared;
    return summary;
}

void BatchState::start() {
    if (phase_ != BatchPhase::prepared) { throw Error("CUDA KV 尚未准备，不能开始设备执行"); }
    phase_ = BatchPhase::executing;
}

void BatchState::commit() {
    if (phase_ != BatchPhase::executing) { throw Error("CUDA KV 不允许提交当前状态"); }
    lengths_.swap(pending_);
    phase_ = BatchPhase::ready;
}

void BatchState::discard_prepared() {
    if (phase_ != BatchPhase::prepared) { throw Error("CUDA KV 仅能撤销未启动的准备状态"); }
    phase_ = BatchPhase::ready;
}

void BatchState::clear(std::int32_t sequence) {
    if (phase_ != BatchPhase::ready) { throw Error("CUDA KV 只能在 ready 完成点清理，不能清除 poisoned"); }
    if (sequence < 0 || std::size_t(sequence) >= lengths_.size()) {
        throw std::invalid_argument("CUDA sequence 越界");
    }
    lengths_[std::size_t(sequence)] = 0;
}

std::size_t BatchState::live_sequences() const noexcept {
    return static_cast<std::size_t>(std::count_if(lengths_.begin(), lengths_.end(), [](auto n) { return n != 0; }));
}
std::size_t BatchState::live_tokens() const noexcept {
    return std::accumulate(lengths_.begin(), lengths_.end(), std::size_t{0});
}

}
