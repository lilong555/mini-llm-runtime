#pragma once

#include "storage.h"

#include <span>

namespace minillm::cuda {

enum class BatchPhase { ready, prepared, executing, poisoned };
struct BatchSummary { std::size_t tokens, logits, max_context; };

// 仅管理连续 KV 的逻辑长度，不拥有设备存储。commit 的调用方必须已检查设备完成。
class BatchState {
public:
    BatchState(StorageLimits limits, std::size_t vocabulary);
    BatchSummary prepare(std::span<const InputToken> tokens);
    void start();
    void commit();
    void discard_prepared();
    void poison() noexcept { phase_ = BatchPhase::poisoned; }
    void clear(std::int32_t sequence);
    BatchPhase phase() const noexcept { return phase_; }
    std::span<const std::size_t> lengths() const noexcept { return lengths_; }
    std::size_t live_sequences() const noexcept;
    std::size_t live_tokens() const noexcept;

private:
    StorageLimits limits_;
    std::size_t vocabulary_;
    std::vector<std::size_t> lengths_, pending_;
    BatchPhase phase_ = BatchPhase::ready;
};

}
