#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace llmserve {

using BlockId = std::uint32_t;

// Logical KV capacity credits. GPU addresses remain owned by the model backend.
class BlockPool {
public:
    explicit BlockPool(std::size_t count);
    std::optional<std::vector<BlockId>> allocate(std::size_t count);
    void retain(std::span<const BlockId> blocks);
    void release(std::span<const BlockId> blocks);
    std::size_t capacity() const noexcept { return refs_.size(); }
    std::size_t free_count() const noexcept { return free_.size(); }
    std::size_t used_count() const noexcept { return capacity() - free_count(); }
    std::size_t references(BlockId id) const;

private:
    void validate_table(std::span<const BlockId> blocks) const;
    std::vector<std::size_t> refs_;
    std::vector<BlockId> free_;
    mutable std::vector<std::uint64_t> seen_;
    mutable std::uint64_t validation_epoch_ = 0;
};

} // namespace llmserve
