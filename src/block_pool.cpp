#include "llmserve/block_pool.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llmserve {

BlockPool::BlockPool(std::size_t count) : refs_(count, 0), seen_(count, 0) {
    if (count == 0 || count > std::numeric_limits<BlockId>::max()) {
        throw std::invalid_argument("invalid block pool size");
    }
    free_.reserve(count);
    for (std::size_t i = count; i > 0; --i) {
        free_.push_back(static_cast<BlockId>(i - 1));
    }
}

std::optional<std::vector<BlockId>> BlockPool::allocate(std::size_t count) {
    if (count > free_.size()) {
        return std::nullopt;
    }
    std::vector<BlockId> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto id = free_.back();
        free_.pop_back();
        refs_[id] = 1;
        result.push_back(id);
    }
    return result;
}

void BlockPool::validate_table(std::span<const BlockId> blocks) const {
    if (++validation_epoch_ == 0) {
        std::fill(seen_.begin(), seen_.end(), 0);
        validation_epoch_ = 1;
    }
    for (const auto id : blocks) {
        if (id >= refs_.size() || refs_[id] == 0 || seen_[id] == validation_epoch_) {
            throw std::logic_error("invalid, free, or duplicate block in a block table");
        }
        seen_[id] = validation_epoch_;
    }
}

void BlockPool::retain(std::span<const BlockId> blocks) {
    validate_table(blocks);
    for (const auto id : blocks) {
        if (refs_[id] == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("block reference count overflow");
        }
    }
    for (const auto id : blocks) {
        ++refs_[id];
    }
}

void BlockPool::release(std::span<const BlockId> blocks) {
    validate_table(blocks);
    for (const auto id : blocks) {
        if (--refs_[id] == 0) {
            free_.push_back(id);
        }
    }
}

std::size_t BlockPool::references(BlockId id) const {
    return refs_.at(id);
}

} // namespace llmserve
