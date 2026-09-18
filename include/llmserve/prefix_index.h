#pragma once

#include "llmserve/block_pool.h"

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmserve {

using Token = std::int32_t;
using SequenceId = std::int32_t;

struct PrefixEntry {
    SequenceId sequence;
    std::string cache_namespace;
    std::vector<Token> tokens;
    std::vector<BlockId> blocks;
    std::uint64_t last_used = 0;
};

struct PrefixMatch {
    SequenceId sequence;
    std::size_t tokens;
};

class PrefixIndex {
public:
    explicit PrefixIndex(std::size_t block_size);
    void insert(PrefixEntry entry);
    PrefixEntry erase(SequenceId sequence);
    std::optional<PrefixMatch> match(const std::string& cache_namespace,
                                     std::span<const Token> tokens,
                                     std::size_t max_reuse);
    std::optional<SequenceId> least_recently_used() const;
    const PrefixEntry& at(SequenceId sequence) const;
    std::size_t size() const noexcept { return entries_.size(); }
    std::size_t token_count() const noexcept { return token_count_; }

private:
    struct Node {
        std::map<Token, Node> children;
        std::set<SequenceId> owners;
    };
    std::size_t block_size_;
    std::size_t token_count_ = 0;
    std::uint64_t clock_ = 0;
    std::unordered_map<std::string, Node> roots_;
    std::map<SequenceId, PrefixEntry> entries_;
};

} // namespace llmserve
