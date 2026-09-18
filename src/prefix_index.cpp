#include "llmserve/prefix_index.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace llmserve {

PrefixIndex::PrefixIndex(std::size_t block_size) : block_size_(block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("prefix block size must be positive");
    }
}

void PrefixIndex::insert(PrefixEntry entry) {
    if (entry.tokens.empty() || entry.tokens.size() % block_size_ != 0 ||
        entry.blocks.size() != entry.tokens.size() / block_size_ ||
        entries_.contains(entry.sequence)) {
        throw std::invalid_argument("invalid or duplicate prefix entry");
    }
    auto* node = &roots_[entry.cache_namespace];
    node->owners.insert(entry.sequence);
    for (const auto token : entry.tokens) {
        node = &node->children[token];
        node->owners.insert(entry.sequence);
    }
    token_count_ += entry.tokens.size();
    entry.last_used = ++clock_;
    entries_.emplace(entry.sequence, std::move(entry));
}

PrefixEntry PrefixIndex::erase(SequenceId sequence) {
    auto entry = entries_.at(sequence);
    auto& root = roots_.at(entry.cache_namespace);
    std::vector<Node*> path{&root};
    for (const auto token : entry.tokens) {
        path.push_back(&path.back()->children.at(token));
    }
    for (auto* node : path) {
        node->owners.erase(sequence);
    }
    // Prune bottom-up without invalidating the unvisited parent pointers.
    for (std::size_t i = entry.tokens.size(); i > 0; --i) {
        if (!path[i]->owners.empty()) {
            break;
        }
        path[i - 1]->children.erase(entry.tokens[i - 1]);
    }
    if (root.owners.empty()) {
        roots_.erase(entry.cache_namespace);
    }
    entries_.erase(sequence);
    token_count_ -= entry.tokens.size();
    return entry;
}

std::optional<PrefixMatch> PrefixIndex::match(const std::string& cache_namespace,
                                             std::span<const Token> tokens,
                                             std::size_t max_reuse) {
    const auto root = roots_.find(cache_namespace);
    if (root == roots_.end()) {
        return std::nullopt;
    }
    const auto limit = std::min(tokens.size(), max_reuse);
    const auto* node = &root->second;
    std::optional<PrefixMatch> result;
    for (std::size_t i = 0; i < limit; ++i) {
        const auto next = node->children.find(tokens[i]);
        if (next == node->children.end()) {
            break;
        }
        node = &next->second;
        if ((i + 1) % block_size_ == 0) {
            result = PrefixMatch{*node->owners.begin(), i + 1};
        }
    }
    if (result) {
        entries_.at(result->sequence).last_used = ++clock_;
    }
    return result;
}

std::optional<SequenceId> PrefixIndex::least_recently_used() const {
    if (entries_.empty()) {
        return std::nullopt;
    }
    const auto entry = std::min_element(entries_.begin(), entries_.end(),
        [](const auto& left, const auto& right) {
            return left.second.last_used < right.second.last_used;
        });
    return entry->first;
}

const PrefixEntry& PrefixIndex::at(SequenceId sequence) const {
    return entries_.at(sequence);
}

} // namespace llmserve
