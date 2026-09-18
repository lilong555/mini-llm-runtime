#include "minillm/paged_kv.h"

#include "minillm/kernels.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace minillm {
namespace {

std::size_t page_elements(const KVConfig& config) {
    std::size_t count = 2;
    for (const auto size : {config.layers, config.page_tokens, config.kv_width}) {
        if (size == 0 || count > std::numeric_limits<std::size_t>::max() / size / 2) {
            throw std::invalid_argument("invalid KV page dimensions");
        }
        count *= size;
    }
    if (config.pages == 0 || config.sequences == 0) {
        throw std::invalid_argument("KV pages and sequence capacity must be positive");
    }
    return count;
}

} // namespace

PagedKV::PagedKV(KVConfig config)
    : config_(config), elements_per_page_(page_elements(config)),
      pages_(config.pages), tables_(config.sequences) {
    free_.reserve(config.pages);
    for (std::size_t i = config.pages; i > 0; --i) {
        free_.push_back(i - 1);
    }
}

std::size_t PagedKV::allocate_page() {
    if (free_.empty()) {
        throw std::runtime_error("physical KV page pool exhausted");
    }
    const auto id = free_.back();
    if (pages_[id].data.empty()) {
        pages_[id].data.resize(elements_per_page_);
    }
    free_.pop_back();
    pages_[id].refs = 1;
    return id;
}

void PagedKV::append(std::size_t sequence, std::size_t position) {
    auto& table = tables_.at(sequence);
    if (position != table.length) {
        throw std::invalid_argument("KV append position must equal the sequence length");
    }
    if (position % config_.page_tokens == 0) {
        table.pages.reserve(table.pages.size() + 1);
        table.pages.push_back(allocate_page());
    } else {
        const auto old = table.pages.back();
        if (pages_[old].refs > 1) {
            const auto replacement = allocate_page();
            std::copy(pages_[old].data.begin(), pages_[old].data.end(), pages_[replacement].data.begin());
            --pages_[old].refs;
            table.pages.back() = replacement;
            ++copy_on_writes_;
        }
    }
    ++table.length;
}

std::size_t PagedKV::offset(std::size_t layer, std::size_t position, bool value) const {
    return ((layer * 2 + static_cast<std::size_t>(value)) * config_.page_tokens +
            position % config_.page_tokens) * config_.kv_width;
}

const PagedKV::Page& PagedKV::page_at(std::size_t sequence, std::size_t layer,
                                    std::size_t position) const {
    const auto& table = tables_.at(sequence);
    if (position >= table.length || layer >= config_.layers) {
        throw std::out_of_range("KV position or layer is out of range");
    }
    return pages_[table.pages[position / config_.page_tokens]];
}

void PagedKV::store(std::size_t sequence, std::size_t layer, std::size_t position,
                    std::span<const float> key_data, std::span<const float> value_data) {
    const auto& page = page_at(sequence, layer, position);
    if (page.refs != 1 || key_data.size() != config_.kv_width || value_data.size() != config_.kv_width) {
        throw std::logic_error("KV writes require an exclusive page and matching dimensions");
    }
    auto& writable = pages_[tables_[sequence].pages[position / config_.page_tokens]];
    const auto k = offset(layer, position, false);
    const auto v = offset(layer, position, true);
    for (std::size_t i = 0; i < config_.kv_width; ++i) {
        writable.data[k + i] = float_to_half(key_data[i]);
        writable.data[v + i] = float_to_half(value_data[i]);
    }
}

std::span<const std::uint16_t> PagedKV::key(std::size_t sequence, std::size_t layer,
                                          std::size_t position) const {
    const auto& page = page_at(sequence, layer, position);
    return {page.data.data() + offset(layer, position, false), config_.kv_width};
}

std::span<const std::uint16_t> PagedKV::value(std::size_t sequence, std::size_t layer,
                                            std::size_t position) const {
    const auto& page = page_at(sequence, layer, position);
    return {page.data.data() + offset(layer, position, true), config_.kv_width};
}

void PagedKV::share_prefix(std::size_t source, std::size_t target, std::size_t tokens) {
    const auto& from = tables_.at(source);
    auto& to = tables_.at(target);
    if (source == target || tokens > from.length) {
        throw std::invalid_argument("invalid KV prefix alias");
    }
    const auto count = (tokens + config_.page_tokens - 1) / config_.page_tokens;
    std::vector<std::size_t> shared(from.pages.begin(),
        from.pages.begin() + static_cast<std::ptrdiff_t>(count));
    for (const auto page : shared) {
        ++pages_[page].refs;
    }
    clear(target);
    to.pages = std::move(shared);
    to.length = tokens;
}

void PagedKV::clear(std::size_t sequence) noexcept {
    if (sequence >= tables_.size()) {
        return;
    }
    auto& table = tables_[sequence];
    for (const auto page : table.pages) {
        if (--pages_[page].refs == 0) {
            free_.push_back(page);
        }
    }
    table.pages.clear();
    table.length = 0;
}

std::size_t PagedKV::length(std::size_t sequence) const { return tables_.at(sequence).length; }
std::span<const std::size_t> PagedKV::page_table(std::size_t sequence) const {
    return tables_.at(sequence).pages;
}
std::size_t PagedKV::references(std::size_t page) const { return pages_.at(page).refs; }
std::size_t PagedKV::resident_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& page : pages_) {
        bytes += page.data.size() * sizeof(std::uint16_t);
    }
    return bytes;
}

} // namespace minillm
