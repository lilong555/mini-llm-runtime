#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace minillm
{

    struct KVConfig
    {
        std::size_t pages;
        std::size_t page_tokens;
        std::size_t layers;
        std::size_t kv_width;
        std::size_t sequences;
    };

    class PagedKV
    {
    public:
        explicit PagedKV(KVConfig config);
        void append(std::size_t sequence, std::size_t position);
        void store(std::size_t sequence, std::size_t layer, std::size_t position,
                   std::span<const float> key, std::span<const float> value);
        std::span<const std::uint16_t> key(std::size_t sequence, std::size_t layer,
                                           std::size_t position) const;
        std::span<const std::uint16_t> value(std::size_t sequence, std::size_t layer,
                                             std::size_t position) const;
        void share_prefix(std::size_t source, std::size_t target, std::size_t tokens);
        void clear(std::size_t sequence) noexcept;
        std::size_t length(std::size_t sequence) const;
        std::span<const std::size_t> page_table(std::size_t sequence) const;
        std::size_t references(std::size_t page) const;
        std::size_t used_pages() const noexcept { return pages_.size() - free_.size(); }
        std::size_t capacity() const noexcept { return pages_.size(); }
        std::size_t resident_bytes() const noexcept;
        std::size_t copy_on_writes() const noexcept { return copy_on_writes_; }

    private:
        struct Page
        {
            std::vector<std::uint16_t> data;
            std::size_t refs = 0;
        };
        struct Table
        {
            std::vector<std::size_t> pages;
            std::size_t length = 0;
        };
        std::size_t allocate_page();
        std::size_t offset(std::size_t layer, std::size_t position, bool value) const;
        const Page &page_at(std::size_t sequence, std::size_t layer, std::size_t position) const;
        KVConfig config_;
        std::size_t elements_per_page_;
        std::vector<Page> pages_;
        std::vector<Table> tables_;
        std::vector<std::size_t> free_;
        std::size_t copy_on_writes_ = 0;
    };

} // namespace minillm
