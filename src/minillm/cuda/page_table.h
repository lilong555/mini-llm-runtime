#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace minillm::cuda {

using PhysicalPageId = std::int32_t;
inline constexpr PhysicalPageId invalid_page = -1;
enum class PageTablePhase { ready, prepared, executing, poisoned };

struct PageTableLimits {
    std::size_t sequences;
    std::size_t max_model_len;
    std::size_t page_tokens;
    std::size_t physical_pages;
};

// 单调用者、独占页；不拥有设备内存，也不维护第二份 token 长度账本。
class PageTableState {
public:
    explicit PageTableState(PageTableLimits limits);
    PageTableState(const PageTableState&) = delete;
    PageTableState& operator=(const PageTableState&) = delete;

    void prepare(std::span<const std::size_t> committed_lengths,
                 std::span<const std::size_t> pending_lengths);
    // 合法转换无分配；错误调用进入 poisoned，不能部分发布后抛异常。
    void begin_execution() noexcept;
    void commit() noexcept;
    void discard_prepared() noexcept;
    void poison() noexcept;
    void clear(std::size_t sequence);

    PageTablePhase phase() const noexcept { return phase_; }
    std::size_t blocks_per_sequence() const noexcept { return blocks_; }
    std::span<const PhysicalPageId> committed_table() const noexcept { return active_; }
    std::span<const PhysicalPageId> free_page_ids() const noexcept { return free_ids_; }
    // 仅 prepared/executing 可供上传，借用保持至 checked completion。
    std::span<const PhysicalPageId> device_table_for_upload() const noexcept { return pending_; }
    bool upload_required() const noexcept { return dirty_ || !journal_.empty(); }
    std::size_t planned_pages() const noexcept { return journal_.size(); }
    // prepared 不消耗 free；executing/poisoned 的已保留新页计入 assigned。
    std::size_t assigned_pages() const noexcept { return limits_.physical_pages - free_ids_.size(); }
    std::size_t free_pages() const noexcept { return free_ids_.size(); }

private:
    std::size_t pages_for(std::size_t length) const noexcept;
    PageTableLimits limits_;
    std::size_t blocks_;
    std::vector<PhysicalPageId> active_, pending_, free_ids_, journal_;
    PageTablePhase phase_ = PageTablePhase::ready;
    bool dirty_ = true;
};

} // namespace minillm::cuda
