#include "page_table.h"
#include "minillm/cuda/kv_layout.h"

#include <algorithm>
#include <numeric>

namespace minillm::cuda {
namespace {
PageTableLimits checked(PageTableLimits limits) {
    // 先限制维度再乘法，所有 table/ID/字节计算均有小于 INT32_MAX 的固定上界。
    if (limits.physical_pages == 0 || limits.physical_pages > 512) {
        throw std::invalid_argument("CUDA 物理页数必须在 [1,512] 内");
    }
    checked_kv_capacity(CudaKvLayout::paged, limits.sequences, limits.max_model_len,
                        limits.physical_pages * kv_page_tokens, limits.page_tokens);
    return limits;
}
}

PageTableState::PageTableState(PageTableLimits limits)
    : limits_(checked(limits)), blocks_(pages_for(limits_.max_model_len)),
      active_(limits_.sequences * blocks_, invalid_page), pending_(active_),
      free_ids_(limits_.physical_pages) {
    std::iota(free_ids_.begin(), free_ids_.end(), PhysicalPageId{0});
    journal_.reserve(limits_.physical_pages);
}

std::size_t PageTableState::pages_for(std::size_t length) const noexcept {
    return (length + limits_.page_tokens - 1) / limits_.page_tokens;
}

void PageTableState::prepare(std::span<const std::size_t> committed,
                             std::span<const std::size_t> pending) {
    if (phase_ != PageTablePhase::ready) {
        throw std::logic_error("CUDA 页状态不是 ready，不能准备执行");
    }
    if (committed.size() != limits_.sequences || pending.size() != limits_.sequences) {
        throw std::invalid_argument("CUDA 页长度视图的 sequence 数量不匹配");
    }
    std::size_t needed = 0;
    for (std::size_t s = 0; s < limits_.sequences; ++s) {
        if (pending[s] > limits_.max_model_len || committed[s] > pending[s]) {
            throw std::invalid_argument("CUDA 页长度必须有界且连续增长");
        }
        const auto old_pages = pages_for(committed[s]);
        for (std::size_t b = 0; b < blocks_; ++b) {
            const auto id = active_[s * blocks_ + b];
            if (b < old_pages ? (id < 0 || std::size_t(id) >= limits_.physical_pages) : id != invalid_page) {
                throw std::invalid_argument("CUDA committed 长度与页映射不一致");
            }
        }
        needed += pages_for(pending[s]) - old_pages;
    }
    if (needed > free_ids_.size()) {
        throw std::length_error("CUDA 物理页池容量不足，尚未开始设备执行");
    }
    std::copy(active_.begin(), active_.end(), pending_.begin());
    journal_.clear();
    for (std::size_t s = 0; s < limits_.sequences; ++s) {
        for (auto b = pages_for(committed[s]); b < pages_for(pending[s]); ++b) {
            const auto id = free_ids_[free_ids_.size() - 1 - journal_.size()];
            pending_[s * blocks_ + b] = id;
            journal_.push_back(id);
        }
    }
    phase_ = PageTablePhase::prepared;
}

void PageTableState::begin_execution() noexcept {
    if (phase_ != PageTablePhase::prepared) { poison(); return; }
    free_ids_.resize(free_ids_.size() - journal_.size());
    phase_ = PageTablePhase::executing;
}

void PageTableState::commit() noexcept {
    if (phase_ != PageTablePhase::executing) { poison(); return; }
    // 调用方已检查 stream/status，交换后不得再执行可能失败的发布操作。
    active_.swap(pending_);
    journal_.clear();
    dirty_ = false;
    phase_ = PageTablePhase::ready;
}

void PageTableState::discard_prepared() noexcept {
    if (phase_ != PageTablePhase::prepared) { poison(); return; }
    journal_.clear();
    phase_ = PageTablePhase::ready;
    // 前一次 clear 的 dirty 必须保留；未上传的 pending 不能成为设备状态。
}

void PageTableState::poison() noexcept {
    if (phase_ == PageTablePhase::prepared) {
        free_ids_.resize(free_ids_.size() - journal_.size());
    }
    phase_ = PageTablePhase::poisoned;
}

void PageTableState::clear(std::size_t sequence) {
    if (phase_ != PageTablePhase::ready) {
        throw std::logic_error("CUDA 页只能在 ready 完成点清理，不能恢复 poisoned");
    }
    if (sequence >= limits_.sequences) { throw std::invalid_argument("CUDA 页 sequence 越界"); }
    const auto row = std::span(active_).subspan(sequence * blocks_, blocks_);
    std::size_t released = 0;
    for (const auto id : row) {
        if (id == invalid_page) { continue; }
        if (id < 0 || std::size_t(id) >= limits_.physical_pages) {
            throw std::logic_error("CUDA 页表包含非法物理页");
        }
        ++released;
    }
    if (released > assigned_pages()) { throw std::logic_error("CUDA 页释放破坏容量守恒"); }
    for (auto& id : row) {
        if (id != invalid_page) { free_ids_.push_back(id); id = invalid_page; }
    }
    dirty_ = dirty_ || released != 0;
}

} // namespace minillm::cuda
