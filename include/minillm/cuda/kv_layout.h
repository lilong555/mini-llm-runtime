#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace minillm::cuda {

enum class CudaKvLayout { contiguous, paged };
inline constexpr std::size_t kv_page_tokens = 16;

inline constexpr std::string_view kv_layout_name(CudaKvLayout layout) {
    switch (layout) {
    case CudaKvLayout::contiguous: return "contiguous";
    case CudaKvLayout::paged: return "paged";
    }
    throw std::invalid_argument("CUDA KV layout 无效");
}

inline CudaKvLayout parse_kv_layout(std::string_view name) {
    if (name == "contiguous") { return CudaKvLayout::contiguous; }
    if (name == "paged") { return CudaKvLayout::paged; }
    throw std::invalid_argument("CUDA KV layout 必须为 contiguous 或 paged");
}

// 仅校验容量合同，不表示 paged 设备路径已经可执行；不依赖 CUDA 头文件。
inline std::size_t checked_kv_capacity(CudaKvLayout layout, std::size_t sequences,
                                      std::size_t max_model_len, std::size_t capacity,
                                      std::size_t page_tokens) {
    kv_layout_name(layout);
    if (sequences == 0 || sequences > 4 || max_model_len == 0 || max_model_len > 2048 ||
        page_tokens != kv_page_tokens) {
        throw std::invalid_argument("CUDA KV 需要 S=1..4、L=1..2048、P=16");
    }
    if (layout == CudaKvLayout::contiguous) {
        const auto slots = sequences * max_model_len;
        if (capacity != 0 && capacity != slots) {
            throw std::invalid_argument("连续 CUDA KV capacity 必须为 0 或 S*L");
        }
        return slots;
    }
    const auto addressable = sequences * ((max_model_len + page_tokens - 1) / page_tokens) * page_tokens;
    if (capacity == 0 || capacity % page_tokens != 0 || capacity > addressable) {
        throw std::invalid_argument("分页 CUDA KV capacity 必须为正的 P 倍数且不超过可寻址容量");
    }
    return capacity;
}

} // namespace minillm::cuda
