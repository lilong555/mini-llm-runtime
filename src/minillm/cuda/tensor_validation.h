#pragma once

#include "minillm/cuda/matrix.h"

#include <cstdint>
#include <limits>

namespace minillm::cuda::detail {

inline int as_int(std::size_t value) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUDA 维度和 stride 必须位于 [1, INT_MAX]");
    }
    return static_cast<int>(value);
}

struct Range { std::uintptr_t begin; std::uintptr_t end; };

template<class T>
DeviceTensorView<const T> read_only(DeviceTensorView<T> view) {
    return {view.data, view.rows, view.columns, view.stride, view.capacity, view.device};
}

template<class T>
Range validate(DeviceTensorView<T> view, int device) {
    as_int(view.rows);
    as_int(view.columns);
    as_int(view.stride);
    if (!view.data || view.device != device || view.stride < view.columns) {
        throw std::invalid_argument("CUDA tensor 指针、设备或 stride 无效");
    }
    const auto preceding = checked_product(view.rows - 1, view.stride);
    if (preceding > view.capacity || view.columns > view.capacity - preceding) {
        throw std::invalid_argument("CUDA tensor 超出 buffer 容量");
    }
    const auto bytes = checked_product(preceding + view.columns, sizeof(T));
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    if (address % alignof(T) != 0 || address > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        throw std::invalid_argument("CUDA tensor 地址范围无效");
    }
    return {address, address + bytes};
}

inline bool overlaps(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

template<class A, class B>
bool same_layout(DeviceTensorView<A> a, DeviceTensorView<B> b) {
    return static_cast<const void*>(a.data) == static_cast<const void*>(b.data) &&
        a.rows == b.rows && a.columns == b.columns && a.stride == b.stride;
}

inline void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

}
