#pragma once

#include "minillm/cuda/context.h"

namespace minillm::cuda {

// dtype 由 T 固定；stride 和 capacity 以元素计。host 只能传递设备指针，不能解引用。
template<class T>
struct DeviceTensorView {
    T* data;
    std::size_t rows;
    std::size_t columns;
    std::size_t stride;
    std::size_t capacity;
    int device;
};

template<class T, class Memory>
DeviceTensorView<T> matrix_view(DeviceBuffer<T, Memory>& buffer, std::size_t rows,
                               std::size_t columns, std::size_t stride = 0) {
    return {buffer.data(), rows, columns, stride ? stride : columns, buffer.size(), buffer.device()};
}

template<class T, class Memory>
DeviceTensorView<const T> matrix_view(const DeviceBuffer<T, Memory>& buffer, std::size_t rows,
                                     std::size_t columns, std::size_t stride = 0) {
    return {buffer.data(), rows, columns, stride ? stride : columns, buffer.size(), buffer.device()};
}

// X[M,K]、W[N,K]、Y[M,N] 均为 row-major；计算 Y = X * transpose(W)。
// 仅入队，不在矩阵调用内部同步；返回结果前调用 CudaContext::synchronize。
void matrix_multiply(const CudaContext& context, DeviceTensorView<const float> x,
                     DeviceTensorView<const float> weights, DeviceTensorView<float> output);

} // namespace minillm::cuda
