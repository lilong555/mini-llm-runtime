#include "minillm/cuda/matrix.h"

#include <cstdint>
#include <limits>

namespace minillm::cuda {
namespace {

int as_int(std::size_t value) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("cuBLAS 维度和 leading dimension 必须位于 [1, INT_MAX]");
    }
    return static_cast<int>(value);
}

struct Range { std::uintptr_t begin; std::uintptr_t end; };

template<class T>
Range validate(DeviceTensorView<T> view, int device) {
    as_int(view.rows);
    as_int(view.columns);
    as_int(view.stride);
    if (!view.data || view.device != device || view.stride < view.columns) {
        throw std::invalid_argument("CUDA 矩阵指针、设备或 stride 无效");
    }
    const auto preceding = checked_product(view.rows - 1, view.stride);
    if (preceding > view.capacity || view.columns > view.capacity - preceding) {
        throw std::invalid_argument("CUDA 矩阵超出 buffer 容量");
    }
    const auto bytes = checked_product(preceding + view.columns, sizeof(T));
    const auto address = reinterpret_cast<std::uintptr_t>(view.data);
    if (address % alignof(T) != 0 || address > std::numeric_limits<std::uintptr_t>::max() - bytes) {
        throw std::invalid_argument("CUDA 矩阵地址范围无效");
    }
    return {address, address + bytes};
}

bool overlaps(Range a, Range b) { return a.begin < b.end && b.begin < a.end; }

} // namespace

void matrix_multiply(const CudaContext& context, DeviceTensorView<const float> x,
                     DeviceTensorView<const float> weights, DeviceTensorView<float> output) {
    const auto x_range = validate(x, context.device());
    const auto w_range = validate(weights, context.device());
    const auto y_range = validate(output, context.device());
    if (x.columns != weights.columns || output.rows != x.rows || output.columns != weights.rows) {
        throw std::invalid_argument("CUDA 矩阵形状不符合 Y[M,N] = X[M,K] * transpose(W[N,K])");
    }
    if (overlaps(x_range, y_range) || overlaps(w_range, y_range)) {
        throw std::invalid_argument("CUDA 矩阵输出不能与输入或权重重叠");
    }
    DeviceScope scope(context.device());
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(cublasGemmEx(context.handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        as_int(weights.rows), as_int(x.rows), as_int(x.columns),
        &alpha, weights.data, CUDA_R_32F, as_int(weights.stride),
        x.data, CUDA_R_32F, as_int(x.stride), &beta,
        output.data, CUDA_R_32F, as_int(output.stride),
        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT), "cublasGemmEx F32 PEDANTIC");
}

} // namespace minillm::cuda
