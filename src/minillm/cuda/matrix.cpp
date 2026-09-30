#include "minillm/cuda/matrix.h"
#include "tensor_validation.h"

namespace minillm::cuda {
using detail::as_int;
using detail::overlaps;
using detail::validate;

namespace {
template<class T>
void multiply(const CudaContext& context, DeviceTensorView<const T> x,
              DeviceTensorView<const T> weights, DeviceTensorView<float> output) {
    constexpr bool half = std::is_same_v<T, std::uint16_t>;
    constexpr auto precision = half ? PrecisionMode::f16_matrix_f32acc : PrecisionMode::f32_pedantic;
    if (context.precision_mode() != precision) {
        throw std::invalid_argument("CUDA 矩阵 operand 类型与 context 精度不一致");
    }
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
        &alpha, weights.data, half ? CUDA_R_16F : CUDA_R_32F, as_int(weights.stride),
        x.data, half ? CUDA_R_16F : CUDA_R_32F, as_int(x.stride), &beta,
        output.data, CUDA_R_32F, as_int(output.stride),
        half ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT),
        half ? "cublasGemmEx F16 F32acc" : "cublasGemmEx F32 PEDANTIC");
}
}

void matrix_multiply(const CudaContext& context, DeviceTensorView<const float> x,
                     DeviceTensorView<const float> weights, DeviceTensorView<float> output) {
    multiply(context, x, weights, output);
}

void matrix_multiply(const CudaContext& context, DeviceTensorView<const std::uint16_t> x,
                     DeviceTensorView<const std::uint16_t> weights, DeviceTensorView<float> output) {
    multiply(context, x, weights, output);
}

void matrix_multiply(const CudaContext& context, const MatrixWeightView& x,
                     const MatrixWeightView& weights, DeviceTensorView<float> output) {
    std::visit([&](auto input, auto weight) {
        if constexpr (std::is_same_v<decltype(input), decltype(weight)>) {
            matrix_multiply(context, input, weight, output);
        } else {
            throw std::invalid_argument("CUDA 矩阵输入与权重 dtype 不一致");
        }
    }, x, weights);
}

} // namespace minillm::cuda
