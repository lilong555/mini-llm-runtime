#include "minillm/cuda/matrix.h"
#include "tensor_validation.h"

namespace minillm::cuda {
using detail::as_int;
using detail::overlaps;
using detail::validate;

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
