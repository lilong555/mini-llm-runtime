#include "ops.h"
#include "tensor_validation.h"

#include <cub/block/block_reduce.cuh>
#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cmath>

namespace minillm::cuda {
namespace {

constexpr unsigned threads = 256;
using detail::as_int;
using detail::overlaps;
using detail::require;
using detail::same_layout;
using detail::validate;

detail::Range status_range(DeviceTensorView<std::int32_t> status, int device) {
    const auto range = validate(status, device);
    require(status.rows == 1 && status.columns == 2, "CUDA status 形状必须为 [1,2]");
    return range;
}

unsigned grid_for(std::size_t count) {
    return static_cast<unsigned>(std::min<std::size_t>((count - 1) / threads + 1, 65535));
}

__device__ void record_error(std::int32_t* status, DeviceError code, int row) {
    atomicOr(status, static_cast<int>(code));
    atomicMin(status + 1, row);
}

__global__ void reset_kernel(std::int32_t* status) {
    status[0] = 0;
    status[1] = INT_MAX;
}

__global__ void gather_kernel(DeviceTensorView<const float> source,
                              DeviceTensorView<const std::int32_t> indices,
                              DeviceTensorView<float> output, std::int32_t* status) {
    const auto count = output.rows * output.columns;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(blockDim.x) * gridDim.x) {
        const auto row = i / output.columns, column = i % output.columns;
        const int index = indices.data[row * indices.stride];
        if (index < 0 || std::size_t(index) >= source.rows) {
            output.data[row * output.stride + column] = CUDART_NAN_F;
            if (column == 0) { record_error(status, DeviceError::invalid_index, static_cast<int>(row)); }
        } else {
            output.data[row * output.stride + column] = source.data[std::size_t(index) * source.stride + column];
        }
    }
}

struct Maximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void norm_kernel(DeviceTensorView<const float> input, DeviceTensorView<const float> weight,
                            DeviceTensorView<float> output, float epsilon) {
    using Reduction = cub::BlockReduce<float, threads>;
    __shared__ typename Reduction::TempStorage temporary;
    __shared__ float largest, denominator;
    const auto width = weight.columns, groups = input.columns / width;
    const auto row = std::size_t(blockIdx.x) / groups, group = std::size_t(blockIdx.x) % groups;
    const auto* x = input.data + row * input.stride + group * width;
    auto* y = output.data + row * output.stride + group * width;
    float local_max = 0;
    for (std::size_t i = threadIdx.x; i < width; i += threads) {
        const float value = x[i];
        local_max = fmaxf(local_max, isfinite(value) ? fabsf(value) : CUDART_INF_F);
    }
    const auto maximum = Reduction(temporary).Reduce(local_max, Maximum{});
    // 同时缩放输入与 epsilon，避免平方、方差加法溢出及极小输入的除法溢出。
    if (threadIdx.x == 0) { largest = fmaxf(maximum, sqrtf(epsilon)); }
    __syncthreads();
    if (!isfinite(largest)) {
        for (std::size_t i = threadIdx.x; i < width; i += threads) {
            y[i] = CUDART_NAN_F;
        }
        return;
    }
    float sum = 0;
    for (std::size_t i = threadIdx.x; i < width; i += threads) {
        const float scaled = x[i] / largest;
        sum += __fmul_rn(scaled, scaled);
    }
    const auto total = Reduction(temporary).Sum(sum);
    if (threadIdx.x == 0) {
        const float mean = total / static_cast<float>(width);
        denominator = sqrtf(mean + (epsilon / largest) / largest);
    }
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < width; i += threads) {
        const float normalized = (x[i] / largest) / denominator;
        y[i] = normalized * weight.data[i];
    }
}

__global__ void rope_kernel(DeviceTensorView<float> input,
                            DeviceTensorView<const std::int32_t> positions,
                            DeviceTensorView<const float> coefficients, std::int32_t* status) {
    const auto width = coefficients.columns, half = width / 2, heads = input.columns / width;
    const auto count = input.rows * input.columns / 2;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(blockDim.x) * gridDim.x) {
        const auto head = i / half, column = i % half, row = head / heads;
        auto* data = input.data + row * input.stride + (head % heads) * width;
        const int position = positions.data[row * positions.stride];
        if (position < 0 || std::size_t(position) >= coefficients.rows) {
            data[column] = data[column + half] = CUDART_NAN_F;
            if (head % heads == 0 && column == 0) {
                record_error(status, DeviceError::invalid_position, static_cast<int>(row));
            }
            continue;
        }
        const auto* coefficient = coefficients.data + std::size_t(position) * coefficients.stride;
        const float left = data[column], right = data[column + half];
        const float cosine = coefficient[column], sine = coefficient[column + half];
        data[column] = __fsub_rn(__fmul_rn(left, cosine), __fmul_rn(right, sine));
        data[column + half] = __fadd_rn(__fmul_rn(left, sine), __fmul_rn(right, cosine));
    }
}

template<bool Activation>
__global__ void pointwise_kernel(DeviceTensorView<float> left, DeviceTensorView<const float> right) {
    const auto count = left.rows * left.columns;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += std::size_t(blockDim.x) * gridDim.x) {
        const auto row = i / left.columns, column = i % left.columns;
        const auto index = row * left.stride + column;
        const float a = left.data[index], b = right.data[row * right.stride + column];
        if constexpr (Activation) { left.data[index] = (a / (1.0f + expf(-a))) * b; }
        else { left.data[index] = a + b; }
    }
}

struct Candidate { float value; int index; };
struct Best {
    __device__ Candidate operator()(Candidate a, Candidate b) const {
        return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
    }
};
struct Any {
    __device__ int operator()(int a, int b) const { return a | b; }
};

__global__ void argmax_kernel(DeviceTensorView<const float> logits,
                              DeviceTensorView<std::int32_t> tokens, std::int32_t* status) {
    using Reduction = cub::BlockReduce<Candidate, threads>;
    using Errors = cub::BlockReduce<int, threads>;
    __shared__ union {
        typename Reduction::TempStorage candidate;
        typename Errors::TempStorage errors;
    } temporary;
    __shared__ int selected;
    const auto row = std::size_t(blockIdx.x);
    Candidate best{-CUDART_INF_F, INT_MAX};
    int bad = 0;
    for (std::size_t i = threadIdx.x; i < logits.columns; i += threads) {
        const float value = logits.data[row * logits.stride + i];
        if (!isfinite(value)) { bad = 1; }
        else { best = Best{}(best, {value, static_cast<int>(i)}); }
    }
    const auto result = Reduction(temporary.candidate).Reduce(best, Best{});
    if (threadIdx.x == 0) { selected = result.index; }
    __syncthreads();
    const auto nonfinite = Errors(temporary.errors).Reduce(bad, Any{});
    if (threadIdx.x == 0) {
        tokens.data[row * tokens.stride] = nonfinite ? -1 : selected;
        if (nonfinite) { record_error(status, DeviceError::nonfinite, static_cast<int>(row)); }
    }
}

template<bool Activation>
void pointwise(const CudaContext& context, DeviceTensorView<float> left, DeviceTensorView<const float> right) {
    const auto a = validate(left, context.device()), b = validate(right, context.device());
    require(left.rows == right.rows && left.columns == right.columns, "CUDA 逐元素算子形状不一致");
    require(!overlaps(a, b) || same_layout(left, right), "CUDA 逐元素算子不支持部分重叠");
    const auto count = checked_product(left.rows, left.columns);
    DeviceScope scope(context.device());
    pointwise_kernel<Activation><<<grid_for(count), threads, 0, context.stream()>>>(left, right);
    check_cuda(cudaGetLastError(), Activation ? "SwiGLU kernel" : "residual kernel");
}

}

void reset_status(const CudaContext& context, DeviceTensorView<std::int32_t> status) {
    status_range(status, context.device());
    DeviceScope scope(context.device());
    reset_kernel<<<1, 1, 0, context.stream()>>>(status.data);
    check_cuda(cudaGetLastError(), "status reset kernel");
}

void gather_rows(const CudaContext& context, DeviceTensorView<const float> source,
                 DeviceTensorView<const std::int32_t> indices, DeviceTensorView<float> output,
                 DeviceTensorView<std::int32_t> status) {
    const auto x = validate(source, context.device()), ids = validate(indices, context.device());
    const auto y = validate(output, context.device()), error = status_range(status, context.device());
    require(indices.columns == 1 && output.rows == indices.rows && output.columns == source.columns,
            "CUDA gather 形状不一致");
    require(!overlaps(x, y) && !overlaps(ids, y) && !overlaps(error, x) &&
            !overlaps(error, ids) && !overlaps(error, y), "CUDA gather 输出或 status 与输入重叠");
    const auto count = checked_product(output.rows, output.columns);
    DeviceScope scope(context.device());
    gather_kernel<<<grid_for(count), threads, 0, context.stream()>>>(source, indices, output, status.data);
    check_cuda(cudaGetLastError(), "gather kernel");
}

void rms_norm(const CudaContext& context, DeviceTensorView<const float> input,
              DeviceTensorView<const float> weight, DeviceTensorView<float> output, float epsilon) {
    const auto x = validate(input, context.device()), w = validate(weight, context.device());
    const auto y = validate(output, context.device());
    require(std::isfinite(epsilon) && epsilon > 0, "CUDA RMSNorm epsilon 必须为有限正数");
    require(weight.rows == 1 && input.columns % weight.columns == 0 &&
            input.rows == output.rows && input.columns == output.columns, "CUDA RMSNorm 形状不一致");
    require((!overlaps(x, y) || same_layout(input, output)) && !overlaps(w, y),
            "CUDA RMSNorm 不支持部分重叠或覆盖权重");
    const auto groups = as_int(checked_product(input.rows, input.columns / weight.columns));
    DeviceScope scope(context.device());
    norm_kernel<<<static_cast<unsigned>(groups), threads, 0, context.stream()>>>(input, weight, output, epsilon);
    check_cuda(cudaGetLastError(), "RMSNorm kernel");
}

void rope(const CudaContext& context, DeviceTensorView<float> input,
          DeviceTensorView<const std::int32_t> positions, DeviceTensorView<const float> coefficients,
          DeviceTensorView<std::int32_t> status) {
    const auto x = validate(input, context.device()), p = validate(positions, context.device());
    const auto table = validate(coefficients, context.device()), error = status_range(status, context.device());
    require(coefficients.columns % 2 == 0 && input.columns % coefficients.columns == 0 &&
            positions.columns == 1 && positions.rows == input.rows, "CUDA RoPE head_dim 或形状无效");
    require(!overlaps(x, p) && !overlaps(x, table) && !overlaps(error, x) &&
            !overlaps(error, p) && !overlaps(error, table), "CUDA RoPE 输入、系数或 status 重叠");
    const auto count = checked_product(input.rows, input.columns) / 2;
    DeviceScope scope(context.device());
    rope_kernel<<<grid_for(count), threads, 0, context.stream()>>>(input, positions, coefficients, status.data);
    check_cuda(cudaGetLastError(), "NeoX RoPE kernel");
}

void residual_add(const CudaContext& context, DeviceTensorView<float> hidden, DeviceTensorView<const float> delta) {
    pointwise<false>(context, hidden, delta);
}
void swiglu(const CudaContext& context, DeviceTensorView<float> gate, DeviceTensorView<const float> up) {
    pointwise<true>(context, gate, up);
}

void argmax(const CudaContext& context, DeviceTensorView<const float> logits,
            DeviceTensorView<std::int32_t> tokens, DeviceTensorView<std::int32_t> status) {
    const auto x = validate(logits, context.device()), y = validate(tokens, context.device());
    const auto error = status_range(status, context.device());
    require(tokens.columns == 1 && tokens.rows == logits.rows, "CUDA argmax 形状不一致");
    require(!overlaps(x, y) && !overlaps(error, x) && !overlaps(error, y), "CUDA argmax 输出或 status 重叠");
    DeviceScope scope(context.device());
    argmax_kernel<<<static_cast<unsigned>(logits.rows), threads, 0, context.stream()>>>(logits, tokens, status.data);
    check_cuda(cudaGetLastError(), "finite argmax kernel");
}

}
