#include "attention.h"
#include "device_helpers.cuh"
#include "tensor_validation.h"

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace minillm::cuda {
namespace {

constexpr unsigned threads = 256, warps = threads / 32;
using detail::as_int;
using detail::overlaps;
using detail::record_error;
using detail::require;
using detail::status_range;
using detail::validate;

template<class T>
detail::Range validate_cache(DeviceTensorView<T> cache, KvShape shape, std::size_t layer, int device) {
    const auto range = validate(cache, device);
    for (auto n : {shape.sequences, shape.layers, shape.max_length, shape.kv_heads, shape.head_dim}) { as_int(n); }
    const auto rows = checked_product(checked_product(shape.sequences, shape.layers), checked_product(2, shape.max_length));
    const auto width = checked_product(shape.kv_heads, shape.head_dim);
    require(cache.rows == rows && cache.columns == width && layer < shape.layers, "CUDA 连续 KV 形状或 layer 无效");
    return range;
}

std::array<detail::Range, 2> validate_metadata(DeviceTensorView<const std::int32_t> slots,
    DeviceTensorView<const std::int32_t> positions, std::size_t rows, int device) {
    const auto s = validate(slots, device), p = validate(positions, device);
    require(slots.rows == rows && positions.rows == rows && slots.columns == 1 && positions.columns == 1,
            "CUDA KV metadata 形状无效");
    return {s, p};
}

__device__ bool valid_metadata(int slot, int position, KvShape shape, std::size_t max_context,
                               std::int32_t* status, int row) {
    int bits = 0;
    if (slot < 0 || std::size_t(slot) >= shape.sequences) { bits |= int(DeviceError::invalid_index); }
    if (position < 0 || std::size_t(position) >= shape.max_length || std::size_t(position) >= max_context) {
        bits |= int(DeviceError::invalid_position);
    }
    if (bits && threadIdx.x == 0) { record_error(status, static_cast<DeviceError>(bits), row); }
    return bits == 0;
}

__device__ std::size_t cache_row(KvShape shape, std::size_t slot, std::size_t layer,
                                 std::size_t kind, std::size_t position) {
    return ((slot * shape.layers + layer) * 2 + kind) * shape.max_length + position;
}

__global__ void store_kernel(DeviceTensorView<std::uint16_t> cache, KvShape shape, std::size_t layer,
                             DeviceTensorView<const float> key, DeviceTensorView<const float> value,
                             DeviceTensorView<const std::int32_t> slots,
                             DeviceTensorView<const std::int32_t> positions, std::int32_t* status) {
    const auto row = std::size_t(blockIdx.x);
    const int slot = slots.data[row * slots.stride], position = positions.data[row * positions.stride];
    if (!valid_metadata(slot, position, shape, shape.max_length, status, static_cast<int>(row))) { return; }
    const auto k_base = cache_row(shape, std::size_t(slot), layer, 0, std::size_t(position)) * cache.stride;
    const auto v_base = cache_row(shape, std::size_t(slot), layer, 1, std::size_t(position)) * cache.stride;
    for (std::size_t c = threadIdx.x; c < key.columns; c += threads) {
        const auto k = __float2half_rn(key.data[row * key.stride + c]);
        const auto v = __float2half_rn(value.data[row * value.stride + c]);
        cache.data[k_base + c] = __half_as_ushort(k);
        cache.data[v_base + c] = __half_as_ushort(v);
        if (!isfinite(__half2float(k)) || !isfinite(__half2float(v))) {
            record_error(status, DeviceError::nonfinite, static_cast<int>(row));
        }
    }
}

__global__ void qk_kernel(DeviceTensorView<const std::uint16_t> cache, KvShape shape, std::size_t layer,
                          DeviceTensorView<const float> query, std::size_t query_heads,
                          DeviceTensorView<const std::int32_t> slots,
                          DeviceTensorView<const std::int32_t> positions, std::size_t max_context,
                          DeviceTensorView<float> scores, std::int32_t* status) {
    using Reduction = cub::WarpReduce<float, 32>;
    __shared__ typename Reduction::TempStorage temporary[warps];
    const auto tiles = (max_context + warps - 1) / warps;
    const auto task = std::size_t(blockIdx.x) / tiles, row = task / query_heads, head = task % query_heads;
    const auto warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const auto position = (std::size_t(blockIdx.x) % tiles) * warps + warp;
    const int slot = slots.data[row * slots.stride], last = positions.data[row * positions.stride];
    const bool valid = valid_metadata(slot, last, shape, max_context, status, static_cast<int>(row));
    float sum = 0;
    if (valid && position <= std::size_t(last)) {
        const auto kv_head = head / (query_heads / shape.kv_heads);
        const auto k_base = cache_row(shape, std::size_t(slot), layer, 0, position) * cache.stride + kv_head * shape.head_dim;
        const auto q_base = row * query.stride + head * shape.head_dim;
        for (std::size_t c = lane; c < shape.head_dim; c += 32) {
            sum = fmaf(query.data[q_base + c], __half2float(__ushort_as_half(cache.data[k_base + c])), sum);
        }
    }
    const float dot = Reduction(temporary[warp]).Sum(sum);
    if (lane == 0 && position < max_context) {
        const auto index = row * scores.stride + head * shape.max_length + position;
        const float result = !valid ? CUDART_NAN_F : position > std::size_t(last) ? -CUDART_INF_F
            : dot * (1.0f / sqrtf(static_cast<float>(shape.head_dim)));
        scores.data[index] = result;
        if (valid && position <= std::size_t(last) && !isfinite(result)) {
            record_error(status, DeviceError::nonfinite, static_cast<int>(row));
        }
    }
}

struct Maximum {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

__global__ void softmax_kernel(KvShape shape, std::size_t query_heads,
                               DeviceTensorView<const std::int32_t> slots,
                               DeviceTensorView<const std::int32_t> positions, std::size_t max_context,
                               DeviceTensorView<float> scores, DeviceTensorView<float> probabilities,
                               std::int32_t* status) {
    using MaximumReduction = cub::BlockReduce<float, threads>;
    using SumReduction = cub::BlockReduce<double, threads>;
    __shared__ union {
        typename MaximumReduction::TempStorage maximum;
        typename SumReduction::TempStorage sum;
    } temporary;
    __shared__ float maximum;
    __shared__ double denominator;
    const auto task = std::size_t(blockIdx.x), row = task / query_heads, head = task % query_heads;
    const int slot = slots.data[row * slots.stride], last = positions.data[row * positions.stride];
    auto* output = probabilities.data + row * probabilities.stride + head * shape.max_length;
    if (!valid_metadata(slot, last, shape, max_context, status, static_cast<int>(row))) {
        for (std::size_t p = threadIdx.x; p < max_context; p += threads) { output[p] = CUDART_NAN_F; }
        return;
    }
    const auto length = std::size_t(last) + 1;
    const auto* input = scores.data + row * scores.stride + head * shape.max_length;
    float local_max = -CUDART_INF_F;
    for (std::size_t p = threadIdx.x; p < length; p += threads) {
        const float value = input[p];
        local_max = fmaxf(local_max, isfinite(value) ? value : CUDART_INF_F);
    }
    const float block_max = MaximumReduction(temporary.maximum).Reduce(local_max, Maximum{});
    if (threadIdx.x == 0) { maximum = block_max; }
    __syncthreads();
    if (!isfinite(maximum)) {
        if (threadIdx.x == 0) { record_error(status, DeviceError::nonfinite, static_cast<int>(row)); }
        for (std::size_t p = threadIdx.x; p < length; p += threads) { output[p] = CUDART_NAN_F; }
        return;
    }
    double sum = 0;
    for (std::size_t p = threadIdx.x; p < length; p += threads) {
        const float value = expf(input[p] - maximum);
        output[p] = value;
        sum += static_cast<double>(value);
    }
    const double total = SumReduction(temporary.sum).Sum(sum);
    if (threadIdx.x == 0) { denominator = total; }
    __syncthreads();
    for (std::size_t p = threadIdx.x; p < max_context; p += threads) {
        output[p] = p < length ? static_cast<float>(static_cast<double>(output[p]) / denominator) : 0.0f;
    }
}

__global__ void pv_kernel(DeviceTensorView<const std::uint16_t> cache, KvShape shape, std::size_t layer,
                          std::size_t query_heads, DeviceTensorView<const std::int32_t> slots,
                          DeviceTensorView<const std::int32_t> positions, std::size_t max_context,
                          DeviceTensorView<float> probabilities, DeviceTensorView<float> output, std::int32_t* status) {
    const auto task = std::size_t(blockIdx.x), row = task / query_heads, head = task % query_heads;
    const int slot = slots.data[row * slots.stride], last = positions.data[row * positions.stride];
    auto* result = output.data + row * output.stride + head * shape.head_dim;
    if (!valid_metadata(slot, last, shape, max_context, status, static_cast<int>(row))) {
        for (std::size_t c = threadIdx.x; c < shape.head_dim; c += threads) { result[c] = CUDART_NAN_F; }
        return;
    }
    const auto kv_head = head / (query_heads / shape.kv_heads);
    const auto* probability = probabilities.data + row * probabilities.stride + head * shape.max_length;
    for (std::size_t c = threadIdx.x; c < shape.head_dim; c += threads) {
        float sum = 0;
        for (std::size_t p = 0; p <= std::size_t(last); ++p) {
            const auto index = cache_row(shape, std::size_t(slot), layer, 1, p) * cache.stride + kv_head * shape.head_dim + c;
            sum = fmaf(probability[p], __half2float(__ushort_as_half(cache.data[index])), sum);
        }
        result[c] = sum;
        if (!isfinite(sum)) { record_error(status, DeviceError::nonfinite, static_cast<int>(row)); }
    }
}

}

void store_kv(const CudaContext& context, DeviceTensorView<std::uint16_t> cache, KvShape shape,
              std::size_t layer, DeviceTensorView<const float> key, DeviceTensorView<const float> value,
              DeviceTensorView<const std::int32_t> slots, DeviceTensorView<const std::int32_t> positions,
              DeviceTensorView<std::int32_t> status) {
    const auto kv = validate_cache(cache, shape, layer, context.device());
    const auto k = validate(key, context.device()), v = validate(value, context.device());
    const auto meta = validate_metadata(slots, positions, key.rows, context.device());
    const auto error = status_range(status, context.device());
    require(key.rows == value.rows && key.columns == value.columns && key.columns == cache.columns,
            "CUDA KV 写入形状不一致");
    for (auto input : {k, v, meta[0], meta[1]}) {
        require(!overlaps(input, kv) && !overlaps(input, error), "CUDA KV 写入与输入或 status 重叠");
    }
    require(!overlaps(kv, error), "CUDA KV 与 status 重叠");
    DeviceScope scope(context.device());
    store_kernel<<<static_cast<unsigned>(key.rows), threads, 0, context.stream()>>>(cache, shape, layer, key, value,
        slots, positions, status.data);
    check_cuda(cudaGetLastError(), "FP16 KV store kernel");
}

void causal_softmax(const CudaContext& context, KvShape shape, std::size_t query_heads,
                    DeviceTensorView<const std::int32_t> slots,
                    DeviceTensorView<const std::int32_t> positions, std::size_t max_context,
                    DeviceTensorView<float> scores, DeviceTensorView<float> probabilities,
                    DeviceTensorView<std::int32_t> status) {
    for (auto n : {shape.sequences, shape.layers, shape.max_length, shape.kv_heads, shape.head_dim, query_heads}) {
        as_int(n);
    }
    const auto s = validate(scores, context.device()), p = validate(probabilities, context.device());
    const auto meta = validate_metadata(slots, positions, scores.rows, context.device());
    const auto error = status_range(status, context.device());
    require(query_heads % shape.kv_heads == 0 && max_context > 0 && max_context <= shape.max_length,
            "CUDA softmax 的 GQA 或 context 上界无效");
    require(scores.columns == checked_product(query_heads, shape.max_length) &&
            probabilities.rows == scores.rows && probabilities.columns == scores.columns,
            "CUDA softmax 输出形状无效");
    for (auto output : {p, error}) {
        for (auto input : {s, meta[0], meta[1]}) {
            require(!overlaps(output, input), "CUDA softmax 输出与输入重叠");
        }
    }
    require(!overlaps(p, error), "CUDA softmax 输出与 status 重叠");
    const auto tasks = as_int(checked_product(scores.rows, query_heads));
    DeviceScope scope(context.device());
    softmax_kernel<<<static_cast<unsigned>(tasks), threads, 0, context.stream()>>>(shape, query_heads, slots,
        positions, max_context, scores, probabilities, status.data);
    check_cuda(cudaGetLastError(), "FP64-denominator softmax kernel");
}

void causal_attention(const CudaContext& context, DeviceTensorView<const std::uint16_t> cache, KvShape shape,
                      std::size_t layer, DeviceTensorView<const float> query, std::size_t query_heads,
                      DeviceTensorView<const std::int32_t> slots,
                      DeviceTensorView<const std::int32_t> positions, std::size_t max_context,
                      DeviceTensorView<float> scores, DeviceTensorView<float> probabilities,
                      DeviceTensorView<float> output, DeviceTensorView<std::int32_t> status) {
    const auto kv = validate_cache(cache, shape, layer, context.device()), q = validate(query, context.device());
    const auto s = validate(scores, context.device()), p = validate(probabilities, context.device());
    const auto y = validate(output, context.device()), error = status_range(status, context.device());
    const auto meta = validate_metadata(slots, positions, query.rows, context.device());
    as_int(query_heads);
    require(query_heads % shape.kv_heads == 0 && query.columns == checked_product(query_heads, shape.head_dim),
            "CUDA attention 的 GQA 或 Q 形状无效");
    require(max_context > 0 && max_context <= shape.max_length, "CUDA attention context 上界无效");
    require(scores.rows == query.rows && probabilities.rows == query.rows && output.rows == query.rows &&
            scores.columns == checked_product(query_heads, shape.max_length) &&
            probabilities.columns == scores.columns && output.columns == query.columns, "CUDA attention 输出形状无效");
    const std::array<detail::Range, 4> outputs{s, p, y, error};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (auto input : {kv, q, meta[0], meta[1]}) {
            require(!overlaps(outputs[i], input), "CUDA attention 输出与输入重叠");
        }
        for (std::size_t j = 0; j < i; ++j) { require(!overlaps(outputs[i], outputs[j]), "CUDA attention 输出互相重叠"); }
    }
    const auto tasks = as_int(checked_product(query.rows, query_heads));
    const auto qk_blocks = as_int(checked_product(std::size_t(tasks), (max_context + warps - 1) / warps));
    DeviceScope scope(context.device());
    qk_kernel<<<static_cast<unsigned>(qk_blocks), threads, 0, context.stream()>>>(cache, shape, layer, query,
        query_heads, slots, positions, max_context, scores, status.data);
    check_cuda(cudaGetLastError(), "causal QK kernel");
    softmax_kernel<<<static_cast<unsigned>(tasks), threads, 0, context.stream()>>>(shape, query_heads, slots,
        positions, max_context, scores, probabilities, status.data);
    check_cuda(cudaGetLastError(), "FP64-denominator softmax kernel");
    pv_kernel<<<static_cast<unsigned>(tasks), threads, 0, context.stream()>>>(cache, shape, layer, query_heads,
        slots, positions, max_context, probabilities, output, status.data);
    check_cuda(cudaGetLastError(), "causal PV kernel");
}

}
