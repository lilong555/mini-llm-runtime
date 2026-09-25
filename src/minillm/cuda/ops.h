#pragma once

#include "minillm/cuda/matrix.h"

namespace minillm::cuda {

enum class DeviceError : std::int32_t { nonfinite = 1, invalid_index = 2, invalid_position = 4 };

// status 为 [1,2] 的 I32：错误位集合、首个错误输入行；只在一次执行开始时重置。
// 所有接口仅向 context 的 stream 入队，借用视图不转移所有权。
void reset_status(const CudaContext& context, DeviceTensorView<std::int32_t> status);
void check_finite(const CudaContext& context, DeviceTensorView<const float> input,
                  DeviceTensorView<std::int32_t> status);

// source[N,D]、indices[M,1]、output[M,D]，允许重复索引，不允许输出与输入重叠。
// 设备索引非法时先屏蔽读取，再将对应输出行写为 NaN 并记录错误。
void gather_rows(const CudaContext& context, DeviceTensorView<const float> source,
                 DeviceTensorView<const std::int32_t> indices, DeviceTensorView<float> output,
                 DeviceTensorView<std::int32_t> status);

// weight[1,D]，每行按 D 分组；普通 norm 为一组，Q/K norm 为每个 head 一组。
// 支持完全相同布局的原地输出，拒绝部分重叠。最大值缩放避免有限输入平方溢出。
void rms_norm(const CudaContext& context, DeviceTensorView<const float> input,
              DeviceTensorView<const float> weight, DeviceTensorView<float> output, float epsilon);

// coefficients[L,D] 每行前 D/2 为 cosine，后 D/2 为 sine；D 是显式 head_dim。
// positions[M,1] 逐 token 指定位置，input[M,H*D] 原地执行 NeoX 两半旋转。
void rope(const CudaContext& context, DeviceTensorView<float> input,
          DeviceTensorView<const std::int32_t> positions, DeviceTensorView<const float> coefficients,
          DeviceTensorView<std::int32_t> status);

void residual_add(const CudaContext& context, DeviceTensorView<float> hidden,
                  DeviceTensorView<const float> delta);
void swiglu(const CudaContext& context, DeviceTensorView<float> gate,
            DeviceTensorView<const float> up);

// logits[M,V] -> tokens[M,1]；相等取最小 ID，任意 NaN/Inf 使该行返回 -1 并记录错误。
void argmax(const CudaContext& context, DeviceTensorView<const float> logits,
            DeviceTensorView<std::int32_t> tokens, DeviceTensorView<std::int32_t> status);

}
