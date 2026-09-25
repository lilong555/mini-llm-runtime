#pragma once

#include "ops.h"

namespace minillm::cuda {

struct KvShape {
    std::size_t sequences, layers, max_length, kv_heads, head_dim;
};

// cache 行为 [sequence][layer][K_or_V][position]，列为 [kv_head * head_dim]。
// 调用方已在 host 预检同批写入位置唯一；设备仍检查 slot/position 与转换的 finite 状态。
void store_kv(const CudaContext& context, DeviceTensorView<std::uint16_t> cache, KvShape shape,
              std::size_t layer, DeviceTensorView<const float> key, DeviceTensorView<const float> value,
              DeviceTensorView<const std::int32_t> slots, DeviceTensorView<const std::int32_t> positions,
              DeviceTensorView<std::int32_t> status);

// scores/probabilities 为 [B,Hq*Lmax]；每个 query 只读自身 position+1 个 KV。
// max_context 是当前 batch 的位置上界，不替代各 query 的因果长度。
// 接口只入队，QK/PV 使用 FP32，softmax 分母使用 FP64；无 host KV gather。
void causal_attention(const CudaContext& context, DeviceTensorView<const std::uint16_t> cache, KvShape shape,
                      std::size_t layer, DeviceTensorView<const float> query, std::size_t query_heads,
                      DeviceTensorView<const std::int32_t> slots,
                      DeviceTensorView<const std::int32_t> positions, std::size_t max_context,
                      DeviceTensorView<float> scores, DeviceTensorView<float> probabilities,
                      DeviceTensorView<float> output, DeviceTensorView<std::int32_t> status);

}
