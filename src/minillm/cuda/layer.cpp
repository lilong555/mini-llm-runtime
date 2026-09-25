#include "layer.h"
#include "tensor_validation.h"

namespace minillm::cuda {

LayerExecutor::LayerExecutor(CudaStorage& storage)
    : storage_(storage), dimensions_(storage.plan().dimensions),
      kv_shape_{storage.plan().limits.max_sequences, dimensions_.layers, storage.plan().limits.max_model_len,
                dimensions_.kv_heads, dimensions_.head_dim} {
    layers_.reserve(dimensions_.layers);
    for (std::size_t layer = 0; layer < dimensions_.layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const auto w = [&](const char* name) { return storage.weight(prefix + name + ".weight"); };
        layers_.push_back({w("attn_norm"), w("attn_q_norm"), w("attn_k_norm"), w("ffn_norm"),
                          w("attn_q"), w("attn_k"), w("attn_v"), w("attn_output"),
                          w("ffn_gate"), w("ffn_up"), w("ffn_down")});
    }
}

void LayerExecutor::enqueue(std::size_t layer, std::size_t tokens, std::size_t max_context) {
    using detail::read_only;
    if (layer >= layers_.size() || tokens == 0 || tokens > storage_.plan().limits.max_batch_tokens ||
        max_context == 0 || max_context > kv_shape_.max_length) {
        throw std::invalid_argument("CUDA layer、batch 或 context 无效");
    }
    detail::as_int(checked_product(checked_product(tokens, dimensions_.heads), (max_context + 7) / 8));
    const auto& w = layers_[layer];
    const auto& context = storage_.context();
    const auto scratch = [&](Workspace name) { return storage_.workspace<float>(name, tokens); };
    const auto hidden = scratch(Workspace::hidden), normalized = scratch(Workspace::normalized);
    const auto query = scratch(Workspace::query), key = scratch(Workspace::key), value = scratch(Workspace::value);
    const auto attention = scratch(Workspace::attention), projected = scratch(Workspace::projected);
    const auto gate = scratch(Workspace::gate), up = scratch(Workspace::up), down = scratch(Workspace::down);
    const auto scores = scratch(Workspace::scores), probabilities = scratch(Workspace::probabilities);
    const auto slots = read_only(storage_.workspace<std::int32_t>(Workspace::slots, tokens));
    const auto positions = read_only(storage_.workspace<std::int32_t>(Workspace::positions, tokens));
    const auto status = storage_.workspace<std::int32_t>(Workspace::status, 1);
    const auto coefficients = read_only(storage_.workspace<float>(Workspace::rope_coefficients, kv_shape_.max_length));
    const auto kv = storage_.kv_view();
    rms_norm(context, read_only(hidden), w.attention_norm, normalized, dimensions_.rms_epsilon);
    matrix_multiply(context, read_only(normalized), w.query, query);
    matrix_multiply(context, read_only(normalized), w.key, key);
    matrix_multiply(context, read_only(normalized), w.value, value);
    rms_norm(context, read_only(query), w.query_norm, query, dimensions_.rms_epsilon);
    rms_norm(context, read_only(key), w.key_norm, key, dimensions_.rms_epsilon);
    rope(context, query, positions, coefficients, status);
    rope(context, key, positions, coefficients, status);
    store_kv(context, kv, kv_shape_, layer, read_only(key), read_only(value), slots, positions, status);
    causal_attention(context, read_only(kv), kv_shape_, layer, read_only(query), dimensions_.heads,
                     slots, positions, max_context, scores, probabilities, attention, status);
    matrix_multiply(context, read_only(attention), w.output, projected);
    residual_add(context, hidden, read_only(projected));
    rms_norm(context, read_only(hidden), w.ffn_norm, normalized, dimensions_.rms_epsilon);
    matrix_multiply(context, read_only(normalized), w.gate, gate);
    matrix_multiply(context, read_only(normalized), w.up, up);
    swiglu(context, gate, read_only(up));
    matrix_multiply(context, read_only(gate), w.down, down);
    residual_add(context, hidden, read_only(down));
    check_finite(context, read_only(hidden), status);
}

}
