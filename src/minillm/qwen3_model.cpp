#include "minillm/qwen3_model.h"

#include <cmath>
#include <stdexcept>

namespace minillm {
namespace {

ModelDimensions load_dimensions(const GgufModel& model) {
    if (model.string_value("general.architecture") != "qwen3") {
        throw std::runtime_error("MiniLLM currently supports the dense Qwen3 architecture");
    }
    const auto integer = [&](const char* name) {
        const auto value = model.integer_value(name);
        if (value == 0 || value > 1000000) {
            throw std::runtime_error(std::string("invalid model dimension: ") + name);
        }
        return static_cast<std::size_t>(value);
    };
    ModelDimensions dims{
        integer("qwen3.embedding_length"), integer("qwen3.block_count"),
        integer("qwen3.attention.head_count"), integer("qwen3.attention.head_count_kv"),
        integer("qwen3.attention.key_length"), integer("qwen3.feed_forward_length"),
        model.tensor("token_embd.weight").rows, integer("qwen3.context_length"),
        model.float_value("qwen3.attention.layer_norm_rms_epsilon"),
        model.float_value("qwen3.rope.freq_base")};
    if (dims.heads % dims.kv_heads != 0 || dims.head_dim % 2 != 0 ||
        dims.head_dim != integer("qwen3.attention.value_length") ||
        (model.contains("qwen3.rope.dimension_count") &&
         dims.head_dim != integer("qwen3.rope.dimension_count")) ||
        !std::isfinite(dims.rms_epsilon) || dims.rms_epsilon <= 0 ||
        !std::isfinite(dims.rope_base) || dims.rope_base <= 0) {
        throw std::runtime_error("unsupported Qwen3 attention or normalization configuration");
    }
    if (model.contains("qwen3.rope.scaling.type") &&
        model.string_value("qwen3.rope.scaling.type") != "none") {
        throw std::runtime_error("RoPE scaling is not supported by MiniLLM");
    }
    return dims;
}

TensorView matrix(const GgufModel& model, const std::string& name,
                  std::size_t rows, std::size_t columns) {
    const auto value = model.tensor(name);
    if (value.rows != rows || value.columns != columns) {
        throw std::runtime_error("unexpected GGUF tensor shape: " + name);
    }
    return value;
}

std::vector<float> norm_weight(const GgufModel& model, const std::string& name, std::size_t size) {
    return matrix(model, name, 1, size).vector();
}

} // namespace

Qwen3Model::Qwen3Model(const std::string& path)
    : weights_(path), dims_(load_dimensions(weights_)),
      embeddings_(matrix(weights_, "token_embd.weight", dims_.vocabulary, dims_.embedding)),
      output_(weights_.has_tensor("output.weight")
          ? matrix(weights_, "output.weight", dims_.vocabulary, dims_.embedding) : embeddings_),
      output_norm_(norm_weight(weights_, "output_norm.weight", dims_.embedding)) {
    const auto q_width = dims_.heads * dims_.head_dim;
    const auto kv_width = dims_.kv_heads * dims_.head_dim;
    for (std::size_t i = 0; i < dims_.layers; ++i) {
        const auto prefix = "blk." + std::to_string(i) + ".";
        layers_.push_back({
            norm_weight(weights_, prefix + "attn_norm.weight", dims_.embedding),
            norm_weight(weights_, prefix + "attn_q_norm.weight", dims_.head_dim),
            norm_weight(weights_, prefix + "attn_k_norm.weight", dims_.head_dim),
            norm_weight(weights_, prefix + "ffn_norm.weight", dims_.embedding),
            matrix(weights_, prefix + "attn_q.weight", q_width, dims_.embedding),
            matrix(weights_, prefix + "attn_k.weight", kv_width, dims_.embedding),
            matrix(weights_, prefix + "attn_v.weight", kv_width, dims_.embedding),
            matrix(weights_, prefix + "attn_output.weight", dims_.embedding, q_width),
            matrix(weights_, prefix + "ffn_gate.weight", dims_.feed_forward, dims_.embedding),
            matrix(weights_, prefix + "ffn_up.weight", dims_.feed_forward, dims_.embedding),
            matrix(weights_, prefix + "ffn_down.weight", dims_.embedding, dims_.feed_forward)});
    }
}

} // namespace minillm
