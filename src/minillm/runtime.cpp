#include "minillm/runtime.h"

#include "minillm/gguf_model.h"
#include "minillm/paged_kv.h"
#include "minillm/parallel.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

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

RuntimeConfig checked(RuntimeConfig config) {
    if (config.context_tokens == 0 || config.page_tokens == 0 ||
        config.context_tokens % config.page_tokens != 0 || config.max_sequences == 0 ||
        config.max_sequences > 256 || config.batch_tokens == 0 ||
        config.batch_tokens > config.context_tokens) {
        throw std::invalid_argument("invalid MiniLLM context, page, sequence, or batch limits");
    }
    return config;
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

struct Runtime::Impl {
    struct Layer {
        std::vector<float> attention_norm;
        std::vector<float> query_norm;
        std::vector<float> key_norm;
        std::vector<float> ffn_norm;
        TensorView query;
        TensorView key;
        TensorView value;
        TensorView attention_output;
        TensorView gate;
        TensorView up;
        TensorView down;
    };

    RuntimeConfig config;
    GgufModel weights;
    ModelDimensions dims;
    ParallelExecutor executor;
    PagedKV cache;
    TensorView embeddings;
    TensorView output;
    std::vector<float> output_norm;
    std::vector<Layer> layers;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> vocabulary_model{nullptr, llama_model_free};
    const llama_vocab* vocab = nullptr;

    explicit Impl(RuntimeConfig cfg)
        : config(checked(std::move(cfg))), weights(config.model_path), dims(load_dimensions(weights)),
          executor(config.threads),
          cache({config.context_tokens / config.page_tokens, config.page_tokens, dims.layers,
                 dims.kv_heads * dims.head_dim, config.max_sequences}),
          embeddings(matrix(weights, "token_embd.weight", dims.vocabulary, dims.embedding)),
          output(weights.has_tensor("output.weight")
              ? matrix(weights, "output.weight", dims.vocabulary, dims.embedding) : embeddings),
          output_norm(norm_weight(weights, "output_norm.weight", dims.embedding)) {
        if (config.context_tokens > dims.trained_context) {
            throw std::invalid_argument("context_tokens exceeds the trained model context");
        }
        const auto q_width = dims.heads * dims.head_dim;
        const auto kv_width = dims.kv_heads * dims.head_dim;
        for (std::size_t i = 0; i < dims.layers; ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            layers.push_back({
                norm_weight(weights, prefix + "attn_norm.weight", dims.embedding),
                norm_weight(weights, prefix + "attn_q_norm.weight", dims.head_dim),
                norm_weight(weights, prefix + "attn_k_norm.weight", dims.head_dim),
                norm_weight(weights, prefix + "ffn_norm.weight", dims.embedding),
                matrix(weights, prefix + "attn_q.weight", q_width, dims.embedding),
                matrix(weights, prefix + "attn_k.weight", kv_width, dims.embedding),
                matrix(weights, prefix + "attn_v.weight", kv_width, dims.embedding),
                matrix(weights, prefix + "attn_output.weight", dims.embedding, q_width),
                matrix(weights, prefix + "ffn_gate.weight", dims.feed_forward, dims.embedding),
                matrix(weights, prefix + "ffn_up.weight", dims.feed_forward, dims.embedding),
                matrix(weights, prefix + "ffn_down.weight", dims.embedding, dims.feed_forward)});
        }
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        vocabulary_model.reset(llama_model_load_from_file(config.model_path.c_str(), params));
        if (!vocabulary_model) {
            throw std::runtime_error("cannot load the GGUF tokenizer");
        }
        vocab = llama_model_get_vocab(vocabulary_model.get());
        if (static_cast<std::size_t>(llama_vocab_n_tokens(vocab)) != dims.vocabulary) {
            throw std::runtime_error("tokenizer vocabulary does not match the embedding matrix");
        }
    }

    void multiply(const TensorView& matrix, const std::vector<float>& input,
                  std::vector<float>& output_values, std::size_t count) {
        output_values.resize(count * matrix.rows);
        executor.run(matrix.rows, 16, [&](auto begin, auto end) {
            for (auto row = begin; row < end; ++row) {
                const auto* data = matrix.row(row);
                for (std::size_t token = 0; token < count; ++token) {
                    output_values[token * matrix.rows + row] = dot_row(
                        matrix.type, data, input.data() + token * matrix.columns,
                        matrix.columns, config.kernels);
                }
            }
        });
    }

    void normalize(const std::vector<float>& input, const std::vector<float>& weight,
                   std::vector<float>& normalized, std::size_t count, std::size_t width) {
        normalized.resize(input.size());
        executor.run(count, 1, [&](auto begin, auto end) {
            for (auto i = begin; i < end; ++i) {
                rms_norm(input.data() + i * width, weight.data(), normalized.data() + i * width,
                          width, dims.rms_epsilon, config.kernels);
            }
        });
    }

    std::vector<Logits> forward(std::span<const InputToken> tokens) {
        if (tokens.empty() || tokens.size() > config.batch_tokens) {
            throw std::invalid_argument("invalid MiniLLM batch size");
        }
        std::vector<std::size_t> lengths(config.max_sequences);
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            lengths[i] = cache.length(i);
        }
        for (const auto& token : tokens) {
            if (token.sequence < 0 || static_cast<std::size_t>(token.sequence) >= lengths.size() ||
                token.token < 0 || static_cast<std::size_t>(token.token) >= dims.vocabulary ||
                token.position < 0 || static_cast<std::size_t>(token.position) >= dims.trained_context ||
                static_cast<std::size_t>(token.position) != lengths[static_cast<std::size_t>(token.sequence)]++) {
                throw std::invalid_argument("invalid token, sequence, or noncontiguous position in MiniLLM batch");
            }
        }
        std::size_t pages_needed = 0;
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            const auto before = cache.length(i);
            if (before == lengths[i]) {
                continue;
            }
            pages_needed += (lengths[i] + config.page_tokens - 1) / config.page_tokens -
                (before + config.page_tokens - 1) / config.page_tokens;
            if (before % config.page_tokens != 0 &&
                cache.references(cache.page_table(i).back()) > 1) {
                ++pages_needed;
            }
        }
        if (pages_needed > cache.capacity() - cache.used_pages()) {
            throw std::runtime_error("MiniLLM KV page capacity exceeded");
        }
        for (const auto& token : tokens) {
            cache.append(static_cast<std::size_t>(token.sequence), static_cast<std::size_t>(token.position));
        }
        const auto count = tokens.size();
        const auto width = dims.embedding;
        const auto q_width = dims.heads * dims.head_dim;
        const auto kv_width = dims.kv_heads * dims.head_dim;
        std::vector<float> hidden(count * width);
        for (std::size_t i = 0; i < count; ++i) {
            decode_row(embeddings.type, embeddings.row(static_cast<std::size_t>(tokens[i].token)),
                       hidden.data() + i * width, width);
        }
        const auto half_head = dims.head_dim / 2;
        std::vector<float> cosine(count * half_head), sine(count * half_head);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < half_head; ++j) {
                const auto frequency = std::pow(dims.rope_base, -2.0f * static_cast<float>(j) /
                                                               static_cast<float>(dims.head_dim));
                const auto angle = static_cast<float>(tokens[i].position) * frequency;
                cosine[i * half_head + j] = std::cos(angle);
                sine[i * half_head + j] = std::sin(angle);
            }
        }
        std::vector<float> normalized, query, key, value, attention(count * q_width);
        std::vector<float> projected, gate, up, down;
        for (std::size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
            const auto& layer = layers[layer_id];
            normalize(hidden, layer.attention_norm, normalized, count, width);
            multiply(layer.query, normalized, query, count);
            multiply(layer.key, normalized, key, count);
            multiply(layer.value, normalized, value, count);
            executor.run(count, 1, [&](auto begin, auto end) {
                for (auto i = begin; i < end; ++i) {
                    const auto apply_rope = [&](float* head, const std::vector<float>& norm) {
                        rms_norm(head, norm.data(), head, dims.head_dim, dims.rms_epsilon, config.kernels);
                        for (std::size_t j = 0; j < half_head; ++j) {
                            const auto left = head[j];
                            const auto right = head[j + half_head];
                            const auto c = cosine[i * half_head + j];
                            const auto s = sine[i * half_head + j];
                            head[j] = left * c - right * s;
                            head[j + half_head] = left * s + right * c;
                        }
                    };
                    for (std::size_t head = 0; head < dims.heads; ++head) {
                        apply_rope(query.data() + i * q_width + head * dims.head_dim, layer.query_norm);
                    }
                    for (std::size_t head = 0; head < dims.kv_heads; ++head) {
                        apply_rope(key.data() + i * kv_width + head * dims.head_dim, layer.key_norm);
                    }
                    cache.store(static_cast<std::size_t>(tokens[i].sequence), layer_id,
                        static_cast<std::size_t>(tokens[i].position),
                        {key.data() + i * kv_width, kv_width}, {value.data() + i * kv_width, kv_width});
                }
            });
            executor.run(count * dims.heads, 1, [&](auto begin, auto end) {
                thread_local std::vector<float> scores;
                for (auto task = begin; task < end; ++task) {
                    const auto i = task / dims.heads;
                    const auto head = task % dims.heads;
                    const auto kv_head = head / (dims.heads / dims.kv_heads);
                    const auto sequence = static_cast<std::size_t>(tokens[i].sequence);
                    const auto length = static_cast<std::size_t>(tokens[i].position) + 1;
                    const auto* q = query.data() + i * q_width + head * dims.head_dim;
                    auto* out = attention.data() + i * q_width + head * dims.head_dim;
                    std::fill(out, out + dims.head_dim, 0.0f);
                    scores.resize(length);
                    auto maximum = -std::numeric_limits<float>::infinity();
                    const auto scale = 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
                    for (std::size_t position = 0; position < length; ++position) {
                        const auto k = cache.key(sequence, layer_id, position);
                        scores[position] = dot_f16(k.data() + kv_head * dims.head_dim, q,
                                                  dims.head_dim, config.kernels) * scale;
                        maximum = std::max(maximum, scores[position]);
                    }
                    double denominator = 0;
                    for (auto& score : scores) {
                        score = std::exp(score - maximum);
                        denominator += score;
                    }
                    for (std::size_t position = 0; position < length; ++position) {
                        const auto probability = static_cast<float>(scores[position] / denominator);
                        const auto v = cache.value(sequence, layer_id, position);
                        for (std::size_t j = 0; j < dims.head_dim; ++j) {
                            out[j] += probability * half_to_float(v[kv_head * dims.head_dim + j]);
                        }
                    }
                }
            });
            multiply(layer.attention_output, attention, projected, count);
            for (std::size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += projected[i];
            }
            normalize(hidden, layer.ffn_norm, normalized, count, width);
            multiply(layer.gate, normalized, gate, count);
            multiply(layer.up, normalized, up, count);
            executor.run(gate.size(), 256, [&](auto begin, auto end) {
                for (auto i = begin; i < end; ++i) {
                    gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
                }
            });
            multiply(layer.down, gate, down, count);
            for (std::size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += down[i];
            }
        }
        std::vector<Logits> result;
        for (std::size_t i = 0; i < count; ++i) {
            if (!tokens[i].logits) {
                continue;
            }
            std::vector<float> final(width);
            rms_norm(hidden.data() + i * width, output_norm.data(), final.data(),
                     width, dims.rms_epsilon, config.kernels);
            Logits logits{tokens[i].sequence, {}};
            multiply(output, final, logits.values, 1);
            result.push_back(std::move(logits));
        }
        return result;
    }
};

Runtime::Runtime(RuntimeConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Runtime::~Runtime() = default;
const ModelDimensions& Runtime::dimensions() const noexcept { return impl_->dims; }
const RuntimeConfig& Runtime::config() const noexcept { return impl_->config; }

std::vector<std::int32_t> Runtime::tokenize(std::string_view text) const {
    if (text.size() > 262144) {
        throw std::invalid_argument("prompt exceeds 256 KiB");
    }
    std::vector<std::int32_t> tokens(text.size() + 8);
    auto count = llama_tokenize(impl_->vocab, text.data(), static_cast<std::int32_t>(text.size()),
                                tokens.data(), static_cast<std::int32_t>(tokens.size()), true, true);
    if (count < 0) {
        tokens.resize(static_cast<std::size_t>(-count));
        count = llama_tokenize(impl_->vocab, text.data(), static_cast<std::int32_t>(text.size()),
                               tokens.data(), static_cast<std::int32_t>(tokens.size()), true, true);
    }
    if (count < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

std::string Runtime::token_piece(std::int32_t token) const {
    std::array<char, 256> buffer{};
    auto count = llama_token_to_piece(impl_->vocab, token, buffer.data(),
                                      static_cast<std::int32_t>(buffer.size()), 0, false);
    if (count >= 0) {
        return {buffer.data(), static_cast<std::size_t>(count)};
    }
    std::string result(static_cast<std::size_t>(-count), '\0');
    count = llama_token_to_piece(impl_->vocab, token, result.data(),
                                 static_cast<std::int32_t>(result.size()), 0, false);
    if (count < 0) {
        throw std::runtime_error("detokenization failed");
    }
    result.resize(static_cast<std::size_t>(count));
    return result;
}

bool Runtime::is_eog(std::int32_t token) const { return llama_vocab_is_eog(impl_->vocab, token); }
std::vector<Logits> Runtime::forward(std::span<const InputToken> tokens) { return impl_->forward(tokens); }
void Runtime::share_prefix(std::int32_t source, std::int32_t target, std::size_t tokens) {
    impl_->cache.share_prefix(static_cast<std::size_t>(source), static_cast<std::size_t>(target), tokens);
}
void Runtime::clear_sequence(std::int32_t sequence) noexcept {
    impl_->cache.clear(static_cast<std::size_t>(sequence));
}
std::size_t Runtime::used_kv_pages() const noexcept { return impl_->cache.used_pages(); }
std::size_t Runtime::resident_kv_bytes() const noexcept { return impl_->cache.resident_bytes(); }

} // namespace minillm
