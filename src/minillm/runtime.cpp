#include "minillm/runtime.h"

#include "minillm/qwen3_model.h"
#include "minillm/tokenizer.h"
#include "minillm/paged_kv.h"
#include "minillm/parallel.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace minillm {
namespace {

using ProfileClock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(ProfileClock::time_point start) noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(ProfileClock::now() - start).count());
}

class ProfileScope {
public:
    ProfileScope(ForwardProfile* profile, ProfileStage stage, std::int32_t layer = -1,
                 std::size_t count = 0, std::size_t logits = 0) {
        if (profile) {
            profile->stages.push_back({stage, layer, count ? count : profile->input_tokens,
                                      count ? logits : profile->logits_tokens});
            record_ = &profile->stages.back();
            started_ = ProfileClock::now();
        }
    }
    ~ProfileScope() { finish(); }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    void finish() noexcept {
        if (record_) {
            record_->wall_ns = elapsed_ns(started_);
            record_ = nullptr;
        }
    }
    ParallelProfile* parallel() noexcept { return record_ ? &record_->parallel : nullptr; }
    void matrix_shape(std::size_t m, std::size_t n, std::size_t k) noexcept {
        if (record_) {
            record_->matrix_m = m;
            record_->matrix_n = n;
            record_->matrix_k = k;
        }
    }
private:
    StageRecord* record_ = nullptr;
    ProfileClock::time_point started_{};
};

class ForwardScope {
public:
    ForwardScope(ForwardProfile* profile, std::size_t capacity, std::size_t count,
                 std::size_t threads, const PagedKV& cache) : profile_(profile), cache_(cache) {
        if (profile_) {
            profile_->reset(capacity);
            profile_->input_tokens = count;
            profile_->threads = threads;
            profile_->kv_pages_before = cache_.used_pages();
            started_ = ProfileClock::now();
        }
    }
    ~ForwardScope() {
        if (profile_) {
            profile_->wall_ns = elapsed_ns(started_);
            profile_->kv_pages_after = cache_.used_pages();
            std::uint64_t accounted = 0;
            for (const auto& stage : profile_->stages) {
                accounted += stage.wall_ns;
            }
            profile_->unaccounted_ns = profile_->wall_ns - accounted;
        }
    }
private:
    ForwardProfile* profile_;
    const PagedKV& cache_;
    ProfileClock::time_point started_{};
};

RuntimeConfig checked(RuntimeConfig config) {
    if (config.context_tokens == 0 || config.page_tokens == 0 ||
        config.context_tokens % config.page_tokens != 0 || config.max_sequences == 0 ||
        config.max_sequences > 256 || config.batch_tokens == 0 ||
        config.batch_tokens > config.context_tokens) {
        throw std::invalid_argument("invalid MiniLLM context, page, sequence, or batch limits");
    }
    return config;
}

const ModelDimensions& checked_dimensions(const RuntimeConfig& config, const Qwen3Model& model) {
    if (config.context_tokens > model.dimensions().trained_context) {
        throw std::invalid_argument("context_tokens exceeds the trained model context");
    }
    return model.dimensions();
}

} // namespace

struct Runtime::Impl {
    RuntimeConfig config;
    Qwen3Model model;
    const ModelDimensions& dims;
    ParallelExecutor executor;
    PagedKV cache;
    const TensorView& embeddings;
    const TensorView& output;
    const std::vector<float>& output_norm;
    const std::vector<Qwen3Layer>& layers;
    Tokenizer tokenizer;

    explicit Impl(RuntimeConfig cfg)
        : config(checked(std::move(cfg))), model(config.model_path),
          dims(checked_dimensions(config, model)), executor(config.threads),
          cache({config.context_tokens / config.page_tokens, config.page_tokens, dims.layers,
                 dims.kv_heads * dims.head_dim, config.max_sequences}),
          embeddings(model.embeddings()), output(model.output()),
          output_norm(model.output_norm()), layers(model.layers()),
          tokenizer(config.model_path, dims.vocabulary) {}

    std::size_t profile_capacity() const noexcept {
        return 3 + layers.size() * 14 + config.batch_tokens * 2;
    }

    void multiply(const TensorView& matrix, const std::vector<float>& input,
                  std::vector<float>& output_values, std::size_t count,
                  ForwardProfile* profile, ProfileStage stage, std::int32_t layer = -1) {
        ProfileScope scope(profile, stage, layer, count,
                           stage == ProfileStage::lm_head ? count : (profile ? profile->logits_tokens : 0));
        scope.matrix_shape(count, matrix.rows, matrix.columns);
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
        }, scope.parallel());
    }

    void normalize(const std::vector<float>& input, const std::vector<float>& weight,
                   std::vector<float>& normalized, std::size_t count, std::size_t width,
                   ForwardProfile* profile, ProfileStage stage, std::int32_t layer) {
        ProfileScope scope(profile, stage, layer);
        normalized.resize(input.size());
        executor.run(count, 1, [&](auto begin, auto end) {
            for (auto i = begin; i < end; ++i) {
                rms_norm(input.data() + i * width, weight.data(), normalized.data() + i * width,
                          width, dims.rms_epsilon, config.kernels);
            }
        }, scope.parallel());
    }

    std::vector<Logits> forward(std::span<const InputToken> tokens, ForwardProfile* profile) {
        ForwardScope forward_scope(profile, profile_capacity(), tokens.size(), config.threads, cache);
        ProfileScope kv_scope(profile, ProfileStage::kv_prepare);
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
        if (profile) {
            std::array<bool, 256> seen{};
            for (const auto& token : tokens) {
                profile->logits_tokens += token.logits ? 1 : 0;
                const auto sequence = static_cast<std::size_t>(token.sequence);
                if (!seen[sequence]) {
                    seen[sequence] = true;
                    ++profile->sequences;
                    const auto before = cache.length(sequence);
                    profile->context_before_sum += before;
                    profile->context_before_max = std::max(profile->context_before_max, before);
                    profile->context_after_sum += lengths[sequence];
                    profile->context_after_max = std::max(profile->context_after_max, lengths[sequence]);
                }
            }
            profile->stages.front().logits_tokens = profile->logits_tokens;
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
        kv_scope.finish();
        const auto count = tokens.size();
        const auto width = dims.embedding;
        const auto q_width = dims.heads * dims.head_dim;
        const auto kv_width = dims.kv_heads * dims.head_dim;
        ProfileScope embedding_scope(profile, ProfileStage::embedding);
        std::vector<float> hidden(count * width);
        for (std::size_t i = 0; i < count; ++i) {
            decode_row(embeddings.type, embeddings.row(static_cast<std::size_t>(tokens[i].token)),
                       hidden.data() + i * width, width);
        }
        embedding_scope.finish();
        ProfileScope rope_scope(profile, ProfileStage::rope_prepare);
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
        rope_scope.finish();
        std::vector<float> normalized, query, key, value, attention(count * q_width);
        std::vector<float> projected, gate, up, down;
        for (std::size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
            const auto& layer = layers[layer_id];
            const auto layer_index = static_cast<std::int32_t>(layer_id);
            normalize(hidden, layer.attention_norm, normalized, count, width,
                      profile, ProfileStage::attention_norm, layer_index);
            multiply(layer.query, normalized, query, count, profile, ProfileStage::query_projection, layer_index);
            multiply(layer.key, normalized, key, count, profile, ProfileStage::key_projection, layer_index);
            multiply(layer.value, normalized, value, count, profile, ProfileStage::value_projection, layer_index);
            ProfileScope qk_scope(profile, ProfileStage::qk_norm_rope_kv, layer_index);
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
            }, qk_scope.parallel());
            qk_scope.finish();
            ProfileScope attention_scope(profile, ProfileStage::attention, layer_index);
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
                        add_scaled_f16(v.data() + kv_head * dims.head_dim, probability, out,
                                       dims.head_dim, config.kernels);
                    }
                }
            }, attention_scope.parallel());
            attention_scope.finish();
            multiply(layer.attention_output, attention, projected, count,
                     profile, ProfileStage::output_projection, layer_index);
            ProfileScope attention_residual(profile, ProfileStage::attention_residual, layer_index);
            for (std::size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += projected[i];
            }
            attention_residual.finish();
            normalize(hidden, layer.ffn_norm, normalized, count, width,
                      profile, ProfileStage::ffn_norm, layer_index);
            multiply(layer.gate, normalized, gate, count, profile, ProfileStage::gate_projection, layer_index);
            multiply(layer.up, normalized, up, count, profile, ProfileStage::up_projection, layer_index);
            ProfileScope swiglu_scope(profile, ProfileStage::swiglu, layer_index);
            executor.run(gate.size(), 256, [&](auto begin, auto end) {
                for (auto i = begin; i < end; ++i) {
                    gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
                }
            }, swiglu_scope.parallel());
            swiglu_scope.finish();
            multiply(layer.down, gate, down, count, profile, ProfileStage::down_projection, layer_index);
            ProfileScope ffn_residual(profile, ProfileStage::ffn_residual, layer_index);
            for (std::size_t i = 0; i < hidden.size(); ++i) {
                hidden[i] += down[i];
            }
        }
        std::vector<Logits> result;
        for (std::size_t i = 0; i < count; ++i) {
            if (!tokens[i].logits) {
                continue;
            }
            ProfileScope final_scope(profile, ProfileStage::final_norm, -1, 1, 1);
            std::vector<float> final(width);
            rms_norm(hidden.data() + i * width, output_norm.data(), final.data(),
                     width, dims.rms_epsilon, config.kernels);
            final_scope.finish();
            Logits logits{tokens[i].sequence, {}};
            multiply(output, final, logits.values, 1, profile, ProfileStage::lm_head);
            result.push_back(std::move(logits));
        }
        if (profile) {
            profile->completed = true;
        }
        return result;
    }
};

Runtime::Runtime(RuntimeConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
Runtime::~Runtime() = default;
const ModelDimensions& Runtime::dimensions() const noexcept { return impl_->dims; }
const RuntimeConfig& Runtime::config() const noexcept { return impl_->config; }

std::vector<std::int32_t> Runtime::tokenize(std::string_view text) const {
    return impl_->tokenizer.tokenize(text);
}
std::string Runtime::token_piece(std::int32_t token) const {
    return impl_->tokenizer.token_piece(token);
}
bool Runtime::is_eog(std::int32_t token) const { return impl_->tokenizer.is_eog(token); }
ForwardProfile Runtime::make_profile() const {
    ForwardProfile result;
    result.stages.reserve(impl_->profile_capacity());
    return result;
}
std::vector<Logits> Runtime::forward(std::span<const InputToken> tokens, ForwardProfile* profile) {
    return impl_->forward(tokens, profile);
}
void Runtime::share_prefix(std::int32_t source, std::int32_t target, std::size_t tokens) {
    impl_->cache.share_prefix(static_cast<std::size_t>(source), static_cast<std::size_t>(target), tokens);
}
void Runtime::clear_sequence(std::int32_t sequence) noexcept {
    impl_->cache.clear(static_cast<std::size_t>(sequence));
}
std::size_t Runtime::used_kv_pages() const noexcept { return impl_->cache.used_pages(); }
std::size_t Runtime::resident_kv_bytes() const noexcept { return impl_->cache.resident_bytes(); }

} // namespace minillm
