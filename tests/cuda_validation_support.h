#pragma once

#include "test_support.h"
#include "../apps/cuda_reports.h"
#include "minillm/runtime.h"
#include "ggml-backend.h"
#include "hash/hash.h"
#include "llama.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <set>

namespace cuda_validation {
using namespace minillm;
using namespace minillm::cuda;
using cuda_reports::json;
using Batch = std::vector<InputToken>;

struct TeacherCase {
    std::size_t corpus, length, chunk, sequences;
};

inline std::string case_id(const json& contract, const TeacherCase& c) {
    return contract.at("corpus").at(c.corpus).at("id").get<std::string>()+
        "-l"+std::to_string(c.length)+"-c"+std::to_string(c.chunk)+"-s"+std::to_string(c.sequences);
}
inline std::vector<TeacherCase> teacher_cases(const json& contract) {
    const auto& recipe = contract.at("teacher_forcing");
    std::vector<TeacherCase> result;
    for (const auto& sequences : recipe.at("sequence_counts")) {
        for (std::size_t corpus = 0; corpus < contract.at("corpus").size(); ++corpus) {
            for (const auto& length : recipe.at("lengths")) {
                for (const auto& chunk : recipe.at("chunk_tokens")) {
                    result.push_back({corpus,length.get<std::size_t>(),chunk.get<std::size_t>(),sequences.get<std::size_t>()});
                }
            }
        }
    }
    return result;
}
inline std::vector<std::int32_t> expanded_tokens(const json& contract, std::size_t corpus, std::size_t length) {
    const auto seed = contract.at("corpus").at(corpus).at("seed_token_ids").get<std::vector<std::int32_t>>();
    if (seed.empty() || length == 0 || length > 2048 ||
        std::any_of(seed.begin(),seed.end(),[&](auto token) {
            return token < 0 || std::size_t(token) >= contract.at("model").at("vocabulary").get<std::size_t>();
        })) { throw std::invalid_argument("数值验证语料或长度无效"); }
    std::vector<std::int32_t> result(length);
    for (std::size_t p = 0; p < length; ++p) { result[p] = seed[p%seed.size()]; }
    return result;
}
inline std::vector<Batch> teacher_batches(const json& contract, const TeacherCase& c) {
    if (c.chunk == 0 || c.chunk > 128 || c.sequences == 0 || c.sequences > 4 ||
        c.corpus >= contract.at("corpus").size()) { throw std::invalid_argument("数值验证 batch 配置无效"); }
    std::vector<std::vector<std::int32_t>> inputs;
    for (std::size_t s = 0; s < c.sequences; ++s) {
        inputs.push_back(expanded_tokens(contract,(c.corpus+s)%contract.at("corpus").size(),c.length));
    }
    const auto positions = contract.at("teacher_forcing").at("positions").get<std::set<std::int32_t>>();
    std::vector<Batch> result;
    // chunk 限制整批输入行数；按 position、sequence 交错，不给多序列额外的 batch 容量。
    for (std::size_t p = 0; p < c.length; ++p) {
        for (std::size_t s = 0; s < c.sequences; ++s) {
            if (result.empty() || result.back().size() == c.chunk) { result.emplace_back(); }
            result.back().push_back({inputs[s][p],std::int32_t(p),std::int32_t(s),positions.contains(std::int32_t(p))});
        }
    }
    return result;
}
inline std::string batch_digest(const std::vector<Batch>& batches) {
    static_assert(std::endian::native == std::endian::little && sizeof(std::int32_t) == 4);
    std::vector<std::int32_t> words;
    for (const auto& batch : batches) {
        words.push_back(static_cast<std::int32_t>(batch.size()));
        for (const auto& token : batch) {
            words.insert(words.end(),{token.token,token.position,token.sequence,token.logits ? 1 : 0});
        }
    }
    return hash_sha256_hex(words.data(),words.size()*sizeof(std::int32_t));
}

class Reference {
    struct Backend {
        Backend() { ggml_backend_load_all(); llama_backend_init(); }
        ~Backend() { llama_backend_free(); }
    } backend_;
    struct OwnedBatch {
        llama_batch value = llama_batch_init(128,0,1);
        ~OwnedBatch() { llama_batch_free(value); }
    } batch_;
public:
    explicit Reference(const std::string& path, bool flash_attention = true) {
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        model_.reset(llama_model_load_from_file(path.c_str(),mp));
        if (!model_) { throw std::runtime_error("F32 参照模型加载失败"); }
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = 128; cp.n_ubatch = 128; cp.n_seq_max = 4;
        cp.n_threads = 8; cp.n_threads_batch = 8; cp.kv_unified = true;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = flash_attention ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_.reset(llama_init_from_model(model_.get(),cp));
        if (!context_) { throw std::runtime_error("F32 参照 context 加载失败"); }
        vocabulary_ = std::size_t(llama_vocab_n_tokens(llama_model_get_vocab(model_.get())));
    }
    void clear() {
        llama_synchronize(context_.get());
        llama_memory_clear(llama_get_memory(context_.get()),false);
    }
    std::vector<Logits> forward(const Batch& tokens) {
        CHECK(!tokens.empty() && tokens.size() <= 128);
        auto& batch = batch_.value;
        batch.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i].token; batch.pos[i] = tokens[i].position;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = tokens[i].sequence;
            batch.logits[i] = static_cast<std::int8_t>(tokens[i].logits);
        }
        CHECK(llama_decode(context_.get(),batch) == 0);
        std::vector<Logits> result;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (!tokens[i].logits) { continue; }
            const auto* values = llama_get_logits_ith(context_.get(),static_cast<std::int32_t>(i));
            CHECK(values);
            result.push_back({tokens[i].sequence,{values,values+vocabulary_}});
        }
        return result;
    }
private:
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model_{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> context_{nullptr,llama_free};
    std::size_t vocabulary_ = 0;
};

inline std::int32_t argmax(const std::vector<float>& values) {
    return static_cast<std::int32_t>(std::max_element(values.begin(),values.end())-values.begin());
}
inline json score(const std::vector<float>& values) {
    CHECK(values.size() > 1 && std::all_of(values.begin(),values.end(),[](float x) { return std::isfinite(x); }));
    const auto top = argmax(values);
    float second = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::int32_t(i) != top) { second = std::max(second,values[i]); }
    }
    return {{"token",top},{"margin",double(values[std::size_t(top)])-second},
            {"sha256",hash_sha256_hex(values.data(),values.size()*sizeof(float))}};
}
inline json compare(const std::vector<float>& actual, const std::vector<float>& expected, const json& thresholds) {
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    CHECK(actual.size() == expected.size() && actual.size() > 1);
    double squared = 0, maximum = 0, dot = 0, a_norm = 0, e_norm = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return {{"passed",false},{"all_finite",false},{"first_nonfinite",i},{"argmax_equal",false}};
        }
        const double error = double(actual[i])-expected[i];
        squared += error*error; maximum = std::max(maximum,std::abs(error));
        dot += double(actual[i])*expected[i]; a_norm += double(actual[i])*actual[i]; e_norm += double(expected[i])*expected[i];
    }
    const double rmse = std::sqrt(squared/double(actual.size())), cosine = dot/std::sqrt(a_norm*e_norm);
    const auto top = argmax(expected), token = argmax(actual);
    float second = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::int32_t(i) != top) { second = std::max(second,expected[i]); }
    }
    const double gap = double(expected[std::size_t(top)])-second;
    const bool near_tie = gap <= 2*maximum;
    return {{"rmse",rmse},{"max_absolute",maximum},{"cosine",cosine},{"all_finite",true},
        {"actual_sha256",hash_sha256_hex(actual.data(),actual.size()*sizeof(float))},
        {"reference_sha256",hash_sha256_hex(expected.data(),expected.size()*sizeof(float))},
        {"actual_argmax",token},{"reference_argmax",top},{"reference_margin",gap},
        {"near_tie",near_tie},{"argmax_equal",token==top},
        {"passed",rmse<thresholds.at("rmse_exclusive").get<double>() &&
                  maximum<thresholds.at("max_absolute_exclusive").get<double>() &&
                  cosine>=thresholds.at("cosine_min_inclusive").get<double>() && (near_tie || token==top)}};
}

json run_full_validation(const std::string& model, const std::string& reference, const json& contract,
                         const std::filesystem::path& output);
}
