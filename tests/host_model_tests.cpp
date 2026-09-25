#include "test_support.h"

#include "minillm/qwen3_model.h"
#include "minillm/tokenizer.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

using minillm::Qwen3Model;
using minillm::Tokenizer;
static_assert(!std::is_copy_constructible_v<Qwen3Model>);
static_assert(!std::is_move_constructible_v<Qwen3Model>);
static_assert(!std::is_copy_constructible_v<Tokenizer>);
static_assert(!std::is_move_constructible_v<Tokenizer>);
static_assert(std::is_same_v<decltype(std::declval<Qwen3Model&>().layers()),
                           const std::vector<minillm::Qwen3Layer>&>);
static_assert(std::is_same_v<decltype(std::declval<Qwen3Model&>().source()),
                           const minillm::GgufModel&>);

class Fixture {
public:
    Fixture() : directory_(std::filesystem::temp_directory_path() /
        ("gguf-test-host-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::create_directory(directory_);
    }
    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }
    std::string write(bool tied = true, const std::function<void(gguf_context*)>& metadata = {},
                      const std::string& bad_shape = {}, const std::string& missing = {}) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
            ggml_init({1024 * 1024, nullptr, false}), ggml_free);
        CHECK(tensors);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> info(gguf_init_empty(), gguf_free);
        gguf_set_val_str(info.get(), "general.architecture", "qwen3");
        gguf_set_val_u32(info.get(), "qwen3.embedding_length", 4);
        gguf_set_val_u32(info.get(), "qwen3.block_count", 2);
        gguf_set_val_u32(info.get(), "qwen3.attention.head_count", 2);
        gguf_set_val_u32(info.get(), "qwen3.attention.head_count_kv", 1);
        // Q 宽度为 8，刻意区别于 embedding，验证按 metadata 使用 head_dim。
        gguf_set_val_u32(info.get(), "qwen3.attention.key_length", 4);
        gguf_set_val_u32(info.get(), "qwen3.attention.value_length", 4);
        gguf_set_val_u32(info.get(), "qwen3.feed_forward_length", 6);
        gguf_set_val_u32(info.get(), "qwen3.context_length", 16);
        gguf_set_val_f32(info.get(), "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
        gguf_set_val_f32(info.get(), "qwen3.rope.freq_base", 10000.0f);
        std::size_t index = 0;
        const auto add = [&](const std::string& name, std::int64_t rows, std::int64_t columns) {
            if (name == missing) { return; }
            auto* tensor = ggml_new_tensor_2d(tensors.get(), GGML_TYPE_F32,
                                             columns + (name == bad_shape ? 1 : 0), rows);
            ggml_set_name(tensor, name.c_str());
            for (std::int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                static_cast<float*>(tensor->data)[i] = static_cast<float>(++index) / 16.0f;
            }
            gguf_add_tensor(info.get(), tensor);
        };
        add("token_embd.weight", 9, 4);
        add("output_norm.weight", 1, 4);
        if (!tied) { add("output.weight", 9, 4); }
        for (int layer = 0; layer < 2; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            add(prefix + "attn_norm.weight", 1, 4);
            add(prefix + "attn_q_norm.weight", 1, 4);
            add(prefix + "attn_k_norm.weight", 1, 4);
            add(prefix + "ffn_norm.weight", 1, 4);
            add(prefix + "attn_q.weight", 8, 4);
            add(prefix + "attn_k.weight", 4, 4);
            add(prefix + "attn_v.weight", 4, 4);
            add(prefix + "attn_output.weight", 4, 8);
            add(prefix + "ffn_gate.weight", 6, 4);
            add(prefix + "ffn_up.weight", 6, 4);
            add(prefix + "ffn_down.weight", 4, 6);
        }
        if (metadata) { metadata(info.get()); }
        const auto path = directory_ / (std::to_string(files_++) + ".gguf");
        CHECK(gguf_write_to_file(info.get(), path.string().c_str(), false));
        return path.string();
    }
private:
    std::filesystem::path directory_;
    std::size_t files_ = 0;
};

void check_matrix(const Qwen3Model& model, const minillm::TensorView& actual, const std::string& name) {
    const auto expected = model.source().tensor(name);
    CHECK(actual.data == expected.data && actual.type == expected.type);
    CHECK(actual.rows == expected.rows && actual.columns == expected.columns && actual.stride == expected.stride);
}

void check_binding(const Qwen3Model& model) {
    const auto& dims = model.dimensions();
    CHECK(model.layers().size() == dims.layers);
    check_matrix(model, model.embeddings(), "token_embd.weight");
    check_matrix(model, model.output(), model.tied_output() ? "token_embd.weight" : "output.weight");
    CHECK(model.output_norm() == model.source().tensor("output_norm.weight").vector());
    for (std::size_t i = 0; i < dims.layers; ++i) {
        const auto prefix = "blk." + std::to_string(i) + ".";
        const auto& layer = model.layers()[i];
        CHECK(layer.attention_norm == model.source().tensor(prefix + "attn_norm.weight").vector());
        CHECK(layer.query_norm == model.source().tensor(prefix + "attn_q_norm.weight").vector());
        CHECK(layer.key_norm == model.source().tensor(prefix + "attn_k_norm.weight").vector());
        CHECK(layer.ffn_norm == model.source().tensor(prefix + "ffn_norm.weight").vector());
        check_matrix(model, layer.query, prefix + "attn_q.weight");
        check_matrix(model, layer.key, prefix + "attn_k.weight");
        check_matrix(model, layer.value, prefix + "attn_v.weight");
        check_matrix(model, layer.attention_output, prefix + "attn_output.weight");
        check_matrix(model, layer.gate, prefix + "ffn_gate.weight");
        check_matrix(model, layer.up, prefix + "ffn_up.weight");
        check_matrix(model, layer.down, prefix + "ffn_down.weight");
    }
}

void check_real_model(const std::string& path) {
    Qwen3Model model(path);
    const auto& dims = model.dimensions();
    CHECK(dims.embedding == 1024 && dims.layers == 28 && dims.heads == 16 && dims.kv_heads == 8);
    CHECK(dims.head_dim == 128 && dims.feed_forward == 3072 && dims.vocabulary == 151936);
    CHECK(model.tied_output() && model.source().tensor_count() == 310);
    check_binding(model);
}

void check_real_tokenizer(const std::string& path, const std::string& contract_path) {
    std::ifstream file(contract_path);
    const auto contract = nlohmann::json::parse(file);
    const auto vocabulary = contract.at("model").at("vocabulary").get<std::size_t>();
    test::throws<std::runtime_error>([&] { Tokenizer invalid(path, vocabulary - 1); });
    Tokenizer tokenizer(path, vocabulary);
    CHECK(tokenizer.vocabulary_size() == vocabulary);
    for (const auto& corpus : contract.at("corpus")) {
        const auto expected = corpus.at("seed_token_ids").get<std::vector<std::int32_t>>();
        CHECK(tokenizer.tokenize(corpus.at("text").get<std::string>()) == expected);
    }
    for (const auto& generation : contract.at("stable_greedy")) {
        CHECK(tokenizer.tokenize(generation.at("text").get<std::string>()) ==
              generation.at("input_token_ids").get<std::vector<std::int32_t>>());
    }
    // 单独加载 vocab-only owner；不借用 Runtime，也不分配模型执行上下文。
    auto params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> reference(
        llama_model_load_from_file(path.c_str(), params), llama_model_free);
    CHECK(reference);
    const auto* vocab = llama_model_get_vocab(reference.get());
    std::vector<char> buffer(4096);
    for (std::size_t i = 0; i < vocabulary; ++i) {
        const auto token = static_cast<std::int32_t>(i);
        CHECK(tokenizer.is_eog(token) == llama_vocab_is_eog(vocab, token));
        auto count = llama_token_to_piece(vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, false);
        if (count < 0) {
            buffer.resize(static_cast<std::size_t>(-count));
            count = llama_token_to_piece(vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, false);
        }
        CHECK(count >= 0);
        CHECK(tokenizer.token_piece(token) == std::string(buffer.data(), static_cast<std::size_t>(count)));
    }
    CHECK(tokenizer.tokenize("").empty());
    test::throws<std::invalid_argument>([&] { tokenizer.tokenize(std::string(262145, 'a')); });
}

} // namespace

TEST(host_model_tied_output_and_dimensions) {
    Fixture fixture;
    Qwen3Model model(fixture.write());
    const auto& dims = model.dimensions();
    CHECK(dims.embedding == 4 && dims.layers == 2 && dims.heads == 2 && dims.kv_heads == 1);
    CHECK(dims.head_dim == 4 && dims.feed_forward == 6 && dims.vocabulary == 9);
    CHECK(dims.trained_context == 16 && dims.rms_epsilon == 1e-6f && dims.rope_base == 10000.0f);
    CHECK(model.tied_output() && model.output().data == model.embeddings().data);
    check_binding(model);
}

TEST(host_model_separate_output_owns_its_view) {
    Fixture fixture;
    Qwen3Model model(fixture.write(false));
    CHECK(!model.tied_output() && model.output().data != model.embeddings().data);
    check_binding(model);
}

TEST(host_model_rejects_architecture_and_invalid_dimensions) {
    Fixture fixture;
    test::throws<std::runtime_error>([&] {
        Qwen3Model model(fixture.write(true, [](auto* info) { gguf_set_val_str(info, "general.architecture", "llama"); }));
    });
    for (const auto* field : {"embedding_length", "block_count", "attention.head_count",
             "attention.head_count_kv", "attention.key_length", "feed_forward_length", "context_length"}) {
        for (const auto value : {0u, 1000001u}) {
            test::throws<std::runtime_error>([&] {
                Qwen3Model model(fixture.write(true, [&](auto* info) {
                    gguf_set_val_u32(info, (std::string("qwen3.") + field).c_str(), value);
                }));
            });
        }
    }
}

TEST(host_model_rejects_attention_norm_and_rope_configuration) {
    Fixture fixture;
    const std::vector<std::function<void(gguf_context*)>> invalid{
        [](auto* info) { gguf_set_val_u32(info, "qwen3.attention.head_count_kv", 3); },
        [](auto* info) { gguf_set_val_u32(info, "qwen3.attention.key_length", 3); },
        [](auto* info) { gguf_set_val_u32(info, "qwen3.attention.value_length", 2); },
        [](auto* info) { gguf_set_val_u32(info, "qwen3.rope.dimension_count", 2); },
        [](auto* info) { gguf_set_val_str(info, "qwen3.rope.scaling.type", "linear"); },
        [](auto* info) { gguf_set_val_f32(info, "qwen3.attention.layer_norm_rms_epsilon", 0.0f); },
        [](auto* info) { gguf_set_val_f32(info, "qwen3.attention.layer_norm_rms_epsilon", std::numeric_limits<float>::quiet_NaN()); },
        [](auto* info) { gguf_set_val_f32(info, "qwen3.rope.freq_base", -1.0f); },
        [](auto* info) { gguf_set_val_f32(info, "qwen3.rope.freq_base", std::numeric_limits<float>::infinity()); }};
    for (const auto& change : invalid) {
        test::throws<std::runtime_error>([&] { Qwen3Model model(fixture.write(true, change)); });
    }
    Qwen3Model valid(fixture.write(true, [](auto* info) {
        gguf_set_val_str(info, "qwen3.rope.scaling.type", "none");
        gguf_set_val_u32(info, "qwen3.rope.dimension_count", 4);
    }));
    check_binding(valid);
}

TEST(host_model_rejects_each_matrix_and_norm_shape) {
    Fixture fixture;
    std::vector<std::string> names{"token_embd.weight", "output.weight", "output_norm.weight"};
    for (const auto* suffix : {"attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm", "attn_q",
             "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}) {
        names.push_back(std::string("blk.1.") + suffix + ".weight");
    }
    for (const auto& name : names) {
        test::throws<std::runtime_error>([&] { Qwen3Model model(fixture.write(false, {}, name)); });
        if (name != "output.weight") {
            test::throws<std::runtime_error>([&] { Qwen3Model model(fixture.write(false, {}, {}, name)); });
        }
    }
    Qwen3Model recovered(fixture.write());
    check_binding(recovered);
}

TEST(tokenizer_missing_file_is_rejected) {
    Fixture fixture;
    test::throws<std::runtime_error>([&] { Tokenizer tokenizer(fixture.write() + ".missing", 9); });
}

int main(int argc, char** argv) {
    llama_log_set([](ggml_log_level level, const char* text, void*) {
        if (level >= GGML_LOG_LEVEL_WARN) { std::cerr << text; }
    }, nullptr);
    if (argc == 5 && std::string_view(argv[1]) == "--model" && std::string_view(argv[3]) == "--contract") {
        const std::string path = argv[2];
        const std::string contract = argv[4];
        test::cases().push_back({"host_model_real_qwen3_binding", [path] { check_real_model(path); }});
        test::cases().push_back({"tokenizer_real_qwen3_contract", [path, contract] { check_real_tokenizer(path, contract); }});
    } else if (argc != 1) {
        std::cerr << "用法：minillm-host-model-tests [--model MODEL.gguf --contract CONTRACT.json]\n";
        return 1;
    }
    return test::run();
}
