#pragma once
#include "test_support.h"
#include "ggml.h"
#include "gguf.h"
#include <chrono>
#include <filesystem>
#include <limits>
#include <memory>

class Qwen3Fixture {
public:
    Qwen3Fixture() : directory_(std::filesystem::temp_directory_path() /
        ("gguf-test-host-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::create_directory(directory_);
    }
    ~Qwen3Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }
    std::string write(bool tied = true, const std::function<void(gguf_context*)>& metadata = {},
                      const std::string& bad_shape = {}, const std::string& missing = {},
                      ggml_type dtype = GGML_TYPE_F32, bool nonfinite = false,
                      float nonfinite_value = std::numeric_limits<float>::infinity()) {
        CHECK(dtype == GGML_TYPE_F32 || dtype == GGML_TYPE_F16);
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
            auto* tensor = ggml_new_tensor_2d(tensors.get(), dtype,
                                             columns + (name == bad_shape ? 1 : 0), rows);
            ggml_set_name(tensor, name.c_str());
            for (std::int64_t i = 0; i < ggml_nelements(tensor); ++i) {
                const float value = nonfinite && name == "blk.1.ffn_down.weight" && i == 0
                    ? nonfinite_value : static_cast<float>(++index) / 16.0f;
                if (dtype == GGML_TYPE_F32) { static_cast<float*>(tensor->data)[i] = value; }
                else { static_cast<ggml_fp16_t*>(tensor->data)[i] = ggml_fp32_to_fp16(value); }
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
