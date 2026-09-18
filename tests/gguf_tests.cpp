#include "test_support.h"

#include "minillm/gguf_model.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>

namespace {

std::filesystem::path artifact(const char* name) {
    static const auto directory = std::filesystem::current_path() /
        ("gguf-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    return directory / name;
}

void fixture(const std::filesystem::path& path, ggml_type type = GGML_TYPE_F32) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(
        ggml_init({1024 * 1024, nullptr, false}), ggml_free);
    CHECK(tensors);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), gguf_free);
    gguf_set_val_str(metadata.get(), "general.architecture", "qwen3");
    gguf_set_val_u32(metadata.get(), "test.count", 8);
    gguf_set_val_f32(metadata.get(), "test.epsilon", 0.000001f);
    auto* tensor = ggml_new_tensor_2d(tensors.get(), type, 4, 2);
    ggml_set_name(tensor, "test.weight");
    if (type == GGML_TYPE_F32) {
        for (int i = 0; i < 8; ++i) {
            static_cast<float*>(tensor->data)[i] = static_cast<float>(i + 1);
        }
    } else {
        std::fill_n(static_cast<std::int32_t*>(tensor->data), 8, 1);
    }
    gguf_add_tensor(metadata.get(), tensor);
    CHECK(gguf_write_to_file(metadata.get(), path.string().c_str(), false));
}

} // namespace

TEST(gguf_mapped_tensor_rows_and_metadata_types) {
    const auto path = artifact("valid.gguf");
    fixture(path);
    minillm::GgufModel model(path.string());
    CHECK(model.version() == 3);
    CHECK(model.tensor_count() == 1);
    CHECK(model.string_value("general.architecture") == "qwen3");
    CHECK(model.integer_value("test.count") == 8);
    CHECK(model.float_value("test.epsilon") == 0.000001f);
    const auto tensor = model.tensor("test.weight");
    CHECK(tensor.rows == 2 && tensor.columns == 4 && tensor.stride == 16);
    std::vector<float> values(4);
    minillm::decode_row(tensor.type, tensor.row(1), values.data(), 4);
    CHECK(values == std::vector<float>({5, 6, 7, 8}));
    test::throws<std::out_of_range>([&] { tensor.row(2); });
    test::throws<std::runtime_error>([&] { model.string_value("test.count"); });
    test::throws<std::runtime_error>([&] { model.integer_value("test.epsilon"); });
    test::throws<std::runtime_error>([&] { model.tensor("absent"); });
}

TEST(gguf_truncated_tensor_data_is_rejected) {
    const auto path = artifact("truncated.gguf");
    fixture(path);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    test::throws<std::runtime_error>([&] { minillm::GgufModel model(path.string()); });
}

TEST(gguf_bad_magic_and_missing_file_are_rejected) {
    const auto path = artifact("bad-magic.gguf");
    {
        std::ofstream file(path, std::ios::binary);
        file << "not a GGUF model";
    }
    test::throws<std::runtime_error>([&] { minillm::GgufModel model(path.string()); });
    test::throws<std::runtime_error>([&] { minillm::GgufModel model(artifact("absent.gguf").string()); });
}

TEST(gguf_unsupported_tensor_type_is_rejected) {
    const auto path = artifact("integer.gguf");
    fixture(path, GGML_TYPE_I32);
    minillm::GgufModel model(path.string());
    test::throws<std::runtime_error>([&] { model.tensor("test.weight"); });
}

int main() { return test::run(); }
