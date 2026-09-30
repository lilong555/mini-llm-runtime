#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace minillm {

// 只持有上游 vocab-only 模型；不创建 llama_context、CPU 执行器或 KV。
class Tokenizer {
public:
    Tokenizer(const std::string& path, std::size_t expected_vocabulary);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    std::size_t vocabulary_size() const noexcept;
    std::vector<std::int32_t> tokenize(std::string_view text) const;
    std::string token_piece(std::int32_t token) const;
    bool is_eog(std::int32_t token) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace minillm
