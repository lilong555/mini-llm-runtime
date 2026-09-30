#include "minillm/tokenizer.h"

#include "llama.h"

#include <array>
#include <stdexcept>

namespace minillm {

struct Tokenizer::Impl {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    const llama_vocab* vocab = nullptr;

    Impl(const std::string& path, std::size_t expected_vocabulary) {
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        model.reset(llama_model_load_from_file(path.c_str(), params));
        if (!model) {
            throw std::runtime_error("cannot load the GGUF tokenizer");
        }
        vocab = llama_model_get_vocab(model.get());
        if (static_cast<std::size_t>(llama_vocab_n_tokens(vocab)) != expected_vocabulary) {
            throw std::runtime_error("tokenizer vocabulary does not match the embedding matrix");
        }
    }
};

Tokenizer::Tokenizer(const std::string& path, std::size_t expected_vocabulary)
    : impl_(std::make_unique<Impl>(path, expected_vocabulary)) {}
Tokenizer::~Tokenizer() = default;
std::size_t Tokenizer::vocabulary_size() const noexcept {
    return static_cast<std::size_t>(llama_vocab_n_tokens(impl_->vocab));
}

std::vector<std::int32_t> Tokenizer::tokenize(std::string_view text) const {
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

std::string Tokenizer::token_piece(std::int32_t token) const {
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

bool Tokenizer::is_eog(std::int32_t token) const { return llama_vocab_is_eog(impl_->vocab, token); }

} // namespace minillm
