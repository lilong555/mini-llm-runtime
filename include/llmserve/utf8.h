#pragma once

#include <string>
#include <string_view>

namespace llmserve {

class Utf8Buffer {
public:
    std::string append(std::string_view bytes);
    std::string finish();

private:
    std::string drain(bool final);
    std::string pending_;
};

} // namespace llmserve
