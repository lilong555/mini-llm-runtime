#pragma once

#include <stdexcept>
#include <string_view>

namespace minillm::cuda {

enum class PrecisionMode { f32_pedantic, f16_matrix_f32acc };

inline constexpr std::string_view precision_mode_name(PrecisionMode mode) {
    switch (mode) {
    case PrecisionMode::f32_pedantic: return "f32-pedantic";
    case PrecisionMode::f16_matrix_f32acc: return "f16-matrix-f32acc";
    }
    throw std::invalid_argument("CUDA precision mode 无效");
}

inline constexpr PrecisionMode parse_precision_mode(std::string_view name) {
    if (name == "f32-pedantic") { return PrecisionMode::f32_pedantic; }
    if (name == "f16-matrix-f32acc") { return PrecisionMode::f16_matrix_f32acc; }
    throw std::invalid_argument("--cuda-precision 必须为 f32-pedantic 或 f16-matrix-f32acc");
}

} // namespace minillm::cuda
