#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace minillm {

enum class KernelMode { automatic, scalar };
enum class WeightType { f32, f16, q8_0 };

float half_to_float(std::uint16_t value) noexcept;
std::uint16_t float_to_half(float value) noexcept;
bool avx2_available() noexcept;
std::string_view kernel_name(KernelMode mode) noexcept;

std::size_t row_bytes(WeightType type, std::size_t columns);
void decode_row(WeightType type, const std::byte* row, float* output, std::size_t columns);
float dot_f32(const float* left, const float* right, std::size_t count,
              KernelMode mode = KernelMode::automatic) noexcept;
float dot_f16(const std::uint16_t* left, const float* right, std::size_t count,
              KernelMode mode = KernelMode::automatic) noexcept;
void add_scaled_f16(const std::uint16_t* input, float scale, float* output, std::size_t count,
                    KernelMode mode = KernelMode::automatic) noexcept;
float dot_row(WeightType type, const std::byte* row, const float* vector,
              std::size_t columns, KernelMode mode = KernelMode::automatic) noexcept;
void rms_norm(const float* input, const float* weight, float* output, std::size_t count,
              float epsilon, KernelMode mode = KernelMode::automatic) noexcept;

namespace detail {
#ifdef MINILLM_HAS_AVX2
float dot_f32_avx2(const float*, const float*, std::size_t) noexcept;
float dot_f16_avx2(const std::uint16_t*, const float*, std::size_t) noexcept;
void add_scaled_f16_avx2(const std::uint16_t*, float, float*, std::size_t) noexcept;
float dot_q8_avx2(const std::byte*, const float*, std::size_t) noexcept;
#endif
}

} // namespace minillm
