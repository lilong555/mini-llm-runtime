#include "minillm/kernels.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(MINILLM_HAS_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#elif defined(MINILLM_HAS_AVX2)
#include <cpuid.h>
#endif

namespace minillm {

float half_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000) << 16;
    auto exponent = static_cast<std::uint32_t>((value >> 10) & 31);
    auto mantissa = static_cast<std::uint32_t>(value & 1023);
    if (exponent == 0) {
        if (mantissa == 0) {
            return std::bit_cast<float>(sign);
        }
        exponent = 113;
        while ((mantissa & 1024) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        return std::bit_cast<float>(sign | (exponent << 23) | ((mantissa & 1023) << 13));
    }
    exponent = exponent == 31 ? 255 : exponent + 112;
    return std::bit_cast<float>(sign | (exponent << 23) | (mantissa << 13));
}

std::uint16_t float_to_half(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
    const auto original_exponent = (bits >> 23) & 255;
    auto mantissa = bits & 0x7fffff;
    int exponent = static_cast<int>(original_exponent) - 112;
    if (original_exponent == 255) {
        return static_cast<std::uint16_t>(sign | 0x7c00 |
            (mantissa ? (0x0200 | (mantissa >> 13)) : 0));
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000;
        const auto shift = static_cast<unsigned int>(14 - exponent);
        auto rounded = mantissa >> shift;
        const auto remainder = mantissa & ((1u << shift) - 1);
        const auto halfway = 1u << (shift - 1);
        rounded += static_cast<std::uint32_t>(remainder > halfway ||
                                              (remainder == halfway && (rounded & 1)));
        return static_cast<std::uint16_t>(sign | rounded);
    }
    mantissa += 0xfff + ((mantissa >> 13) & 1);
    if (mantissa & 0x800000) {
        mantissa = 0;
        ++exponent;
    }
    return static_cast<std::uint16_t>(sign | (static_cast<unsigned int>(exponent) << 10) |
                                       (mantissa >> 13));
}

bool avx2_available() noexcept {
    static const bool available = [] {
#if defined(MINILLM_HAS_AVX2) && defined(_MSC_VER)
        int registers[4]{};
        __cpuid(registers, 0);
        if (registers[0] < 7) {
            return false;
        }
        __cpuidex(registers, 1, 0);
        const auto flags = static_cast<unsigned int>(registers[2]);
        constexpr unsigned int required = (1u << 27) | (1u << 28) | (1u << 29) | (1u << 12);
        if ((flags & required) != required || (_xgetbv(0) & 6) != 6) {
            return false;
        }
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 5)) != 0;
#elif defined(MINILLM_HAS_AVX2)
        __builtin_cpu_init();
        return static_cast<bool>(__builtin_cpu_supports("avx2") &&
                                  __builtin_cpu_supports("fma") &&
                                  __builtin_cpu_supports("f16c"));
#else
        return false;
#endif
    }();
    return available;
}

std::string_view kernel_name(KernelMode mode) noexcept {
    return mode == KernelMode::automatic && avx2_available() ? "avx2-fma-f16c" : "scalar";
}

std::size_t row_bytes(WeightType type, std::size_t columns) {
    if (type == WeightType::q8_0 && columns % 32 != 0) {
        throw std::invalid_argument("Q8_0 rows must contain complete 32-element blocks");
    }
    return type == WeightType::f32 ? columns * 4 :
        type == WeightType::f16 ? columns * 2 : columns / 32 * 34;
}

void decode_row(WeightType type, const std::byte* row, float* output, std::size_t columns) {
    if (type == WeightType::f32) {
        std::memcpy(output, row, columns * sizeof(float));
    } else if (type == WeightType::f16) {
        for (std::size_t i = 0; i < columns; ++i) {
            std::uint16_t half;
            std::memcpy(&half, row + i * 2, 2);
            output[i] = half_to_float(half);
        }
    } else {
        if (columns % 32 != 0) {
            throw std::invalid_argument("incomplete Q8_0 row");
        }
        for (std::size_t block = 0; block < columns / 32; ++block) {
            std::uint16_t half;
            std::memcpy(&half, row + block * 34, 2);
            const auto scale = half_to_float(half);
            for (std::size_t j = 0; j < 32; ++j) {
                std::int8_t value;
                std::memcpy(&value, row + block * 34 + 2 + j, 1);
                output[block * 32 + j] = scale * value;
            }
        }
    }
}

float dot_f32(const float* left, const float* right, std::size_t count, KernelMode mode) noexcept {
#ifdef MINILLM_HAS_AVX2
    if (mode == KernelMode::automatic && avx2_available()) {
        return detail::dot_f32_avx2(left, right, count);
    }
#else
    (void)mode;
#endif
    float sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += left[i] * right[i];
    }
    return sum;
}

float dot_f16(const std::uint16_t* left, const float* right, std::size_t count,
              KernelMode mode) noexcept {
#ifdef MINILLM_HAS_AVX2
    if (mode == KernelMode::automatic && avx2_available()) {
        return detail::dot_f16_avx2(left, right, count);
    }
#else
    (void)mode;
#endif
    float sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += half_to_float(left[i]) * right[i];
    }
    return sum;
}

float dot_row(WeightType type, const std::byte* row, const float* vector,
              std::size_t columns, KernelMode mode) noexcept {
    if (type == WeightType::f32) {
        return dot_f32(reinterpret_cast<const float*>(row), vector, columns, mode);
    }
    if (type == WeightType::f16) {
        return dot_f16(reinterpret_cast<const std::uint16_t*>(row), vector, columns, mode);
    }
#ifdef MINILLM_HAS_AVX2
    if (mode == KernelMode::automatic && avx2_available()) {
        return detail::dot_q8_avx2(row, vector, columns);
    }
#endif
    float sum = 0;
    for (std::size_t block = 0; block < columns / 32; ++block) {
        std::uint16_t half;
        std::memcpy(&half, row + block * 34, 2);
        const auto scale = half_to_float(half);
        for (std::size_t j = 0; j < 32; ++j) {
            std::int8_t value;
            std::memcpy(&value, row + block * 34 + 2 + j, 1);
            sum += (scale * value) * vector[block * 32 + j];
        }
    }
    return sum;
}

void rms_norm(const float* input, const float* weight, float* output, std::size_t count,
              float epsilon, KernelMode mode) noexcept {
    const auto scale = 1.0f / std::sqrt(dot_f32(input, input, count, mode) /
                                       static_cast<float>(count) + epsilon);
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = (input[i] * scale) * weight[i];
    }
}

} // namespace minillm
