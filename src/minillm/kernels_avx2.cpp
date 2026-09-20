#include "minillm/kernels.h"

#include <cstring>
#include <immintrin.h>

namespace minillm::detail {
namespace {

float horizontal_sum(__m256 value) noexcept {
    const auto pairs = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    auto sum = _mm_hadd_ps(pairs, pairs);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

} // namespace

float dot_f32_avx2(const float* left, const float* right, std::size_t count) noexcept {
    auto sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(left + i), _mm256_loadu_ps(right + i), sum);
    }
    auto result = horizontal_sum(sum);
    for (; i < count; ++i) {
        result += left[i] * right[i];
    }
    return result;
}

float dot_f16_avx2(const std::uint16_t* left, const float* right, std::size_t count) noexcept {
    auto sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(left + i));
        sum = _mm256_fmadd_ps(_mm256_cvtph_ps(half), _mm256_loadu_ps(right + i), sum);
    }
    auto result = horizontal_sum(sum);
    for (; i < count; ++i) {
        result += half_to_float(left[i]) * right[i];
    }
    return result;
}

void add_scaled_f16_avx2(const std::uint16_t* input, float scale, float* output,
                         std::size_t count) noexcept {
    const auto factor = _mm256_set1_ps(scale);
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto half = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const auto values = _mm256_cvtph_ps(half);
        const auto accumulated = _mm256_fmadd_ps(values, factor, _mm256_loadu_ps(output + i));
        _mm256_storeu_ps(output + i, accumulated);
    }
    for (; i < count; ++i) {
        output[i] += scale * half_to_float(input[i]);
    }
}

float dot_q8_avx2(const std::byte* row, const float* vector, std::size_t columns) noexcept {
    auto sum = _mm256_setzero_ps();
    for (std::size_t block = 0; block < columns / 32; ++block) {
        std::uint16_t half;
        std::memcpy(&half, row + block * 34, 2);
        const auto scale = _mm256_set1_ps(half_to_float(half));
        for (std::size_t part = 0; part < 4; ++part) {
            const auto bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(row + block * 34 + 2 + part * 8));
            const auto values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            const auto input = _mm256_loadu_ps(vector + block * 32 + part * 8);
            sum = _mm256_fmadd_ps(_mm256_mul_ps(values, scale), input, sum);
        }
    }
    return horizontal_sum(sum);
}

} // namespace minillm::detail
