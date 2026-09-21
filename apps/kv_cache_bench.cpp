#include "options.h"

#include "minillm/kernels.h"
#include "minillm/paged_kv.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Shape {
    std::size_t layers = 28;
    std::size_t heads = 16;
    std::size_t kv_heads = 8;
    std::size_t head_dim = 128;
    std::size_t page_tokens = 16;

    std::size_t kv_width() const noexcept { return kv_heads * head_dim; }
};

struct Result {
    double checksum = 0;
    double nanoseconds = 0;
};

template <class KeyAccessor, class ValueAccessor>
double attention(const Shape& shape, std::size_t length, const std::vector<float>& query,
                 KeyAccessor&& key_at, ValueAccessor&& value_at) {
    std::vector<float> scores(length);
    std::vector<float> output(shape.heads * shape.head_dim);
    const auto scale = 1.0f / std::sqrt(static_cast<float>(shape.head_dim));
    for (std::size_t head = 0; head < shape.heads; ++head) {
        const auto kv_head = head / (shape.heads / shape.kv_heads);
        const auto* q = query.data() + head * shape.head_dim;
        auto maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t position = 0; position < length; ++position) {
            const auto* key = key_at(position);
            scores[position] = minillm::dot_f16(key + kv_head * shape.head_dim, q,
                shape.head_dim) * scale;
            maximum = std::max(maximum, scores[position]);
        }
        double denominator = 0;
        for (auto& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        auto* out = output.data() + head * shape.head_dim;
        for (std::size_t position = 0; position < length; ++position) {
            const auto* value = value_at(position);
            const auto probability = static_cast<float>(scores[position] / denominator);
            minillm::add_scaled_f16(value + kv_head * shape.head_dim, probability, out,
                                    shape.head_dim);
        }
    }
    double checksum = 0;
    for (const auto value : output) {
        checksum += value;
    }
    return checksum;
}

template <class Function>
Result measure(std::size_t repeats, Function&& function) {
    const auto start = Clock::now();
    double checksum = 0;
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        checksum += function();
    }
    return {checksum, std::chrono::duration<double, std::nano>(Clock::now() - start).count() /
                          static_cast<double>(repeats)};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options(argc, argv, {"--output", "--rounds"});
        if (options.has("--help")) {
            std::cout << "mini-kv-cache-bench [--output REPORT.json] [--rounds 7]\n";
            return 0;
        }
        const auto rounds = static_cast<std::size_t>(options.integer("--rounds", 7, 3, 31));
        const Shape shape;
        constexpr std::size_t max_length = 1536;
        constexpr std::size_t layer = 14;
        minillm::PagedKV paged({max_length / shape.page_tokens, shape.page_tokens, shape.layers,
                                shape.kv_width(), 1});
        std::vector<std::uint16_t> contiguous_key(max_length * shape.kv_width());
        std::vector<std::uint16_t> contiguous_value(max_length * shape.kv_width());
        std::vector<float> key(shape.kv_width()), value(shape.kv_width());
        std::vector<float> query(shape.heads * shape.head_dim);
        std::mt19937 random(20260921);
        std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
        for (auto& item : query) {
            item = distribution(random);
        }
        for (std::size_t position = 0; position < max_length; ++position) {
            paged.append(0, position);
            for (std::size_t i = 0; i < shape.kv_width(); ++i) {
                key[i] = distribution(random);
                value[i] = distribution(random);
                contiguous_key[position * shape.kv_width() + i] = minillm::float_to_half(key[i]);
                contiguous_value[position * shape.kv_width() + i] = minillm::float_to_half(value[i]);
            }
            paged.store(0, layer, position, key, value);
        }

        nlohmann::json measurements = nlohmann::json::array();
        double report_checksum = 0;
        for (const auto length : {std::size_t{16}, std::size_t{256},
                                  std::size_t{1024}, std::size_t{1536}}) {
            const auto paged_attention = [&] {
                return attention(shape, length, query,
                    [&](std::size_t position) { return paged.key(0, layer, position).data(); },
                    [&](std::size_t position) { return paged.value(0, layer, position).data(); });
            };
            const auto contiguous_attention = [&] {
                return attention(shape, length, query,
                    [&](std::size_t position) {
                        return contiguous_key.data() + position * shape.kv_width();
                    },
                    [&](std::size_t position) {
                        return contiguous_value.data() + position * shape.kv_width();
                    });
            };
            paged_attention();
            contiguous_attention();
            const auto repeats = std::max<std::size_t>(32, 100000 / (length * shape.heads));
            std::vector<double> paged_times, contiguous_times;
            for (std::size_t round = 0; round < rounds; ++round) {
                if (round % 2 == 0) {
                    const auto first = measure(repeats, paged_attention);
                    const auto second = measure(repeats, contiguous_attention);
                    paged_times.push_back(first.nanoseconds);
                    contiguous_times.push_back(second.nanoseconds);
                    report_checksum += first.checksum + second.checksum;
                } else {
                    const auto first = measure(repeats, contiguous_attention);
                    const auto second = measure(repeats, paged_attention);
                    contiguous_times.push_back(first.nanoseconds);
                    paged_times.push_back(second.nanoseconds);
                    report_checksum += first.checksum + second.checksum;
                }
            }
            const auto paged_median = median(paged_times);
            const auto contiguous_median = median(contiguous_times);
            measurements.push_back({{"context_tokens", length}, {"repeats_per_round", repeats},
                {"paged_ns_per_layer_token", paged_times},
                {"contiguous_ns_per_layer_token", contiguous_times},
                {"paged_median_ns", paged_median}, {"contiguous_median_ns", contiguous_median},
                {"paged_over_contiguous", paged_median / contiguous_median}});
        }
        const nlohmann::json report{{"scope",
            "single-thread resident-memory one-layer attention; identical AVX2/F16C math; compares MiniLLM PagedKV lookup and page discontinuities with a per-layer contiguous F16 layout; not an actual llama.cpp kernel benchmark"},
            {"kernel", minillm::kernel_name(minillm::KernelMode::automatic)},
            {"shape", {{"layers", shape.layers}, {"heads", shape.heads},
                       {"kv_heads", shape.kv_heads}, {"head_dim", shape.head_dim},
                       {"page_tokens", shape.page_tokens}}},
            {"rounds", rounds}, {"resident_paged_bytes", paged.resident_bytes()},
            {"measurements", measurements}, {"checksum", report_checksum}};
        if (options.has("--output")) {
            const auto path = std::filesystem::path(options.get("--output"));
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream file(path);
            if (!file) {
                throw std::runtime_error("cannot create KV benchmark report");
            }
            file << report.dump(2) << '\n';
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mini-kv-cache-bench: " << error.what() << '\n';
        return 1;
    }
}
