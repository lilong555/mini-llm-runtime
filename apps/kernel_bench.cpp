#include "options.h"

#include "minillm/kernels.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    try {
        Options options(argc, argv, {"--output", "--repeats"});
        if (options.has("--help")) {
            std::cout << "mini-kernel-bench [--output REPORT.json] [--repeats 200]\n";
            return 0;
        }
        constexpr std::size_t rows = 128, columns = 1024;
        const auto repeats = static_cast<std::size_t>(options.integer("--repeats", 200, 1, 100000));
        const auto stride = minillm::row_bytes(minillm::WeightType::q8_0, columns);
        std::vector<std::byte> weights(rows * stride);
        std::vector<float> input(columns);
        std::mt19937 random(42);
        for (auto& value : input) {
            value = static_cast<float>(static_cast<int>(random() % 2001) - 1000) / 1000;
        }
        for (std::size_t i = 0; i < rows * columns / 32; ++i) {
            const auto scale = minillm::float_to_half(0.005f * static_cast<float>(1 + i % 4));
            std::memcpy(weights.data() + i * 34, &scale, 2);
            for (std::size_t j = 0; j < 32; ++j) {
                const auto value = static_cast<std::int8_t>(static_cast<int>(random() % 255) - 127);
                std::memcpy(weights.data() + i * 34 + 2 + j, &value, 1);
            }
        }
        double max_error = 0;
        for (std::size_t row = 0; row < rows; ++row) {
            const auto scalar = minillm::dot_row(minillm::WeightType::q8_0,
                weights.data() + row * stride, input.data(), columns, minillm::KernelMode::scalar);
            const auto simd = minillm::dot_row(minillm::WeightType::q8_0,
                weights.data() + row * stride, input.data(), columns);
            max_error = std::max(max_error, static_cast<double>(std::abs(scalar - simd)));
        }
        if (max_error > 0.001) {
            throw std::runtime_error("SIMD kernel disagrees with the scalar reference");
        }
        double checksum = 0;
        const auto measure = [&](minillm::KernelMode mode) {
            const auto start = std::chrono::steady_clock::now();
            double sum = 0;
            for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
                for (std::size_t row = 0; row < rows; ++row) {
                    sum += minillm::dot_row(minillm::WeightType::q8_0,
                        weights.data() + row * stride, input.data(), columns, mode);
                }
            }
            checksum += sum;
            return std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now() - start).count() / static_cast<double>(repeats * rows);
        };
        std::vector<double> scalar, simd;
        measure(minillm::KernelMode::scalar);
        measure(minillm::KernelMode::automatic);
        for (int round = 0; round < 7; ++round) {
            if (round % 2 == 0) {
                scalar.push_back(measure(minillm::KernelMode::scalar));
                simd.push_back(measure(minillm::KernelMode::automatic));
            } else {
                simd.push_back(measure(minillm::KernelMode::automatic));
                scalar.push_back(measure(minillm::KernelMode::scalar));
            }
        }
        auto ordered_scalar = scalar;
        auto ordered_simd = simd;
        std::sort(ordered_scalar.begin(), ordered_scalar.end());
        std::sort(ordered_simd.begin(), ordered_simd.end());
        const auto scalar_ns = ordered_scalar[3];
        const auto simd_ns = ordered_simd[3];
        const nlohmann::json report{
            {"scope", "single-thread hot-cache Q8_0 weight x F32 vector dot; not end-to-end serving"},
            {"kernel", minillm::kernel_name(minillm::KernelMode::automatic)},
            {"rows", rows}, {"columns", columns}, {"repeats_per_round", repeats},
            {"scalar_ns_per_dot", scalar}, {"simd_ns_per_dot", simd},
            {"scalar_median_ns", scalar_ns}, {"simd_median_ns", simd_ns},
            {"median_speedup", scalar_ns / simd_ns}, {"max_absolute_error", max_error},
            {"checksum", checksum}};
        if (options.has("--output")) {
            const auto path = std::filesystem::path(options.get("--output"));
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            std::ofstream file(path);
            if (!file) {
                throw std::runtime_error("cannot create kernel benchmark report");
            }
            file << report.dump(2) << '\n';
        }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mini-kernel-bench: " << error.what() << '\n';
        return 1;
    }
}
