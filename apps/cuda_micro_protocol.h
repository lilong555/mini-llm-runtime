#pragma once

#include "minillm/model_types.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_micro {

using json = nlohmann::ordered_json;
inline constexpr std::size_t calls_per_sample = 32, repetitions = 5, max_length = 2048, max_batch = 128;

struct Case {
    std::string name, operation, role, tensor;
    std::size_t m = 0, n = 0, k = 0, group_width = 0, max_context = 0;
    std::vector<std::int32_t> slots, positions;
    std::size_t seed = 0;
};

inline std::vector<std::size_t> sample_rows(std::size_t count) {
    if (count == 0) { throw std::invalid_argument("micro 抽样行数不能为零"); }
    std::vector<std::size_t> result{0, count / 2, count - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

inline std::vector<std::size_t> sample_columns(std::size_t count) {
    if (count == 0) { throw std::invalid_argument("micro 抽样列数不能为零"); }
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < std::min<std::size_t>(8, count); ++i) {
        result.push_back(count <= 8 ? i : i * (count - 1) / 7);
    }
    return result;
}

inline std::vector<Case> make_cases(const json& recipe, const minillm::ModelDimensions& d) {
    std::vector<Case> result;
    const auto rows = recipe.at("matrix").at("rows").get<std::vector<std::size_t>>();
    const auto add = [&](Case item) {
        item.seed = result.size() + 1;
        result.push_back(std::move(item));
    };
    const auto q = d.heads * d.head_dim, kv = d.kv_heads * d.head_dim;
    for (const auto& role_json : recipe.at("matrix").at("roles")) {
        const auto role = role_json.get<std::string>();
        std::string tensor = "blk.0.";
        std::size_t n = 0, k = d.embedding;
        if (role == "Q") { tensor += "attn_q.weight"; n = q; }
        else if (role == "K") { tensor += "attn_k.weight"; n = kv; }
        else if (role == "V") { tensor += "attn_v.weight"; n = kv; }
        else if (role == "attention_output") { tensor += "attn_output.weight"; n = d.embedding; k = q; }
        else if (role == "gate" || role == "up") { tensor += "ffn_" + role + ".weight"; n = d.feed_forward; }
        else if (role == "down") { tensor += "ffn_down.weight"; n = d.embedding; k = d.feed_forward; }
        else if (role == "LM_head") { tensor = "output.weight"; n = d.vocabulary; }
        else { throw std::invalid_argument("micro 矩阵角色无效"); }
        for (auto m : rows) {
            add({"matrix-" + role + "-m" + std::to_string(m), "matrix", role, tensor, m, n, k});
        }
    }
    for (const auto& role_json : recipe.at("ops").at("rms_norm_roles")) {
        const auto role = role_json.get<std::string>();
        const auto width = role == "hidden" ? d.embedding : role == "query" ? q : kv;
        const auto group = role == "hidden" ? d.embedding : d.head_dim;
        const auto tensor = role == "hidden" ? "blk.0.attn_norm.weight" :
                            role == "query" ? "blk.0.attn_q_norm.weight" : "blk.0.attn_k_norm.weight";
        for (auto m : rows) {
            add({"rms_norm-" + role + "-m" + std::to_string(m), "rms_norm", role, tensor, m, width, 0, group});
        }
    }
    for (const auto& role_json : recipe.at("ops").at("rope_roles")) {
        const auto role = role_json.get<std::string>();
        for (auto m : rows) {
            for (auto position : recipe.at("ops").at("rope_positions").get<std::vector<std::int32_t>>()) {
                Case item{"rope-" + role + "-m" + std::to_string(m) + "-p" + std::to_string(position),
                          "rope", role, "", m, role == "query" ? q : kv, 0, d.head_dim};
                item.positions.assign(m, position);
                add(std::move(item));
            }
        }
    }
    const auto attention = [&](const std::string& role, const std::vector<std::int32_t>& slots,
                               const std::vector<std::int32_t>& positions) {
        const auto length = std::size_t(*std::max_element(positions.begin(), positions.end())) + 1;
        for (const auto* operation : {"softmax", "attention"}) {
            Case item{std::string(operation) + "-" + role + "-m" + std::to_string(slots.size()) + "-l" + std::to_string(length),
                      operation, role, "", slots.size(), std::string(operation) == "softmax" ? d.heads * max_length : q,
                      0, d.head_dim, length, slots, positions};
            add(std::move(item));
        }
    };
    for (auto length : recipe.at("ops").at("attention_contexts").get<std::vector<std::size_t>>()) {
        for (auto m : recipe.at("ops").at("decode_sequences").get<std::vector<std::size_t>>()) {
            std::vector<std::int32_t> slots(m), positions(m, static_cast<std::int32_t>(length - 1));
            for (std::size_t i = 0; i < m; ++i) { slots[i] = static_cast<std::int32_t>(i); }
            attention("decode", slots, positions);
        }
        for (auto m : rows) {
            if (m > length) { continue; }
            std::vector<std::int32_t> slots(m, 0), positions(m);
            for (std::size_t i = 0; i < m; ++i) { positions[i] = static_cast<std::int32_t>(length - m + i); }
            attention("prefill", slots, positions);
        }
    }
    for (auto prefix : recipe.at("ops").at("mixed").at("prefix_tokens").get<std::vector<std::int32_t>>()) {
        std::vector<std::int32_t> slots(16, 0), positions;
        for (std::int32_t i = 0; i < 16; ++i) { positions.push_back(i); }
        for (std::int32_t i = 1; i <= 2; ++i) { slots.push_back(i); positions.push_back(prefix); }
        attention("mixed", slots, positions);
    }
    return result;
}

inline json describe(const Case& c, const minillm::ModelDimensions& d) {
    return {{"name", c.name}, {"operation", c.operation}, {"role", c.role}, {"tensor", c.tensor},
        {"m", c.m}, {"n", c.n}, {"k", c.k}, {"group_width", c.group_width}, {"max_context", c.max_context},
        {"slots", c.slots}, {"positions", c.positions}, {"input_seed", c.seed},
        {"logical_flops_per_call", c.operation == "matrix" ? json(2*c.m*c.n*c.k) : json(nullptr)},
        {"output_elements", c.m*c.n}, {"query_heads", d.heads}, {"kv_heads", d.kv_heads},
        {"head_dim", d.head_dim}, {"kv_max_length", max_length}};
}

inline float input_value(std::size_t index, std::size_t seed) {
    return float(int((index * 17 + seed * 13) % 127) - 63) / 4096.0f;
}

inline float kv_value(std::size_t slot, std::size_t kind, std::size_t position, std::size_t column) {
    return float(int((slot * 19 + kind * 23 + position * 7 + column * 11) % 127) - 63) / 256.0f;
}

} // namespace cuda_micro
