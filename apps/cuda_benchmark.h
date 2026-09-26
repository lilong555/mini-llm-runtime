#pragma once

#include "minillm/model_types.h"
#include "hash/hash.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_benchmark {

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
using Batch = std::vector<minillm::InputToken>;
using Lengths = std::array<std::size_t, 4>;
inline constexpr auto input_sha256 = "f5a311a0d7c993640ba5b761844a39e70a5ae5015db3ce9dcd07c01b6ad2a6c6";

inline std::uint64_t elapsed(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

struct Sample {
    std::int32_t sequence;
    std::size_t input_index;
    std::int32_t token;
};

struct Forward {
    std::vector<Sample> samples;
    std::uint64_t host_forward_to_token_ns = 0;
    // 正式模型基准关闭设备 events；此处不把 CPU wall time 冒充设备时间。
    json device_elapsed_ms = nullptr;
};

struct Workload {
    std::string name, mode;
    std::vector<Batch> setup, measured;
    std::size_t generation_tokens = 0, prompt_tokens = 0;
};

inline std::int32_t fixed_token(const json& input, std::size_t position) {
    const auto& ids = input.at("token_ids");
    if (!ids.is_array() || ids.empty()) { throw std::invalid_argument("固定 token 列表不能为空"); }
    const auto& id = ids.at(position % ids.size());
    if (!id.is_number_integer() || id.get<std::int64_t>() < 0 || id.get<std::int64_t>() > INT32_MAX) {
        throw std::invalid_argument("固定 token ID 无效");
    }
    return id.get<std::int32_t>();
}

inline std::vector<Batch> prefix(const json& input, std::size_t count, std::int32_t sequence, bool logits) {
    if (count == 0 || count > 2048 || sequence < 0 || sequence >= 4) {
        throw std::invalid_argument("基准前缀范围无效");
    }
    std::vector<Batch> batches;
    for (std::size_t position = 0; position < count;) {
        Batch batch;
        const auto end = std::min(count, position + 128);
        for (; position < end; ++position) {
            batch.push_back({fixed_token(input, position), static_cast<std::int32_t>(position), sequence,
                             logits && position + 1 == count});
        }
        batches.push_back(std::move(batch));
    }
    return batches;
}

inline std::vector<Workload> make_workloads(const json& input) {
    std::vector<Workload> result;
    for (const auto& item : input.at("workloads")) {
        Workload work{item.at("name").get<std::string>(), item.at("mode").get<std::string>(), {}, {}, 0, 0};
        if (work.mode == "prefill" || work.mode == "natural_generation") {
            work.prompt_tokens = item.at(work.mode == "prefill" ? "tokens" : "prompt_tokens").get<std::size_t>();
            work.measured = prefix(input, work.prompt_tokens, 0, true);
            if (work.mode == "natural_generation") {
                work.generation_tokens = item.at("output_tokens").get<std::size_t>();
                if (work.generation_tokens != 32 || work.prompt_tokens + work.generation_tokens - 1 > 2048) {
                    throw std::invalid_argument("自然生成基准必须包含 32 个输出 token");
                }
            }
        } else {
            const bool mixed = work.mode == "mixed";
            const auto sequences = mixed ? item.at("decode_sequences").get<std::size_t>() :
                work.mode == "independent_sequences" ? item.at("sequences").get<std::size_t>() : 1;
            const auto length = item.at("prefix_tokens").get<std::size_t>();
            if ((work.mode != "fixed_context_decode" && work.mode != "independent_sequences" && !mixed) ||
                sequences == 0 || sequences + std::size_t(mixed) > 4 || length >= 2048) {
                throw std::invalid_argument("decode 基准参数无效");
            }
            Batch batch;
            if (mixed) {
                work.prompt_tokens = item.at("prefill_tokens").get<std::size_t>();
                const auto prompt = prefix(input, work.prompt_tokens, 0, true);
                if (prompt.size() != 1) { throw std::invalid_argument("mixed prompt 必须位于一个 batch"); }
                batch = prompt.front();
            }
            // 每个序列从空 KV 独立构建，不调用 CPU share_prefix。
            for (std::size_t i = 0; i < sequences; ++i) {
                const auto sequence = static_cast<std::int32_t>(i + std::size_t(mixed));
                auto setup = prefix(input, length, sequence, false);
                work.setup.insert(work.setup.end(), setup.begin(), setup.end());
                batch.push_back({fixed_token(input, length), static_cast<std::int32_t>(length), sequence, true});
            }
            if (batch.size() > 128) { throw std::invalid_argument("基准 batch 超过 128 个 token"); }
            work.measured.push_back(std::move(batch));
        }
        result.push_back(std::move(work));
    }
    return result;
}

inline std::string input_digest(std::span<const minillm::InputToken> batch) {
    std::vector<unsigned char> bytes;
    const auto append = [&](std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<unsigned char>(value >> shift));
        }
    };
    append(static_cast<std::uint32_t>(batch.size()));
    for (const auto& row : batch) {
        append(static_cast<std::uint32_t>(row.token));
        append(static_cast<std::uint32_t>(row.position));
        append(static_cast<std::uint32_t>(row.sequence));
        append(static_cast<std::uint32_t>(row.logits));
    }
    return hash_sha256_hex(bytes.data(), bytes.size());
}

inline Lengths after_batch(std::span<const minillm::InputToken> batch, Lengths lengths) {
    if (batch.empty() || batch.size() > 128) { throw std::invalid_argument("基准 batch 大小无效"); }
    for (const auto& row : batch) {
        if (row.sequence < 0 || row.sequence >= 4 || row.position < 0 ||
            static_cast<std::size_t>(row.position) != lengths[static_cast<std::size_t>(row.sequence)] ||
            row.position >= 2048 || row.token < 0) {
            throw std::invalid_argument("基准输入不是连续的独立 KV 序列");
        }
        ++lengths[static_cast<std::size_t>(row.sequence)];
    }
    return lengths;
}

template<class Backend>
json run_call(Backend& backend, const Batch& batch, const char* phase, Lengths& lengths) {
    const auto after = after_batch(batch, lengths);
    // descriptors 和身份摘要均已就绪；JSON 与资源查询不进入主计时区间。
    const auto digest = input_digest(batch);
    const auto output = backend.forward(batch);
    json samples = json::array();
    std::size_t selected = 0;
    for (std::size_t i = 0; i < batch.size(); ++i) {
        if (!batch[i].logits) { continue; }
        if (selected >= output.samples.size()) { throw std::runtime_error("基准输出缺少 greedy token"); }
        const auto& sample = output.samples[selected++];
        if (sample.sequence != batch[i].sequence || sample.input_index != i || sample.token < 0 ||
            static_cast<std::size_t>(sample.token) >= backend.vocabulary()) {
            throw std::runtime_error("基准输出的顺序或 token 范围无效");
        }
        samples.push_back({{"sequence", sample.sequence}, {"input_index", i}, {"token", sample.token}});
    }
    if (selected != output.samples.size() || output.host_forward_to_token_ns == 0 ||
        !output.device_elapsed_ms.is_null()) {
        throw std::runtime_error("基准输出数量、时钟或 events 模式无效");
    }
    json record = {{"phase", phase}, {"input_sha256", digest}, {"input_tokens", batch.size()},
        {"logits_rows", selected}, {"context_before", lengths}, {"context_after", after},
        {"host_forward_to_token_ns", output.host_forward_to_token_ns},
        {"device_elapsed_ms", nullptr}, {"samples", std::move(samples)}};
    lengths = after;
    return record;
}

template<class Backend>
void run_workload(Backend& backend, const Workload& work, json& report) {
    report = {{"name", work.name}, {"mode", work.mode}, {"status", "running"}, {"iterations", json::array()}};
    json expected_tokens;
    for (std::size_t iteration = 0; iteration < 5; ++iteration) {
        // warmup 与 measured 均遵守同一 clear/rebuild 语义，重建顺序也进入原始报告。
        report["iterations"].push_back({{"index", iteration}, {"phase", iteration < 2 ? "warmup" : "measured"},
            {"clear_sequences", {0, 1, 2, 3}}, {"setup", json::array()}, {"forwards", json::array()}});
        auto& row = report["iterations"].back();
        const auto reset_start = Clock::now();
        for (std::int32_t sequence = 0; sequence < 4; ++sequence) { backend.clear_sequence(sequence); }
        row["clear_ns"] = elapsed(reset_start);
        Lengths lengths{};
        row["after_clear"] = backend.snapshot(lengths);
        for (const auto& batch : work.setup) {
            row["setup"].push_back(run_call(backend, batch, "setup", lengths));
        }
        row["before_measured"] = backend.snapshot(lengths);
        const auto* phase = work.mode == "prefill" || work.mode == "natural_generation" ? "prefill" :
                            work.mode == "mixed" ? "mixed" : "decode";
        for (const auto& batch : work.measured) {
            row["forwards"].push_back(run_call(backend, batch, phase, lengths));
        }
        if (work.generation_tokens) {
            for (std::size_t step = 1; step < work.generation_tokens; ++step) {
                const auto token = row["forwards"].back().at("samples").at(0).at("token").template get<std::int32_t>();
                const Batch batch{{token, static_cast<std::int32_t>(work.prompt_tokens + step - 1), 0, true}};
                row["forwards"].push_back(run_call(backend, batch, "decode", lengths));
            }
        }
        row["after_measured"] = backend.snapshot(lengths);
        std::uint64_t setup_ns = 0, total_ns = 0, prefill_ns = 0, decode_ns = 0;
        json tokens = json::array();
        for (const auto& call : row["setup"]) { setup_ns += call.at("host_forward_to_token_ns").template get<std::uint64_t>(); }
        for (const auto& call : row["forwards"]) {
            const auto ns = call.at("host_forward_to_token_ns").template get<std::uint64_t>();
            total_ns += ns;
            if (call.at("phase") == "prefill") { prefill_ns += ns; }
            if (call.at("phase") == "decode") { decode_ns += ns; }
            for (const auto& sample : call.at("samples")) { tokens.push_back(sample.at("token")); }
        }
        row["setup_forward_ns"] = setup_ns;
        row["host_forward_to_token_ns"] = total_ns;
        row["prefill_forward_ns"] = prefill_ns;
        row["decode_forward_ns"] = decode_ns;
        row["token_ids"] = tokens;
        if (iteration == 0) { expected_tokens = tokens; }
        if (tokens != expected_tokens) { throw std::runtime_error("同一 backend 重复执行的 greedy token 不一致"); }
    }
    report["status"] = "passed";
}

} // namespace cuda_benchmark
