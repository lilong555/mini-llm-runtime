#include "options.h"

#include "minillm/runtime.h"
#include "hash/hash.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot read benchmark input: " + path);
    }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void write_report(const std::string& path, const json& report) {
    if (path.empty()) {
        return;
    }
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream stream(path);
    stream << report.dump() << '\n';
    if (!stream) {
        throw std::runtime_error("cannot write runtime benchmark report");
    }
}

std::size_t size_value(const json& value, const char* key, std::size_t minimum, std::size_t maximum) {
    const auto& item = value.at(key);
    if (!item.is_number_integer() || item.get<std::int64_t>() < static_cast<std::int64_t>(minimum) ||
        item.get<std::uint64_t>() > maximum) {
        throw std::invalid_argument(std::string("invalid runtime input integer: ") + key);
    }
    return item.get<std::size_t>();
}

struct Workload {
    std::string name;
    std::string mode;
    std::size_t prefill;
    std::size_t decode;
    std::size_t context;
};

std::vector<Workload> read_workloads(const json& input, const minillm::RuntimeConfig& config) {
    std::vector<Workload> result;
    std::set<std::string> names;
    const auto& workloads = input.at("workloads");
    if (!workloads.is_array() || workloads.empty() || workloads.size() > 64) {
        throw std::invalid_argument("runtime workloads must be a nonempty array of at most 64 cases");
    }
    for (const auto& item : workloads) {
        Workload work{item.at("name").get<std::string>(), item.at("mode").get<std::string>(),
            size_value(item, "prefill_tokens", 0, config.batch_tokens),
            size_value(item, "decode_sequences", 0, config.max_sequences - 1),
            size_value(item, "kv_tokens", 0, config.context_tokens)};
        const auto valid_name = !work.name.empty() && work.name.size() <= 64 &&
            std::all_of(work.name.begin(), work.name.end(), [](char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
            });
        if (!valid_name || !names.insert(work.name).second ||
            (work.mode != "prefill" && work.mode != "decode" && work.mode != "mixed") ||
            ((work.mode != "decode") != (work.prefill > 0)) ||
            ((work.mode != "prefill") != (work.decode > 0 && work.context > 0)) ||
            (work.mode == "prefill" && (work.decode != 0 || work.context != 0)) ||
            work.prefill + work.decode > config.batch_tokens ||
            work.decode + static_cast<std::size_t>(work.prefill > 0) >= config.max_sequences) {
            throw std::invalid_argument("inconsistent runtime workload: " + work.name);
        }
        const auto pages = [&](std::size_t n) { return (n + config.page_tokens - 1) / config.page_tokens; };
        if (pages(work.context) + pages(work.prefill) + work.decode > config.context_tokens / config.page_tokens) {
            throw std::invalid_argument("insufficient physical KV pages for runtime workload: " + work.name);
        }
        result.push_back(std::move(work));
    }
    return result;
}

json parallel_json(const minillm::ParallelProfile& p) {
    if (p.threads == 0) {
        return nullptr;
    }
    return {{"count", p.count}, {"grain", p.grain}, {"threads", p.threads},
        {"chunks", p.chunks}, {"participating_threads", p.participating_threads},
        {"wall_ns", p.wall_ns}, {"dispatch_ns", p.dispatch_ns}, {"caller_work_ns", p.caller_work_ns},
        {"caller_wait_ns", p.caller_wait_ns}, {"worker_work_sum_ns", p.worker_work_sum_ns},
        {"worker_work_max_ns", p.worker_work_max_ns}, {"worker_start_delay_max_ns", p.worker_start_delay_max_ns},
        {"completed", p.completed}};
}

json profile_json(const minillm::ForwardProfile& p) {
    json stages = json::array();
    std::uint64_t accounted = 0;
    for (const auto& stage : p.stages) {
        accounted += stage.wall_ns;
        stages.push_back({{"batch_id", p.batch_id}, {"stage", minillm::profile_stage_name(stage.stage)},
            {"layer", stage.layer}, {"wall_ns", stage.wall_ns}, {"input_tokens", stage.input_tokens},
            {"logits_tokens", stage.logits_tokens}, {"matrix_m", stage.matrix_m},
            {"matrix_n", stage.matrix_n}, {"matrix_k", stage.matrix_k},
            {"threads", p.threads}, {"parallel", parallel_json(stage.parallel)}});
    }
    if (!p.completed || accounted + p.unaccounted_ns != p.wall_ns) {
        throw std::runtime_error("incomplete or overlapping runtime stage profile");
    }
    return {{"batch_id", p.batch_id}, {"completed", p.completed}, {"wall_ns", p.wall_ns},
        {"unaccounted_ns", p.unaccounted_ns}, {"input_tokens", p.input_tokens},
        {"logits_tokens", p.logits_tokens}, {"sequences", p.sequences},
        {"context_before_sum", p.context_before_sum}, {"context_before_max", p.context_before_max},
        {"context_after_sum", p.context_after_sum}, {"context_after_max", p.context_after_max},
        {"threads", p.threads}, {"kv_pages_before", p.kv_pages_before}, {"kv_pages_after", p.kv_pages_after},
        {"stages", std::move(stages)}};
}

json fingerprint(const std::vector<minillm::Logits>& output, std::uint64_t& sampling_ns) {
    json tokens = json::array();
    std::vector<std::uint8_t> bytes;
    const auto append = [&](std::uint32_t word) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<std::uint8_t>(word >> shift));
        }
    };
    const auto start = Clock::now();
    std::vector<std::int32_t> greedy;
    for (const auto& logits : output) {
        greedy.push_back(static_cast<std::int32_t>(
            std::max_element(logits.values.begin(), logits.values.end()) - logits.values.begin()));
    }
    sampling_ns = elapsed(start);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const auto& logits = output[i];
        tokens.push_back({{"sequence", logits.sequence}, {"token", greedy[i]}});
        append(static_cast<std::uint32_t>(logits.sequence));
        append(static_cast<std::uint32_t>(logits.values.size()));
        for (const auto value : logits.values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("nonfinite runtime benchmark logits");
            }
            append(std::bit_cast<std::uint32_t>(value));
        }
    }
    return {{"logits_sha256", hash_sha256_hex(bytes.data(), bytes.size())},
            {"logits_vectors", output.size()}, {"greedy_tokens", std::move(tokens)}};
}

json run_workload(minillm::Runtime& runtime, const Workload& work,
                  const std::vector<std::int32_t>& pool, std::size_t warmup,
                  std::size_t repeats, bool profiling, std::uint64_t& batch_id) {
    const auto& cfg = runtime.config();
    for (std::size_t seq = 0; seq < cfg.max_sequences; ++seq) {
        runtime.clear_sequence(static_cast<std::int32_t>(seq));
    }
    const auto setup_start = Clock::now();
    std::vector<std::int32_t> prefix;
    std::size_t setup_calls = 0;
    for (std::size_t offset = 0; offset < work.context;) {
        std::vector<minillm::InputToken> setup;
        const auto end = std::min(work.context, offset + cfg.batch_tokens);
        for (; offset < end; ++offset) {
            prefix.push_back(pool[offset % pool.size()]);
            setup.push_back({prefix.back(), static_cast<std::int32_t>(offset), 0, false});
        }
        if (!runtime.forward(setup).empty()) {
            throw std::runtime_error("unexpected setup logits");
        }
        ++setup_calls;
    }
    const auto setup_ns = elapsed(setup_start);
    std::vector<minillm::InputToken> batch;
    json input_tokens = json::array();
    const auto add = [&](minillm::InputToken token, const char* phase) {
        batch.push_back(token);
        input_tokens.push_back({{"token", token.token}, {"position", token.position},
                                {"sequence", token.sequence}, {"logits", token.logits}, {"phase", phase}});
    };
    for (std::size_t i = 0; i < work.prefill; ++i) {
        add({pool[i % pool.size()], static_cast<std::int32_t>(i), 1, i + 1 == work.prefill}, "prefill");
    }
    const auto decode_start = work.prefill ? 2 : 1;
    for (std::size_t i = 0; i < work.decode; ++i) {
        const auto seq = static_cast<std::int32_t>(decode_start + i);
        add({pool[(work.context + static_cast<std::size_t>(seq)) % pool.size()],
             static_cast<std::int32_t>(work.context), seq, true}, "decode");
    }
    auto profile = runtime.make_profile();
    json samples = json::array();
    json expected = nullptr;
    for (std::size_t repeat = 0; repeat < warmup + repeats; ++repeat) {
        const auto reset_start = Clock::now();
        for (std::size_t seq = 1; seq < cfg.max_sequences; ++seq) {
            runtime.clear_sequence(static_cast<std::int32_t>(seq));
        }
        for (std::size_t i = 0; i < work.decode; ++i) {
            runtime.share_prefix(0, static_cast<std::int32_t>(decode_start + i), work.context);
        }
        const auto reset_ns = elapsed(reset_start);
        profile.batch_id = ++batch_id;
        const auto pages_before = runtime.used_kv_pages();
        const auto resident_before = runtime.resident_kv_bytes();
        const auto start = Clock::now();
        const auto output = runtime.forward(batch, profiling ? &profile : nullptr);
        const auto wall_ns = elapsed(start);
        std::uint64_t sampling_ns = 0;
        const auto signature = fingerprint(output, sampling_ns);
        if (!expected.is_null() && signature != expected) {
            throw std::runtime_error("runtime output changed across fixed-input repetitions");
        }
        expected = signature;
        if (repeat >= warmup) {
            samples.push_back({{"repeat", repeat - warmup}, {"batch_id", profile.batch_id},
                {"wall_ns", wall_ns}, {"sampling_ns", sampling_ns}, {"reset_ns", reset_ns},
                {"output", signature}, {"kv_pages_before", pages_before},
                {"kv_pages_after", runtime.used_kv_pages()}, {"resident_bytes_before", resident_before},
                {"resident_bytes_after", runtime.resident_kv_bytes()},
                {"profile", profiling ? profile_json(profile) : json(nullptr)}});
        }
    }
    for (std::size_t seq = 0; seq < cfg.max_sequences; ++seq) {
        runtime.clear_sequence(static_cast<std::int32_t>(seq));
    }
    if (runtime.used_kv_pages() != 0) {
        throw std::runtime_error("runtime benchmark leaked KV pages");
    }
    return {{"name", work.name}, {"mode", work.mode}, {"prefill_tokens", work.prefill},
        {"decode_sequences", work.decode}, {"kv_tokens", work.context},
        {"prefix_token_ids", prefix}, {"input_tokens", input_tokens},
        {"setup_ns", setup_ns}, {"setup_forward_calls", setup_calls},
        {"used_pages_after_clear", runtime.used_kv_pages()}, {"samples", std::move(samples)}};
}

} // namespace

int main(int argc, char** argv) {
    json report{{"schema_version", 1}, {"benchmark", "minillm-runtime"}, {"status", "failed"},
                {"workloads", json::array()}};
    std::string report_path;
    try {
        Options options(argc, argv, {"--model", "--input", "--output", "--threads", "--kernel",
            "--profiler", "--warmup", "--repeats", "--manifest", "--trial"});
        report_path = options.get("--output");
        if (options.has("--help") || !options.has("--model") || !options.has("--input")) {
            std::cout << "mini-runtime-bench --model MODEL.gguf --input INPUT.json [--output REPORT.json]\n"
                         "                   [--threads 8] [--kernel auto|scalar] [--profiler none|stages]\n"
                         "                   [--warmup 1] [--repeats 3] [--manifest MANIFEST.json --trial 0]\n";
            return options.has("--help") ? 0 : 1;
        }
        const auto raw_input = read_file(options.get("--input"));
        const auto input = json::parse(raw_input);
        if (input.at("schema_version") != 1) {
            throw std::invalid_argument("unsupported runtime input schema");
        }
        const auto input_hash = hash_sha256_hex(raw_input.data(), raw_input.size());
        const auto& config = input.at("runtime");
        minillm::RuntimeConfig cfg;
        cfg.model_path = options.get("--model");
        cfg.context_tokens = size_value(config, "context_tokens", 1, 1048576);
        cfg.page_tokens = size_value(config, "page_tokens", 1, 256);
        cfg.max_sequences = size_value(config, "max_sequences", 2, 256);
        cfg.batch_tokens = size_value(config, "batch_tokens", 1, cfg.context_tokens);
        cfg.threads = static_cast<std::size_t>(options.integer("--threads", 8, 1, 256));
        const auto kernel = options.get("--kernel", "auto");
        const auto profiler = options.get("--profiler", "none");
        if ((kernel != "auto" && kernel != "scalar") || (profiler != "none" && profiler != "stages") ||
            cfg.context_tokens % cfg.page_tokens != 0) {
            throw std::invalid_argument("invalid runtime kernel, profiler, or page alignment");
        }
        cfg.kernels = kernel == "scalar" ? minillm::KernelMode::scalar : minillm::KernelMode::automatic;
        const auto warmup = static_cast<std::size_t>(options.integer("--warmup", 1, 0, 100));
        const auto repeats = static_cast<std::size_t>(options.integer("--repeats", 3, 1, 100));
        const auto workloads = read_workloads(input, cfg);
        const auto& ids = input.at("token_ids");
        if (!ids.is_array() || ids.empty() || ids.size() > 65536) {
            throw std::invalid_argument("runtime token_ids must be a nonempty array");
        }
        std::vector<std::int32_t> pool;
        for (const auto& token : ids) {
            if (!token.is_number_integer() || token.get<std::int64_t>() < 0 ||
                token.get<std::uint64_t>() > 1000000) {
                throw std::invalid_argument("invalid fixed runtime token ID");
            }
            pool.push_back(token.get<std::int32_t>());
        }
        report["input_sha256"] = input_hash;
        report["trial"] = options.integer("--trial", 0, 0, 1000000);
        report["run_identity"] = nullptr;
        if (options.has("--manifest")) {
            const auto bytes = read_file(options.get("--manifest"));
            const auto manifest = json::parse(bytes);
            if (manifest.at("benchmark") != "minillm-runtime" || manifest.at("schema_version") != 1 ||
                manifest.at("input").at("sha256") != input_hash) {
                throw std::invalid_argument("runtime input does not match the manifest");
            }
            report["run_identity"] = {{"run_id", manifest.at("run_id")},
                {"manifest_sha256", hash_sha256_hex(bytes.data(), bytes.size())},
                {"model_sha256", manifest.at("model").at("sha256")},
                {"binary_sha256", manifest.at("binary").at("sha256")}};
        }
        report["runtime"] = config;
        report["runtime"]["threads"] = cfg.threads;
        report["runtime"]["kernel"] = kernel;
        report["runtime"]["effective_kernel"] = minillm::kernel_name(cfg.kernels);
        report["protocol"] = {{"profiler", profiler}, {"warmup", warmup}, {"repeats", repeats},
            {"clock", "steady_clock"}, {"setup", "pinned_prefix_share_reset"},
            {"sampling", "greedy_argmax_separate"}, {"activation_dtype", "F32"}, {"kv_dtype", "F16"},
            {"logits_digest", "sha256_seq_i32le_length_u32le_logits_f32le"}};
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::cerr << text;
            }
        }, nullptr);
        const auto load_start = Clock::now();
        minillm::Runtime runtime(cfg);
        report["load_ns"] = elapsed(load_start);
        const auto& dims = runtime.dimensions();
        report["dimensions"] = {{"embedding", dims.embedding}, {"layers", dims.layers},
            {"heads", dims.heads}, {"kv_heads", dims.kv_heads}, {"head_dim", dims.head_dim},
            {"feed_forward", dims.feed_forward}, {"vocabulary", dims.vocabulary}};
        if (std::any_of(pool.begin(), pool.end(), [&](auto token) {
                return static_cast<std::size_t>(token) >= dims.vocabulary;
            })) {
            throw std::invalid_argument("fixed runtime token exceeds model vocabulary");
        }
        std::uint64_t batch_id = 0;
        for (const auto& work : workloads) {
            report["workloads"].push_back(run_workload(runtime, work, pool, warmup, repeats,
                                                       profiler == "stages", batch_id));
        }
        report["status"] = "passed";
        write_report(report_path, report);
        std::cout << "mini-runtime-bench: " << workloads.size() << " workloads passed; threads="
                  << cfg.threads << " profiler=" << profiler << '\n';
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        try {
            write_report(report_path, report);
        } catch (const std::exception& report_error) {
            std::cerr << "mini-runtime-bench: " << report_error.what() << '\n';
        }
        std::cerr << "mini-runtime-bench: " << error.what() << '\n';
        return 1;
    }
}
