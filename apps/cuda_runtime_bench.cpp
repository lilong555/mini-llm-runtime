#include "cuda_benchmark.h"
#include "cuda_reports.h"
#include "options.h"
#include "minillm/cuda/device_buffer.h"
#include "minillm/runtime.h"
#include "llama.h"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

using namespace cuda_benchmark;
namespace gpu = minillm::cuda;

json read(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::invalid_argument("无法读取基准 JSON：" + path); }
    return json::parse(file);
}

json allocation_counts() {
    const auto stats = gpu::allocation_stats();
    return {{"allocation_calls", stats.allocation_calls}, {"allocations", stats.allocations},
        {"release_calls", stats.release_calls}, {"releases", stats.releases}, {"allocated_bytes", stats.allocated_bytes}};
}

class Backend {
public:
    Backend(const std::string& name, const std::string& model) {
        const auto started = Clock::now();
        if (name == "cuda") {
            gpu::CudaRuntimeConfig config;
            config.model_path = model;
            cuda_ = std::make_unique<gpu::CudaRuntime>(config);
        } else {
            minillm::RuntimeConfig config;
            config.model_path = model;
            config.context_tokens = 4 * 2048; // CPU 参数表示全池容量，不是单序列上限。
            config.max_sequences = 4;
            config.batch_tokens = 128;
            config.page_tokens = 16;
            config.threads = name == "cpu8" ? 8 : 16;
            cpu_ = std::make_unique<minillm::Runtime>(config);
        }
        initialization_ns_ = elapsed(started);
    }

    std::size_t vocabulary() const { return dimensions().vocabulary; }
    const minillm::ModelDimensions& dimensions() const { return cuda_ ? cuda_->dimensions() : cpu_->dimensions(); }
    void clear_sequence(std::int32_t sequence) {
        if (cuda_) { cuda_->clear_sequence(sequence); }
        else { cpu_->clear_sequence(sequence); }
    }
    Forward forward(std::span<const minillm::InputToken> batch) {
        Forward result;
        const auto started = Clock::now();
        if (cuda_) {
            auto output = cuda_->forward(batch, gpu::CudaOutputMode::greedy, false);
            result.samples.reserve(output.samples.size());
            for (const auto& sample : output.samples) {
                result.samples.push_back({sample.sequence, sample.input_index, sample.token});
            }
            if (!output.logits.empty() || output.device_elapsed_ms) {
                throw std::runtime_error("正式 CUDA greedy 基准不能下载 logits 或启用 events");
            }
        } else {
            const auto output = cpu_->forward(batch);
            result.samples.reserve(output.size());
            std::size_t selected = 0;
            for (std::size_t i = 0; i < batch.size(); ++i) {
                if (!batch[i].logits) { continue; }
                const auto& row = output.at(selected++);
                if (row.sequence != batch[i].sequence || row.values.size() != vocabulary()) {
                    throw std::runtime_error("CPU logits 形状或序列不一致");
                }
                float best = -std::numeric_limits<float>::infinity();
                std::int32_t token = -1;
                for (std::size_t id = 0; id < row.values.size(); ++id) {
                    const auto value = row.values[id];
                    if (!std::isfinite(value)) { throw std::runtime_error("CPU 基准 logits 含非有限值"); }
                    if (value > best) { best = value; token = static_cast<std::int32_t>(id); }
                }
                result.samples.push_back({row.sequence, i, token});
            }
            if (selected != output.size()) { throw std::runtime_error("CPU 输出存在多余 logits"); }
        }
        result.host_forward_to_token_ns = elapsed(started);
        return result;
    }
    json snapshot(const Lengths& lengths) const {
        json result = {{"allocation_stats", allocation_counts()}, {"cuda", nullptr}, {"cpu", nullptr}};
        if (cuda_) {
            const auto state = cuda_->diagnostics();
            if (state.state != gpu::CudaRuntimeState::ready ||
                state.sequence_lengths != std::vector<std::size_t>(lengths.begin(), lengths.end())) {
                throw std::runtime_error("CUDA 基准逻辑 KV 状态不一致");
            }
            result["cuda"] = cuda_reports::diagnostics(state);
        } else {
            std::size_t pages = 0;
            for (const auto length : lengths) { pages += (length + 15) / 16; }
            if (cpu_->used_kv_pages() != pages) { throw std::runtime_error("CPU 基准独立 KV 页数不一致"); }
            result["cpu"] = {{"used_kv_pages", cpu_->used_kv_pages()}, {"resident_kv_bytes", cpu_->resident_kv_bytes()}};
        }
        return result;
    }
    json metadata() const {
        const auto& d = dimensions();
        return {{"dimensions", {{"embedding", d.embedding}, {"layers", d.layers}, {"heads", d.heads},
                    {"kv_heads", d.kv_heads}, {"head_dim", d.head_dim}, {"feed_forward", d.feed_forward},
                    {"vocabulary", d.vocabulary}, {"trained_context", d.trained_context}}},
            {"configuration", {{"max_sequences", 4}, {"max_model_len", 2048}, {"batch_tokens", 128},
                {"context_pool_tokens", 8192}, {"threads", cuda_ ? json(nullptr) : json(cpu_->config().threads)},
                {"kernel", cuda_ ? json(nullptr) : json("auto")},
                {"effective_kernel", cuda_ ? json(nullptr) : json(minillm::kernel_name(cpu_->config().kernels))},
                {"device", cuda_ ? json(0) : json(nullptr)}, {"streams", cuda_ ? json(1) : json(nullptr)},
                {"kv_layout", cuda_ ? "contiguous" : "paged"}, {"page_tokens", cuda_ ? json(nullptr) : json(16)}}},
            {"device", cuda_ ? cuda_reports::device(cuda_->device_info()) : json(nullptr)},
            {"arithmetic", cuda_ ? cuda_reports::arithmetic(*cuda_) :
                json{{"source_weight_dtype", "Q8_0"}, {"effective_weight_dtype", "F32"}, {"activation_dtype", "F32"},
                     {"kv_dtype", "F16"}, {"softmax_denominator_dtype", "F64"}, {"kernel", "auto"}}},
            {"initialization", {{"runtime_constructor_ns", initialization_ns_},
                {"model_load_ns", cuda_ ? json(cuda_->diagnostics().model_load_ns) : json(nullptr)},
                {"weight_decode_upload_ns", cuda_ ? json(cuda_->diagnostics().weight_decode_upload_ns) : json(nullptr)},
                {"storage_initialization_ns", cuda_ ? json(cuda_->diagnostics().storage_initialization_ns) : json(nullptr)}}},
            {"weights", cuda_ ? cuda_reports::weights(cuda_->weight_manifest()) : json(nullptr)}};
    }
private:
    std::unique_ptr<minillm::Runtime> cpu_;
    std::unique_ptr<gpu::CudaRuntime> cuda_;
    std::uint64_t initialization_ns_ = 0;
};
}

int main(int argc, char** argv) {
    using namespace cuda_benchmark;
    json report = {{"schema_version", 1}, {"benchmark", "minillm-cuda-runtime"}, {"status", "failed"},
                   {"workloads", json::array()}};
    std::string output;
    try {
        Options options(argc, argv, {"--model", "--input", "--output", "--backend", "--manifest", "--order"});
        if (options.has("--help")) {
            std::cout << "mini-cuda-runtime-bench --model MODEL.gguf --input INPUT.json --output NEW_REPORT.json\n"
                "                        --backend cpu8|cpu16|cuda [--manifest MANIFEST.json --order N]\n"
                "单进程构造一个 Runtime；12 个 workload，每项 2 次 warmup、3 次正式测量。\n";
            return 0;
        }
        const auto candidate = options.get("--output");
        if (candidate.empty() || std::filesystem::exists(candidate)) {
            throw std::invalid_argument("--output 必须是尚不存在的新文件");
        }
        output = candidate;
        const auto backend = options.get("--backend");
        if (backend != "cpu8" && backend != "cpu16" && backend != "cuda") {
            throw std::invalid_argument("--backend 必须为 cpu8、cpu16 或 cuda");
        }
        if (options.has("--manifest") != options.has("--order")) {
            throw std::invalid_argument("--manifest 与 --order 必须同时指定");
        }
        const auto input = read(options.get("--input"));
        const auto input_hash = cuda_reports::file_hash(options.get("--input"));
        // 原始冻结 recipe 的摘要由采集器与分析器共同核对；应用也拒绝配置缩减。
        if (input_hash != input_sha256 || input.at("schema_version") != 1 ||
            input.at("protocol_id") != "qwen3-cuda-model-v0" ||
            input.at("gpu") != json{{"device", 0}, {"max_sequences", 4}, {"max_model_len", 2048},
                {"batch_tokens", 128}, {"streams", 1}, {"kv_layout", "contiguous"}, {"page_tokens", nullptr}} ||
            input.at("cpu") != json{{"threads", {8, 16}}, {"kernel", "auto"}, {"page_tokens", 16}} ||
            input.at("measurement").at("independent_trials") != 5 || input.at("measurement").at("warmup") != 2 ||
            input.at("measurement").at("measured_repeats") != 3 || input.at("workloads").size() != 12) {
            throw std::invalid_argument("输入不是冻结的 qwen3-cuda-model-v0 配置");
        }
        const auto model_hash = cuda_reports::file_hash(options.get("--model"));
        if (model_hash != input.at("model_sha256") ||
            model_hash != "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031") {
            throw std::invalid_argument("模型摘要与固定 Q8_0 checkpoint 不符");
        }
        const auto works = make_workloads(input);
        report["backend"] = backend;
        report["input_sha256"] = input_hash;
        report["model_sha256"] = model_hash;
        report["run_identity"] = nullptr;
        report["process"] = nullptr;
        report["scope"] = "diagnostic_single_process";
        if (options.has("--manifest")) {
            const auto manifest = read(options.get("--manifest"));
            const auto order = static_cast<std::size_t>(options.integer("--order", 0, 0, 10000));
            const auto& slot = manifest.at("reports").at(order);
            const auto binary_hash = cuda_reports::file_hash(std::filesystem::canonical(argv[0]));
            if (manifest.at("schema_version") != 1 || manifest.at("benchmark") != "minillm-cuda-runtime" ||
                manifest.at("input").at("sha256") != input_hash ||
                manifest.at("model").at("sha256") != model_hash ||
                manifest.at("binary").at("sha256") != binary_hash ||
                slot.at("order") != order || slot.at("backend") != backend ||
                slot.at("file") != std::filesystem::path(output).filename().string()) {
                throw std::invalid_argument("基准进程与 manifest 身份不一致");
            }
            report["scope"] = "paired_model_baseline";
            report["process"] = slot;
            report["run_identity"] = {{"run_id", manifest.at("run_id")},
                {"manifest_sha256", cuda_reports::file_hash(options.get("--manifest"))},
                {"source_state_sha256", manifest.at("source").at("worktree_state_sha256")},
                {"binary_sha256", binary_hash}};
        }
        report["protocol"] = {{"warmup", 2}, {"measured_repeats", 3}, {"clock", "steady_clock"},
            {"primary", "host_forward_to_token_ns"}, {"profiler", "none"}, {"device_events", false},
            {"setup", "clear_and_rebuild_independent_prefix_outside_timing"},
            {"sampling", "finite_check_and_greedy_argmax_inside_timing"}, {"reference_model_resident", false},
            {"generation_stop", "ignore_eos_fixed_32"}, {"input_digest", "sha256_count_u32le_token_position_sequence_logits_i32le"}};
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) { std::cerr << text; }
        }, nullptr);
        report["before_initialization_allocations"] = allocation_counts();
        Backend runtime(backend, options.get("--model"));
        report["runtime"] = runtime.metadata();
        for (const auto& id : input.at("token_ids")) {
            if (!id.is_number_integer() || id.get<std::int64_t>() < 0 ||
                id.get<std::uint64_t>() >= runtime.vocabulary()) {
                throw std::invalid_argument("固定 token 超出模型词表");
            }
        }
        report["initial_state"] = runtime.snapshot({});
        for (const auto& work : works) {
            report["workloads"].push_back(json::object());
            run_workload(runtime, work, report["workloads"].back());
            std::cout << backend << ": " << work.name << " 通过\n";
        }
        for (std::int32_t sequence = 0; sequence < 4; ++sequence) { runtime.clear_sequence(sequence); }
        report["final_state"] = runtime.snapshot({});
        report["status"] = "passed";
        cuda_reports::write(output, report);
        return 0;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        if (!output.empty()) {
            try { cuda_reports::write(output, report); }
            catch (const std::exception& failure) { std::cerr << "报告写入失败：" << failure.what() << '\n'; }
        }
        std::cerr << "mini-cuda-runtime-bench: " << error.what() << '\n';
        return 1;
    }
}
