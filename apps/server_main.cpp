#include "options.h"
#include "telemetry_output.h"

#include "llmserve/engine.h"
#include "llmserve/http_server.h"
#include "llama.h"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <climits>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        Options options(argc, argv, {"--model", "--backend", "--port", "--threads", "--gpu-layers",
            "--context", "--max-model-len", "--batch-tokens", "--prefill-chunk", "--max-active",
            "--queue-capacity", "--prefix-entries", "--prefix-tokens", "--page-size", "--policy",
            "--kernel", "--shutdown-file", "--event-buffer", "--telemetry", "--telemetry-output",
            "--telemetry-capacity", "--device", "--device-budget-bytes"});
        if (options.has("--help") || !options.has("--model")) {
            std::cout << "llmserve --model MODEL.gguf [--backend mini|mini-cuda|llama] [--port 8000]\n"
                         "         [--threads 8] [--gpu-layers 0] [--context 8192]\n"
                         "         [--max-model-len 2048] [--batch-tokens 256] [--prefill-chunk 32]\n"
                         "         [--max-active 8] [--queue-capacity 64] [--page-size 16]\n"
                         "         [--prefix-entries 4] [--prefix-tokens 2048]\n"
                         "         [--policy mixed|prefill_first] [--kernel auto|scalar]\n"
                         "         [--event-buffer 128] [--shutdown-file PATH]\n"
                         "         [--telemetry off|batches|stages] [--telemetry-output PATH]\n"
                         "         [--telemetry-capacity 1024] [--device 0] [--device-budget-bytes 0]\n"
                         "mini-cuda 默认：max-active=4, batch-tokens=128, prefix-entries=0, prefix-tokens=0\n";
            return options.has("--help") ? 0 : 1;
        }
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::cerr << text;
            }
        }, nullptr);
        const auto backend = options.get("--backend", "mini");
        if (backend != "mini" && backend != "mini-cuda" && backend != "llama") {
            throw std::invalid_argument("--backend must be mini, mini-cuda or llama");
        }
        const bool own_cuda = backend == "mini-cuda";
        llmserve::EngineConfig config;
        const auto telemetry = options.get("--telemetry", "off");
        if (telemetry != "off" && telemetry != "batches" && telemetry != "stages") {
            throw std::invalid_argument("--telemetry must be off, batches or stages");
        }
        config.telemetry_mode = telemetry == "off" ? llmserve::TelemetryMode::off :
            telemetry == "batches" ? llmserve::TelemetryMode::batches : llmserve::TelemetryMode::stages;
        config.telemetry_capacity = static_cast<std::size_t>(options.integer("--telemetry-capacity", 1024, 1, 16384));
        const auto telemetry_path = options.get("--telemetry-output");
        if ((telemetry == "off") != telemetry_path.empty()) {
            throw std::invalid_argument("enabled telemetry requires --telemetry-output; off mode accepts no output path");
        }
        std::ofstream telemetry_file;
        if (!telemetry_path.empty()) {
            if (std::filesystem::exists(telemetry_path)) {
                throw std::invalid_argument("telemetry output already exists");
            }
            telemetry_file.open(telemetry_path, std::ios::binary);
            if (!telemetry_file) { throw std::runtime_error("cannot open telemetry output"); }
        }
        config.context_tokens = static_cast<std::size_t>(options.integer("--context", 8192, 16, 1048576));
        config.max_model_len = static_cast<std::size_t>(options.integer("--max-model-len", 2048, 2, 1048576));
        config.batch_tokens = static_cast<std::size_t>(options.integer("--batch-tokens", own_cuda ? 128 : 256, 1, 65536));
        config.prefill_chunk = static_cast<std::size_t>(options.integer("--prefill-chunk", 32, 1, 65536));
        config.max_active = static_cast<std::size_t>(options.integer("--max-active", own_cuda ? 4 : 8, 1, 128));
        config.queue_capacity = static_cast<std::size_t>(options.integer("--queue-capacity", 64, 1, 65536));
        config.block_size = static_cast<std::size_t>(options.integer("--page-size", 16, 1, 256));
        config.prefix_cache_entries = static_cast<std::size_t>(options.integer("--prefix-entries", own_cuda ? 0 : 4, 0, 128));
        config.prefix_cache_tokens = static_cast<std::size_t>(options.integer("--prefix-tokens", own_cuda ? 0 : 2048, 0, 1048576));
        config.event_buffer_size = static_cast<std::size_t>(options.integer("--event-buffer", 128, 1, 65536));
        const auto policy = options.get("--policy", "mixed");
        if (policy != "mixed" && policy != "prefill_first") {
            throw std::invalid_argument("--policy must be mixed or prefill_first");
        }
        config.policy = policy == "mixed" ? llmserve::SchedulingPolicy::mixed : llmserve::SchedulingPolicy::prefill_first;
        config.validate();
        llmserve::ModelConfig model;
        model.path = options.get("--model");
        model.threads = static_cast<int>(options.integer("--threads", 8, 1, 256));
        model.gpu_layers = static_cast<int>(options.integer("--gpu-layers", backend == "llama" ? 99 : 0, 0, 10000));
        const auto kernel = options.get("--kernel", "auto");
        if (kernel != "auto" && kernel != "scalar") {
            throw std::invalid_argument("--kernel must be auto or scalar");
        }
        if (backend == "llama" && kernel != "auto") {
            throw std::invalid_argument("--kernel applies only to MiniLLM");
        }
        model.scalar_kernels = kernel == "scalar";
        model.device = static_cast<int>(options.integer("--device", 0, 0, INT_MAX));
        model.device_budget_bytes = static_cast<std::size_t>(options.integer("--device-budget-bytes", 0, 0, INT64_MAX));
        if (!own_cuda && (options.has("--device") || options.has("--device-budget-bytes"))) {
            throw std::invalid_argument("--device 和 --device-budget-bytes 仅适用于 mini-cuda");
        }
        if (own_cuda) { llmserve::validate_mini_cuda_config(model, config); }
        const auto port = static_cast<int>(options.integer("--port", 8000, 1024, 65535));
        std::unique_ptr<llmserve::ModelRunner> runner;
        if (own_cuda) {
#ifdef MINILLM_ENABLE_CUDA
            runner = llmserve::make_mini_cuda_runner(model, config);
#else
            throw std::invalid_argument("mini-cuda 需要使用 MINILLM_ENABLE_CUDA=ON 构建");
#endif
        } else if (backend == "mini") {
            runner = llmserve::make_mini_runner(model, config);
        } else {
            runner = llmserve::make_llama_runner(model, config);
        }
        llmserve::Engine engine(config, std::move(runner));
        const auto served = llmserve::serve_http(engine, port, options.get("--shutdown-file"));
        engine.stop();
        if (!telemetry_path.empty()) { write_telemetry(telemetry_file, engine); }
        if (!served) {
            throw std::runtime_error("HTTP server could not listen; select an unused port");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "llmserve: " << error.what() << '\n';
        return 1;
    }
}
