#include "options.h"

#include "llmserve/engine.h"
#include "llmserve/http_server.h"
#include "llama.h"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        Options options(argc, argv, {"--model", "--backend", "--port", "--threads", "--gpu-layers",
            "--context", "--max-model-len", "--batch-tokens", "--prefill-chunk", "--max-active",
            "--queue-capacity", "--prefix-entries", "--prefix-tokens", "--page-size", "--policy",
            "--kernel", "--shutdown-file", "--event-buffer"});
        if (options.has("--help") || !options.has("--model")) {
            std::cout << "llmserve --model MODEL.gguf [--backend mini|llama] [--port 8000]\n"
                         "         [--threads 8] [--gpu-layers 0] [--context 8192]\n"
                         "         [--max-model-len 2048] [--batch-tokens 256] [--prefill-chunk 32]\n"
                         "         [--max-active 8] [--queue-capacity 64] [--page-size 16]\n"
                         "         [--prefix-entries 4] [--prefix-tokens 2048]\n"
                         "         [--policy mixed|prefill_first] [--kernel auto|scalar]\n"
                         "         [--event-buffer 128] [--shutdown-file PATH]\n";
            return options.has("--help") ? 0 : 1;
        }
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::cerr << text;
            }
        }, nullptr);
        const auto backend = options.get("--backend", "mini");
        if (backend != "mini" && backend != "llama") {
            throw std::invalid_argument("--backend must be mini or llama");
        }
        llmserve::EngineConfig config;
        config.context_tokens = static_cast<std::size_t>(options.integer("--context", 8192, 16, 1048576));
        config.max_model_len = static_cast<std::size_t>(options.integer("--max-model-len", 2048, 2, 1048576));
        config.batch_tokens = static_cast<std::size_t>(options.integer("--batch-tokens", 256, 1, 65536));
        config.prefill_chunk = static_cast<std::size_t>(options.integer("--prefill-chunk", 32, 1, 65536));
        config.max_active = static_cast<std::size_t>(options.integer("--max-active", 8, 1, 128));
        config.queue_capacity = static_cast<std::size_t>(options.integer("--queue-capacity", 64, 1, 65536));
        config.block_size = static_cast<std::size_t>(options.integer("--page-size", 16, 1, 256));
        config.prefix_cache_entries = static_cast<std::size_t>(options.integer("--prefix-entries", 4, 0, 128));
        config.prefix_cache_tokens = static_cast<std::size_t>(options.integer("--prefix-tokens", 2048, 0, 1048576));
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
        model.gpu_layers = static_cast<int>(options.integer("--gpu-layers", backend == "mini" ? 0 : 99, 0, 10000));
        const auto kernel = options.get("--kernel", "auto");
        if (kernel != "auto" && kernel != "scalar") {
            throw std::invalid_argument("--kernel must be auto or scalar");
        }
        if (backend == "llama" && kernel != "auto") {
            throw std::invalid_argument("--kernel applies only to MiniLLM");
        }
        model.scalar_kernels = kernel == "scalar";
        const auto port = static_cast<int>(options.integer("--port", 8000, 1024, 65535));
        auto runner = backend == "mini" ? llmserve::make_mini_runner(model, config) :
                                         llmserve::make_llama_runner(model, config);
        llmserve::Engine engine(config, std::move(runner));
        if (!llmserve::serve_http(engine, port, options.get("--shutdown-file"))) {
            throw std::runtime_error("HTTP server could not listen; select an unused port");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "llmserve: " << error.what() << '\n';
        return 1;
    }
}
