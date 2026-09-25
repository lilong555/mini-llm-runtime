#include "cuda_reports.h"
#include "options.h"
#include "minillm/cuda/device_buffer.h"
#include "llmserve/utf8.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>

using namespace minillm;
using namespace minillm::cuda;
using cuda_reports::json;

int main(int argc, char** argv) {
    try {
        Options options(argc,argv,{"--model","--prompt","--tokens","--context","--batch","--sequences","--device",
            "--device-budget-bytes","--chunk","--output"},{"--help","--ignore-eos","--device-time"});
        if (options.has("--help") || !options.has("--model")) {
            std::cout << "mini-cuda-llm --model MODEL.gguf [--prompt TEXT] [--tokens 32]\n"
                "              [--context 2048] [--batch 128] [--chunk 128] [--sequences 4]\n"
                "              [--device 0] [--device-budget-bytes 0] [--ignore-eos]\n"
                "              [--device-time] [--output NEW_REPORT.json]\n";
            return options.has("--help") ? 0 : 1;
        }
        const auto output_path = options.get("--output");
        if (options.has("--output") && (output_path.empty() || std::filesystem::exists(output_path))) {
            throw std::invalid_argument("--output 必须是尚不存在的新文件");
        }
        CudaRuntimeConfig config;
        config.model_path = options.get("--model");
        config.device = static_cast<int>(options.integer("--device",0,0,INT_MAX));
        config.max_sequences = static_cast<std::size_t>(options.integer("--sequences",4,1,4));
        config.max_model_len = static_cast<std::size_t>(options.integer("--context",2048,1,INT_MAX));
        config.batch_tokens = static_cast<std::size_t>(options.integer("--batch",128,1,CudaRuntimeConfig::max_supported_batch_tokens));
        config.device_budget_bytes = static_cast<std::size_t>(options.integer("--device-budget-bytes",0,0,INT64_MAX));
        const auto chunk = static_cast<std::size_t>(options.integer("--chunk",static_cast<std::int64_t>(config.batch_tokens),
                                                                    1,static_cast<std::int64_t>(config.batch_tokens)));
        const auto count = static_cast<std::size_t>(options.integer("--tokens",32,1,4096));
        const bool timing = options.has("--device-time");
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) { std::cerr << text; }
        },nullptr);
        const auto model_sha = cuda_reports::file_hash(config.model_path);
        CudaRuntime runtime(config);
        const auto prompt = runtime.tokenize(options.get("--prompt","The capital of France is"));
        if (prompt.empty() || prompt.size() > config.max_model_len || count-1 > config.max_model_len-prompt.size()) {
            throw std::invalid_argument("prompt 和实际写入 KV 的输出 token 超过 context 上限");
        }
        const auto before = runtime.diagnostics();
        const auto allocations_before = allocation_stats();
        json forwards = json::array();
        const auto record = [&](const CudaForwardResult& result, const char* phase, std::size_t tokens) {
            forwards.push_back({{"phase",phase},{"input_tokens",tokens},{"logits_rows",result.samples.size()},
                {"host_forward_to_token_ns",result.host_forward_to_token_ns},
                {"device_elapsed_ms",result.device_elapsed_ms ? json(*result.device_elapsed_ms) : json(nullptr)}});
        };
        const auto started = std::chrono::steady_clock::now();
        CudaForwardResult result;
        for (std::size_t position = 0; position < prompt.size();) {
            std::vector<InputToken> batch;
            const auto end = std::min(prompt.size(),position+chunk);
            for (; position < end; ++position) {
                batch.push_back({prompt[position],static_cast<std::int32_t>(position),0,position+1==prompt.size()});
            }
            result = runtime.forward(batch,CudaOutputMode::greedy,timing);
            record(result,"prefill",batch.size());
        }
        const auto prefill_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        std::vector<std::int32_t> generated;
        std::string output;
        llmserve::Utf8Buffer text;
        for (std::size_t step = 0; step < count; ++step) {
            const auto token = result.samples.at(0).token;
            generated.push_back(token);
            const bool stop = runtime.is_eog(token) && !options.has("--ignore-eos");
            if (!stop) { output += text.append(runtime.token_piece(token)); }
            if (stop || step+1 == count) { break; }
            const InputToken input{token,static_cast<std::int32_t>(prompt.size()+step),0,true};
            result = runtime.forward({&input,1},CudaOutputMode::greedy,timing);
            record(result,"decode",1);
        }
        output += text.finish();
        const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count();
        const auto after = runtime.diagnostics();
        const auto allocations_after = allocation_stats();
        runtime.clear_sequence(0);
        const json report = {{"schema_version",1},{"status","passed"},{"backend","minillm-cuda"},
            {"execution","项目 CUDA forward；cuBLAS GEMM；llama.cpp 仅提供词表"},
            {"model_sha256",model_sha},{"arithmetic",cuda_reports::arithmetic(runtime)},
            {"device",cuda_reports::device(runtime.device_info())},{"kv_layout","contiguous"},{"streams",1},
            {"limits",{{"max_sequences",config.max_sequences},{"max_model_len",config.max_model_len},
                {"batch_tokens",config.batch_tokens},{"device_budget_bytes",config.device_budget_bytes}}},
            {"input_token_ids",prompt},{"token_ids",generated},{"text",output},{"prompt_tokens",prompt.size()},
            {"completion_tokens",generated.size()},{"prefill_wall_ns",prefill_ns},{"generation_wall_ns",elapsed_ns},
            {"forwards",forwards},{"before",cuda_reports::diagnostics(before)},{"after",cuda_reports::diagnostics(after)},
            {"steady_project_allocation_calls",allocations_after.allocation_calls-allocations_before.allocation_calls},
            {"steady_project_release_calls",allocations_after.release_calls-allocations_before.release_calls},
            {"after_clear",cuda_reports::diagnostics(runtime.diagnostics())},
            {"scope","模型到 token CLI；不是 HTTP TTFT、GPU Serving 或性能基线"}};
        if (!output_path.empty()) { cuda_reports::write(output_path,report); }
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mini-cuda-llm: " << error.what() << '\n';
        return 1;
    }
}
