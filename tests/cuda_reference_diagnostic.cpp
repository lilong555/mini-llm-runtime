#include "cuda_validation_support.h"
#include "../apps/options.h"

using namespace cuda_validation;

namespace {
template<class Forward>
std::vector<std::vector<float>> replay(const std::vector<std::int32_t>& prompt,
                                     const std::vector<std::int32_t>& continuation, Forward forward) {
    std::vector<Logits> rows;
    for (std::size_t first = 0; first < prompt.size(); first += 128) {
        Batch batch;
        for (std::size_t p = first; p < std::min(first+128,prompt.size()); ++p) {
            batch.push_back({prompt[p],std::int32_t(p),0,p+1==prompt.size()});
        }
        rows = forward(batch);
    }
    std::vector<std::vector<float>> result;
    for (std::size_t step = 0; step < continuation.size(); ++step) {
        CHECK(rows.size() == 1 && rows[0].sequence == 0);
        result.push_back(std::move(rows[0].values));
        if (step+1 < continuation.size()) {
            rows = forward(Batch{{continuation[step],std::int32_t(prompt.size()+step),0,true}});
        }
    }
    return result;
}
}

int main(int argc, char** argv) {
    std::filesystem::path output;
    bool owns_output = false;
    try {
        Options options(argc,argv,{"--model","--reference-model","--contract","--generation-report","--output"},{"--help"});
        if (options.has("--help")) {
            std::cout << "minillm-cuda-reference-diagnostic --model MODEL --reference-model F32 --contract JSON "
                         "--generation-report JSON --output NEW_DIRECTORY\n";
            return 0;
        }
        for (const auto* key : {"--model","--reference-model","--contract","--generation-report","--output"}) {
            CHECK(!options.get(key).empty());
        }
        output = options.get("--output");
        if (!output.parent_path().empty()) { std::filesystem::create_directories(output.parent_path()); }
        if (!std::filesystem::create_directory(output)) { throw std::runtime_error("诊断目录必须尚不存在"); }
        owns_output = true;
        cuda_reports::write(output/"summary.json",{{"status","incomplete"},{"passed",false}});
        const auto load = [](const std::string& path) { std::ifstream stream(path); return json::parse(stream); };
        const auto contract = load(options.get("--contract"));
        const auto generation = load(options.get("--generation-report"));
        CHECK(contract.at("model").at("sha256") == cuda_reports::file_hash(options.get("--model")));
        CHECK(contract.at("reference").at("sha256") == cuda_reports::file_hash(options.get("--reference-model")));
        CHECK(generation.at("id") == "repeated-l1536-g32" && generation.at("rows").size() == 32);
        const auto prompt = expanded_tokens(contract,2,1536);
        std::vector<std::int32_t> continuation;
        for (const auto& row : generation.at("rows")) { continuation.push_back(row.at("cuda").at("token").get<std::int32_t>()); }
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) { std::cerr << text; }
        },nullptr);
        std::array<std::vector<std::vector<float>>,4> values;
        {
            std::cout << "固定输入重放：自有 CPU" << std::endl;
            Runtime cpu({options.get("--model"),2048,16,4,128,8,KernelMode::automatic});
            values[0] = replay(prompt,continuation,[&](const Batch& batch) { return cpu.forward(batch); });
        }
        for (std::size_t index = 1; index <= 2; ++index) {
            std::cout << "固定输入重放：F32 llama，FlashAttention=" << (index==1 ? "ON" : "OFF") << std::endl;
            Reference reference(options.get("--reference-model"),index==1);
            values[index] = replay(prompt,continuation,[&](const Batch& batch) { return reference.forward(batch); });
        }
        {
            std::cout << "固定输入重放：自有 CUDA" << std::endl;
            CudaRuntime cuda({options.get("--model"),0,4,2048,128,0});
            values[3] = replay(prompt,continuation,[&](const Batch& batch) {
                return cuda.forward(batch,CudaOutputMode::debug_logits).logits;
            });
        }
        json rows = json::array();
        std::array<std::size_t,3> failures{}, digest_matches{};
        const std::array<const char*,4> names{"cpu","llama_fused","llama_unfused","cuda"};
        for (std::size_t step = 0; step < continuation.size(); ++step) {
            json row = {{"step",step},{"scores",json::object()},{"comparisons",json::object()}};
            for (std::size_t backend = 0; backend < names.size(); ++backend) {
                row["scores"][names[backend]] = score(values[backend][step]);
            }
            for (std::size_t backend = 0; backend < 3; ++backend) {
                auto result = compare(values[3][step],values[backend][step],contract.at("thresholds"));
                failures[backend] += result.at("passed").get<bool>() ? 0 : 1;
                row["comparisons"][names[backend]] = std::move(result);
            }
            const auto& original = generation.at("rows").at(step);
            digest_matches[0] += original.at("references").at("cpu").at("score") == row.at("scores").at("cpu");
            digest_matches[1] += original.at("references").at("llama_f32").at("score") == row.at("scores").at("llama_fused");
            digest_matches[2] += original.at("cuda") == row.at("scores").at("cuda");
            rows.push_back(std::move(row));
        }
        const bool supported = failures[0] == 0 && failures[1] == 2 && failures[2] == 0 &&
            digest_matches == (std::array<std::size_t,3>{32,32,32});
        cuda_reports::write(output/"attention-reference.json",{{"schema_version",1},
            {"scope","独立 attention 参照诊断，不是完整数值门禁"},{"case","repeated-l1536-g32"},
            {"input_mode","固定重放原始 GPU 输出，所有后端输入完全相同"},{"prompt",prompt},{"continuation",continuation},
            {"threads",8},{"kv_dtype","F16"},{"fused_failures",failures[1]},{"unfused_failures",failures[2]},
            {"cpu_failures",failures[0]},{"original_score_matches",digest_matches},{"rows",rows},
            {"hypothesis_supported",supported}});
        cuda_reports::write(output/"summary.json",{{"schema_version",1},{"status",supported ? "passed" : "failed"},
            {"passed",supported},{"hypothesis_supported",supported},{"full_corpus_contract",false},
            {"fused_failures",failures[1]},{"unfused_failures",failures[2]},{"original_score_matches",digest_matches}});
        return supported ? 0 : 1;
    } catch (const std::exception& error) {
        if (owns_output) {
            try { cuda_reports::write(output/"summary.json",{{"status","failed"},{"passed",false},{"error",error.what()}}); }
            catch (...) {}
        }
        std::cerr << "参照诊断失败：" << error.what() << '\n';
        return 1;
    }
}
