#include "test_support.h"
#include "../apps/cuda_reports.h"
#include "../apps/options.h"
#include "minillm/cuda/device_buffer.h"
#include "minillm/runtime.h"
#include "ggml-backend.h"
#include "hash/hash.h"
#include "llama.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <set>

using namespace minillm;
using namespace minillm::cuda;
using cuda_reports::json;

namespace {
using Batch = std::vector<InputToken>;
using Outputs = std::vector<std::vector<Logits>>;
struct Scenario {
    std::string id;
    std::size_t sequences;
    std::vector<Batch> batches;
    Outputs cpu, reference;
};

class Reference {
    struct Backend {
        Backend() { ggml_backend_load_all(); llama_backend_init(); }
        ~Backend() { llama_backend_free(); }
    } backend_;
    struct OwnedBatch {
        llama_batch value = llama_batch_init(128,0,1);
        ~OwnedBatch() { llama_batch_free(value); }
    } batch_;
public:
    explicit Reference(const std::string& path) {
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        model_.reset(llama_model_load_from_file(path.c_str(),mp));
        if (!model_) { throw std::runtime_error("F32 参照模型加载失败"); }
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = 128; cp.n_ubatch = 128; cp.n_seq_max = 4;
        cp.n_threads = 8; cp.n_threads_batch = 8; cp.kv_unified = true;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_.reset(llama_init_from_model(model_.get(),cp));
        if (!context_) { throw std::runtime_error("F32 参照 context 加载失败"); }
        vocabulary_ = std::size_t(llama_vocab_n_tokens(llama_model_get_vocab(model_.get())));
    }
    void clear() {
        llama_synchronize(context_.get());
        llama_memory_clear(llama_get_memory(context_.get()),false);
    }
    std::vector<Logits> forward(const Batch& tokens) {
        CHECK(!tokens.empty() && tokens.size() <= 128);
        auto& batch = batch_.value;
        batch.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i].token; batch.pos[i] = tokens[i].position;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = tokens[i].sequence;
            batch.logits[i] = static_cast<std::int8_t>(tokens[i].logits);
        }
        CHECK(llama_decode(context_.get(),batch) == 0);
        std::vector<Logits> result;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (!tokens[i].logits) { continue; }
            const auto* values = llama_get_logits_ith(context_.get(),static_cast<std::int32_t>(i));
            CHECK(values);
            result.push_back({tokens[i].sequence,{values,values+vocabulary_}});
        }
        return result;
    }
private:
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model_{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> context_{nullptr,llama_free};
    std::size_t vocabulary_ = 0;
};

std::int32_t argmax(const std::vector<float>& values) {
    return static_cast<std::int32_t>(std::max_element(values.begin(),values.end())-values.begin());
}
json compare(const std::vector<float>& actual, const std::vector<float>& expected, const json& thresholds) {
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    CHECK(actual.size() == expected.size() && actual.size() > 1);
    double squared = 0, maximum = 0, dot = 0, a_norm = 0, e_norm = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return {{"passed",false},{"all_finite",false},{"first_nonfinite",i},{"argmax_equal",false}};
        }
        const double error = double(actual[i])-expected[i];
        squared += error*error; maximum = std::max(maximum,std::abs(error));
        dot += double(actual[i])*expected[i]; a_norm += double(actual[i])*actual[i]; e_norm += double(expected[i])*expected[i];
    }
    const double rmse = std::sqrt(squared/double(actual.size())), cosine = dot/std::sqrt(a_norm*e_norm);
    const auto top = argmax(expected), token = argmax(actual);
    float second = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::int32_t(i) != top) { second = std::max(second,expected[i]); }
    }
    const double gap = double(expected[std::size_t(top)])-second;
    const bool near_tie = gap <= 2*maximum;
    return {{"rmse",rmse},{"max_absolute",maximum},{"cosine",cosine},{"all_finite",true},
        {"actual_sha256",hash_sha256_hex(actual.data(),actual.size()*sizeof(float))},
        {"reference_sha256",hash_sha256_hex(expected.data(),expected.size()*sizeof(float))},
        {"actual_argmax",token},{"reference_argmax",top},{"reference_margin",gap},
        {"near_tie",near_tie},{"argmax_equal",token==top},
        {"passed",rmse<thresholds.at("rmse_exclusive").get<double>() &&
                  maximum<thresholds.at("max_absolute_exclusive").get<double>() &&
                  cosine>=thresholds.at("cosine_min_inclusive").get<double>() && (near_tie || token==top)}};
}

std::vector<Scenario> scenarios(const json& contract) {
    std::set<std::int32_t> positions;
    for (const auto& p : contract.at("teacher_forcing").at("positions")) { positions.insert(p.get<std::int32_t>()); }
    std::vector<Scenario> result;
    for (const auto& corpus : contract.at("corpus")) {
        const auto seed = corpus.at("seed_token_ids").get<std::vector<std::int32_t>>();
        Scenario c{corpus.at("id").get<std::string>()+"-33",1,{},{},{}};
        for (std::size_t first = 0; first < 33; first += 16) {
            Batch batch;
            for (auto p = first; p < std::min(first+16,std::size_t{33}); ++p) {
                batch.push_back({seed[p%seed.size()],std::int32_t(p),0,positions.contains(std::int32_t(p))});
            }
            c.batches.push_back(std::move(batch));
        }
        result.push_back(std::move(c));
    }
    const auto english = contract.at("corpus").at(1).at("seed_token_ids").get<std::vector<std::int32_t>>();
    Scenario maximum{"en-128",1,{Batch{}},{},{}};
    for (std::size_t p = 0; p < 128; ++p) {
        maximum.batches[0].push_back({english[p%english.size()],std::int32_t(p),0,positions.contains(std::int32_t(p))});
    }
    result.push_back(std::move(maximum));
    Scenario mixed{"mixed-16-plus-2",4,{Batch{},Batch{},{{198,0,3,true},{323,17,2,true},{315,16,0,true},{629,17,1,true}}},{},{}};
    for (std::int32_t p = 0; p < 16; ++p) {
        mixed.batches[0].push_back({english[std::size_t(p)%english.size()],p,1,p==15});
        mixed.batches[0].push_back({14990,p,2,p==15});
        mixed.batches[1].push_back({p%2 ? 104455 : 14990,p,0,p==15});
    }
    mixed.batches[1].push_back({13598,16,1,true}); mixed.batches[1].push_back({14324,16,2,true});
    result.push_back(std::move(mixed));
    Scenario four{"interleaved-four-33",4,{Batch{},Batch{}},{},{}};
    for (std::int32_t p = 0; p < 33; ++p) {
        for (std::int32_t sequence = 0; sequence < 4; ++sequence) {
            const auto seed = contract.at("corpus").at(std::size_t(sequence)).at("seed_token_ids").get<std::vector<std::int32_t>>();
            four.batches[p<32 ? 0 : 1].push_back({seed[std::size_t(p)%seed.size()],p,sequence,positions.contains(p)});
        }
    }
    result.push_back(std::move(four));
    return result;
}

class Validation {
public:
    Validation(std::string path, std::string reference, json contract, std::filesystem::path output)
        : path_(std::move(path)), reference_path_(std::move(reference)), contract_(std::move(contract)),
          output_(std::move(output)), cases_(scenarios(contract_)) {
        report_ = {{"schema_version",1},{"scope","CUDA-VS-001 Step 7"},{"contract",contract_},
            {"references",{{"cpu","自有 CPU Runtime，Q8_0，FP16 KV，auto SIMD，8 线程"},
                {"llama","同有效权重 F32，CPU，FP16 KV，8 线程"},
                {"context_tokens",2048},{"batch_tokens",128},{"max_sequences",4},
                {"lifetime","CPU、F32 参照与 GPU 顺序构造，不同时驻留"}}},
            {"comparisons",json::array()},{"golden",json::array()},{"configurations",json::array()},
            {"logits_digest","sha256_raw_f32_le"},{"first_numeric_failure",nullptr},{"first_argmax_divergence",nullptr}};
        json inputs = json::array();
        for (const auto& c : cases_) {
            json batches = json::array();
            for (const auto& batch : c.batches) {
                json tokens = json::array();
                for (const auto& token : batch) {
                    tokens.push_back({{"token",token.token},{"position",token.position},
                        {"sequence",token.sequence},{"logits",token.logits}});
                }
                batches.push_back(tokens);
            }
            inputs.push_back({{"id",c.id},{"max_sequences",c.sequences},{"batches",batches}});
        }
        cuda_reports::write(output_/"input.json",{{"schema_version",1},{"cases",inputs}});
    }
    void prepare_references() {
        {
            Runtime cpu({path_,2048,16,4,128,8,KernelMode::automatic});
            for (auto& c : cases_) {
                for (std::int32_t seq = 0; seq < 4; ++seq) { cpu.clear_sequence(seq); }
                for (const auto& batch : c.batches) { c.cpu.push_back(cpu.forward(batch)); }
            }
        }
        {
            Reference reference(reference_path_);
            for (auto& c : cases_) {
                reference.clear();
                for (const auto& batch : c.batches) { c.reference.push_back(reference.forward(batch)); }
            }
        }
    }
    void run(std::size_t sequences) {
        CudaRuntime runtime({path_,0,sequences,2048,128,0});
        CHECK(runtime.dimensions().layers == 28 && runtime.dimensions().head_dim == 128);
        CHECK(runtime.dimensions().vocabulary == contract_.at("model").at("vocabulary"));
        const auto before = runtime.diagnostics();
        const auto allocations = allocation_stats();
        std::size_t timing_checks = 0;
        for (const auto& c : cases_) {
            if (c.sequences != sequences) { continue; }
            clear(runtime);
            std::vector<CudaForwardResult> baseline;
            for (std::size_t i = 0; i < c.batches.size(); ++i) {
                const auto& batch = c.batches[i];
                auto actual = runtime.forward(batch,CudaOutputMode::debug_logits);
                CHECK(actual.samples.size() == c.cpu[i].size() && actual.samples.size() == c.reference[i].size());
                std::size_t row = 0;
                for (std::size_t input = 0; input < batch.size(); ++input) {
                    if (!batch[input].logits) { continue; }
                    CHECK(actual.samples[row].sequence == batch[input].sequence && actual.samples[row].input_index == input);
                    CHECK(actual.samples[row].token == argmax(actual.logits[row].values));
                    for (const auto* name : {"cpu","llama_f32"}) {
                        const auto& expected = std::string_view(name)=="cpu" ? c.cpu[i][row] : c.reference[i][row];
                        CHECK(expected.sequence == batch[input].sequence);
                        auto comparison = compare(actual.logits[row].values,expected.values,contract_.at("thresholds"));
                        comparison["case"] = c.id; comparison["reference"] = name;
                        comparison["sequence"] = batch[input].sequence; comparison["position"] = batch[input].position;
                        comparison["input_index"] = input; comparison["batch_tokens"] = batch.size();
                        if (!comparison.at("argmax_equal").get<bool>() && report_["first_argmax_divergence"].is_null()) {
                            report_["first_argmax_divergence"] = comparison;
                        }
                        if (!comparison.at("passed").get<bool>() && report_["first_numeric_failure"].is_null()) {
                            report_["first_numeric_failure"] = comparison;
                        }
                        report_["comparisons"].push_back(comparison);
                        if (!comparison.at("passed").get<bool>()) { save(); }
                        CHECK(comparison.at("passed").get<bool>());
                    }
                    ++row;
                }
                baseline.push_back(std::move(actual));
            }
            if (c.id == "zh-33") {
                clear(runtime);
                for (std::size_t i = 0; i < c.batches.size(); ++i) {
                    const auto timed = runtime.forward(c.batches[i],CudaOutputMode::debug_logits,true);
                    CHECK(timed.device_elapsed_ms && *timed.device_elapsed_ms >= 0);
                    for (std::size_t row = 0; row < timed.logits.size(); ++row) {
                        const auto& expected = baseline[i].logits[row].values;
                        CHECK(timed.samples[row].token == baseline[i].samples[row].token);
                        CHECK(timed.logits[row].values.size() == expected.size());
                        CHECK(std::memcmp(timed.logits[row].values.data(),expected.data(),expected.size()*sizeof(float)) == 0);
                    }
                    ++timing_checks;
                }
            }
        }
        const auto before_greedy = runtime.diagnostics();
        for (const auto& golden : contract_.at("stable_greedy")) {
            clear(runtime);
            const auto input = golden.at("input_token_ids").get<std::vector<std::int32_t>>();
            CHECK(runtime.tokenize(golden.at("text").get<std::string>()) == input);
            const auto expected = golden.at("expected_token_ids").get<std::vector<std::int32_t>>();
            CudaForwardResult output;
            for (std::size_t first = 0; first < input.size(); first += 2) {
                Batch batch;
                for (std::size_t i = first; i < std::min(first+2,input.size()); ++i) {
                    batch.push_back({input[i],std::int32_t(i),0,i+1==input.size()});
                }
                output = runtime.forward(batch);
                CHECK(output.logits.empty());
            }
            std::vector<std::int32_t> actual;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                actual.push_back(output.samples.at(0).token);
                if (i+1 != expected.size()) {
                    output = runtime.forward(std::array<InputToken,1>{{{actual.back(),std::int32_t(input.size()+i),0,true}}});
                    CHECK(output.logits.empty());
                }
            }
            report_["golden"].push_back({{"max_sequences",sequences},{"text",golden.at("text")},
                {"expected",expected},{"actual",actual},{"passed",actual==expected}});
            if (actual != expected) { save(); }
            CHECK(actual == expected);
        }
        CHECK(runtime.diagnostics().debug_d2h_bytes == before_greedy.debug_d2h_bytes);
        clear(runtime);
        runtime.forward(std::array<InputToken,1>{{{785,0,0,false}}});
        const auto valid = runtime.diagnostics();
        test::throws<std::invalid_argument>([&] {
            runtime.forward(std::array<InputToken,2>{{{785,1,0,false},{785,3,0,true}}});
        });
        const auto rejected = runtime.diagnostics();
        CHECK(rejected.sequence_lengths == valid.sequence_lengths && rejected.state == CudaRuntimeState::ready);
        CHECK(rejected.metadata_h2d_bytes == valid.metadata_h2d_bytes && rejected.completed_forwards == valid.completed_forwards);
        CHECK(runtime.forward(std::array<InputToken,1>{{{13,1,0,true}}}).samples.size() == 1);
        clear(runtime);
        const auto after = runtime.diagnostics();
        const auto end = allocation_stats();
        CHECK(after.weight_h2d_bytes == before.weight_h2d_bytes && after.rope_h2d_bytes == before.rope_h2d_bytes);
        CHECK(after.intermediate_h2d_bytes == 0 && after.intermediate_d2h_bytes == 0 && after.post_launch_failures == 0);
        CHECK(after.live_kv_tokens == 0 && after.state == CudaRuntimeState::ready);
        CHECK(allocations.allocation_calls == end.allocation_calls && allocations.release_calls == end.release_calls);
        report_["configurations"].push_back({{"max_sequences",sequences},{"max_model_len",2048},{"batch_tokens",128},
            {"before",cuda_reports::diagnostics(before)},{"after",cuda_reports::diagnostics(after)},
            {"steady_project_allocation_calls",0},{"steady_project_release_calls",0},
            {"timing_bitwise_batches",timing_checks},{"device",cuda_reports::device(runtime.device_info())},
            {"arithmetic",cuda_reports::arithmetic(runtime)},{"passed",true}});
        if (sequences == 4) { cuda_reports::write(output_/"weight-plan.json",cuda_reports::weights(runtime.weight_manifest())); }
        save();
    }
    void save() const { cuda_reports::write(output_/"model-validation.json",report_); }
    std::size_t comparisons() const { return report_.at("comparisons").size(); }
    std::size_t golden_cases() const { return report_.at("golden").size(); }
private:
    static void clear(CudaRuntime& runtime) {
        for (std::size_t s = 0; s < runtime.config().max_sequences; ++s) { runtime.clear_sequence(std::int32_t(s)); }
    }
    std::string path_, reference_path_;
    json contract_, report_;
    std::filesystem::path output_;
    std::vector<Scenario> cases_;
};
}

int main(int argc, char** argv) {
    std::filesystem::path output;
    bool owns_output = false;
    try {
        Options options(argc,argv,{"--model","--reference-model","--contract","--output"});
        if (options.has("--help")) {
            std::cout << "minillm-cuda-model-tests --model MODEL --reference-model F32 --contract JSON --output NEW_DIRECTORY\n";
            return 0;
        }
        for (const auto* name : {"--model","--reference-model","--contract","--output"}) {
            if (options.get(name).empty()) { throw std::invalid_argument(std::string("缺少参数：")+name); }
        }
        output = options.get("--output");
        if (!output.parent_path().empty()) { std::filesystem::create_directories(output.parent_path()); }
        if (!std::filesystem::create_directory(output)) { throw std::runtime_error("模型验证目录必须尚不存在"); }
        owns_output = true;
        cuda_reports::write(output/"validation-summary.json",{{"status","incomplete"},{"passed",false}});
        std::ifstream input(options.get("--contract"));
        const auto contract = json::parse(input);
        const auto model_sha = cuda_reports::file_hash(options.get("--model"));
        const auto reference_sha = cuda_reports::file_hash(options.get("--reference-model"));
        CHECK(contract.at("schema_version") == 1 && contract.at("model").at("sha256") == model_sha);
        CHECK(contract.at("reference").at("sha256") == reference_sha && contract.at("reference").at("source_sha256") == model_sha);
        std::filesystem::copy_file(options.get("--contract"),output/"validation-contract.json");
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) { std::cerr << text; }
        },nullptr);
        Validation validation(options.get("--model"),options.get("--reference-model"),contract,output);
        validation.prepare_references();
        test::cases().push_back({"cuda_model_single_sequence",[&] { validation.run(1); }});
        test::cases().push_back({"cuda_model_four_sequences",[&] { validation.run(4); }});
        const auto result = test::run();
        validation.save();
        cuda_reports::write(output/"validation-summary.json",{{"schema_version",1},{"status",result ? "failed" : "passed"},
            {"passed",result==0},{"complete_gpu_model",result==0},{"full_corpus_contract",false},{"performance_baseline",false},
            {"model_sha256",model_sha},{"reference_sha256",reference_sha},{"logits_comparisons",validation.comparisons()},
            {"golden_cases",validation.golden_cases()}});
        return result;
    } catch (const std::exception& error) {
        if (owns_output) {
            try { cuda_reports::write(output/"validation-summary.json",{{"status","failed"},{"passed",false}}); }
            catch (const std::exception& report_error) { std::cerr << "记录失败摘要失败：" << report_error.what() << '\n'; }
        }
        std::cerr << "CUDA 模型验证失败：" << error.what() << '\n';
        return 1;
    }
}
