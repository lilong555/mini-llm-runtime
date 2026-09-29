#include "test_support.h"
#include "cuda_validation_support.h"
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
using cuda_validation::Reference;
using cuda_validation::argmax;
using cuda_validation::compare;

namespace {
using Batch = std::vector<InputToken>;
using Outputs = std::vector<std::vector<Logits>>;
struct Scenario {
    std::string id;
    std::size_t sequences;
    std::vector<Batch> batches;
    Outputs cpu, reference;
};

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

class PagedValidation {
    struct Case {
        Scenario input;
        std::size_t continuation = 0;
        std::vector<std::int32_t> golden;
    };
public:
    PagedValidation(std::string path, const json& contract, std::filesystem::path output)
        : path_(std::move(path)), output_(std::move(output)) {
        report_ = {{"schema_version",1},{"spec_id","GPU-KV-001"},{"passed",false},
            {"reference","同一 F32 Runtime 的 contiguous 布局，按相同批次逐位比较"},
            {"performance_baseline",false},{"comparisons",json::array()},{"cases",json::array()},
            {"configurations",json::array()},{"first_failure",nullptr}};
        for (auto& c : scenarios(contract)) { cases_.push_back({std::move(c),0,{}}); }
        auto recipe = contract;
        recipe["teacher_forcing"]["positions"] = {0,1,15,16,17,32,127,128,129,255,1535,2047};
        for (const auto length : {128u,1536u,2048u}) {
            const cuda_validation::TeacherCase c{1,length,length == 128 ? 16u : 128u,1};
            cases_.push_back({{cuda_validation::case_id(recipe,c),1,
                cuda_validation::teacher_batches(recipe,c),{},{}},length == 1536 ? 32u : 0u,{}});
        }
        for (std::size_t s : {1u,4u}) {
            for (std::size_t i = 0; i < contract.at("stable_greedy").size(); ++i) {
                const auto& golden = contract.at("stable_greedy").at(i);
                const auto ids = golden.at("input_token_ids").get<std::vector<std::int32_t>>();
                Case c{{"golden-"+std::to_string(i),s,{},{},{}},8,
                    golden.at("expected_token_ids").get<std::vector<std::int32_t>>()};
                for (std::size_t first = 0; first < ids.size(); first += 2) {
                    Batch batch;
                    for (auto p = first; p < std::min(first+2,ids.size()); ++p) {
                        batch.push_back({ids[p],std::int32_t(p),0,p+1==ids.size()});
                    }
                    c.input.batches.push_back(std::move(batch));
                }
                cases_.push_back(std::move(c));
            }
        }
    }

    void run(std::size_t sequences) {
        // 两个布局顺序构造；仅在 host 保留参照 logits 和冻结的续写输入。
        for (const auto layout : {CudaKvLayout::contiguous,CudaKvLayout::paged}) {
            CudaRuntimeConfig config{path_,0,sequences,2048,128,0};
            config.kv_layout = layout;
            if (layout == CudaKvLayout::paged) { config.kv_capacity_tokens = sequences == 1 ? 2048 : 2560; }
            CudaRuntime runtime(config);
            const auto before = runtime.diagnostics();
            const auto allocations = allocation_stats();
            for (auto& c : cases_) {
                auto& input = c.input;
                if (input.sequences != sequences) { continue; }
                clear(runtime);
                std::vector<std::int32_t> generated;
                const auto prompt_batches = input.batches.size();
                if (layout == CudaKvLayout::contiguous) {
                    for (const auto& batch : input.batches) {
                        auto result = runtime.forward(batch,CudaOutputMode::debug_logits);
                        validate_samples(batch,result);
                        input.cpu.push_back(std::move(result.logits));
                    }
                    if (c.continuation) {
                        auto next = argmax(input.cpu.back().at(0).values);
                        auto position = input.batches.back().back().position+1;
                        generated.push_back(next);
                        for (std::size_t step = 1; step < c.continuation; ++step) {
                            input.batches.push_back({{next,position++,0,true}});
                            auto result = runtime.forward(input.batches.back(),CudaOutputMode::debug_logits);
                            validate_samples(input.batches.back(),result);
                            next = result.samples.at(0).token;
                            generated.push_back(next);
                            input.cpu.push_back(std::move(result.logits));
                        }
                    }
                    json batches = json::array();
                    for (const auto& batch : input.batches) {
                        json tokens = json::array();
                        for (const auto& token : batch) {
                            tokens.push_back({{"token",token.token},{"position",token.position},
                                {"sequence",token.sequence},{"logits",token.logits}});
                        }
                        batches.push_back(std::move(tokens));
                    }
                    report_["cases"].push_back({{"id",input.id},{"max_sequences",sequences},
                        {"prompt_batches",prompt_batches},{"continuation_tokens",c.continuation},
                        {"batches",batches},{"batch_sha256",cuda_validation::batch_digest(input.batches)},
                        {"generated_tokens",generated},{"golden",c.golden}});
                } else {
                    const auto prefill_batches = input.batches.size()-(c.continuation ? c.continuation-1 : 0);
                    for (std::size_t i = 0; i < input.batches.size(); ++i) {
                        const auto& batch = input.batches[i];
                        const bool timed = input.id == "zh-33";
                        const auto result = runtime.forward(batch,CudaOutputMode::debug_logits,timed);
                        CHECK(!timed || (result.device_elapsed_ms && *result.device_elapsed_ms >= 0));
                        validate_samples(batch,result);
                        CHECK(result.logits.size() == input.cpu[i].size());
                        for (std::size_t row = 0; row < result.logits.size(); ++row) {
                            const auto& actual = result.logits[row].values;
                            const auto& expected = input.cpu[i][row].values;
                            const auto& token = batch[result.samples[row].input_index];
                            const bool equal = actual.size() == expected.size() &&
                                std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)) == 0;
                            const auto a = cuda_validation::score(actual), e = cuda_validation::score(expected);
                            json comparison{{"case",input.id},{"max_sequences",sequences},{"batch",i},
                                {"sequence",token.sequence},{"position",token.position},
                                {"input_index",result.samples[row].input_index},{"actual",a},{"reference",e},
                                {"bitwise_equal",equal},{"passed",equal}};
                            report_["comparisons"].push_back(comparison);
                            if (!equal) {
                                report_["first_failure"] = comparison;
                                cuda_reports::write(output_/"first-failure-logits.json",
                                    {{"comparison",comparison},{"contiguous",expected},{"paged",actual}});
                                save();
                            }
                            CHECK(equal);
                        }
                        if (c.continuation && i+1 >= prefill_batches) { generated.push_back(result.samples.at(0).token); }
                        const auto d = runtime.diagnostics();
                        std::size_t pages = 0;
                        for (auto length : d.sequence_lengths) { pages += (length+15)/16; }
                        CHECK(runtime.live_kv_pages() == pages && pages <= *d.capacity_pages);
                    }
                }
                CHECK(c.golden.empty() || generated == c.golden);
                if (input.id == "en-l2048-c128-s1") {
                    const auto valid = runtime.diagnostics();
                    test::throws<std::invalid_argument>([&] {
                        runtime.forward(std::array<InputToken,1>{{{785,2048,0,true}}});
                    });
                    const auto rejected = runtime.diagnostics();
                    CHECK(rejected.state == CudaRuntimeState::ready);
                    CHECK(rejected.sequence_lengths == valid.sequence_lengths);
                    CHECK(rejected.live_kv_pages == valid.live_kv_pages);
                    CHECK(rejected.completed_forwards == valid.completed_forwards);
                    CHECK(rejected.metadata_h2d_bytes == valid.metadata_h2d_bytes);
                    CHECK(rejected.page_table_h2d_bytes == valid.page_table_h2d_bytes);
                }
                save();
            }
            clear(runtime);
            const auto after = runtime.diagnostics();
            const auto end = allocation_stats();
            CHECK(after.weight_h2d_bytes == before.weight_h2d_bytes && after.rope_h2d_bytes == before.rope_h2d_bytes);
            CHECK(after.intermediate_h2d_bytes == 0 && after.intermediate_d2h_bytes == 0);
            CHECK(after.state == CudaRuntimeState::ready && after.post_launch_failures == 0 && after.live_kv_tokens == 0);
            CHECK(allocations.allocation_calls == end.allocation_calls && allocations.release_calls == end.release_calls);
            CHECK(layout != CudaKvLayout::paged || (after.live_kv_pages == 0 && after.page_table_h2d_bytes > 0));
            report_["configurations"].push_back({{"max_sequences",sequences},
                {"layout",kv_layout_name(layout)},{"before",cuda_reports::diagnostics(before)},
                {"after",cuda_reports::diagnostics(after)},{"device",cuda_reports::device(runtime.device_info())},
                {"arithmetic",cuda_reports::arithmetic(runtime)},{"steady_device_allocations",0},{"passed",true}});
            save();
        }
        for (auto& c : cases_) { if (c.input.sequences == sequences) { c.input.cpu.clear(); } }
    }
    void save() const { cuda_reports::write(output_/"paged-model-validation.json",report_); }
    json finish(bool passed) {
        report_["passed"] = passed;
        save();
        return {{"schema_version",1},{"status",passed ? "passed" : "failed"},{"passed",passed},
            {"spec_id","GPU-KV-001"},{"paged_gpu_model",passed},{"full_corpus_contract",false},
            {"performance_baseline",false},{"cases",report_.at("cases").size()},
            {"bitwise_comparisons",report_.at("comparisons").size()}};
    }
private:
    static void clear(CudaRuntime& runtime) {
        const auto before = runtime.diagnostics();
        for (std::size_t s = 0; s < runtime.config().max_sequences; ++s) { runtime.clear_sequence(std::int32_t(s)); }
        const auto after = runtime.diagnostics();
        CHECK(after.live_kv_tokens == 0 && after.owned_device_bytes == before.owned_device_bytes);
        CHECK(after.kv_layout != CudaKvLayout::paged || after.live_kv_pages == 0);
    }
    static void validate_samples(const Batch& batch, const CudaForwardResult& result) {
        std::size_t row = 0;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!batch[i].logits) { continue; }
            const auto& sample = result.samples.at(row);
            const auto& logits = result.logits.at(row++);
            CHECK(sample.input_index == i && sample.sequence == batch[i].sequence);
            CHECK(logits.sequence == sample.sequence && sample.token == argmax(logits.values));
            CHECK(std::all_of(logits.values.begin(),logits.values.end(),[](float v) { return std::isfinite(v); }));
        }
        CHECK(row == result.samples.size() && row == result.logits.size());
    }
    std::string path_;
    std::filesystem::path output_;
    json report_;
    std::vector<Case> cases_;
};
}

int main(int argc, char** argv) {
    std::filesystem::path output;
    bool owns_output = false;
    try {
        Options options(argc,argv,{"--model","--reference-model","--contract","--output","--precision-study"},{"--help","--full","--paged-kv"});
        if (options.has("--help")) {
            std::cout << "minillm-cuda-model-tests --model MODEL --contract JSON --output NEW_DIRECTORY\n"
                         "  --reference-model F32 [--full] 或 --precision-study INPUT_JSON 或 --paged-kv\n";
            return 0;
        }
        const bool precision = options.has("--precision-study");
        const bool paged = options.has("--paged-kv");
        if (paged && (precision || options.has("--full") || options.has("--reference-model"))) {
            throw std::invalid_argument("--paged-kv 不能与其他模型验证模式同时使用");
        }
        if (precision && (options.has("--full") || options.has("--reference-model"))) {
            throw std::invalid_argument("--precision-study 不能与 --full 或 --reference-model 同时使用");
        }
        for (const auto* name : {"--model","--contract","--output"}) {
            if (options.get(name).empty()) { throw std::invalid_argument(std::string("缺少参数：")+name); }
        }
        if (!precision && !paged && options.get("--reference-model").empty()) { throw std::invalid_argument("缺少参数：--reference-model"); }
        output = options.get("--output");
        if (!output.parent_path().empty()) { std::filesystem::create_directories(output.parent_path()); }
        if (!std::filesystem::create_directory(output)) { throw std::runtime_error("模型验证目录必须尚不存在"); }
        owns_output = true;
        cuda_reports::write(output/"validation-summary.json",{{"status","incomplete"},{"passed",false}});
        std::ifstream input(options.get("--contract"));
        const auto contract = json::parse(input);
        const auto model_sha = cuda_reports::file_hash(options.get("--model"));
        CHECK(contract.at("schema_version") == 1 && contract.at("model").at("sha256") == model_sha);
        std::filesystem::copy_file(options.get("--contract"),output/"validation-contract.json");
        llama_log_set([](ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) { std::cerr << text; }
        },nullptr);
        if (paged) {
            PagedValidation validation(options.get("--model"),contract,output);
            test::cases().push_back({"cuda_paged_model_single_sequence",[&] { validation.run(1); }});
            test::cases().push_back({"cuda_paged_model_four_sequences",[&] { validation.run(4); }});
            const auto result = test::run();
            auto summary = validation.finish(result == 0);
            summary["model_sha256"] = model_sha;
            cuda_reports::write(output/"validation-summary.json",summary);
            return result;
        }
        if (precision) {
            const auto experiment_path = options.get("--precision-study");
            const auto input_sha = cuda_reports::file_hash(experiment_path);
            CHECK(input_sha == "3b9ec80ee5e9cc83865378f21c46d5dedf4975530e7686a1dfb61d5f4af992b8");
            std::ifstream experiment_file(experiment_path);
            const auto experiment = json::parse(experiment_file);
            CHECK(experiment.at("model_sha256") == model_sha);
            CHECK(experiment.at("validation_contract").at("sha256") == cuda_reports::file_hash(options.get("--contract")));
            std::filesystem::copy_file(experiment_path,output/"precision-input.json");
            auto summary = cuda_validation::run_precision_validation(options.get("--model"),contract,experiment,output);
            summary["model_sha256"] = model_sha; summary["precision_input_sha256"] = input_sha;
            cuda_reports::write(output/"validation-summary.json",summary);
            return summary.at("passed").get<bool>() ? 0 : 1;
        }
        const auto reference_sha = cuda_reports::file_hash(options.get("--reference-model"));
        CHECK(contract.at("reference").at("sha256") == reference_sha && contract.at("reference").at("source_sha256") == model_sha);
        if (options.has("--full")) {
            auto summary = cuda_validation::run_full_validation(options.get("--model"),options.get("--reference-model"),contract,output);
            summary["model_sha256"] = model_sha; summary["reference_sha256"] = reference_sha;
            cuda_reports::write(output/"validation-summary.json",summary);
            return summary.at("passed").get<bool>() ? 0 : 1;
        }
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
