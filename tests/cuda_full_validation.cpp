#include "cuda_validation_support.h"
#include "minillm/cuda/device_buffer.h"

#include <cstring>
#include <map>
#include <numeric>

namespace cuda_validation {
namespace {

using Rows = std::map<std::int32_t,std::vector<float>>;
using Canonical = std::map<std::pair<std::size_t,std::size_t>,Rows>;
constexpr bool reference_flash_attention = false;

struct Generation {
    std::size_t corpus, length;
    std::array<std::vector<std::vector<float>>,2> reference;
};

class FullValidation {
public:
    FullValidation(const std::string& model, const std::string& reference, const json& contract,
                   const std::filesystem::path& output)
        : model_(model), reference_(reference), contract_(contract), output_(output), cases_(teacher_cases(contract)) {
        report_ = {{"schema_version",1},{"scope","CUDA-VS-001 Step 8 numerical"},{"status","incomplete"},
            {"complete",false},{"passed",false},{"teacher_forcing",json::array()},{"canonical",json::array()},
            {"generation",json::array()},{"golden",json::array()},{"configurations",json::array()},
            {"timing",nullptr},{"boundary",nullptr},{"first_numeric_failure",nullptr},{"first_argmax_divergence",nullptr},
            {"references",{{"cpu","自有 CPU Runtime，Q8_0，FP16 KV，auto SIMD，8 线程"},
                {"llama_f32","同有效权重 F32，CPU，FP16 KV，8 线程，非融合 attention"},
                {"llama_f32_settings",{{"flash_attention",reference_flash_attention},{"kv_dtype","F16"},{"threads",8},
                    {"qk_accumulation_dtype","F32"},{"pv_accumulation_dtype","F32"}}},
                {"cuda_canonical","独立 S=1、chunk=128 的自有 CUDA Runtime"},
                {"lifetime","CPU、F32、GPU 顺序构造；每次只常驻一个模型执行实例"}}},
            {"totals",{{"teacher_cases",0},{"teacher_rows",0},{"teacher_comparisons",0},
                {"generation_comparisons",0},{"numeric_failures",0},{"argmax_divergences",0},{"near_ties",0}}},
            {"extrema",{{"rmse_max",0.0},{"absolute_max",0.0},{"cosine_min",1.0}}}};
        std::filesystem::create_directory(output_/"canonical");
        std::filesystem::create_directory(output_/"teacher");
        std::filesystem::create_directory(output_/"generation");
        json input = {{"schema_version",1},{"contract_id",contract_.at("contract_id")},
            {"chunk_unit","total_batch_input_rows"},{"sequence_corpus","(case_corpus_index + sequence) % corpus_count"},
            {"token_order","position_then_sequence"},{"canonical",{{"sequences",1},{"chunk",128}}},
            {"input_digest","sha256_i32le_batch_size_then_token_position_sequence_logits"},
            {"natural_generation",{{"prompt_lengths",{16,128,1536}},{"output_tokens",32},{"chunk",128},
                {"eog_policy","固定生成 32 个 token，不因 EOG 提前结束"}}},
            {"corpora",json::array()},{"teacher_cases",json::array()}};
        std::size_t expected_rows = 0;
        for (std::size_t corpus = 0; corpus < contract_.at("corpus").size(); ++corpus) {
            for (const auto& length : contract_.at("teacher_forcing").at("lengths")) {
                const auto count = length.get<std::size_t>();
                input["corpora"].push_back({{"corpus",corpus},{"length",count},
                    {"token_ids",expanded_tokens(contract_,corpus,count)}});
            }
            for (const std::size_t length : {16,128,1536}) { generations_.push_back({corpus,length,{}}); }
        }
        for (const auto& c : cases_) {
            const auto batches = teacher_batches(contract_,c);
            std::size_t rows = 0;
            for (const auto& batch : batches) {
                rows += std::count_if(batch.begin(),batch.end(),[](const auto& token) { return token.logits; });
            }
            input["teacher_cases"].push_back({{"id",case_id(contract_,c)},{"corpus",c.corpus},{"length",c.length},
                {"chunk",c.chunk},{"sequences",c.sequences},{"batches",batches.size()},
                {"sampled_rows",rows},{"input_sha256",batch_digest(batches)}});
            expected_rows += rows;
        }
        report_["expected"] = {{"teacher_cases",cases_.size()},{"teacher_rows",expected_rows},
            {"teacher_comparisons",expected_rows*3},{"canonical_cases",60},{"generation_cases",generations_.size()},
            {"golden_cases",6}};
        CHECK(cases_.size() == 240 && expected_rows == 3920 && generations_.size() == 12);
        cuda_reports::write(output_/"input.json",input);
        save();
    }

    json run() {
        try {
            stage_ = "cpu_reference";
            {
                Runtime cpu({model_,2048,16,4,128,8,KernelMode::automatic});
                auto clear_cpu = [&] { for (std::int32_t s = 0; s < 4; ++s) { cpu.clear_sequence(s); } };
                auto forward_cpu = [&](const Batch& batch) { return cpu.forward(batch); };
                prepare(0,clear_cpu,forward_cpu);
            }
            stage_ = "llama_f32_reference";
            {
                Reference reference(reference_,reference_flash_attention);
                prepare(1,[&] { reference.clear(); },[&](const Batch& batch) { return reference.forward(batch); });
            }
            for (const std::size_t sequences : {1,2,4}) { run_gpu(sequences); }
            CHECK(report_.at("totals").at("teacher_cases") == report_.at("expected").at("teacher_cases"));
            CHECK(report_.at("totals").at("teacher_rows") == report_.at("expected").at("teacher_rows"));
            CHECK(report_.at("totals").at("teacher_comparisons") == report_.at("expected").at("teacher_comparisons"));
            CHECK(report_.at("canonical").size() == 60 && report_.at("golden").size() == 6);
            CHECK(report_.at("generation").size() == 12 && report_.at("configurations").size() == 3);
            report_["complete"] = true;
            report_["passed"] = report_.at("totals").at("numeric_failures") == 0;
            report_["status"] = report_.at("passed").get<bool>() ? "passed" : "failed";
            save();
            return {{"schema_version",1},{"scope",report_.at("scope")},{"status",report_.at("status")},
                {"complete",true},{"passed",report_.at("passed")},{"complete_gpu_model",report_.at("passed")},
                {"full_corpus_contract",report_.at("passed")},{"performance_baseline",false},{"gpu_serving",false},
                {"reference_attention","unfused"},
                {"totals",report_.at("totals")},{"extrema",report_.at("extrema")},
                {"golden_cases",report_.at("golden").size()},{"generation_cases",report_.at("generation").size()},
                {"first_numeric_failure",report_.at("first_numeric_failure")},
                {"first_argmax_divergence",report_.at("first_argmax_divergence")}};
        } catch (const std::exception& error) {
            report_["status"] = "failed";
            report_["failure"] = {{"stage",stage_},{"message",error.what()}};
            save();
            throw;
        }
    }

private:
    static constexpr std::array<const char*,3> names_{"cpu","llama_f32","cuda_canonical"};

    void save() const { cuda_reports::write(output_/"full-validation.json",report_); }
    static void clear(CudaRuntime& runtime) {
        for (std::size_t s = 0; s < runtime.config().max_sequences; ++s) { runtime.clear_sequence(std::int32_t(s)); }
    }
    void check_rows(const Batch& batch, const std::vector<Logits>& rows) const {
        std::size_t selected = 0;
        for (const auto& input : batch) {
            if (!input.logits) { continue; }
            CHECK(selected < rows.size() && rows[selected].sequence == input.sequence);
            CHECK(rows[selected].values.size() == contract_.at("model").at("vocabulary").get<std::size_t>());
            ++selected;
        }
        CHECK(selected == rows.size());
    }
    CudaForwardResult forward(CudaRuntime& runtime, const Batch& batch, bool timing = false) const {
        auto result = runtime.forward(batch,CudaOutputMode::debug_logits,timing);
        check_rows(batch,result.logits);
        CHECK(result.samples.size() == result.logits.size() && result.host_forward_to_token_ns > 0);
        CHECK(result.device_elapsed_ms.has_value() == timing);
        if (timing) { CHECK(std::isfinite(*result.device_elapsed_ms) && *result.device_elapsed_ms >= 0); }
        std::size_t row = 0;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!batch[i].logits) { continue; }
            CHECK(result.samples[row].sequence == batch[i].sequence && result.samples[row].input_index == i);
            CHECK(result.samples[row].token == argmax(result.logits[row].values));
            ++row;
        }
        return result;
    }
    template<class Clear, class Forward>
    void canonical(std::size_t backend, Clear clear_runtime, Forward forward_runtime) {
        for (std::size_t corpus = 0; corpus < contract_.at("corpus").size(); ++corpus) {
            for (const auto& length : contract_.at("teacher_forcing").at("lengths")) {
                const auto count = length.get<std::size_t>();
                const TeacherCase c{corpus,count,128,1};
                const auto id = case_id(contract_,c);
                std::cout << "参照 " << names_[backend] << ' ' << id << std::endl;
                clear_runtime();
                const auto batches = teacher_batches(contract_,c);
                Rows collected;
                json samples = json::array();
                for (const auto& batch : batches) {
                    auto rows = forward_runtime(batch);
                    check_rows(batch,rows);
                    std::size_t selected = 0;
                    for (const auto& input : batch) {
                        if (!input.logits) { continue; }
                        auto value = score(rows[selected].values);
                        value["position"] = input.position;
                        samples.push_back(std::move(value));
                        CHECK(collected.emplace(input.position,std::move(rows[selected].values)).second);
                        ++selected;
                    }
                }
                canonical_[backend].emplace(std::pair{corpus,count},std::move(collected));
                const auto file = std::string("canonical/")+names_[backend]+"-"+id+".json";
                cuda_reports::write(output_/file,{{"schema_version",1},{"backend",names_[backend]},{"id",id},
                    {"corpus",corpus},{"length",count},{"input_sha256",batch_digest(batches)},{"samples",samples}});
                report_["canonical"].push_back({{"backend",names_[backend]},{"id",id},{"file",file}});
                save();
            }
        }
    }
    template<class Clear, class Forward>
    std::vector<std::vector<float>> generate(const Generation& c, Clear clear_runtime, Forward forward_runtime) {
        clear_runtime();
        const auto input = expanded_tokens(contract_,c.corpus,c.length);
        std::vector<Logits> rows;
        for (std::size_t first = 0; first < input.size(); first += 128) {
            Batch batch;
            for (std::size_t p = first; p < std::min(first+128,input.size()); ++p) {
                batch.push_back({input[p],std::int32_t(p),0,p+1==input.size()});
            }
            rows = forward_runtime(batch);
            check_rows(batch,rows);
        }
        std::vector<std::vector<float>> result;
        for (std::size_t i = 0; i < 32; ++i) {
            CHECK(rows.size() == 1 && rows[0].sequence == 0);
            const auto token = score(rows[0].values).at("token").get<std::int32_t>();
            result.push_back(std::move(rows[0].values));
            if (i+1 < 32) {
                rows = forward_runtime(Batch{{token,std::int32_t(c.length+i),0,true}});
            }
        }
        return result;
    }
    template<class Clear, class Forward>
    void prepare(std::size_t backend, Clear clear_runtime, Forward forward_runtime) {
        canonical(backend,clear_runtime,forward_runtime);
        for (auto& c : generations_) {
            std::cout << "生成参照 " << names_[backend] << " corpus=" << c.corpus << " length=" << c.length << std::endl;
            c.reference[backend] = generate(c,clear_runtime,forward_runtime);
            json samples = json::array();
            for (const auto& values : c.reference[backend]) { samples.push_back(score(values)); }
            const auto file = std::string("generation/")+names_[backend]+"-"+generation_id(c)+".json";
            cuda_reports::write(output_/file,{{"schema_version",1},{"backend",names_[backend]},
                {"corpus",c.corpus},{"prompt_tokens",c.length},{"output_tokens",32},{"final_kv_tokens",c.length+31},
                {"samples",samples}});
        }
        clear_runtime();
    }
    std::string generation_id(const Generation& c) const {
        return contract_.at("corpus").at(c.corpus).at("id").get<std::string>()+"-l"+std::to_string(c.length)+"-g32";
    }
    json comparison(const std::vector<float>& actual, const std::vector<float>& expected, const json& identity,
                    bool teacher) {
        auto value = compare(actual,expected,contract_.at("thresholds"));
        value.update(identity);
        auto& totals = report_["totals"];
        const char* counter = teacher ? "teacher_comparisons" : "generation_comparisons";
        totals[counter] = totals.at(counter).get<std::size_t>()+1;
        if (!value.at("passed").get<bool>()) {
            totals["numeric_failures"] = totals.at("numeric_failures").get<std::size_t>()+1;
            if (report_.at("first_numeric_failure").is_null()) { report_["first_numeric_failure"] = value; }
        }
        if (!value.at("argmax_equal").get<bool>()) {
            totals["argmax_divergences"] = totals.at("argmax_divergences").get<std::size_t>()+1;
            if (report_.at("first_argmax_divergence").is_null()) { report_["first_argmax_divergence"] = value; }
        }
        if (value.value("near_tie",false)) { totals["near_ties"] = totals.at("near_ties").get<std::size_t>()+1; }
        if (value.at("all_finite").get<bool>()) {
            auto& e = report_["extrema"];
            e["rmse_max"] = std::max(e.at("rmse_max").get<double>(),value.at("rmse").get<double>());
            e["absolute_max"] = std::max(e.at("absolute_max").get<double>(),value.at("max_absolute").get<double>());
            if (value.at("cosine").is_number()) {
                e["cosine_min"] = std::min(e.at("cosine_min").get<double>(),value.at("cosine").get<double>());
            }
        }
        return value;
    }
    void teacher(CudaRuntime& runtime, const TeacherCase& c) {
        stage_ = case_id(contract_,c);
        std::cout << "验证 " << stage_ << std::endl;
        clear(runtime);
        const auto batches = teacher_batches(contract_,c);
        const bool timing_case = c.sequences == 4 && c.corpus == 0 && c.length == 1536 && c.chunk == 33;
        std::map<std::pair<std::int32_t,std::int32_t>,std::vector<float>> untimed;
        json records = json::array();
        bool passed = true;
        std::size_t sampled_rows = 0;
        for (std::size_t b = 0; b < batches.size(); ++b) {
            const auto& batch = batches[b];
            const auto actual = forward(runtime,batch);
            std::size_t row = 0;
            for (std::size_t i = 0; i < batch.size(); ++i) {
                const auto& token = batch[i];
                if (!token.logits) { continue; }
                const auto corpus = (c.corpus+std::size_t(token.sequence))%contract_.at("corpus").size();
                for (std::size_t backend = 0; backend < names_.size(); ++backend) {
                    const auto& expected = canonical_[backend].at({corpus,c.length}).at(token.position);
                    auto value = comparison(actual.logits[row].values,expected,
                        {{"case",stage_},{"reference",names_[backend]},{"corpus",corpus},
                            {"sequence",token.sequence},{"position",token.position},{"batch",b},
                            {"input_index",i},{"batch_tokens",batch.size()}},true);
                    passed = passed && value.at("passed").get<bool>();
                    records.push_back(std::move(value));
                }
                if (timing_case) { untimed.emplace(std::pair{token.sequence,token.position},actual.logits[row].values); }
                ++sampled_rows;
                ++row;
            }
        }
        const auto after = runtime.diagnostics();
        CHECK(after.sequence_lengths == std::vector<std::size_t>(c.sequences,c.length));
        CHECK(after.live_sequences == c.sequences && after.live_kv_tokens == c.sequences*c.length);
        const auto file = "teacher/"+stage_+".json";
        cuda_reports::write(output_/file,{{"schema_version",1},{"id",stage_},{"corpus",c.corpus},{"length",c.length},
            {"chunk",c.chunk},{"sequences",c.sequences},{"batches",batches.size()},{"input_sha256",batch_digest(batches)},
            {"sampled_rows",sampled_rows},{"comparisons",records},{"after",cuda_reports::diagnostics(after)},{"passed",passed}});
        report_["teacher_forcing"].push_back({{"id",stage_},{"file",file},{"sampled_rows",sampled_rows},{"passed",passed}});
        auto& totals = report_["totals"];
        totals["teacher_cases"] = totals.at("teacher_cases").get<std::size_t>()+1;
        totals["teacher_rows"] = totals.at("teacher_rows").get<std::size_t>()+sampled_rows;
        save();
        if (timing_case) {
            clear(runtime);
            json checks = json::array();
            std::size_t bitwise_rows = 0;
            for (std::size_t b = 0; b < batches.size(); ++b) {
                const auto& batch = batches[b];
                const auto timed = forward(runtime,batch,true);
                std::size_t row = 0;
                for (const auto& token : batch) {
                    if (!token.logits) { continue; }
                    const auto& expected = untimed.at({token.sequence,token.position});
                    CHECK(timed.logits[row].values.size() == expected.size());
                    CHECK(std::memcmp(timed.logits[row].values.data(),expected.data(),expected.size()*sizeof(float)) == 0);
                    ++row;
                    ++bitwise_rows;
                }
                checks.push_back({{"batch",b},{"input_rows",batch.size()},{"logits_rows",row},
                    {"device_elapsed_ms",*timed.device_elapsed_ms},{"bitwise_equal",true}});
            }
            CHECK(runtime.diagnostics().sequence_lengths == after.sequence_lengths && bitwise_rows == sampled_rows);
            report_["timing"] = {{"case",stage_},{"bitwise_rows",bitwise_rows},{"batches",checks},{"passed",true}};
            save();
        }
    }
    void golden(CudaRuntime& runtime) {
        const auto before = runtime.diagnostics();
        for (const auto& c : contract_.at("stable_greedy")) {
            clear(runtime);
            const auto input = c.at("input_token_ids").get<std::vector<std::int32_t>>();
            const auto expected = c.at("expected_token_ids").get<std::vector<std::int32_t>>();
            CHECK(runtime.tokenize(c.at("text").get<std::string>()) == input);
            Batch batch;
            for (std::size_t p = 0; p < input.size(); ++p) { batch.push_back({input[p],std::int32_t(p),0,p+1==input.size()}); }
            auto result = runtime.forward(batch);
            std::vector<std::int32_t> tokens;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                CHECK(result.logits.empty() && result.samples.size() == 1);
                tokens.push_back(result.samples[0].token);
                if (i+1 < expected.size()) {
                    result = runtime.forward(Batch{{tokens.back(),std::int32_t(input.size()+i),0,true}});
                }
            }
            report_["golden"].push_back({{"max_sequences",runtime.config().max_sequences},{"text",c.at("text")},
                {"expected",expected},{"actual",tokens},{"passed",tokens==expected}});
            save();
            CHECK(tokens == expected);
        }
        CHECK(runtime.diagnostics().debug_d2h_bytes == before.debug_d2h_bytes);
    }
    void generation(CudaRuntime& runtime) {
        for (auto& c : generations_) {
            stage_ = generation_id(c);
            std::cout << "生成 " << stage_ << std::endl;
            const auto actual = generate(c,[&] { clear(runtime); },
                [&](const Batch& batch) { return forward(runtime,batch).logits; });
            const auto after = runtime.diagnostics();
            CHECK(after.sequence_lengths == (std::vector<std::size_t>{c.length+31,0,0,0}));
            json rows = json::array(), first = {{"cpu",nullptr},{"llama_f32",nullptr}};
            std::array<bool,2> common{true,true};
            bool passed = true;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                json row = {{"step",i},{"cuda",score(actual[i])},{"references",json::object()}};
                for (std::size_t backend = 0; backend < 2; ++backend) {
                    const auto expected = score(c.reference[backend].at(i));
                    json entry = {{"score",expected},{"input_prefix_equal",common[backend]},{"comparison",nullptr}};
                    if (common[backend]) {
                        entry["comparison"] = comparison(actual[i],c.reference[backend][i],
                            {{"case",stage_},{"reference",names_[backend]},{"step",i},{"sequence",0},
                                {"position",c.length-1+i}},false);
                        passed = passed && entry.at("comparison").at("passed").get<bool>();
                        if (row.at("cuda").at("token") != expected.at("token")) {
                            first[names_[backend]] = entry.at("comparison");
                            common[backend] = false;
                        }
                    }
                    row["references"][names_[backend]] = std::move(entry);
                }
                rows.push_back(std::move(row));
            }
            const auto file = "generation/cuda-"+stage_+".json";
            cuda_reports::write(output_/file,{{"schema_version",1},{"id",stage_},{"corpus",c.corpus},
                {"prompt_tokens",c.length},{"output_tokens",32},{"final_kv_tokens",c.length+31},
                {"rows",rows},{"first_divergence",first},{"passed",passed},{"after",cuda_reports::diagnostics(after)}});
            report_["generation"].push_back({{"id",stage_},{"file",file},{"passed",passed},{"first_divergence",first}});
            for (auto& reference : c.reference) { std::vector<std::vector<float>>{}.swap(reference); }
            save();
        }
    }
    void boundary(CudaRuntime& runtime) {
        stage_ = "slot-3-context-2048";
        clear(runtime);
        const auto input = expanded_tokens(contract_,0,2048);
        Batch short_batch;
        for (std::int32_t p = 0; p < 16; ++p) { short_batch.push_back({input[std::size_t(p)],p,3,p==15}); }
        const auto fresh = forward(runtime,short_batch);
        clear(runtime);
        CudaForwardResult last;
        for (std::size_t start = 0; start < input.size(); start += 128) {
            Batch batch;
            for (std::size_t p = start; p < std::min(start+128,input.size()); ++p) {
                batch.push_back({input[p],std::int32_t(p),3,p+1==input.size()});
            }
            last = forward(runtime,batch);
        }
        CHECK(last.logits.size() == 1);
        const auto before = runtime.diagnostics();
        CHECK(before.sequence_lengths == (std::vector<std::size_t>{0,0,0,2048}));
        test::throws<std::invalid_argument>([&] { runtime.forward(Batch{{input[0],2048,3,true}}); });
        const auto rejected = runtime.diagnostics();
        CHECK(cuda_reports::diagnostics(before) == cuda_reports::diagnostics(rejected));
        runtime.clear_sequence(3);
        CHECK(runtime.diagnostics().live_kv_tokens == 0);
        const auto reused = forward(runtime,short_batch);
        CHECK(reused.logits[0].values == fresh.logits[0].values);
        CHECK(std::memcmp(reused.logits[0].values.data(),fresh.logits[0].values.data(),
                          fresh.logits[0].values.size()*sizeof(float)) == 0);
        report_["boundary"] = {{"slot",3},{"length",2048},{"last",score(last.logits[0].values)},
            {"before_rejected_append",cuda_reports::diagnostics(before)},
            {"after_rejected_append",cuda_reports::diagnostics(rejected)},{"reuse",score(reused.logits[0].values)},
            {"fresh",score(fresh.logits[0].values)},{"reuse_bitwise_equal",true},{"passed",true}};
        save();
    }
    void run_gpu(std::size_t sequences) {
        stage_ = "cuda-s"+std::to_string(sequences);
        CudaRuntime runtime({model_,0,sequences,2048,128,0});
        CHECK(runtime.dimensions().layers == 28 && runtime.dimensions().head_dim == 128);
        CHECK(runtime.dimensions().vocabulary == contract_.at("model").at("vocabulary").get<std::size_t>());
        const auto before = runtime.diagnostics();
        const auto allocations = allocation_stats();
        if (sequences == 1) {
            canonical(2,[&] { clear(runtime); },[&](const Batch& batch) { return forward(runtime,batch).logits; });
        }
        for (const auto& c : cases_) { if (c.sequences == sequences) { teacher(runtime,c); } }
        if (sequences == 1 || sequences == 4) { golden(runtime); }
        if (sequences == 4) {
            generation(runtime);
            boundary(runtime);
            cuda_reports::write(output_/"weight-plan.json",cuda_reports::weights(runtime.weight_manifest()));
            cuda_reports::write(output_/"memory-plan.json",cuda_reports::memory(before.resident));
        }
        clear(runtime);
        const auto after = runtime.diagnostics();
        const auto final_allocations = allocation_stats();
        CHECK(after.weight_h2d_bytes == before.weight_h2d_bytes && after.rope_h2d_bytes == before.rope_h2d_bytes);
        CHECK(after.intermediate_h2d_bytes == 0 && after.intermediate_d2h_bytes == 0 && after.post_launch_failures == 0);
        CHECK(after.live_kv_tokens == 0 && after.state == CudaRuntimeState::ready);
        CHECK(allocations.allocation_calls == final_allocations.allocation_calls);
        CHECK(allocations.release_calls == final_allocations.release_calls);
        CHECK(after.owned_device_bytes == after.resident.total_owned_bytes && after.owned_device_allocations == 4);
        report_["configurations"].push_back({{"max_sequences",sequences},{"max_model_len",2048},{"batch_tokens",128},
            {"before",cuda_reports::diagnostics(before)},{"after",cuda_reports::diagnostics(after)},
            {"steady_project_allocation_calls",0},{"steady_project_release_calls",0},
            {"device",cuda_reports::device(runtime.device_info())},{"arithmetic",cuda_reports::arithmetic(runtime)},{"passed",true}});
        save();
    }

    std::string model_, reference_, stage_;
    const json& contract_;
    std::filesystem::path output_;
    std::vector<TeacherCase> cases_;
    std::array<Canonical,3> canonical_;
    std::vector<Generation> generations_;
    json report_;
};
}

json run_full_validation(const std::string& model, const std::string& reference, const json& contract,
                         const std::filesystem::path& output) {
    return FullValidation(model,reference,contract,output).run();
}
}
