#include "cuda_validation_support.h"
#include "minillm/cuda/device_buffer.h"

#include <cstring>
#include <map>

namespace cuda_validation {
namespace {
using Rows = std::vector<std::vector<float>>;

class PrecisionValidation {
public:
    PrecisionValidation(const std::string& model, const json& contract, const json& experiment,
                        const std::filesystem::path& output)
        : model_(model), contract_(contract), experiment_(experiment), output_(output) {
        auto recipe = contract_;
        recipe["teacher_forcing"] = experiment_.at("numerical");
        cases_ = teacher_cases(recipe);
        CHECK(cases_.size() == experiment_.at("numerical").at("configuration_count"));
        report_ = {{"schema_version",1},{"scope","CUDA-PREC-001 numerical"},{"status","incomplete"},
            {"complete",false},{"passed",false},{"performance_baseline",false},{"gpu_serving",false},
            {"teacher_forcing",json::array()},{"golden",json::array()},{"continuations",json::array()},
            {"state_checks",json::array()},{"configurations",json::array()},
            {"first_numeric_failure",nullptr},{"first_argmax_divergence",nullptr},
            {"totals",{{"teacher_cases",0},{"teacher_rows",0},{"comparisons",0},{"numeric_failures",0},
                {"golden_failures",0},{"argmax_divergences",0},{"near_ties",0}}},
            {"extrema",{{"rmse_max",0.0},{"absolute_max",0.0},{"cosine_min",1.0}}}};
        json inputs = {{"chunk_unit","total_batch_input_rows"},{"sequence_corpus","(case_corpus + sequence) % 4"},
            {"gpu_lifetime","同一 S 的 F32 全部完成并析构后才构造 F16"},{"teacher_cases",json::array()}};
        for (const auto& c : cases_) {
            const auto batches = precision_batches(contract_,experiment_.at("numerical"),c);
            std::size_t rows = 0;
            std::set<std::int32_t> positions;
            for (const auto& batch : batches) {
                for (const auto& token : batch) {
                    if (token.logits) { ++rows; positions.insert(token.position); }
                }
            }
            expected_rows_ += rows;
            inputs["teacher_cases"].push_back({{"id",case_id(contract_,c)},{"corpus",c.corpus},
                {"length",c.length},{"chunk",c.chunk},{"sequences",c.sequences},{"positions",positions},
                {"batches",batches.size()},{"sampled_rows",rows},{"input_sha256",batch_digest(batches)}});
        }
        cuda_reports::write(output_/"input.json",inputs);
        save();
    }

    json run() {
        try {
            for (const std::size_t sequences : {1,4}) {
                run_mode(sequences,PrecisionMode::f32_pedantic);
                run_mode(sequences,PrecisionMode::f16_matrix_f32acc);
                CHECK(baseline_.empty());
            }
            CHECK(report_["totals"]["teacher_cases"] == cases_.size());
            CHECK(report_["totals"]["teacher_rows"] == expected_rows_);
            CHECK(report_["golden"].size() == 12 && report_["continuations"].size() == 4);
            CHECK(report_["configurations"].size() == 4 && report_["state_checks"].size() == 4);
            report_["complete"] = true;
            report_["passed"] = report_["totals"]["numeric_failures"] == 0 && report_["totals"]["golden_failures"] == 0;
            report_["status"] = report_["passed"].get<bool>() ? "passed" : "failed";
            save();
            auto summary = report_;
            summary.erase("teacher_forcing");
            return summary;
        } catch (const std::exception& error) {
            report_["status"] = "failed"; report_["failure"] = {{"stage",stage_},{"message",error.what()}};
            save();
            throw;
        }
    }

private:
    void save() const { cuda_reports::write(output_/"precision-validation.json",report_); }
    void count(const char* key, std::size_t n = 1) {
        report_["totals"][key] = report_["totals"].at(key).get<std::size_t>()+n;
    }
    static bool half(const CudaRuntime& runtime) {
        return runtime.config().precision_mode == PrecisionMode::f16_matrix_f32acc;
    }
    static void clear(CudaRuntime& runtime) {
        for (std::size_t s = 0; s < runtime.config().max_sequences; ++s) { runtime.clear_sequence(std::int32_t(s)); }
    }
    CudaForwardResult forward(CudaRuntime& runtime, const Batch& batch, bool timing = false,
                              CudaOutputMode mode = CudaOutputMode::debug_logits) const {
        const auto before = runtime.diagnostics();
        auto result = runtime.forward(batch,mode,timing);
        CHECK(result.host_forward_to_token_ns > 0 && result.device_elapsed_ms.has_value() == timing);
        if (timing) { CHECK(std::isfinite(*result.device_elapsed_ms) && *result.device_elapsed_ms >= 0); }
        std::size_t row = 0;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (!batch[i].logits) { continue; }
            CHECK(row < result.samples.size());
            CHECK(result.samples[row].sequence == batch[i].sequence && result.samples[row].input_index == i);
            if (mode == CudaOutputMode::debug_logits) {
                CHECK(row < result.logits.size() && result.logits[row].sequence == batch[i].sequence);
                CHECK(result.logits[row].values.size() == contract_.at("model").at("vocabulary"));
                CHECK(result.samples[row].token == score(result.logits[row].values).at("token"));
            }
            ++row;
        }
        CHECK(row == result.samples.size());
        CHECK(result.logits.size() == (mode == CudaOutputMode::debug_logits ? row : 0));
        const auto casts = half(runtime) ? 4*runtime.dimensions().layers+(row ? 1 : 0) : 0;
        CHECK(runtime.diagnostics().matrix_cast_calls-before.matrix_cast_calls == casts);
        return result;
    }
    json comparison(const std::vector<float>& actual, const std::vector<float>& expected, const json& identity) {
        auto value = compare(actual,expected,experiment_.at("numerical"));
        value.update(identity);
        count("comparisons");
        if (!value.at("passed").get<bool>()) {
            count("numeric_failures");
            if (report_["first_numeric_failure"].is_null()) {
                report_["first_numeric_failure"] = value;
                cuda_reports::write(output_/"first-numeric-failure-logits.json",
                    {{"identity",value},{"f32_logits",expected},{"f16_logits",actual}});
            }
        }
        if (!value.at("argmax_equal").get<bool>()) {
            count("argmax_divergences");
            if (report_["first_argmax_divergence"].is_null()) {
                report_["first_argmax_divergence"] = value;
                cuda_reports::write(output_/"first-divergence-logits.json",
                    {{"identity",value},{"f32_logits",expected},{"f16_logits",actual}});
            }
        }
        if (value.value("near_tie",false)) { count("near_ties"); }
        if (value.at("all_finite").get<bool>()) {
            auto& e = report_["extrema"];
            e["rmse_max"] = std::max(e["rmse_max"].get<double>(),value["rmse"].get<double>());
            e["absolute_max"] = std::max(e["absolute_max"].get<double>(),value["max_absolute"].get<double>());
            e["cosine_min"] = std::min(e["cosine_min"].get<double>(),value["cosine"].get<double>());
        }
        return value;
    }
    Rows execute(CudaRuntime& runtime, const std::vector<Batch>& batches, bool timing = false) const {
        Rows rows;
        for (const auto& batch : batches) {
            auto result = forward(runtime,batch,timing);
            for (auto& row : result.logits) { rows.push_back(std::move(row.values)); }
        }
        return rows;
    }
    void teacher(CudaRuntime& runtime, const TeacherCase& c) {
        stage_ = case_id(contract_,c);
        std::cout << precision_mode_name(runtime.config().precision_mode) << ' ' << stage_ << std::endl;
        clear(runtime);
        const auto batches = precision_batches(contract_,experiment_.at("numerical"),c);
        auto rows = execute(runtime,batches);
        CHECK(runtime.diagnostics().sequence_lengths == std::vector<std::size_t>(c.sequences,c.length));
        CHECK(runtime.diagnostics().live_kv_tokens == c.sequences*c.length);
        if (c.corpus == 0 && c.length == 128 && c.chunk == 16 && c.sequences == 4) {
            clear(runtime);
            const auto timed = execute(runtime,batches,true);
            CHECK(timed.size() == rows.size());
            for (std::size_t r = 0; r < rows.size(); ++r) {
                CHECK(timed[r].size() == rows[r].size());
                CHECK(std::memcmp(timed[r].data(),rows[r].data(),rows[r].size()*sizeof(float)) == 0);
            }
            report_["state_checks"].push_back({{"check","timing_bitwise"},{"case",stage_},
                {"mode",precision_mode_name(runtime.config().precision_mode)},{"rows",rows.size()},{"passed",true}});
        }
        json records = json::array();
        std::size_t row = 0;
        for (std::size_t b = 0; b < batches.size(); ++b) {
            for (std::size_t i = 0; i < batches[b].size(); ++i) {
                const auto& token = batches[b][i];
                if (!token.logits) { continue; }
                const json identity = {{"case",stage_},{"sequence",token.sequence},{"position",token.position},
                    {"batch",b},{"input_index",i},{"batch_tokens",batches[b].size()}};
                if (half(runtime)) {
                    records.push_back(comparison(rows.at(row),baseline_.at(stage_).at(row),identity));
                } else {
                    auto value = score(rows.at(row)); value.update(identity); records.push_back(std::move(value));
                }
                ++row;
            }
        }
        CHECK(row == rows.size());
        const auto file = std::string(precision_mode_name(runtime.config().precision_mode))+"-"+stage_+".json";
        cuda_reports::write(output_/file,{{"input_sha256",batch_digest(batches)},{"rows",records}});
        if (half(runtime)) {
            CHECK(rows.size() == baseline_.at(stage_).size());
            count("teacher_cases"); count("teacher_rows",row);
            report_["teacher_forcing"].push_back({{"id",stage_},{"file",file},{"sampled_rows",row}});
            baseline_.erase(stage_);
        } else { CHECK(baseline_.emplace(stage_,std::move(rows)).second); }
        save();
    }
    void golden(CudaRuntime& runtime) {
        const auto before = runtime.diagnostics();
        for (const auto& c : contract_.at("stable_greedy")) {
            clear(runtime);
            const auto input = c.at("input_token_ids").get<std::vector<std::int32_t>>();
            CHECK(runtime.tokenize(c.at("text").get<std::string>()) == input);
            Batch batch;
            for (std::size_t p = 0; p < input.size(); ++p) { batch.push_back({input[p],std::int32_t(p),0,p+1==input.size()}); }
            auto result = forward(runtime,batch,false,CudaOutputMode::greedy);
            const auto expected = c.at("expected_token_ids").get<std::vector<std::int32_t>>();
            std::vector<std::int32_t> tokens;
            for (std::size_t i = 0; i < expected.size(); ++i) {
                tokens.push_back(result.samples.at(0).token);
                if (i+1 < expected.size()) {
                    result = forward(runtime,Batch{{tokens.back(),std::int32_t(input.size()+i),0,true}},
                                     false,CudaOutputMode::greedy);
                }
            }
            report_["golden"].push_back({{"mode",precision_mode_name(runtime.config().precision_mode)},
                {"max_sequences",runtime.config().max_sequences},{"text",c.at("text")},
                {"expected",expected},{"actual",tokens},{"passed",tokens==expected}});
            if (tokens != expected) { count("golden_failures"); }
            save();
        }
        CHECK(runtime.diagnostics().debug_d2h_bytes == before.debug_d2h_bytes);
    }
    Rows generate(CudaRuntime& runtime, std::size_t corpus, std::size_t length, const Rows* forced = nullptr) const {
        clear(runtime);
        auto prompt = teacher_batches(contract_,{corpus,length,128,1});
        for (auto& batch : prompt) { for (auto& token : batch) { token.logits = std::size_t(token.position)+1 == length; } }
        auto rows = execute(runtime,prompt);
        CHECK(rows.size() == 1);
        const auto count = experiment_.at("numerical").at("continuation_tokens").get<std::size_t>();
        for (std::size_t i = 1; i < count; ++i) {
            const auto token = argmax(forced ? forced->at(i-1) : rows.back());
            auto result = forward(runtime,Batch{{token,std::int32_t(length+i-1),0,true}});
            rows.push_back(std::move(result.logits.at(0).values));
        }
        CHECK(runtime.diagnostics().sequence_lengths[0] == length+count-1);
        return rows;
    }
    void continuations(CudaRuntime& runtime) {
        for (const auto& c : experiment_.at("numerical").at("continuations")) {
            std::size_t corpus = 0;
            while (contract_.at("corpus").at(corpus).at("id") != c.at("corpus")) { ++corpus; }
            const auto length = c.at("prompt_tokens").get<std::size_t>();
            stage_ = "generation-"+c.at("corpus").get<std::string>();
            std::cout << precision_mode_name(runtime.config().precision_mode) << ' ' << stage_ << std::endl;
            auto rows = generate(runtime,corpus,length);
            if (!half(runtime)) { baseline_.emplace(stage_,std::move(rows)); continue; }
            const auto& expected = baseline_.at(stage_);
            CHECK(rows.size() == expected.size());
            json natural = json::array(), teacher = json::array(), first = nullptr;
            bool common = true;
            for (std::size_t i = 0; i < rows.size(); ++i) {
                json row = {{"step",i},{"f32",score(expected[i])},{"f16",score(rows[i])},
                    {"input_prefix_equal",common},{"comparison",nullptr}};
                if (common) {
                    row["comparison"] = comparison(rows[i],expected[i],{{"case",stage_},{"step",i},{"trajectory","common_prefix"}});
                    if (argmax(rows[i]) != argmax(expected[i])) {
                        first = row; common = false;
                        cuda_reports::write(output_/(stage_+"-divergence-logits.json"),
                            {{"identity",row},{"f32_logits",expected[i]},{"f16_logits",rows[i]}});
                    }
                }
                natural.push_back(std::move(row));
            }
            // 分叉后只用固定 F32 token 轨迹比较数值，不把两条自由续写当作同输入。
            const auto forced = common ? rows : generate(runtime,corpus,length,&expected);
            for (std::size_t i = 0; i < forced.size(); ++i) {
                teacher.push_back(comparison(forced[i],expected[i],{{"case",stage_},{"step",i},{"trajectory","f32_teacher_forced"}}));
            }
            const auto file = stage_+".json";
            cuda_reports::write(output_/file,{{"prompt_tokens",length},{"natural",natural},
                {"teacher_forced",teacher},{"first_divergence",first}});
            report_["continuations"].push_back({{"id",stage_},{"file",file},{"first_divergence",first}});
            baseline_.erase(stage_);
            save();
        }
    }
    void boundary_and_mixed(CudaRuntime& runtime) {
        stage_ = "boundary-repeated-2048";
        clear(runtime);
        const Batch short_batch{{14990,0,3,false},{14990,1,3,true}};
        const auto fresh = execute(runtime,{short_batch});
        clear(runtime);
        auto batches = teacher_batches(contract_,{2,2048,128,1});
        for (auto& batch : batches) {
            for (auto& token : batch) { token.sequence = 3; token.logits = token.position == 2047; }
        }
        auto last = execute(runtime,batches);
        const auto before = runtime.diagnostics();
        CHECK(before.sequence_lengths == (std::vector<std::size_t>{0,0,0,2048}));
        test::throws<std::invalid_argument>([&] { runtime.forward(Batch{{14990,2048,3,true}}); });
        CHECK(cuda_reports::diagnostics(before) == cuda_reports::diagnostics(runtime.diagnostics()));
        CHECK(before.matrix_cast_calls == runtime.diagnostics().matrix_cast_calls);
        runtime.clear_sequence(3);
        CHECK(runtime.diagnostics().live_kv_tokens == 0);
        CHECK(execute(runtime,{short_batch}) == fresh);
        const Batch mixed{{785,0,0,false},{14990,2,3,true},{15592,1,0,true}};
        const auto mixed_rows = execute(runtime,{mixed});
        last.insert(last.end(),mixed_rows.begin(),mixed_rows.end());
        json comparisons = json::array();
        if (half(runtime)) {
            CHECK(last.size() == baseline_.at(stage_).size());
            for (std::size_t i = 0; i < last.size(); ++i) {
                comparisons.push_back(comparison(last[i],baseline_.at(stage_)[i],{{"case",stage_},{"row",i}}));
            }
            baseline_.erase(stage_);
        } else { baseline_.emplace(stage_,std::move(last)); }
        report_["state_checks"].push_back({{"check",stage_},{"mode",precision_mode_name(runtime.config().precision_mode)},
            {"boundary_slot",3},{"rejected_append_unchanged",true},{"reuse_values_equal",true},{"mixed_rows",mixed_rows.size()},
            {"comparisons",comparisons},{"passed",true}});
        save();
    }
    void run_mode(std::size_t sequences, PrecisionMode mode) {
        stage_ = std::string(precision_mode_name(mode))+"-s"+std::to_string(sequences);
        const auto lifetime = allocation_stats();
        {
            CudaRuntimeConfig config{model_,0,sequences,2048,128,0};
            config.precision_mode = mode;
            CudaRuntime runtime(config);
            const auto before = runtime.diagnostics();
            const auto allocations = allocation_stats();
            for (const auto& c : cases_) { if (c.sequences == sequences) { teacher(runtime,c); } }
            golden(runtime);
            if (sequences == 4) {
                boundary_and_mixed(runtime);
                continuations(runtime);
                cuda_reports::write(output_/(std::string(precision_mode_name(mode))+"-weights.json"),
                                    cuda_reports::weights(runtime.weight_manifest(),true));
            }
            clear(runtime);
            const auto after = runtime.diagnostics();
            CHECK(after.state == CudaRuntimeState::ready && after.live_kv_tokens == 0);
            CHECK(after.weight_h2d_bytes == before.weight_h2d_bytes && after.rope_h2d_bytes == before.rope_h2d_bytes);
            CHECK(after.intermediate_h2d_bytes == 0 && after.intermediate_d2h_bytes == 0 && after.post_launch_failures == 0);
            CHECK(allocation_stats().allocation_calls == allocations.allocation_calls);
            CHECK(allocation_stats().release_calls == allocations.release_calls);
            CHECK(after.owned_device_bytes == after.resident.total_owned_bytes && after.owned_device_allocations == 4);
            report_["configurations"].push_back({{"mode",precision_mode_name(mode)},{"max_sequences",sequences},
                {"before",cuda_reports::diagnostics(before)},{"after",cuda_reports::diagnostics(after)},
                {"matrix_cast_calls",after.matrix_cast_calls},{"steady_project_allocation_calls",0},
                {"steady_project_release_calls",0},{"device",cuda_reports::device(runtime.device_info())},
                {"arithmetic",cuda_reports::arithmetic(runtime,true)},{"passed",true}});
            save();
        }
        const auto released = allocation_stats();
        CHECK(released.allocations-lifetime.allocations == released.releases-lifetime.releases);
    }

    std::string model_, stage_;
    const json& contract_;
    const json& experiment_;
    std::filesystem::path output_;
    std::vector<TeacherCase> cases_;
    std::size_t expected_rows_ = 0;
    std::map<std::string,Rows> baseline_;
    json report_;
};
}

json run_precision_validation(const std::string& model, const json& contract, const json& experiment,
                              const std::filesystem::path& output) {
    return PrecisionValidation(model,contract,experiment,output).run();
}
}
