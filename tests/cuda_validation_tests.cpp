#include "cuda_validation_support.h"

#include <array>

using namespace cuda_validation;

namespace {
json fixture() {
    return {{"model",{{"vocabulary",128}}},
        {"corpus",{{{"id","a"},{"seed_token_ids",{1,2,3}}},{{"id","b"},{"seed_token_ids",{4,5}}},
                   {{"id","c"},{"seed_token_ids",{6}}},{{"id","d"},{"seed_token_ids",{7,8,9,10}}}}},
        {"teacher_forcing",{{"lengths",{16,33,128,256,1536}},{"positions",{0,1,15,16,17,32,127,128,255,1535}},
            {"chunk_tokens",{1,16,33,128}},{"sequence_counts",{1,2,4}}}}};
}
json thresholds() { return {{"rmse_exclusive",0.05},{"max_absolute_exclusive",0.5},{"cosine_min_inclusive",0.9999}}; }
}

TEST(validation_full_matrix_has_exact_coverage) {
    const auto contract = fixture();
    const auto cases = teacher_cases(contract);
    CHECK(cases.size() == 240);
    std::set<std::string> ids;
    std::size_t sampled_rows = 0;
    for (const auto& c : cases) {
        CHECK(ids.insert(case_id(contract,c)).second);
        const auto batches = teacher_batches(contract,c);
        std::vector<std::size_t> lengths(c.sequences);
        std::size_t tokens = 0;
        for (const auto& batch : batches) {
            CHECK(!batch.empty() && batch.size() <= c.chunk && batch.size() <= 128);
            for (const auto& token : batch) {
                CHECK(token.sequence == std::int32_t(tokens%c.sequences));
                CHECK(token.position == std::int32_t(tokens/c.sequences));
                CHECK(std::size_t(token.position) == lengths[std::size_t(token.sequence)]++);
                ++tokens;
                sampled_rows += token.logits;
            }
        }
        CHECK(tokens == c.length*c.sequences && lengths == std::vector<std::size_t>(c.sequences,c.length));
    }
    CHECK(sampled_rows == 3920 && sampled_rows*3 == 11760);
}

TEST(validation_interleaved_chunk_uses_independent_corpora) {
    const auto contract = fixture();
    const auto batches = teacher_batches(contract,{2,33,33,4});
    CHECK(batches.size() == 4 && batches[1][0].position == 8 && batches[1][0].sequence == 1);
    const std::array<std::vector<std::int32_t>,4> seeds{{{6},{7,8,9,10},{1,2,3},{4,5}}};
    for (const auto& batch : batches) {
        for (const auto& token : batch) {
            const auto& seed = seeds[std::size_t(token.sequence)];
            CHECK(token.token == seed[std::size_t(token.position)%seed.size()]);
        }
    }
}

TEST(validation_digest_includes_batch_boundaries_and_flags) {
    const auto contract = fixture();
    auto batches = teacher_batches(contract,{0,33,33,1});
    const auto digest = batch_digest(batches);
    CHECK(digest.size() == 64 && digest == batch_digest(teacher_batches(contract,{0,33,33,1})));
    CHECK(digest != batch_digest(teacher_batches(contract,{0,33,16,1})));
    batches[0][0].logits = !batches[0][0].logits;
    CHECK(digest != batch_digest(batches));
}

TEST(validation_rejects_invalid_recipe_before_execution) {
    auto contract = fixture();
    for (const auto& c : std::vector<TeacherCase>{{0,16,0,1},{0,16,129,1},{0,16,1,0},{0,16,1,5},{4,16,1,1},
                                               {0,0,1,1},{0,2049,1,1}}) {
        test::throws<std::invalid_argument>([&] { teacher_batches(contract,c); });
    }
    contract["corpus"][0]["seed_token_ids"] = json::array();
    test::throws<std::invalid_argument>([&] { teacher_batches(contract,{0,16,1,1}); });
    contract["corpus"][0]["seed_token_ids"] = {-1};
    test::throws<std::invalid_argument>([&] { teacher_batches(contract,{0,16,1,1}); });
    contract["corpus"][0]["seed_token_ids"] = {128};
    test::throws<std::invalid_argument>([&] { teacher_batches(contract,{0,16,1,1}); });
}

TEST(validation_near_tie_preserves_actual_divergence) {
    std::vector<float> expected(100,0.5F), actual(100,0.5F);
    expected[0] = actual[1] = 1.0F;
    expected[1] = actual[0] = 0.999F;
    const auto result = compare(actual,expected,thresholds());
    CHECK(result.at("passed") && result.at("near_tie") && !result.at("argmax_equal").get<bool>());
    CHECK(result.at("actual_argmax") == 1 && result.at("reference_argmax") == 0);
    CHECK(result.at("near_tie") == (result.at("reference_margin").get<double>() <= 2*result.at("max_absolute").get<double>()));
    actual[1] = 4.0F;
    CHECK(!compare(actual,expected,thresholds()).at("passed").get<bool>());
}

TEST(validation_score_ties_and_nonfinite_are_explicit) {
    const auto tied = score({1.0F,3.0F,3.0F});
    CHECK(tied.at("token") == 1 && tied.at("margin") == 0);
    CHECK(tied.at("sha256").get<std::string>().size() == 64);
    for (const auto value : {std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        const auto result = compare({value,1.0F},{1.0F,2.0F},thresholds());
        CHECK(!result.at("passed").get<bool>() && !result.at("all_finite").get<bool>());
        test::throws<std::runtime_error>([&] { score({value,1.0F}); });
    }
    test::throws<std::runtime_error>([] { compare({1.0F},{1.0F,2.0F},thresholds()); });
}

int main() { return test::run(); }
