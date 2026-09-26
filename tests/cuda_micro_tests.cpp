#include "../apps/cuda_micro_protocol.h"
#include "test_support.h"

#include <fstream>
#include <map>
#include <set>

using namespace cuda_micro;

namespace {
json input;
minillm::ModelDimensions dimensions() {
    minillm::ModelDimensions d{};
    d.embedding = 1024; d.layers = 28; d.heads = 16; d.kv_heads = 8;
    d.head_dim = 128; d.feed_forward = 3072; d.vocabulary = 151936;
    return d;
}
}

TEST(micro_plan_has_all_shapes_and_stable_seeds) {
    const auto cases = make_cases(input,dimensions());
    CHECK(cases.size() == 375);
    std::map<std::string,std::size_t> counts;
    std::set<std::string> names;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        CHECK(c.seed == i+1 && names.insert(c.name).second && c.m > 0 && c.m <= 128);
        ++counts[c.operation];
        const auto description = describe(c,dimensions());
        CHECK(description.at("output_elements") == c.m*c.n && description.at("input_seed") == i+1);
    }
    CHECK(counts["matrix"] == 72 && counts["rms_norm"] == 27 && counts["rope"] == 108);
    CHECK(counts["softmax"] == 84 && counts["attention"] == 84);
}

TEST(micro_matrix_dimensions_follow_metadata) {
    auto d = dimensions();
    for (bool smaller : {false,true}) {
        if (smaller) {
            d.embedding = 512; d.heads = 8; d.kv_heads = 4;
            d.head_dim = 64; d.feed_forward = 1536; d.vocabulary = 4096;
        }
        std::map<std::string,std::pair<std::size_t,std::size_t>> shapes{
            {"Q",{d.heads*d.head_dim,d.embedding}}, {"K",{d.kv_heads*d.head_dim,d.embedding}},
            {"V",{d.kv_heads*d.head_dim,d.embedding}}, {"attention_output",{d.embedding,d.heads*d.head_dim}},
            {"gate",{d.feed_forward,d.embedding}}, {"up",{d.feed_forward,d.embedding}},
            {"down",{d.embedding,d.feed_forward}}, {"LM_head",{d.vocabulary,d.embedding}}};
        for (const auto& c : make_cases(input,d)) {
            if (c.operation == "matrix") {
                CHECK(std::make_pair(c.n,c.k) == shapes.at(c.role));
                CHECK(describe(c,d).at("logical_flops_per_call") == 2*c.m*c.n*c.k);
            }
        }
    }
}

TEST(micro_attention_metadata_is_causal_and_independent) {
    std::size_t mixed = 0;
    for (const auto& c : make_cases(input,dimensions())) {
        if (c.operation != "attention" && c.operation != "softmax") { continue; }
        CHECK(c.slots.size() == c.m && c.positions.size() == c.m);
        CHECK(c.max_context == std::size_t(*std::max_element(c.positions.begin(),c.positions.end()))+1);
        CHECK(c.max_context <= max_length);
        for (std::size_t row = 0; row < c.m; ++row) {
            CHECK(c.slots[row] >= 0 && c.slots[row] < 4 && c.positions[row] >= 0);
            CHECK(std::size_t(c.positions[row]) < c.max_context);
            if (c.role == "decode") { CHECK(c.slots[row] == std::int32_t(row)); }
            if (c.role == "prefill") {
                CHECK(c.slots[row] == 0 && std::size_t(c.positions[row]) == c.max_context-c.m+row);
            }
        }
        if (c.role == "mixed") {
            ++mixed;
            CHECK(c.m == 18 && c.slots[16] == 1 && c.slots[17] == 2);
            CHECK(c.positions[15] == 15 && c.positions[16] == std::int32_t(c.max_context-1));
            CHECK(c.positions[17] == c.positions[16]);
        }
    }
    CHECK(mixed == 4);
}

TEST(micro_rope_norm_and_sampling_boundaries) {
    for (const auto& c : make_cases(input,dimensions())) {
        if (c.operation == "rope") {
            CHECK(c.group_width == 128 && c.positions.size() == c.m);
            CHECK(std::all_of(c.positions.begin(),c.positions.end(),[&](auto p) { return p == c.positions[0]; }));
        }
        if (c.operation == "rms_norm") {
            CHECK(c.n % c.group_width == 0 && c.group_width == (c.role == "hidden" ? 1024 : 128));
        }
    }
    CHECK(sample_rows(1) == std::vector<std::size_t>{0});
    CHECK(sample_rows(2) == (std::vector<std::size_t>{0,1}));
    CHECK(sample_rows(128) == (std::vector<std::size_t>{0,64,127}));
    CHECK(sample_columns(1) == std::vector<std::size_t>{0});
    CHECK(sample_columns(151936).size() == 8 && sample_columns(151936).back() == 151935);
    test::throws<std::invalid_argument>([] { sample_rows(0); });
    test::throws<std::invalid_argument>([] { sample_columns(0); });
    auto invalid = input;
    invalid["matrix"]["roles"][0] = "unknown";
    test::throws<std::invalid_argument>([&] { make_cases(invalid,dimensions()); });
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "需要冻结的 micro 输入 JSON\n"; return 1; }
    std::ifstream file(argv[1]);
    input = json::parse(file);
    return test::run();
}
