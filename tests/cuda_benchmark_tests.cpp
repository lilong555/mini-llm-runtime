#include "../apps/cuda_benchmark.h"
#include "test_support.h"

#include <fstream>

using namespace cuda_benchmark;

namespace {
json input;

struct FakeBackend {
    Lengths lengths{};
    std::size_t clears = 0, forwards = 0;
    bool reverse_output = false;
    std::size_t vocabulary() const { return 151936; }
    void clear_sequence(std::int32_t sequence) { lengths.at(static_cast<std::size_t>(sequence)) = 0; ++clears; }
    Forward forward(const Batch& batch) {
        lengths = after_batch(batch, lengths);
        ++forwards;
        Forward result;
        result.host_forward_to_token_ns = 100;
        for (std::size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].logits) { result.samples.push_back({batch[i].sequence, i, (batch[i].token + 1) % 151936}); }
        }
        if (reverse_output) { std::reverse(result.samples.begin(), result.samples.end()); }
        return result;
    }
    json snapshot(const Lengths& expected) const {
        CHECK(lengths == expected);
        return {{"lengths", lengths}};
    }
};
}

TEST(benchmark_all_cases_have_independent_contiguous_inputs) {
    const auto works = make_workloads(input);
    CHECK(works.size() == 12);
    for (const auto& work : works) {
        Lengths lengths{};
        for (const auto& batch : work.setup) {
            CHECK(std::none_of(batch.begin(), batch.end(), [](const auto& row) { return row.logits; }));
            lengths = after_batch(batch, lengths);
        }
        for (const auto& batch : work.measured) { lengths = after_batch(batch, lengths); }
    }
    CHECK(works[2].measured.size() == 2 && works[3].measured.size() == 12);
    CHECK(works[6].setup.size() == 12 && works[6].measured.front().front().position == 1536);
    CHECK(works[7].setup.size() == 4 && works[8].setup.size() == 8);
    CHECK(works[9].setup.size() == 4 && works[9].measured.front().size() == 18);
    const auto& mixed = works[9].measured.front();
    CHECK(mixed[15].logits && mixed[15].sequence == 0 && mixed[16].sequence == 1 && mixed[17].sequence == 2);
}

TEST(benchmark_warmup_and_measured_always_clear_and_rebuild) {
    for (const auto& work : make_workloads(input)) {
        FakeBackend backend;
        json report;
        run_workload(backend, work, report);
        CHECK(backend.clears == 20 && report.at("iterations").size() == 5 && report.at("status") == "passed");
        const auto measured = work.measured.size() + (work.generation_tokens ? work.generation_tokens - 1 : 0);
        CHECK(backend.forwards == 5 * (work.setup.size() + measured));
        for (std::size_t i = 0; i < 5; ++i) {
            const auto& row = report.at("iterations").at(i);
            CHECK(row.at("phase") == (i < 2 ? "warmup" : "measured"));
            CHECK(row.at("setup_forward_ns") == work.setup.size() * 100);
            CHECK(row.at("host_forward_to_token_ns") == measured * 100);
            CHECK(row.at("after_clear").at("lengths") == Lengths{});
        }
    }
}

TEST(benchmark_generation_feeds_actual_previous_output_for_31_decodes) {
    const auto works = make_workloads(input);
    for (const auto index : {10, 11}) {
        FakeBackend backend;
        json report;
        run_workload(backend, works[index], report);
        for (const auto& row : report.at("iterations")) {
            CHECK(row.at("forwards").size() == 32 && row.at("token_ids").size() == 32);
            CHECK(row.at("prefill_forward_ns") == 100 && row.at("decode_forward_ns") == 3100);
            CHECK(row.at("after_measured").at("lengths").at(0) == works[index].prompt_tokens + 31);
            const auto& tokens = row.at("token_ids");
            for (std::size_t i = 1; i < tokens.size(); ++i) {
                CHECK(tokens.at(i).get<int>() == (tokens.at(i - 1).get<int>() + 1) % 151936);
            }
        }
    }
}

TEST(benchmark_digest_covers_positions_sequences_and_logits) {
    Batch batch{{1, 0, 0, false}, {2, 1, 0, true}};
    const auto hash = input_digest(batch);
    CHECK(hash.size() == 64 && hash == input_digest(batch));
    for (int field = 0; field < 4; ++field) {
        auto changed = batch;
        if (field == 0) { ++changed[0].token; }
        if (field == 1) { ++changed[0].position; }
        if (field == 2) { ++changed[0].sequence; }
        if (field == 3) { changed[0].logits = true; }
        CHECK(input_digest(changed) != hash);
    }
    CHECK(input_digest({batch.data(), 1}) != hash);
}

TEST(benchmark_refuses_wrong_state_and_output_order) {
    for (const auto& row : Batch{{1, 1, 0, true}, {1, 0, 4, true}, {1, 0, -1, true}, {-1, 0, 0, true}}) {
        test::throws<std::invalid_argument>([&] { after_batch({&row, 1}, {}); });
    }
    FakeBackend backend;
    backend.reverse_output = true;
    Lengths lengths{};
    test::throws<std::runtime_error>([&] {
        run_call(backend, {{1, 0, 0, true}, {2, 0, 1, true}}, "decode", lengths);
    });
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "需要冻结的性能输入 JSON\n"; return 1; }
    std::ifstream file(argv[1]);
    input = json::parse(file);
    return test::run();
}
