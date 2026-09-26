#include "test_support.h"
#include "gated_runner.h"

#include "llmserve/block_pool.h"
#include "llmserve/engine.h"
#include "llmserve/prefix_index.h"
#include "llmserve/scheduler.h"
#include "llmserve/utf8.h"
#include "minillm/kernels.h"
#include "minillm/paged_kv.h"
#include "minillm/parallel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <thread>

using namespace llmserve;
using namespace std::chrono_literals;

TEST(block_pool_allocation_is_atomic) {
    BlockPool pool(4);
    auto first = pool.allocate(3);
    CHECK(first && first->size() == 3);
    CHECK(!pool.allocate(2));
    CHECK(pool.free_count() == 1);
    pool.release(*first);
    CHECK(pool.free_count() == 4);
}

TEST(block_pool_reference_counts_and_invalid_tables) {
    BlockPool pool(4);
    auto blocks = *pool.allocate(2);
    pool.retain(blocks);
    pool.release(blocks);
    CHECK(pool.references(blocks[0]) == 1);
    const std::vector<BlockId> duplicate{blocks[0], blocks[0]};
    test::throws<std::logic_error>([&] { pool.release(duplicate); });
    CHECK(pool.references(blocks[0]) == 1);
    pool.release(blocks);
    test::throws<std::logic_error>([&] { pool.release(blocks); });
    CHECK(pool.free_count() == 4);
}

TEST(block_pool_randomized_accounting) {
    BlockPool pool(64);
    std::mt19937 rng(17);
    std::vector<std::vector<BlockId>> tables;
    for (int i = 0; i < 5000; ++i) {
        if (tables.empty() || rng() % 3 == 0) {
            if (auto table = pool.allocate(rng() % 7 + 1)) {
                tables.push_back(std::move(*table));
            }
        } else {
            const auto index = rng() % tables.size();
            if (rng() % 2 == 0 && tables.size() < 128) {
                pool.retain(tables[index]);
                tables.push_back(tables[index]);
            } else {
                pool.release(tables[index]);
                tables.erase(tables.begin() + static_cast<std::ptrdiff_t>(index));
            }
        }
        std::vector<std::size_t> expected(64);
        for (const auto& table : tables) {
            for (auto id : table) {
                ++expected[id];
            }
        }
        for (BlockId id = 0; id < 64; ++id) {
            CHECK(pool.references(id) == expected[id]);
        }
    }
    for (const auto& table : tables) {
        pool.release(table);
    }
    CHECK(pool.free_count() == 64);
}

TEST(prefix_index_alignment_namespace_and_pruning) {
    PrefixIndex index(2);
    index.insert({7, "a", {1, 2, 3, 4, 5, 6}, {0, 1, 2}});
    index.insert({8, "a", {1, 2, 7, 8}, {3, 4}});
    const std::vector<Token> query{1, 2, 3, 4, 9};
    CHECK(index.match("a", query, 5)->tokens == 4);
    CHECK(index.match("a", query, 3)->tokens == 2);
    CHECK(!index.match("b", query, 5));
    index.erase(7);
    CHECK(index.match("a", query, 5)->tokens == 2);
    index.erase(8);
    CHECK(!index.match("a", query, 5));
    CHECK(index.token_count() == 0);
}

TEST(prefix_index_lru_and_full_hit_recompute) {
    PrefixIndex index(2);
    index.insert({7, "a", {1, 2, 3, 4}, {0, 1}});
    index.insert({8, "a", {5, 6}, {2}});
    const std::vector<Token> query{1, 2, 3, 4};
    CHECK(index.match("a", query, query.size() - 1)->tokens == 2);
    CHECK(index.least_recently_used() == 8);
}

TEST(utf8_split_and_invalid_sequences) {
    Utf8Buffer text;
    CHECK(text.append("A\xe4").compare("A") == 0);
    CHECK(text.append("\xb8").empty());
    CHECK(text.append("\xad\xf0\x9f").compare("\xe4\xb8\xad") == 0);
    CHECK(text.append("\x98\x80").compare("\xf0\x9f\x98\x80") == 0);
    CHECK(text.append("\xff").compare("\xef\xbf\xbd") == 0);
    CHECK(text.append("\xe2\x82").empty());
    CHECK(text.finish().compare("\xef\xbf\xbd") == 0);
}

TEST(scheduler_mixed_budget_and_decode_protection) {
    EngineConfig config;
    config.batch_tokens = 8;
    config.prefill_chunk = 4;
    std::vector<ScheduleItem> items{{0, 0, 0, 0, 0}, {1, 10, 3, 0, 1}, {2, 10, 0, 0, 2}};
    const auto plan = schedule_batch(items, config);
    CHECK(plan.mixed());
    CHECK(plan.token_count() == 8);
    CHECK(plan.slices[0].key == 0);
    CHECK(plan.slices[1].tokens == 4);
    CHECK(plan.slices[2].tokens == 3);
}

TEST(scheduler_prefill_first_and_aging) {
    EngineConfig config;
    config.policy = SchedulingPolicy::prefill_first;
    config.batch_tokens = 8;
    config.prefill_chunk = 8;
    std::vector<ScheduleItem> items{{0, 0, 0, 0, 0}, {1, 20, 3, 0, 1}, {2, 20, 0, 2000, 2}};
    const auto plan = schedule_batch(items, config);
    CHECK(plan.decode_tokens == 0);
    CHECK(plan.slices[0].key == 2);
    CHECK(plan.token_count() == 8);
}

TEST(config_rejects_unprogressable_settings) {
    EngineConfig config;
    config.max_active = 0;
    test::throws<std::invalid_argument>([&] { config.validate(); });
    config = {};
    config.batch_tokens = 4;
    test::throws<std::invalid_argument>([&] { config.validate(); });
    config = {};
    config.context_tokens = 65;
    test::throws<std::invalid_argument>([&] { config.validate(); });
}

TEST(cuda_serving_configuration_is_checked_without_loading_a_model) {
    ModelConfig model{"不存在的模型.gguf", 0, 8};
    EngineConfig valid;
    valid.max_active = 4;
    valid.batch_tokens = 128;
    valid.prefill_chunk = 32;
    valid.prefix_cache_entries = valid.prefix_cache_tokens = 0;
    validate_mini_cuda_config(model, valid);
    for (int failure = 0; failure < 8; ++failure) {
        auto config = valid;
        switch (failure) {
        case 0: config.max_active = 8; break;
        case 1: config.batch_tokens = 256; break;
        case 2: config.max_model_len = 2049; break;
        case 3: config.prefix_cache_entries = 1; config.prefix_cache_tokens = 16; break;
        case 4: config.prefix_cache_tokens = 16; break;
        case 5: config.context_tokens = 8208; break;
        case 6: config.max_active = std::numeric_limits<std::size_t>::max(); break;
        case 7: config.max_model_len = std::numeric_limits<std::size_t>::max(); break;
        }
        test::throws<std::invalid_argument>([&] { validate_mini_cuda_config(model, config); });
    }
    for (int failure = 0; failure < 5; ++failure) {
        auto invalid = model;
        switch (failure) {
        case 0: invalid.gpu_layers = 1; break;
        case 1: invalid.scalar_kernels = true; break;
        case 2: invalid.device = -1; break;
        case 3: invalid.threads = 0; break;
        case 4: invalid.path.clear(); break;
        }
        test::throws<std::invalid_argument>([&] { validate_mini_cuda_config(invalid, valid); });
    }
}

TEST(half_conversion_all_finite_patterns) {
    for (std::uint32_t bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00) == 0x7c00 && (bits & 1023)) {
            continue;
        }
        const auto half = static_cast<std::uint16_t>(bits);
        CHECK(minillm::float_to_half(minillm::half_to_float(half)) == half);
    }
    CHECK(minillm::float_to_half(1.00048828125f) == 0x3c00);
    CHECK(minillm::float_to_half(1.00146484375f) == 0x3c02);
}

TEST(simd_float_and_half_dot_match_scalar) {
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> sample(-2, 2);
    for (const auto length : {1, 7, 8, 9, 31, 128, 1024, 4097}) {
        std::vector<float> a(length), b(length);
        std::vector<std::uint16_t> h(length);
        for (int i = 0; i < length; ++i) {
            a[i] = sample(rng);
            b[i] = sample(rng);
            h[i] = minillm::float_to_half(a[i]);
        }
        const auto scalar = minillm::dot_f32(a.data(), b.data(), a.size(), minillm::KernelMode::scalar);
        const auto simd = minillm::dot_f32(a.data(), b.data(), a.size());
        CHECK(std::abs(scalar - simd) < 1e-3f);
        const auto hs = minillm::dot_f16(h.data(), b.data(), h.size(), minillm::KernelMode::scalar);
        CHECK(std::abs(hs - minillm::dot_f16(h.data(), b.data(), h.size())) < 1e-3f);
    }
}

TEST(simd_half_value_accumulation_matches_scalar) {
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> sample(-2, 2);
    for (const auto length : {1, 7, 8, 9, 31, 128, 1024, 4097}) {
        std::vector<std::uint16_t> input(length);
        std::vector<float> scalar(length), simd(length);
        for (int i = 0; i < length; ++i) {
            input[i] = minillm::float_to_half(sample(rng));
            scalar[i] = sample(rng);
        }
        simd = scalar;
        minillm::add_scaled_f16(input.data(), -0.375f, scalar.data(), input.size(),
                                minillm::KernelMode::scalar);
        minillm::add_scaled_f16(input.data(), -0.375f, simd.data(), input.size());
        for (int i = 0; i < length; ++i) {
            CHECK(std::abs(scalar[i] - simd[i]) < 1e-6f);
        }
    }
}

TEST(simd_q8_dot_matches_dequantized_weights) {
    std::mt19937 rng(23);
    for (const auto length : {32, 64, 1024, 4096}) {
        std::vector<std::byte> row(static_cast<std::size_t>(length) / 32 * 34);
        std::vector<float> input(length), decoded(length);
        for (int block = 0; block < length / 32; ++block) {
            const auto scale = minillm::float_to_half(0.01f * static_cast<float>(block % 3 + 1));
            std::memcpy(row.data() + block * 34, &scale, 2);
            for (int i = 0; i < 32; ++i) {
                const auto q = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127);
                std::memcpy(row.data() + block * 34 + 2 + i, &q, 1);
                input[block * 32 + i] = static_cast<float>(static_cast<int>(rng() % 100) - 50) / 100;
            }
        }
        minillm::decode_row(minillm::WeightType::q8_0, row.data(), decoded.data(), length);
        const auto expected = minillm::dot_f32(decoded.data(), input.data(), length, minillm::KernelMode::scalar);
        const auto actual = minillm::dot_row(minillm::WeightType::q8_0, row.data(), input.data(), length);
        CHECK(std::abs(expected - actual) < 1e-3f);
    }
}

TEST(rms_norm_in_place_and_simd) {
    std::vector<float> input{1, -2, 3, -4, 5, 6, -7, 8, 9};
    std::vector<float> weights(input.size(), 2);
    auto scalar = input;
    minillm::rms_norm(input.data(), weights.data(), scalar.data(), input.size(), 1e-6f,
                     minillm::KernelMode::scalar);
    minillm::rms_norm(input.data(), weights.data(), input.data(), input.size(), 1e-6f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        CHECK(std::abs(input[i] - scalar[i]) < 1e-5f);
    }
}

static void append_kv(minillm::PagedKV& cache, std::size_t seq, std::size_t pos, float value) {
    cache.append(seq, pos);
    const std::array<float, 2> key{value, value + 1};
    const std::array<float, 2> val{value + 2, value + 3};
    cache.store(seq, 0, pos, key, val);
    cache.store(seq, 1, pos, val, key);
}

TEST(paged_kv_layer_layout_and_full_page_sharing) {
    minillm::PagedKV cache({8, 2, 2, 2, 4});
    append_kv(cache, 0, 0, 1);
    append_kv(cache, 0, 1, 5);
    cache.share_prefix(0, 1, 2);
    CHECK(cache.used_pages() == 1);
    CHECK(cache.references(cache.page_table(0)[0]) == 2);
    CHECK(minillm::half_to_float(cache.value(1, 1, 1)[1]) == 6);
    append_kv(cache, 1, 2, 9);
    CHECK(cache.used_pages() == 2);
    CHECK(cache.copy_on_writes() == 0);
    cache.clear(0);
    CHECK(cache.used_pages() == 2);
    cache.clear(1);
    CHECK(cache.used_pages() == 0);
}

TEST(paged_kv_partial_tail_copy_on_write) {
    minillm::PagedKV cache({8, 4, 2, 2, 4});
    append_kv(cache, 0, 0, 1);
    append_kv(cache, 0, 1, 5);
    cache.share_prefix(0, 1, 1);
    append_kv(cache, 1, 1, 20);
    CHECK(cache.copy_on_writes() == 1);
    CHECK(cache.used_pages() == 2);
    CHECK(minillm::half_to_float(cache.key(0, 0, 1)[0]) == 5);
    CHECK(minillm::half_to_float(cache.key(1, 0, 1)[0]) == 20);
    CHECK(cache.page_table(0)[0] != cache.page_table(1)[0]);
}

TEST(paged_kv_exhaustion_does_not_corrupt_aliases) {
    minillm::PagedKV cache({1, 2, 2, 2, 4});
    append_kv(cache, 0, 0, 1);
    cache.share_prefix(0, 1, 1);
    test::throws<std::runtime_error>([&] { cache.append(1, 1); });
    CHECK(cache.length(1) == 1);
    CHECK(cache.references(0) == 2);
    cache.clear(0);
    append_kv(cache, 1, 1, 7);
    CHECK(cache.length(1) == 2);
    test::throws<std::invalid_argument>([&] { cache.append(1, 3); });
    cache.clear(1);
    CHECK(cache.used_pages() == 0);
}

TEST(paged_kv_randomized_alias_append_and_reclamation) {
    minillm::PagedKV cache({64, 4, 2, 2, 8});
    std::array<std::vector<float>, 8> history;
    std::mt19937 random(29);
    for (int step = 0; step < 1000; ++step) {
        const auto target = random() % history.size();
        const auto operation = random() % 3;
        if (operation == 0 && history[target].size() < 32) {
            const auto value = static_cast<float>(random() % 500);
            append_kv(cache, target, history[target].size(), value);
            history[target].push_back(value);
        } else if (operation == 1) {
            const auto source = (target + 1 + random() % 7) % history.size();
            const auto length = random() % (history[source].size() + 1);
            cache.share_prefix(source, target, length);
            history[target] = {history[source].begin(), history[source].begin() +
                               static_cast<std::ptrdiff_t>(length)};
        } else {
            cache.clear(target);
            history[target].clear();
        }
        std::array<std::size_t, 64> refs{};
        for (std::size_t sequence = 0; sequence < history.size(); ++sequence) {
            CHECK(cache.length(sequence) == history[sequence].size());
            for (const auto page : cache.page_table(sequence)) {
                ++refs[page];
            }
            for (std::size_t position = 0; position < history[sequence].size(); ++position) {
                CHECK(minillm::half_to_float(cache.key(sequence, 0, position)[0]) ==
                      history[sequence][position]);
                CHECK(minillm::half_to_float(cache.value(sequence, 1, position)[1]) ==
                      history[sequence][position] + 1);
            }
        }
        for (std::size_t page = 0; page < refs.size(); ++page) {
            CHECK(cache.references(page) == refs[page]);
        }
    }
    for (std::size_t sequence = 0; sequence < history.size(); ++sequence) {
        cache.clear(sequence);
    }
    CHECK(cache.used_pages() == 0);
}

TEST(parallel_executor_exact_coverage_and_exception_recovery) {
    minillm::ParallelExecutor executor(4);
    for (int repeat = 0; repeat < 50; ++repeat) {
        std::vector<int> counts(1003, 0);
        executor.run(counts.size(), 7, [&](auto begin, auto end) {
            for (auto i = begin; i < end; ++i) {
                ++counts[i];
            }
        });
        CHECK(std::all_of(counts.begin(), counts.end(), [](int value) { return value == 1; }));
    }
    test::throws<std::runtime_error>([&] {
        executor.run(10, 1, [](auto, auto) { throw std::runtime_error("injected"); });
    });
    std::atomic<int> count = 0;
    executor.run(10, 1, [&](auto, auto) { ++count; });
    CHECK(count == 10);
}

TEST(parallel_profile_accounting_and_reuse) {
    for (const std::size_t threads : {1, 4}) {
        minillm::ParallelExecutor executor(threads);
        minillm::ParallelProfile profile;
        for (int repeat = 0; repeat < 20; ++repeat) {
            std::vector<int> values(1003);
            executor.run(values.size(), 7, [&](auto begin, auto end) {
                for (auto i = begin; i < end; ++i) {
                    ++values[i];
                }
            }, &profile);
            CHECK(std::all_of(values.begin(), values.end(), [](int n) { return n == 1; }));
            CHECK(profile.completed);
            CHECK(profile.count == values.size() && profile.grain == 7 && profile.threads == threads);
            CHECK(profile.chunks == (values.size() + 6) / 7);
            CHECK(profile.participating_threads >= 1 && profile.participating_threads <= threads);
            CHECK(profile.dispatch_ns + profile.caller_work_ns + profile.caller_wait_ns <= profile.wall_ns);
            CHECK(profile.worker_work_max_ns <= profile.wall_ns);
            CHECK(profile.worker_start_delay_max_ns <= profile.wall_ns);
            CHECK(profile.worker_work_sum_ns <= profile.wall_ns * (threads - 1));
            if (threads == 1) {
                CHECK(profile.worker_work_sum_ns == 0 && profile.worker_start_delay_max_ns == 0);
            }
        }
        executor.run(0, 1, [](auto, auto) { throw std::runtime_error("must not execute"); }, &profile);
        CHECK(profile.completed && profile.count == 0 && profile.chunks == 0);
        test::throws<std::invalid_argument>([&] { executor.run(1, 0, [](auto, auto) {}, &profile); });
        CHECK(!profile.completed && profile.count == 1 && profile.chunks == 0);
        test::throws<std::runtime_error>([&] {
            executor.run(10, 1, [](auto, auto) { throw std::runtime_error("injected"); }, &profile);
        });
        CHECK(!profile.completed && profile.count == 10 && profile.wall_ns > 0);
        std::atomic<int> count{0};
        executor.run(17, 1, [&](auto, auto) { ++count; });
        CHECK(count == 17);
        executor.run(17, 1, [](auto, auto) {}, &profile);
        CHECK(profile.completed && profile.chunks == 17);
    }
}

TEST(forward_profile_reserve_failure_clears_stale_success) {
    minillm::ForwardProfile profile;
    profile.batch_id = 17;
    profile.completed = true;
    profile.wall_ns = 42;
    profile.input_tokens = 2;
    profile.stages.reserve(2);
    profile.stages.push_back({minillm::ProfileStage::embedding});
    const auto capacity = profile.stages.capacity();
    test::throws<std::length_error>([&] { profile.reset(profile.stages.max_size() + 1); });
    CHECK(!profile.completed && profile.wall_ns == 0 && profile.input_tokens == 0);
    CHECK(profile.batch_id == 17 && profile.stages.empty() && profile.stages.capacity() == capacity);
    profile.reset(4);
    CHECK(profile.batch_id == 17 && !profile.completed && profile.stages.capacity() >= 4);
}

struct Probe {
    std::atomic<int> delay_ms{0};
    std::atomic<bool> fail{false};
    std::atomic<bool> fail_copy{false};
    std::atomic<bool> missing_sample{false};
    std::atomic<int> invalid_sample{0};
    std::atomic<int> clear_calls{0};
    std::atomic<int> fail_clear_at{0};
    std::atomic<bool> fail_sync{false};
    std::atomic<bool> invalid_state{false};
    std::atomic<bool> invalid_clear_slot{false};
    std::atomic<std::size_t> resource_calls{0};
    std::atomic<bool> wrong_resource_thread{false};
    std::thread::id resource_thread;
    bool report_resources = false;
    BackendCapabilities capabilities{256, 8192, 8192, true, false, true};
};

class FakeRunner final : public ModelRunner {
public:
    explicit FakeRunner(std::shared_ptr<Probe> probe = std::make_shared<Probe>())
        : probe_(std::move(probe)) {}
    const ModelInfo& info() const noexcept override { return info_; }
    BackendCapabilities capabilities() const noexcept override { return probe_->capabilities; }
    std::optional<RunnerResources> resources() const noexcept override {
        if (!probe_->report_resources) { return std::nullopt; }
        ++probe_->resource_calls;
        if (probe_->resource_thread == std::thread::id{}) { probe_->resource_thread = std::this_thread::get_id(); }
        if (probe_->resource_thread != std::this_thread::get_id()) { probe_->wrong_resource_thread = true; }
        const bool valid = !probe_->invalid_state.load();
        return RunnerResources{std::nullopt, 4096, KvLayout::contiguous, 8192,
                               valid ? std::optional<std::size_t>(0) : std::nullopt, 8192, valid, valid};
    }
    std::vector<Token> tokenize(std::string_view text) const override {
        return {text.begin(), text.end()};
    }
    std::string token_piece(Token token) const override { return std::string(1, static_cast<char>(token)); }
    bool is_eog(Token token) const override { return token == 0; }
    std::vector<Sample> execute(std::span<const BatchToken> batch) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(probe_->delay_ms.load()));
        if (probe_->fail.load()) {
            throw std::runtime_error("injected model failure");
        }
        std::vector<Sample> result;
        for (const auto& token : batch) {
            auto& sequence = sequences_[token.sequence];
            if (static_cast<std::size_t>(token.position) != sequence.size()) {
                throw std::runtime_error("wrong position in fake runner");
            }
            sequence.push_back(token.token);
            if (token.logits) {
                std::uint64_t hash = 0;
                for (auto t : sequence) {
                    hash = hash * 31 + static_cast<std::uint64_t>(t);
                }
                result.push_back({token.sequence, static_cast<Token>(65 + hash % 26)});
            }
        }
        if (probe_->missing_sample.load()) {
            result.clear();
        }
        if (result.size() > 1) {
            switch (probe_->invalid_sample.load()) {
            case 1: result.back() = result.front(); break;
            case 2: result.back().sequence = -1; break;
            case 3: result.back().token = 512; break;
            case 4: result.back().sequence = 7; break;
            default: break;
            }
        }
        return result;
    }
    void copy_sequence(SequenceId source, SequenceId target, std::size_t tokens) override {
        if (probe_->fail_copy.load()) {
            throw std::runtime_error("injected prefix copy failure");
        }
        const auto& from = sequences_.at(source);
        sequences_[target] = {from.begin(), from.begin() + static_cast<std::ptrdiff_t>(tokens)};
    }
    void clear_sequence(SequenceId sequence) noexcept override {
        if (sequence < 0 || (probe_->capabilities.max_sequences &&
            static_cast<std::size_t>(sequence) >= probe_->capabilities.max_sequences)) {
            probe_->invalid_clear_slot = true;
        }
        if (++probe_->clear_calls == probe_->fail_clear_at.load()) { probe_->invalid_state = true; }
        if (!probe_->invalid_state.load()) { sequences_.erase(sequence); }
    }
    void synchronize() noexcept override {
        if (probe_->fail_sync.load()) { probe_->invalid_state = true; }
    }
private:
    ModelInfo info_{"fake", "test-model", "test", "CPU", 8192, 512, false};
    std::shared_ptr<Probe> probe_;
    std::map<SequenceId, std::vector<Token>> sequences_;
};

struct Collected {
    std::string text;
    std::vector<Token> tokens;
    Event terminal;
    std::vector<TokenTelemetry> telemetry;
};

static Collected collect(const std::shared_ptr<RequestHandle>& handle) {
    Collected result;
    const auto deadline = Clock::now() + 5s;
    while (Clock::now() < deadline) {
        if (auto event = handle->next(20ms)) {
            result.text += event->text;
            if (event->token) {
                result.tokens.push_back(*event->token);
                if (event->telemetry) { result.telemetry.push_back(*event->telemetry); }
            }
            if (event->kind != Event::Kind::token) {
                result.terminal = std::move(*event);
                return result;
            }
        }
    }
    throw std::runtime_error("test request did not terminate");
}

static void check_idle(Engine& engine) {
    for (int i = 0; i < 200; ++i) {
        const auto stats = engine.statistics();
        if (stats.active_requests == 0 && stats.outstanding_requests == 0) {
            CHECK(stats.kv_active_unique_blocks == 0);
            return;
        }
        std::this_thread::sleep_for(1ms);
    }
    throw std::runtime_error("engine did not reclaim active requests");
}

static RequestInput request(std::string prompt, std::size_t max_tokens = 8) {
    RequestInput result;
    result.prompt = std::move(prompt);
    result.max_tokens = max_tokens;
    result.ignore_eos = true;
    return result;
}

TEST(engine_checks_runner_capacities_and_prefix_support) {
    EngineConfig config;
    for (int failure = 0; failure < 5; ++failure) {
        auto probe = std::make_shared<Probe>();
        switch (failure) {
        case 0: probe->capabilities.max_sequences = config.max_active + config.prefix_cache_entries - 1; break;
        case 1: probe->capabilities.max_batch_tokens = config.batch_tokens - 1; break;
        case 2: probe->capabilities.max_model_len = config.max_model_len - 1; break;
        case 3: probe->capabilities.prefix_copy = false; break;
        case 4: probe->capabilities.synchronous_execute = false; break;
        }
        test::throws<std::invalid_argument>([&] { Engine engine(config, std::make_unique<FakeRunner>(probe)); });
        CHECK(probe->clear_calls == 0);
    }
    auto probe = std::make_shared<Probe>();
    probe->capabilities = {0, 0, 0, true, false, true};
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    CHECK(collect(engine.submit(request("unknown upper bound", 1))).terminal.status == 200);
}

TEST(gated_runner_preserves_capabilities_and_nullable_resources) {
    auto probe = std::make_shared<Probe>();
    probe->capabilities = {4, 128, 2048, false, false, true};
    probe->report_resources = true;
    test::GatedRunner runner(std::make_unique<FakeRunner>(probe), std::make_shared<test::RunnerGate>());
    const auto caps = runner.capabilities();
    CHECK(caps.max_sequences == 4 && caps.max_batch_tokens == 128 && caps.max_model_len == 2048);
    CHECK(!caps.prefix_copy && !caps.runtime_stage_profile && caps.synchronous_execute);
    CHECK(runner.resources()->layout == KvLayout::contiguous && !runner.resources()->live_kv_pages);
    CHECK(runner.resources()->live_tokens == 0 && runner.resources()->resident_kv_payload_bytes == 4096);
    probe->invalid_state = true;
    CHECK(!runner.resources()->state_valid && !runner.resources()->reusable && !runner.resources()->live_tokens);
    CHECK(runner.resources()->resident_kv_payload_bytes == 4096);
    RunnerResources unknown;
    CHECK(unknown.layout == KvLayout::unknown && !unknown.live_kv_pages && !unknown.live_tokens);
    RunnerResources paged{3, 4096, KvLayout::paged, 64};
    CHECK(paged.live_kv_pages == 3 && !paged.live_tokens && !paged.owned_device_bytes);
}

TEST(engine_statistics_only_reads_published_resource_copies) {
    auto probe = std::make_shared<Probe>();
    probe->report_resources = true;
    auto gate = std::make_shared<test::RunnerGate>();
    Engine engine({}, std::make_unique<test::GatedRunner>(std::make_unique<FakeRunner>(probe), gate));
    CHECK(engine.statistics().resources && engine.statistics().resources_batch_id == 0);
    const auto handle = engine.submit(request("snapshot", 4));
    try {
        gate->wait_until_sampled();
        const auto calls = probe->resource_calls.load();
        for (int i = 0; i < 100; ++i) {
            const auto snapshot = engine.statistics();
            CHECK(snapshot.resources_batch_id == 0 && snapshot.resources->state_valid);
        }
        CHECK(probe->resource_calls == calls && !probe->wrong_resource_thread);
    } catch (...) {
        gate->release();
        throw;
    }
    gate->release();
    CHECK(collect(handle).terminal.status == 200);
    engine.stop();
    CHECK(!probe->wrong_resource_thread);
    CHECK(engine.statistics().resources_batch_id == engine.statistics().batches);
}

TEST(engine_idle_stop_does_not_lose_worker_wakeup) {
    for (int i = 0; i < 256; ++i) {
        Engine engine({}, std::make_unique<FakeRunner>());
        engine.stop();
        CHECK(!engine.statistics().ready && engine.statistics().outstanding_requests == 0);
    }
}

TEST(engine_noexcept_cleanup_failure_has_one_terminal_and_returns_credits) {
    for (int fail_at : {1, 2}) {
        auto probe = std::make_shared<Probe>();
        probe->report_resources = true;
        probe->fail_clear_at = fail_at;
        EngineConfig config;
        config.prefix_cache_entries = config.prefix_cache_tokens = 0;
        Engine engine(config, std::make_unique<FakeRunner>(probe));
        const auto handle = engine.submit(request("clear failure", 1));
        CHECK(collect(handle).terminal.error_code == "backend_error");
        CHECK(!handle->next(1ms));
        engine.stop();
        CHECK(engine.statistics().failed == 1 && engine.statistics().completed == 0);
        CHECK(engine.statistics().kv_used_blocks == 0 && !engine.statistics().ready);
        CHECK(!probe->invalid_clear_slot);
        test::throws<RequestError>([&] { engine.submit(request("cannot reuse")); });
    }
    auto probe = std::make_shared<Probe>();
    probe->report_resources = true;
    probe->fail_sync = true;
    auto gate = std::make_shared<test::RunnerGate>();
    Engine engine({}, std::make_unique<test::GatedRunner>(std::make_unique<FakeRunner>(probe), gate));
    const auto handle = engine.submit(request("sync failure", 100));
    try {
        gate->wait_until_sampled();
    } catch (...) {
        gate->release();
        throw;
    }
    auto stopped = std::async(std::launch::async, [&] { engine.stop(); });
    while (engine.statistics().ready) { std::this_thread::yield(); }
    gate->release();
    CHECK(collect(handle).terminal.error_code == "backend_error" && !handle->next(1ms));
    stopped.get();
    CHECK(engine.statistics().kv_used_blocks == 0 && !engine.statistics().last_error.empty());
}

TEST(engine_rejects_entire_invalid_sample_batch_before_emission) {
    for (int failure : {1, 2, 3, 4}) {
        auto probe = std::make_shared<Probe>();
        probe->invalid_sample = failure;
        auto gate = std::make_shared<test::RunnerGate>();
        EngineConfig config;
        config.prefix_cache_entries = config.prefix_cache_tokens = 0;
        Engine engine(config, std::make_unique<test::GatedRunner>(std::make_unique<FakeRunner>(probe), gate));
        auto first = engine.submit(request("a", 4));
        std::shared_ptr<RequestHandle> second;
        try {
            gate->wait_until_sampled();
            second = engine.submit(request("b", 4));
        } catch (...) {
            gate->release();
            throw;
        }
        gate->release();
        const auto one = collect(first), two = collect(second);
        CHECK(one.terminal.error_code == "backend_error" && two.terminal.error_code == "backend_error");
        CHECK(one.tokens.size() == 1 && two.tokens.empty());
        CHECK(!first->next(1ms) && !second->next(1ms));
        engine.stop();
        CHECK(engine.statistics().kv_used_blocks == 0 && !probe->invalid_clear_slot);
    }
}

TEST(engine_serial_batched_and_cached_outputs_agree) {
    EngineConfig config;
    config.prefill_chunk = 8;
    config.block_size = 4;
    Engine engine(config, std::make_unique<FakeRunner>());
    const auto input = request("Shared prompt with repeated prefix.");
    const auto serial = collect(engine.submit(input));
    CHECK(serial.terminal.status == 200);
    std::vector<std::shared_ptr<RequestHandle>> batch;
    for (int i = 0; i < 8; ++i) {
        batch.push_back(engine.submit(input));
    }
    for (const auto& handle : batch) {
        const auto output = collect(handle);
        CHECK(output.tokens == serial.tokens);
        CHECK(output.text == serial.text);
        CHECK(output.terminal.usage.cached_tokens > 0);
        CHECK(output.terminal.usage.completion_tokens == 8);
    }
    check_idle(engine);
    CHECK(engine.statistics().cache_hits >= 8);
}

TEST(engine_mixed_batch_after_deterministic_request_injection) {
    for (int repeat = 0; repeat < 10; ++repeat) {
        EngineConfig config;
        config.max_active = 3;
        config.batch_tokens = 8;
        config.prefill_chunk = 4;
        auto gate = std::make_shared<test::RunnerGate>();
        Engine engine(config, std::make_unique<test::GatedRunner>(
            std::make_unique<FakeRunner>(), gate));
        std::vector<std::shared_ptr<RequestHandle>> handles;
        try {
            handles.push_back(engine.submit(request("first prompt", 4)));
            gate->wait_until_sampled();
            handles.push_back(engine.submit(request("second prompt", 4)));
            handles.push_back(engine.submit(request("third prompt", 4)));
        } catch (...) {
            gate->release();
            throw;
        }
        gate->release();
        for (const auto& handle : handles) {
            const auto result = collect(handle);
            CHECK(result.terminal.status == 200);
            CHECK(result.tokens.size() == 4);
        }
        check_idle(engine);
        CHECK(engine.statistics().mixed_batches > 0);
        CHECK(engine.statistics().max_batch_sequences > 1);
    }
}

TEST(engine_cache_namespace_and_full_hit_recompute) {
    EngineConfig config;
    config.block_size = 4;
    Engine engine(config, std::make_unique<FakeRunner>());
    auto input = request("12345678");
    const auto first = collect(engine.submit(input));
    const auto second = collect(engine.submit(input));
    CHECK(second.terminal.usage.cached_tokens == 4);
    CHECK(first.tokens == second.tokens);
    input.cache_namespace = "isolated";
    CHECK(collect(engine.submit(input)).terminal.usage.cached_tokens == 0);
    check_idle(engine);
}

TEST(engine_telemetry_links_mixed_batches_and_cached_tokens) {
    for (const auto mode : {TelemetryMode::batches, TelemetryMode::stages}) {
        EngineConfig config;
        config.telemetry_mode = mode;
        config.telemetry_capacity = 64;
        config.max_active = 3;
        config.batch_tokens = 8;
        config.prefill_chunk = 4;
        config.block_size = 4;
        auto gate = std::make_shared<test::RunnerGate>();
        Engine engine(config, std::make_unique<test::GatedRunner>(std::make_unique<FakeRunner>(), gate));
        test::throws<std::logic_error>([&] { engine.telemetry(); });
        std::vector<std::shared_ptr<RequestHandle>> handles;
        try {
            handles.push_back(engine.submit(request("a cached first prompt", 4)));
            gate->wait_until_sampled();
            handles.push_back(engine.submit(request("second prompt", 4)));
            handles.push_back(engine.submit(request("third prompt", 4)));
        } catch (...) {
            gate->release();
            throw;
        }
        gate->release();
        std::map<std::string, Collected> results;
        for (const auto& handle : handles) { results.emplace(handle->id(), collect(handle)); }
        auto cached = engine.submit(request("a cached first prompt", 4));
        results.emplace(cached->id(), collect(cached));
        CHECK(results.at(cached->id()).tokens == results.at(handles.front()->id()).tokens);
        CHECK(results.at(cached->id()).terminal.usage.cached_tokens > 0);
        engine.stop();
        const auto& capture = engine.telemetry();
        CHECK(capture.recorded == engine.statistics().batches && capture.dropped == 0);
        CHECK(capture.storage_bytes > 0 && !capture.resources_final);
        std::size_t emitted = 0, mixed = 0;
        std::uint64_t previous_finish = 0;
        for (std::size_t i = 0; i < capture.recorded; ++i) {
            const auto& batch = capture.batches[i];
            CHECK(batch.batch_id == i + 1 && batch.completed && batch.runner_completed);
            CHECK(batch.start_ns >= previous_finish);
            CHECK(batch.start_ns + batch.admission_ns + batch.scheduler_ns + batch.prepare_ns == batch.runner_start_ns);
            CHECK(batch.runner_start_ns + batch.runner_ns <= batch.finish_ns);
            CHECK(!batch.runner.available && !batch.resources_before && !batch.resources_after);
            mixed += batch.prefill_tokens > 0 && batch.decode_tokens > 0;
            std::size_t before = 0, after = 0, logits = 0, tokens = 0;
            for (std::size_t j = 0; j < batch.sequences; ++j) {
                const auto& slice = batch.slices[j];
                before += slice.context_before;
                after += slice.context_before + slice.tokens;
                logits += slice.logits_tokens;
                tokens += slice.tokens;
                if (!slice.emitted) { continue; }
                ++emitted;
                const auto& result = results.at(slice.request_id.data());
                CHECK(result.terminal.status == 200 && result.telemetry.size() == result.tokens.size());
                const auto& event = result.telemetry.at(slice.token_index);
                CHECK(event.batch_id == batch.batch_id && event.request_order == slice.request_order);
                CHECK(event.engine_elapsed_ns == slice.emitted_ns && event.token_index == slice.token_index);
                CHECK(slice.sampled_token == result.tokens.at(slice.token_index));
                CHECK(slice.emitted_ns >= batch.runner_start_ns + batch.runner_ns);
                CHECK(slice.emitted_ns <= batch.finish_ns);
            }
            CHECK(before == batch.context_before_sum && after == batch.context_after_sum);
            CHECK(tokens == batch.prefill_tokens + batch.decode_tokens && logits == batch.logits_tokens);
            previous_finish = batch.finish_ns;
        }
        CHECK(emitted == 16 && mixed > 0);
    }
}

TEST(engine_telemetry_overflow_preserves_output_and_reports_loss) {
    std::vector<Token> baseline;
    for (const auto mode : {TelemetryMode::off, TelemetryMode::batches, TelemetryMode::stages}) {
        EngineConfig config;
        config.telemetry_mode = mode;
        config.telemetry_capacity = 1;
        Engine engine(config, std::make_unique<FakeRunner>());
        const auto result = collect(engine.submit(request("bounded capture", 8)));
        CHECK(result.terminal.status == 200);
        engine.stop();
        const auto& capture = engine.telemetry();
        if (mode == TelemetryMode::off) {
            baseline = result.tokens;
            CHECK(capture.batches.empty() && capture.storage_bytes == 0 && result.telemetry.empty());
        } else {
            CHECK(result.tokens == baseline && result.telemetry.size() == baseline.size());
            CHECK(capture.recorded == 1 && capture.batches.size() == 1);
            CHECK(capture.dropped + 1 == engine.statistics().batches);
        }
    }
}

TEST(engine_telemetry_backend_failure_is_incomplete) {
    auto probe = std::make_shared<Probe>();
    probe->fail = true;
    EngineConfig config;
    config.telemetry_mode = TelemetryMode::stages;
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    CHECK(collect(engine.submit(request("failure"))).terminal.error_code == "backend_error");
    engine.stop();
    const auto& capture = engine.telemetry();
    CHECK(capture.recorded == 1 && capture.dropped == 0);
    CHECK(!capture.batches[0].completed && !capture.batches[0].runner_completed);
    CHECK(capture.batches[0].finish_ns >= capture.batches[0].runner_start_ns + capture.batches[0].runner_ns);
    CHECK(engine.statistics().kv_used_blocks == 0);
}

TEST(engine_context_validation_and_duplicate_ids) {
    EngineConfig config;
    auto probe = std::make_shared<Probe>();
    probe->delay_ms = 20;
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    test::throws<RequestError>([&] { engine.submit(request("", 1)); });
    test::throws<RequestError>([&] { engine.submit(request(std::string(2048, 'x'), 1)); });
    auto input = request("abc", 20);
    input.id = "duplicate";
    auto handle = engine.submit(input);
    test::throws<RequestError>([&] { engine.submit(input); });
    CHECK(engine.cancel(input.id));
    CHECK(collect(handle).terminal.error_code == "cancelled");
    check_idle(engine);
}

TEST(engine_bounded_queue_and_cancellation) {
    EngineConfig config;
    config.queue_capacity = 1;
    auto probe = std::make_shared<Probe>();
    probe->delay_ms = 20;
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    auto handle = engine.submit(request("abc", 100));
    test::throws<RequestError>([&] { engine.submit(request("def")); });
    handle->cancel();
    CHECK(collect(handle).terminal.error_code == "cancelled");
    check_idle(engine);
    CHECK(!engine.cancel("absent"));
}

TEST(engine_deadline_and_slow_consumer_reclaim_resources) {
    EngineConfig config;
    config.event_buffer_size = 2;
    auto probe = std::make_shared<Probe>();
    probe->delay_ms = 10;
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    auto input = request("deadline", 20);
    input.timeout_ms = 1;
    CHECK(collect(engine.submit(input)).terminal.error_code == "timeout");
    auto slow = engine.submit(request("slow", 20));
    std::this_thread::sleep_for(150ms);
    CHECK(collect(slow).terminal.error_code == "backpressure");
    check_idle(engine);
}

TEST(engine_backend_failure_is_terminal_and_rejects_new_work) {
    auto probe = std::make_shared<Probe>();
    probe->fail = true;
    Engine engine({}, std::make_unique<FakeRunner>(probe));
    const auto result = collect(engine.submit(request("failure")));
    CHECK(result.terminal.error_code == "backend_error");
    check_idle(engine);
    CHECK(!engine.statistics().ready);
    test::throws<RequestError>([&] { engine.submit(request("again")); });
}

TEST(engine_prefix_copy_failure_has_one_terminal_and_no_reservation_leak) {
    auto probe = std::make_shared<Probe>();
    EngineConfig config;
    config.block_size = 4;
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    CHECK(collect(engine.submit(request("a reusable prefix"))).terminal.status == 200);
    probe->fail_copy = true;
    const auto failed = engine.submit(request("a reusable prefix"));
    CHECK(collect(failed).terminal.error_code == "backend_error");
    CHECK(!failed->next(1ms));
    engine.stop();
    check_idle(engine);
    CHECK(engine.statistics().failed == 1);
    CHECK(engine.statistics().kv_used_blocks == 0);
}

TEST(engine_missing_logits_is_a_backend_error) {
    auto probe = std::make_shared<Probe>();
    probe->missing_sample = true;
    Engine engine({}, std::make_unique<FakeRunner>(probe));
    CHECK(collect(engine.submit(request("missing logits"))).terminal.error_code == "backend_error");
    check_idle(engine);
    CHECK(!engine.statistics().ready);
}

TEST(engine_small_kv_pool_evicts_and_makes_progress) {
    EngineConfig config;
    config.context_tokens = 64;
    config.max_model_len = 48;
    config.batch_tokens = 16;
    config.prefill_chunk = 4;
    config.block_size = 4;
    config.prefix_cache_tokens = 32;
    Engine engine(config, std::make_unique<FakeRunner>());
    std::vector<std::shared_ptr<RequestHandle>> handles;
    for (int i = 0; i < 8; ++i) {
        handles.push_back(engine.submit(request(std::string(24, static_cast<char>('a' + i)))));
    }
    for (const auto& handle : handles) {
        CHECK(collect(handle).terminal.status == 200);
    }
    check_idle(engine);
    CHECK(engine.statistics().cache_evictions > 0);
}

TEST(engine_shutdown_terminates_queued_and_running_requests) {
    auto probe = std::make_shared<Probe>();
    probe->delay_ms = 10;
    EngineConfig config;
    config.max_active = 1;
    Engine engine(config, std::make_unique<FakeRunner>(probe));
    auto one = engine.submit(request("one", 100));
    auto two = engine.submit(request("two", 100));
    engine.stop();
    CHECK(collect(one).terminal.error_code == "cancelled");
    CHECK(collect(two).terminal.error_code == "cancelled");
    CHECK(engine.statistics().kv_used_blocks == 0);
    test::throws<RequestError>([&] { engine.submit(request("three")); });
}

int main() {
    std::cout << std::unitbuf;
    return test::run();
}
