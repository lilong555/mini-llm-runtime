#include "test_support.h"

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

struct Probe {
    std::atomic<int> delay_ms{0};
    std::atomic<bool> fail{false};
    std::atomic<bool> fail_copy{false};
    std::atomic<bool> missing_sample{false};
};

class FakeRunner final : public ModelRunner {
public:
    explicit FakeRunner(std::shared_ptr<Probe> probe = std::make_shared<Probe>())
        : probe_(std::move(probe)) {}
    const ModelInfo& info() const noexcept override { return info_; }
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
        return result;
    }
    void copy_sequence(SequenceId source, SequenceId target, std::size_t tokens) override {
        if (probe_->fail_copy.load()) {
            throw std::runtime_error("injected prefix copy failure");
        }
        const auto& from = sequences_.at(source);
        sequences_[target] = {from.begin(), from.begin() + static_cast<std::ptrdiff_t>(tokens)};
    }
    void clear_sequence(SequenceId sequence) noexcept override { sequences_.erase(sequence); }
    void synchronize() noexcept override {}
private:
    ModelInfo info_{"fake", "test-model", "test", "CPU", 8192, 512, false};
    std::shared_ptr<Probe> probe_;
    std::map<SequenceId, std::vector<Token>> sequences_;
};

struct Collected {
    std::string text;
    std::vector<Token> tokens;
    Event terminal;
};

static Collected collect(const std::shared_ptr<RequestHandle>& handle) {
    Collected result;
    const auto deadline = Clock::now() + 5s;
    while (Clock::now() < deadline) {
        if (auto event = handle->next(20ms)) {
            result.text += event->text;
            if (event->token) {
                result.tokens.push_back(*event->token);
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

int main() { return test::run(); }
