#include "test_support.h"
#include "qwen3_fixture.h"
#include "batch_state.h"
#include "layer.h"
#include "tensor_validation.h"

#include <nlohmann/json.hpp>
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <climits>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <sstream>

using namespace minillm;
using namespace minillm::cuda;
using minillm::cuda::detail::read_only;
using json = nlohmann::ordered_json;

namespace {
template<class T>
void upload(const CudaContext& context, DeviceTensorView<T> view, const std::vector<T>& data) {
    CHECK(view.stride == view.columns && data.size() == view.rows * view.columns);
    check_cuda(cudaMemcpyAsync(view.data, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice,
                               context.stream()), "上传层测试输入");
    context.synchronize();
}
template<class T>
std::vector<T> download(const CudaContext& context, DeviceTensorView<T> view) {
    CHECK(view.stride == view.columns);
    std::vector<T> data(view.rows * view.columns);
    check_cuda(cudaMemcpyAsync(data.data(), view.data, data.size() * sizeof(T), cudaMemcpyDeviceToHost,
                               context.stream()), "下载层测试结果");
    context.synchronize();
    return data;
}
void near(float actual, double expected) {
    CHECK(std::isfinite(actual) && std::isfinite(expected));
    const double error = std::abs(double(actual) - expected), limit = 2e-4 + 2e-4 * std::abs(expected);
    if (error > limit) {
        std::ostringstream message;
        message << std::setprecision(12) << "层数值超出容差：actual=" << actual << " expected=" << expected
                << " absolute=" << error << " limit=" << limit;
        throw std::runtime_error(message.str());
    }
}
void status_is(const CudaContext& context, DeviceBuffer<std::int32_t>& status, int bits = 0, int first = INT_MAX) {
    const auto value = download(context, matrix_view(status, 1, 2));
    CHECK(value[0] == bits && value[1] == first);
}
std::size_t cache_index(KvShape s, std::size_t layer, std::size_t slot, std::size_t kind,
                        std::size_t position, std::size_t column, std::size_t stride) {
    return (((slot * s.layers + layer) * 2 + kind) * s.max_length + position) * stride + column;
}
std::vector<std::size_t> lengths(const BatchState& state) {
    return {state.lengths().begin(), state.lengths().end()};
}
}

TEST(batch_state_prepare_commit_clear_and_preflight_failure) {
    BatchState state({3, 17, 8, 0}, 101);
    const std::vector<InputToken> batch{{3,0,2,true}, {4,0,0,false}, {5,1,2,true}};
    const auto summary = state.prepare(batch);
    CHECK(summary.tokens == 3 && summary.logits == 2 && summary.max_context == 2);
    CHECK(state.live_tokens() == 0 && state.phase() == BatchPhase::prepared);
    test::throws<Error>([&] { state.clear(0); });
    test::throws<Error>([&] { state.prepare(batch); });
    test::throws<Error>([&] { state.commit(); });
    state.discard_prepared();
    CHECK(state.live_tokens() == 0);
    state.prepare(batch); state.start();
    CHECK(state.live_tokens() == 0);
    test::throws<Error>([&] { state.clear(0); });
    state.commit();
    CHECK(lengths(state) == (std::vector<std::size_t>{1,0,2}));
    CHECK(state.live_sequences() == 2 && state.live_tokens() == 3);
    for (const auto& invalid : std::vector<std::vector<InputToken>>{
        {}, {{0,0,3,true}}, {{-1,1,0,true}}, {{101,1,0,true}}, {{0,-1,0,true}},
        {{0,17,0,true}}, {{0,2,0,true}}, {{0,1,0,true},{1,1,0,true}},
        std::vector<InputToken>(9, {0,1,0,true})}) {
        const auto before = lengths(state);
        test::throws<std::invalid_argument>([&] { state.prepare(invalid); });
        CHECK(state.phase() == BatchPhase::ready && lengths(state) == before);
    }
    state.clear(0);
    CHECK(lengths(state) == (std::vector<std::size_t>{0,0,2}));
    test::throws<std::invalid_argument>([&] { state.clear(-1); });
    test::throws<std::invalid_argument>([&] { state.clear(3); });
}

TEST(batch_state_random_append_clear_and_poison_is_terminal) {
    BatchState state({3,17,8,0}, 101);
    std::array<std::size_t,3> expected{};
    std::mt19937 random(20260926);
    for (int step = 0; step < 1000; ++step) {
        const auto sequence = random() % 3;
        if (random() % 5 == 0 || expected[sequence] == 17) {
            state.clear(std::int32_t(sequence)); expected[sequence] = 0;
        } else {
            const auto count = random() % 8 + 1;
            auto pending = expected;
            std::vector<InputToken> batch;
            for (std::size_t i = 0; i < count; ++i) {
                const auto slot = i == 0 ? sequence : random() % 3;
                if (pending[slot] == 17) { continue; }
                batch.push_back({std::int32_t(random() % 101), std::int32_t(pending[slot]++),
                                 std::int32_t(slot), i + 1 == count});
            }
            state.prepare(batch);
            if (random() % 7 == 0) { state.discard_prepared(); }
            else { state.start(); state.commit(); expected = pending; }
        }
        CHECK(std::equal(expected.begin(), expected.end(), state.lengths().begin()));
        CHECK(state.live_tokens() == std::accumulate(expected.begin(), expected.end(), std::size_t{0}));
    }
    state.clear(0);
    state.prepare(std::array<InputToken,1>{{{1,0,0,true}}}); state.start();
    const auto before = lengths(state);
    state.poison();
    test::throws<Error>([&] { state.commit(); });
    test::throws<Error>([&] { state.clear(0); });
    test::throws<Error>([&] { state.discard_prepared(); });
    test::throws<Error>([&] { state.prepare(std::array<InputToken,1>{{{1,0,0,true}}}); });
    CHECK(state.phase() == BatchPhase::poisoned && lengths(state) == before);
}

TEST(kv_store_round_to_even_layers_slots_and_guards) {
    CudaContext context;
    const KvShape shape{2,2,5,2,33};
    const std::size_t width = 66, stride = 71, cache_rows = 40, rows = 3, guard = 4;
    std::vector<std::uint16_t> initial(cache_rows * stride + 2 * guard, 0xa5a5);
    for (std::size_t r = 0; r < cache_rows; ++r) { std::fill_n(initial.begin() + guard + r * stride, width, 0xffff); }
    const std::vector<float> edge{0, -0.0f, std::ldexp(1.0f,-24), -std::ldexp(1.0f,-24),
        std::ldexp(1.0f,-14), 1.0f + std::ldexp(1.0f,-11), 1.0f + 3 * std::ldexp(1.0f,-11),
        65504, -65504, 1e-40f, -1e-40f};
    std::vector<float> key(rows * width), value(rows * width);
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = edge[i % edge.size()]; value[i] = edge[(i * 3 + 1) % edge.size()];
    }
    const std::vector<std::int32_t> slots{1,0,1}, positions{4,0,2};
    DeviceBuffer<std::uint16_t> cache(initial.size());
    DeviceBuffer<float> k(key.size()), v(value.size());
    DeviceBuffer<std::int32_t> s(rows), p(rows), status(2);
    upload(context, matrix_view(cache, 1, initial.size()), initial);
    upload(context, matrix_view(k, rows, width), key); upload(context, matrix_view(v, rows, width), value);
    upload(context, matrix_view(s, rows, 1), slots); upload(context, matrix_view(p, rows, 1), positions);
    DeviceTensorView<std::uint16_t> kv{cache.data() + guard, cache_rows, width, stride, cache.size() - 2 * guard, 0};
    reset_status(context, matrix_view(status, 1, 2));
    store_kv(context, kv, shape, 1, read_only(matrix_view(k, rows, width)), read_only(matrix_view(v, rows, width)),
             read_only(matrix_view(s, rows, 1)), read_only(matrix_view(p, rows, 1)), matrix_view(status, 1, 2));
    status_is(context, status);
    auto expected = initial;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < width; ++c) {
            expected[guard + cache_index(shape, 1, slots[r], 0, positions[r], c, stride)] = float_to_half(key[r * width + c]);
            expected[guard + cache_index(shape, 1, slots[r], 1, positions[r], c, stride)] = float_to_half(value[r * width + c]);
        }
    }
    CHECK(download(context, matrix_view(cache, 1, cache.size())) == expected);
}

TEST(kv_store_device_failure_keeps_logical_lengths_uncommitted) {
    CudaContext context;
    const KvShape shape{1,1,4,1,4};
    DeviceBuffer<std::uint16_t> cache(32);
    DeviceBuffer<float> key(16), value(16);
    DeviceBuffer<std::int32_t> slots(4), positions(4), status(2);
    std::vector<float> input(16, 1.0f);
    input[4] = 65520.0f;
    input[8] = std::numeric_limits<float>::quiet_NaN();
    input[12] = -std::numeric_limits<float>::infinity();
    upload(context, matrix_view(key,4,4), input);
    upload(context, matrix_view(value,4,4), std::vector<float>(16, 2));
    upload(context, matrix_view(slots,4,1), std::vector<std::int32_t>(4,0));
    upload(context, matrix_view(positions,4,1), std::vector<std::int32_t>{0,1,2,3});
    BatchState state({1,4,4,0}, 9);
    state.prepare(std::array<InputToken,4>{{{1,0,0,false},{2,1,0,false},{3,2,0,false},{4,3,0,true}}});
    state.start();
    reset_status(context, matrix_view(status,1,2));
    store_kv(context, matrix_view(cache,8,4), shape, 0, read_only(matrix_view(key,4,4)), read_only(matrix_view(value,4,4)),
             read_only(matrix_view(slots,4,1)), read_only(matrix_view(positions,4,1)), matrix_view(status,1,2));
    status_is(context, status, int(DeviceError::nonfinite), 1);
    state.poison();
    CHECK(state.live_tokens() == 0 && state.phase() == BatchPhase::poisoned);
    test::throws<Error>([&] { state.clear(0); });
    test::throws<Error>([&] { state.commit(); });
    const auto data = download(context, matrix_view(cache,8,4));
    CHECK(data[0] == float_to_half(1.0f));
    CHECK(data[4] == float_to_half(65520.0f));
}

TEST(softmax_standalone_real_width_causal_nan_tail_and_preflight) {
    CudaContext context;
    constexpr std::size_t rows = 4, heads = 16, capacity = 2048, width = heads * capacity, stride = width + 3;
    const KvShape shape{4,28,capacity,8,128};
    const std::vector<std::int32_t> slots{0,1,2,3}, positions{0,16,1535,2047};
    std::vector<float> input(rows * stride, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t h = 0; h < heads; ++h) {
            for (std::size_t p = 0; p <= std::size_t(positions[r]); ++p) {
                input[r*stride+h*capacity+p] = float(int((p*7+h*3+r*11)%67)-33)/4.0f;
            }
        }
    }
    DeviceBuffer<float> scores(input.size()), probabilities(input.size());
    DeviceBuffer<std::int32_t> s(rows), p(rows), status(2);
    upload(context, matrix_view(scores,1,input.size()), input);
    upload(context, matrix_view(probabilities,1,input.size()), std::vector<float>(input.size(), 719.25f));
    upload(context, matrix_view(s,rows,1), slots); upload(context, matrix_view(p,rows,1), positions);
    const DeviceTensorView<float> a{scores.data(),rows,width,stride,scores.size(),0};
    const DeviceTensorView<float> b{probabilities.data(),rows,width,stride,probabilities.size(),0};
    reset_status(context, matrix_view(status,1,2));
    const auto before = allocation_stats();
    causal_softmax(context, shape, heads, read_only(matrix_view(s,rows,1)), read_only(matrix_view(p,rows,1)),
                   capacity, a, b, matrix_view(status,1,2));
    status_is(context, status);
    const auto actual = download(context, matrix_view(probabilities,1,probabilities.size()));
    for (std::size_t r = 0; r < rows; ++r) {
        const auto used = std::size_t(positions[r]) + 1;
        for (std::size_t h = 0; h < heads; ++h) {
            const auto first = input.begin() + r*stride+h*capacity;
            const float maximum = *std::max_element(first, first+used);
            double denominator = 0;
            for (std::size_t i = 0; i < used; ++i) { denominator += std::exp(double(first[i])-maximum); }
            for (std::size_t i = 0; i < capacity; ++i) {
                near(actual[r*stride+h*capacity+i], i < used ? std::exp(double(first[i])-maximum)/denominator : 0.0);
            }
        }
        for (std::size_t i = width; i < stride; ++i) { CHECK(actual[r*stride+i] == 719.25f); }
    }
    const auto observed = download(context, matrix_view(scores,1,scores.size()));
    for (std::size_t i = 0; i < input.size(); ++i) {
        CHECK(std::isnan(input[i]) ? std::isnan(observed[i]) : input[i] == observed[i]);
    }
    test::throws<std::invalid_argument>([&] {
        causal_softmax(context, shape, heads, read_only(matrix_view(s,rows,1)), read_only(matrix_view(p,rows,1)),
                       capacity, a, a, matrix_view(status,1,2));
    });
    test::throws<std::invalid_argument>([&] {
        causal_softmax(context, shape, heads, read_only(matrix_view(s,rows,1)), read_only(matrix_view(p,rows,1)),
                       capacity+1, a, b, matrix_view(status,1,2));
    });
    const auto after = allocation_stats();
    CHECK(after.allocation_calls == before.allocation_calls && after.release_calls == before.release_calls);
}

TEST(softmax_standalone_device_errors_are_explicit) {
    CudaContext context;
    const KvShape shape{1,1,17,1,128};
    DeviceBuffer<float> scores(3*17), probabilities(3*17);
    DeviceBuffer<std::int32_t> slots(3), positions(3), status(2);
    std::vector<float> input(3*17, 1.0f);
    input[0] = std::numeric_limits<float>::quiet_NaN();
    upload(context, matrix_view(scores,3,17), input);
    upload(context, matrix_view(slots,3,1), std::vector<std::int32_t>{0,1,0});
    upload(context, matrix_view(positions,3,1), std::vector<std::int32_t>{0,0,17});
    reset_status(context, matrix_view(status,1,2));
    causal_softmax(context, shape, 1, read_only(matrix_view(slots,3,1)), read_only(matrix_view(positions,3,1)),
                   17, matrix_view(scores,3,17), matrix_view(probabilities,3,17), matrix_view(status,1,2));
    status_is(context, status, 7, 0);
    const auto output = download(context, matrix_view(probabilities,3,17));
    CHECK(std::isnan(output[0]) && std::isnan(output[17]) && std::isnan(output[34]));
}

namespace {
void attention_case(std::size_t capacity, std::size_t length, std::size_t kv_heads,
                     std::size_t heads, std::size_t dimension) {
    CudaContext context;
    const KvShape shape{2,2,capacity,kv_heads,dimension};
    constexpr std::size_t rows = 3, guard = 4;
    const auto width = kv_heads * dimension, q_width = heads * dimension, cache_rows = 8 * capacity;
    const auto cache_stride = width + 3, q_stride = q_width + 5, score_stride = heads * capacity + 7;
    const std::vector<std::int32_t> slots{0,1,0}, positions{0,std::int32_t(length - 1),std::int32_t(length / 2)};
    const auto active = std::max<std::size_t>(length, std::size_t(positions[2]) + 1);
    std::vector<std::uint16_t> kv(cache_rows * cache_stride, 0xffff);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        for (std::size_t p = 0; p < length; ++p) {
            for (std::size_t c = 0; c < width; ++c) {
                const float k = float(int((p * 7 + c * 3 + slot * 13) % 43) - 21) / 31.0f;
                const float v = float(slot * 5 + (c / dimension) * 3) + float((p * 3 + c) % 59) / 31.0f;
                kv[cache_index(shape,1,slot,0,p,c,cache_stride)] = float_to_half(k);
                kv[cache_index(shape,1,slot,1,p,c,cache_stride)] = float_to_half(v);
            }
        }
    }
    std::vector<float> query(rows * q_stride, std::numeric_limits<float>::quiet_NaN());
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < q_width; ++c) { query[r * q_stride + c] = float(int((r * 7 + c * 3) % 37) - 18) / 29.0f; }
    }
    std::vector<float> score(rows * score_stride, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> initial(rows * q_stride + 2 * guard, 719.25f);
    DeviceBuffer<std::uint16_t> cache(kv.size());
    DeviceBuffer<float> q(query.size()), scores(score.size()), probabilities(score.size()), out(initial.size());
    DeviceBuffer<std::int32_t> s(rows), p(rows), status(2);
    upload(context, matrix_view(cache,1,kv.size()), kv); upload(context, matrix_view(q,1,query.size()), query);
    upload(context, matrix_view(scores,1,score.size()), score); upload(context, matrix_view(probabilities,1,score.size()), score);
    upload(context, matrix_view(out,1,initial.size()), initial);
    upload(context, matrix_view(s,rows,1), slots); upload(context, matrix_view(p,rows,1), positions);
    const auto before = allocation_stats();
    reset_status(context, matrix_view(status,1,2));
    causal_attention(context, {cache.data(),cache_rows,width,cache_stride,cache.size(),0}, shape, 1,
        {q.data(),rows,q_width,q_stride,q.size(),0}, heads, read_only(matrix_view(s,rows,1)),
        read_only(matrix_view(p,rows,1)), active,
        {scores.data(),rows,heads*capacity,score_stride,scores.size(),0},
        {probabilities.data(),rows,heads*capacity,score_stride,probabilities.size(),0},
        {out.data()+guard,rows,q_width,q_stride,out.size()-2*guard,0}, matrix_view(status,1,2));
    status_is(context, status);
    const auto actual = download(context, matrix_view(out,1,out.size()));
    const auto probability = download(context, matrix_view(probabilities,1,probabilities.size()));
    const auto observed_scores = download(context, matrix_view(scores,1,scores.size()));
    for (std::size_t i = 0; i < guard; ++i) { CHECK(actual[i] == 719.25f && actual[actual.size()-1-i] == 719.25f); }
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = q_width; c < q_stride; ++c) { CHECK(actual[guard+r*q_stride+c] == 719.25f); }
        const auto used = std::size_t(positions[r]) + 1;
        for (std::size_t h = 0; h < heads; ++h) {
            const auto kh = h / (heads / kv_heads);
            std::vector<float> expected_score(used);
            for (std::size_t pos = 0; pos < used; ++pos) {
                double sum = 0;
                for (std::size_t c = 0; c < dimension; ++c) {
                    sum += double(query[r*q_stride+h*dimension+c]) *
                        half_to_float(kv[cache_index(shape,1,slots[r],0,pos,kh*dimension+c,cache_stride)]);
                }
                expected_score[pos] = float(sum / std::sqrt(double(dimension)));
                near(observed_scores[r*score_stride+h*capacity+pos], expected_score[pos]);
            }
            const float maximum = *std::max_element(expected_score.begin(), expected_score.end());
            double denominator = 0;
            for (auto& value : expected_score) { value = std::exp(value - maximum); denominator += value; }
            for (auto& value : expected_score) { value = float(value / denominator); }
            for (std::size_t pos = 0; pos < active; ++pos) {
                near(probability[r*score_stride+h*capacity+pos], pos < used ? expected_score[pos] : 0.0);
            }
            for (std::size_t pos = active; pos < capacity; ++pos) { CHECK(std::isnan(probability[r*score_stride+h*capacity+pos])); }
            for (std::size_t c = 0; c < dimension; ++c) {
                double sum = 0;
                for (std::size_t pos = 0; pos < used; ++pos) {
                    sum += double(expected_score[pos]) *
                        half_to_float(kv[cache_index(shape,1,slots[r],1,pos,kh*dimension+c,cache_stride)]);
                }
                near(actual[guard+r*q_stride+h*dimension+c], sum);
            }
        }
    }
    const auto after = allocation_stats();
    CHECK(before.allocation_calls == after.allocation_calls && before.release_calls == after.release_calls);
}
}

TEST(attention_causal_gqa_nonwarp_and_long_context) {
    attention_case(1,1,1,1,1);
    attention_case(23,17,1,4,33);
    attention_case(40,33,2,4,128);
    attention_case(2048,1536,8,16,128);
}

TEST(attention_invalid_metadata_nonfinite_and_preflight) {
    CudaContext context;
    const KvShape shape{1,1,4,1,4};
    DeviceBuffer<std::uint16_t> cache(32);
    DeviceBuffer<float> q(8), scores(8), probabilities(8), output(8);
    DeviceBuffer<std::int32_t> slots(2), positions(2), status(2);
    upload(context, matrix_view(cache,8,4), std::vector<std::uint16_t>(32, float_to_half(1)));
    upload(context, matrix_view(q,2,4), std::vector<float>(8,1));
    const auto run = [&](std::vector<std::int32_t> s, std::vector<std::int32_t> p, std::size_t maximum) {
        upload(context, matrix_view(slots,2,1), s); upload(context, matrix_view(positions,2,1), p);
        reset_status(context, matrix_view(status,1,2));
        causal_attention(context, read_only(matrix_view(cache,8,4)), shape, 0, read_only(matrix_view(q,2,4)), 1,
            read_only(matrix_view(slots,2,1)), read_only(matrix_view(positions,2,1)), maximum,
            matrix_view(scores,2,4), matrix_view(probabilities,2,4), matrix_view(output,2,4), matrix_view(status,1,2));
    };
    run({0,1},{0,0},1); status_is(context, status, int(DeviceError::invalid_index), 1);
    run({0,0},{0,4},4); status_is(context, status, int(DeviceError::invalid_position), 1);
    run({0,0},{0,3},2); status_is(context, status, int(DeviceError::invalid_position), 1);
    run({0,0},{-1,0},1); status_is(context, status, int(DeviceError::invalid_position), 0);
    auto bad_cache = std::vector<std::uint16_t>(32, float_to_half(1));
    bad_cache[0] = 0x7e00;
    upload(context, matrix_view(cache,8,4), bad_cache);
    run({0,0},{0,0},1); status_is(context, status, int(DeviceError::nonfinite), 0);
    const auto bad = download(context, matrix_view(output,2,4));
    CHECK(std::all_of(bad.begin(), bad.end(), [](float value) { return std::isnan(value); }));
    test::throws<std::invalid_argument>([&] {
        causal_attention(context, read_only(matrix_view(cache,8,4)), shape, 1, read_only(matrix_view(q,2,4)), 1,
            read_only(matrix_view(slots,2,1)), read_only(matrix_view(positions,2,1)), 1,
            matrix_view(scores,2,4), matrix_view(probabilities,2,4), matrix_view(output,2,4), matrix_view(status,1,2));
    });
    test::throws<std::invalid_argument>([&] {
        causal_attention(context, read_only(matrix_view(cache,8,4)), shape, 0, read_only(matrix_view(q,2,4)), 1,
            read_only(matrix_view(slots,2,1)), read_only(matrix_view(positions,2,1)), 1,
            matrix_view(scores,2,4), matrix_view(scores,2,4), matrix_view(output,2,4), matrix_view(status,1,2));
    });
}

namespace {
class ReferenceLayer {
public:
    ReferenceLayer(const Qwen3Model& model, std::size_t layer, StorageLimits limits)
        : dimensions_(model.dimensions()), weights_(model.layers().at(layer)), limits_(limits),
          k_(limits.max_sequences * limits.max_model_len * dimensions_.kv_heads * dimensions_.head_dim, 0xffff),
          v_(k_.size(), 0xffff) {}
    std::vector<float> forward(std::vector<float> hidden, const std::vector<InputToken>& tokens,
                              const std::map<Workspace, std::vector<float>>* shared_qkv = nullptr) {
        const auto& d = dimensions_;
        const auto rows = tokens.size(), qw = d.heads * d.head_dim, kw = d.kv_heads * d.head_dim;
        const auto normalized = norm(hidden, weights_.attention_norm, rows, d.embedding);
        auto q = multiply(weights_.query, normalized, rows), k = multiply(weights_.key, normalized, rows);
        auto v = multiply(weights_.value, normalized, rows);
        q = norm(q, weights_.query_norm, rows * d.heads, d.head_dim);
        k = norm(k, weights_.key_norm, rows * d.kv_heads, d.head_dim);
        for (std::size_t r = 0; r < rows; ++r) {
            const auto rotate = [&](std::vector<float>& data, std::size_t width) {
                for (std::size_t h = 0; h < width / d.head_dim; ++h) {
                    for (std::size_t j = 0; j < d.head_dim / 2; ++j) {
                        const float angle = float(tokens[r].position) * std::pow(d.rope_base, -2.0f * float(j) / float(d.head_dim));
                        const float cosine = std::cos(angle), sine = std::sin(angle);
                        const auto a = r * width + h * d.head_dim + j, b = a + d.head_dim / 2;
                        const float left = data[a], right = data[b];
                        data[a] = left * cosine - right * sine; data[b] = left * sine + right * cosine;
                    }
                }
            };
            rotate(q,qw); rotate(k,kw);
        }
        checkpoints[Workspace::query] = q; checkpoints[Workspace::key] = k; checkpoints[Workspace::value] = v;
        if (shared_qkv) {
            q = shared_qkv->at(Workspace::query); k = shared_qkv->at(Workspace::key); v = shared_qkv->at(Workspace::value);
        }
        for (std::size_t r = 0; r < rows; ++r) {
            const auto offset = (std::size_t(tokens[r].sequence) * limits_.max_model_len + std::size_t(tokens[r].position)) * kw;
            for (std::size_t c = 0; c < kw; ++c) { k_[offset+c] = float_to_half(k[r*kw+c]); v_[offset+c] = float_to_half(v[r*kw+c]); }
        }
        std::vector<float> attention(rows * qw);
        for (std::size_t r = 0; r < rows; ++r) {
            const auto length = std::size_t(tokens[r].position) + 1, seq_base = std::size_t(tokens[r].sequence) * limits_.max_model_len * kw;
            for (std::size_t h = 0; h < d.heads; ++h) {
                const auto kh = h / (d.heads / d.kv_heads);
                std::vector<float> probability(length);
                for (std::size_t p = 0; p < length; ++p) {
                    double score = 0;
                    for (std::size_t c = 0; c < d.head_dim; ++c) {
                        score += double(q[r*qw+h*d.head_dim+c]) * half_to_float(k_[seq_base+p*kw+kh*d.head_dim+c]);
                    }
                    probability[p] = float(score / std::sqrt(double(d.head_dim)));
                }
                const float maximum = *std::max_element(probability.begin(), probability.end());
                double denominator = 0;
                for (auto& p : probability) { p = std::exp(p - maximum); denominator += p; }
                for (auto& p : probability) { p = float(p / denominator); }
                for (std::size_t c = 0; c < d.head_dim; ++c) {
                    double sum = 0;
                    for (std::size_t p = 0; p < length; ++p) { sum += double(probability[p]) * half_to_float(v_[seq_base+p*kw+kh*d.head_dim+c]); }
                    attention[r*qw+h*d.head_dim+c] = float(sum);
                }
            }
        }
        const auto projected = multiply(weights_.attention_output, attention, rows);
        checkpoints[Workspace::attention] = attention; checkpoints[Workspace::projected] = projected;
        for (std::size_t i = 0; i < hidden.size(); ++i) { hidden[i] += projected[i]; }
        const auto ffn = norm(hidden, weights_.ffn_norm, rows, d.embedding);
        auto gate = multiply(weights_.gate, ffn, rows);
        const auto up = multiply(weights_.up, ffn, rows);
        for (std::size_t i = 0; i < gate.size(); ++i) { gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i]; }
        const auto down = multiply(weights_.down, gate, rows);
        checkpoints[Workspace::gate] = gate; checkpoints[Workspace::up] = up; checkpoints[Workspace::down] = down;
        for (std::size_t i = 0; i < hidden.size(); ++i) { hidden[i] += down[i]; }
        return hidden;
    }
    std::map<Workspace, std::vector<float>> checkpoints;
private:
    std::vector<float> norm(const std::vector<float>& input, const std::vector<float>& weight,
                            std::size_t rows, std::size_t width) {
        std::vector<float> output(input.size());
        for (std::size_t r = 0; r < rows; ++r) {
            double sum = 0;
            for (std::size_t c = 0; c < width; ++c) { sum += double(input[r*width+c]) * input[r*width+c]; }
            const double denominator = std::sqrt(sum / double(width) + dimensions_.rms_epsilon);
            for (std::size_t c = 0; c < width; ++c) { output[r*width+c] = float((double(input[r*width+c]) / denominator) * weight[c]); }
        }
        return output;
    }
    std::vector<float> multiply(const TensorView& w, const std::vector<float>& input, std::size_t rows) {
        std::vector<float> result(rows * w.rows), decoded(w.columns);
        for (std::size_t n = 0; n < w.rows; ++n) {
            decode_row(w.type, w.row(n), decoded.data(), w.columns);
            for (std::size_t r = 0; r < rows; ++r) {
                double sum = 0;
                for (std::size_t k = 0; k < w.columns; ++k) { sum += double(input[r*w.columns+k]) * decoded[k]; }
                result[r*w.rows+n] = float(sum);
            }
        }
        return result;
    }
    ModelDimensions dimensions_;
    const Qwen3Layer& weights_;
    StorageLimits limits_;
    std::vector<std::uint16_t> k_, v_;
};

json layer_batch(const Qwen3Model& model, CudaStorage& storage, LayerExecutor& executor,
                  ReferenceLayer& reference, BatchState& state, std::size_t layer, const std::vector<InputToken>& tokens,
                  ReferenceLayer* shared_reference = nullptr, const json* thresholds = nullptr) {
    CHECK((shared_reference == nullptr) == (thresholds == nullptr));
    const auto summary = state.prepare(tokens);
    const auto width = model.dimensions().embedding;
    std::vector<float> hidden(tokens.size() * width);
    std::vector<std::int32_t> slots, positions;
    for (std::size_t r = 0; r < tokens.size(); ++r) {
        decode_row(model.embeddings().type, model.embeddings().row(std::size_t(tokens[r].token)), hidden.data()+r*width, width);
        slots.push_back(tokens[r].sequence); positions.push_back(tokens[r].position);
    }
    const auto expected = reference.forward(hidden, tokens);
    const auto& context = storage.context();
    state.start();
    upload(context, storage.workspace<float>(Workspace::hidden,tokens.size()), hidden);
    upload(context, storage.workspace<std::int32_t>(Workspace::slots,tokens.size()), slots);
    upload(context, storage.workspace<std::int32_t>(Workspace::positions,tokens.size()), positions);
    reset_status(context, storage.workspace<std::int32_t>(Workspace::status,1));
    const auto before = allocation_stats();
    executor.enqueue(layer,tokens.size(),summary.max_context);
    const auto status = download(context, storage.workspace<std::int32_t>(Workspace::status,1));
    if (status[0]) { state.poison(); throw Error("层执行返回设备错误：" + std::to_string(status[0])); }
    const auto actual = download(context, storage.workspace<float>(Workspace::hidden,tokens.size()));
    const auto after = allocation_stats();
    CHECK(before.allocation_calls == after.allocation_calls && before.release_calls == after.release_calls);
    double squared = 0, maximum = 0, ratio = 0, dot = 0, norm_a = 0, norm_b = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::isfinite(expected[i]));
        const double error = double(actual[i]) - expected[i];
        squared += error * error; maximum = std::max(maximum,std::abs(error));
        ratio = std::max(ratio, std::abs(error) / (2e-4 + 2e-4 * std::abs(double(expected[i]))));
        dot += double(actual[i])*expected[i]; norm_a += double(actual[i])*actual[i]; norm_b += double(expected[i])*expected[i];
    }
    const double rmse = std::sqrt(squared/double(actual.size())), cosine = dot / std::sqrt(norm_a*norm_b);
    json report = {{"layer",layer},{"tokens",tokens.size()},{"max_context",summary.max_context},
        {"rmse",rmse},{"max_absolute",maximum},{"cosine",cosine},{"max_unit_tolerance_ratio",ratio},
        {"project_allocation_calls",0},{"project_release_calls",0}};
    if (shared_reference) {
        std::map<Workspace, std::vector<float>> shared;
        report["stages"] = json::array();
        for (const auto& [id, values] : reference.checkpoints) {
            auto device = download(context, storage.workspace<float>(id,tokens.size()));
            double max_error = 0, sum = 0;
            std::size_t half_differences = 0;
            for (std::size_t i = 0; i < device.size(); ++i) {
                const double e = double(device[i])-values[i];
                max_error = std::max(max_error,std::abs(e)); sum += e*e;
                half_differences += float_to_half(device[i]) != float_to_half(values[i]) ? 1 : 0;
            }
            report["stages"].push_back({{"stage",storage.region(id).name},{"max_absolute",max_error},
                {"rmse",std::sqrt(sum/double(device.size()))},{"fp16_rounding_differences",half_differences}});
            if (id == Workspace::query || id == Workspace::key || id == Workspace::value) {
                for (std::size_t i = 0; i < device.size(); ++i) { near(device[i],values[i]); }
                shared.emplace(id,std::move(device));
            }
        }
        // 独立整层误差与 FP16 边界对照分开；后者复用实际 Q/K/V，其他数学仍由 CPU FP64 计算。
        const auto boundary_expected = shared_reference->forward(hidden,tokens,&shared);
        double boundary_maximum = 0, boundary_squared = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            near(actual[i],boundary_expected[i]);
            const double error = double(actual[i])-boundary_expected[i];
            boundary_maximum = std::max(boundary_maximum,std::abs(error)); boundary_squared += error*error;
        }
        report["shared_qkv"] = {{"atol",2e-4},{"rtol",2e-4},{"max_absolute",boundary_maximum},
            {"rmse",std::sqrt(boundary_squared/double(actual.size()))},{"passed",true}};
        const bool accepted = rmse < thresholds->at("rmse_exclusive").get<double>() &&
            maximum < thresholds->at("max_absolute_exclusive").get<double>() &&
            cosine >= thresholds->at("cosine_min_inclusive").get<double>();
        if (!accepted) { std::cerr << report.dump(2) << '\n'; throw std::runtime_error("独立真实层对照超出数值契约"); }
    } else {
        for (std::size_t i = 0; i < actual.size(); ++i) { near(actual[i],expected[i]); }
    }
    state.commit();
    report["passed"] = true;
    return report;
}
}

TEST(layer_real_execution_interleaved_append_and_clear) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    const StorageLimits limits{2,16,8,0};
    CudaStorage storage(model,limits);
    LayerExecutor executor(storage);
    BatchState state(limits,model.dimensions().vocabulary);
    ReferenceLayer reference(model,0,limits);
    layer_batch(model,storage,executor,reference,state,0,{{1,0,0,false},{2,0,1,true},{3,1,0,true}});
    layer_batch(model,storage,executor,reference,state,0,{{4,2,0,true},{5,1,1,true}});
    CHECK(state.live_tokens() == 5);
    state.clear(0); state.clear(1);
    ReferenceLayer fresh(model,0,limits);
    layer_batch(model,storage,executor,fresh,state,0,{{7,0,0,true},{8,0,1,true}});
    CHECK(state.live_tokens() == 2);
    test::throws<std::invalid_argument>([&] { executor.enqueue(2,1,1); });
    test::throws<std::invalid_argument>([&] { executor.enqueue(0,0,1); });
    test::throws<std::invalid_argument>([&] { executor.enqueue(0,1,17); });
}

namespace {
std::string file_hash(const std::string& path) {
    std::ifstream file(path,std::ios::binary);
    if (!file) { throw std::runtime_error("无法读取模型：" + path); }
    sha256_t state; sha256_init(&state);
    std::array<char,65536> buffer{};
    while (file.read(buffer.data(),buffer.size()) || file.gcount()) {
        sha256_update(&state,reinterpret_cast<const unsigned char*>(buffer.data()),std::size_t(file.gcount()));
    }
    if (!file.eof()) { throw std::runtime_error("模型读取失败"); }
    unsigned char digest[32]; sha256_final(&state,digest);
    std::ostringstream out;
    for (auto byte : digest) { out << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); }
    return out.str();
}
void write_json(const std::filesystem::path& path, const json& report) {
    std::ofstream file(path);
    file << report.dump(2) << '\n'; file.close();
    if (!file) { throw std::runtime_error("写入层报告失败"); }
}
void real_model(const std::string& path, const std::string& contract_path, const std::filesystem::path& output) {
    std::ifstream contract_file(contract_path);
    const auto contract = json::parse(contract_file);
    const auto sha = file_hash(path);
    CHECK(contract.at("schema_version") == 1 && contract.at("model").at("sha256") == sha);
    std::filesystem::copy_file(contract_path,output / "validation-contract.json");
    Qwen3Model model(path);
    CudaStorage storage(model);
    LayerExecutor executor(storage);
    json cases = json::array();
    for (std::size_t layer : {std::size_t{0},model.dimensions().layers-1}) {
        BatchState state(storage.plan().limits,model.dimensions().vocabulary);
        ReferenceLayer reference(model,layer,storage.plan().limits);
        ReferenceLayer shared_reference(model,layer,storage.plan().limits);
        const auto check = [&](const std::vector<InputToken>& tokens) {
            cases.push_back(layer_batch(model,storage,executor,reference,state,layer,tokens,
                                        &shared_reference,&contract.at("thresholds")));
        };
        check({{785,0,1,true},{15592,0,2,true}});
        std::vector<InputToken> mixed;
        for (int p = 0; p < 16; ++p) { mixed.push_back({p % 2 ? 104455 : 14990,p,0,p==15}); }
        mixed.push_back({13598,1,1,true}); mixed.push_back({14324,1,2,true});
        check(mixed);
        check({{315,16,0,true},{323,2,1,true},{629,2,2,true},{198,0,3,true}});
        CHECK(state.live_sequences() == 4 && state.live_tokens() == 24);
    }
    write_json(output / "layer-validation.json",{{"model_sha256",sha},{"thresholds",contract.at("thresholds")},
        {"oracle","CPU FP64；算子边界为 FP32，KV 为 FP16"},{"cases",cases},
        {"boundary_oracle","实际 Q/K/V 共享输入，CPU FP64 attention/FFN；固定 unit_atol/unit_rtol"},
        {"scope","首层与末层使用真实权重及受控输入；不是完整模型验证"}});
    write_json(output / "validation-summary.json",{{"schema_version",1},{"passed",true},
        {"model_sha256",sha},{"layer_cases",cases.size()},{"complete_gpu_model",false}});
}
}

int main(int argc, char** argv) {
    if (argc == 1) { return test::run(); }
    if (argc != 7 || std::string_view(argv[1]) != "--model" || std::string_view(argv[3]) != "--contract" ||
        std::string_view(argv[5]) != "--output") {
        std::cerr << "用法：minillm-cuda-layer-tests [--model MODEL --contract CONTRACT --output NEW_DIRECTORY]\n";
        return 1;
    }
    try {
        const std::string path=argv[2],contract=argv[4];
        const std::filesystem::path output=argv[6];
        if (!output.parent_path().empty()) { std::filesystem::create_directories(output.parent_path()); }
        if (!std::filesystem::create_directory(output)) { throw std::runtime_error("层报告目录必须尚不存在"); }
        write_json(output / "validation-summary.json",{{"status","incomplete"},{"passed",false}});
        test::cases().push_back({"layer_qwen3_matched_weights",[path,contract,output]{real_model(path,contract,output);}});
        const auto result=test::run();
        if (result) { write_json(output / "validation-summary.json",{{"status","failed"},{"passed",false}}); }
        return result;
    } catch (const std::exception& error) {
        std::cerr << "CUDA 层验证失败：" << error.what() << '\n'; return 1;
    }
}
