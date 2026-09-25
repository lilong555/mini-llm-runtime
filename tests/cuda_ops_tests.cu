#include "test_support.h"
#include "ops.h"
#include "tensor_validation.h"

#include <algorithm>
#include <array>
#include <climits>
#include <limits>
#include <span>

using namespace minillm::cuda;
using minillm::cuda::detail::read_only;

namespace {
constexpr float guard_value = 913.25f;
constexpr std::size_t guard = 4;

template<class T>
void upload(const CudaContext& context, DeviceBuffer<T>& device, const std::vector<T>& values) {
    CHECK(device.size() == values.size());
    check_cuda(cudaMemcpyAsync(device.data(), values.data(), device.bytes(), cudaMemcpyHostToDevice,
                               context.stream()), "上传算子测试输入");
    context.synchronize();
}
template<class T>
std::vector<T> download(const CudaContext& context, const DeviceBuffer<T>& device) {
    std::vector<T> values(device.size());
    check_cuda(cudaMemcpyAsync(values.data(), device.data(), device.bytes(), cudaMemcpyDeviceToHost,
                               context.stream()), "下载算子测试输出");
    context.synchronize();
    return values;
}
template<class T>
DeviceTensorView<T> padded(DeviceBuffer<T>& buffer, std::size_t rows, std::size_t columns, std::size_t stride) {
    return {buffer.data() + guard, rows, columns, stride, buffer.size() - 2 * guard, buffer.device()};
}
void near(float actual, double expected) {
    CHECK(std::isfinite(actual) && std::isfinite(expected));
    CHECK(std::abs(double(actual) - expected) <= 2e-4 + 2e-4 * std::abs(expected));
}
void check_guards(const std::vector<float>& values, std::size_t rows, std::size_t columns, std::size_t stride) {
    for (std::size_t i = 0; i < guard; ++i) {
        CHECK(values[i] == guard_value && values[values.size() - 1 - i] == guard_value);
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t i = columns; i < stride; ++i) { CHECK(values[guard + row * stride + i] == guard_value); }
    }
}
void check_status(const CudaContext& context, const DeviceBuffer<std::int32_t>& status,
                  int error = 0, int row = INT_MAX) {
    const auto values = download(context, status);
    CHECK(values[0] == error && values[1] == row);
}
std::vector<float> coefficients(std::size_t length, std::size_t dimension) {
    std::vector<float> table(length * dimension);
    for (std::size_t p = 0; p < length; ++p) {
        for (std::size_t i = 0; i < dimension / 2; ++i) {
            const float frequency = std::pow(1000000.0f, -2.0f * float(i) / float(dimension));
            const float angle = float(p) * frequency;
            table[p * dimension + i] = std::cos(angle);
            table[p * dimension + i + dimension / 2] = std::sin(angle);
        }
    }
    return table;
}
void reference_norm(std::span<float> input, std::span<const float> weight, float epsilon) {
    double sum = 0;
    for (float value : input) { sum += double(value) * value; }
    const double denominator = std::sqrt(sum / double(input.size()) + epsilon);
    for (std::size_t i = 0; i < input.size(); ++i) { input[i] = float((double(input[i]) / denominator) * weight[i]); }
}
}

TEST(ops_gather_strides_duplicates_and_real_width) {
    CudaContext context;
    DeviceBuffer<std::int32_t> status(2);
    for (std::size_t columns : {37, 1024}) {
        constexpr std::size_t rows = 18, source_rows = 7;
        const auto xs = columns + 3, ys = columns + 5;
        std::vector<float> input(source_rows * xs, std::numeric_limits<float>::quiet_NaN());
        for (std::size_t r = 0; r < source_rows; ++r) {
            for (std::size_t c = 0; c < columns; ++c) { input[r * xs + c] = float(r * 19 + c) / 31.0f; }
        }
        std::vector<std::int32_t> indices(rows * 2, INT_MAX);
        for (std::size_t i = 0; i < rows; ++i) { indices[i * 2] = std::int32_t((i * 3 + 6) % source_rows); }
        std::vector<float> output(rows * ys + 2 * guard, guard_value);
        DeviceBuffer<float> x(input.size()), y(output.size());
        DeviceBuffer<std::int32_t> ids(indices.size());
        upload(context, x, input); upload(context, y, output); upload(context, ids, indices);
        reset_status(context, matrix_view(status, 1, 2));
        gather_rows(context, read_only(matrix_view(x, source_rows, columns, xs)),
                    read_only(matrix_view(ids, rows, 1, 2)), padded(y, rows, columns, ys), matrix_view(status, 1, 2));
        output = download(context, y);
        check_guards(output, rows, columns, ys);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < columns; ++c) {
                CHECK(output[guard + r * ys + c] == input[std::size_t(indices[r * 2]) * xs + c]);
            }
        }
        check_status(context, status);
    }
}

TEST(ops_gather_masks_invalid_indices_and_accumulates_status) {
    CudaContext context;
    DeviceBuffer<float> source(3 * 33), output(4 * 33);
    DeviceBuffer<std::int32_t> indices(4), status(2);
    upload(context, source, std::vector<float>(source.size(), 0.75f));
    upload(context, indices, std::vector<std::int32_t>{2, -1, 3, 0});
    reset_status(context, matrix_view(status, 1, 2));
    for (int i = 0; i < 2; ++i) {
        gather_rows(context, read_only(matrix_view(source, 3, 33)), read_only(matrix_view(indices, 4, 1)),
                    matrix_view(output, 4, 33), matrix_view(status, 1, 2));
    }
    const auto result = download(context, output);
    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 33; ++c) {
            if (r == 1 || r == 2) { CHECK(std::isnan(result[r * 33 + c])); }
            else { CHECK(result[r * 33 + c] == 0.75f); }
        }
    }
    check_status(context, status, int(DeviceError::invalid_index), 1);
    reset_status(context, matrix_view(status, 1, 2));
    check_status(context, status);
}

TEST(ops_rms_norm_groups_tails_and_exact_inplace) {
    CudaContext context;
    for (std::size_t width : {1, 17, 31, 32, 33, 128, 1024, 3072}) {
        const std::size_t groups = width == 128 ? 16 : 2, rows = 3;
        const auto columns = groups * width, stride = columns + 7;
        std::vector<float> input(rows * stride + 2 * guard, guard_value), weight(width);
        for (std::size_t i = 0; i < width; ++i) { weight[i] = float(int(i % 17) - 8) / 7.0f; }
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < columns; ++c) {
                input[guard + r * stride + c] = float(int((r * 23 + c * 7) % 41) - 20) / 19.0f;
            }
        }
        auto expected = input;
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t g = 0; g < groups; ++g) {
                reference_norm({expected.data() + guard + r * stride + g * width, width}, weight, 1e-6f);
            }
        }
        DeviceBuffer<float> x(input.size()), y(input.size()), w(width);
        upload(context, w, weight);
        for (bool inplace : {false, true}) {
            upload(context, x, input);
            upload(context, y, std::vector<float>(input.size(), guard_value));
            auto& destination = inplace ? x : y;
            rms_norm(context, read_only(padded(x, rows, columns, stride)), read_only(matrix_view(w, 1, width)),
                     padded(destination, rows, columns, stride), 1e-6f);
            const auto actual = download(context, destination);
            check_guards(actual, rows, columns, stride);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < columns; ++c) { near(actual[guard + r * stride + c], expected[guard + r * stride + c]); }
            }
        }
    }
}

TEST(ops_rms_norm_finite_extremes_and_epsilon) {
    CudaContext context;
    constexpr std::size_t width = 257, rows = 6;
    const float maximum = std::numeric_limits<float>::max();
    const std::array<float, rows> amplitudes{0, 1e-30f, 1e-15f, 1e15f, 1e30f, maximum};
    std::vector<float> input(rows * width), weight(width, 1.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < width; ++c) {
            input[r * width + c] = c % 3 == 0 ? -amplitudes[r] : amplitudes[r] * 0.5f;
        }
    }
    DeviceBuffer<float> x(input.size()), y(input.size()), w(width);
    upload(context, x, input); upload(context, w, weight);
    for (float epsilon : {std::numeric_limits<float>::denorm_min(), 1e-6f, maximum}) {
        rms_norm(context, read_only(matrix_view(x, rows, width)), read_only(matrix_view(w, 1, width)),
                 matrix_view(y, rows, width), epsilon);
        const auto actual = download(context, y);
        auto expected = input;
        for (std::size_t r = 0; r < rows; ++r) { reference_norm({expected.data() + r * width, width}, weight, epsilon); }
        for (std::size_t i = 0; i < actual.size(); ++i) { near(actual[i], expected[i]); }
    }
}

TEST(ops_rms_norm_nonfinite_cannot_become_valid_token) {
    CudaContext context;
    constexpr std::size_t rows = 4, width = 128;
    std::vector<float> input(rows * width, 0.0f);
    input[width + 17] = std::numeric_limits<float>::quiet_NaN();
    input[2 * width + 31] = std::numeric_limits<float>::infinity();
    input[3 * width + 127] = -std::numeric_limits<float>::infinity();
    DeviceBuffer<float> x(input.size()), w(width);
    DeviceBuffer<std::int32_t> tokens(rows), status(2);
    upload(context, x, input); upload(context, w, std::vector<float>(width, 1.0f));
    reset_status(context, matrix_view(status, 1, 2));
    rms_norm(context, read_only(matrix_view(x, rows, width)), read_only(matrix_view(w, 1, width)),
             matrix_view(x, rows, width), 1e-6f);
    check_finite(context, read_only(matrix_view(x, rows, width)), matrix_view(status, 1, 2));
    check_status(context, status, int(DeviceError::nonfinite), 1);
    argmax(context, read_only(matrix_view(x, rows, width)), matrix_view(tokens, rows, 1), matrix_view(status, 1, 2));
    CHECK(download(context, tokens) == (std::vector<std::int32_t>{0, -1, -1, -1}));
    check_status(context, status, int(DeviceError::nonfinite), 1);
}

TEST(ops_rope_neox_positions_heads_and_padding) {
    CudaContext context;
    constexpr std::size_t length = 2048, dimension = 128, rows = 7;
    const std::vector<std::int32_t> positions{0, 1, 15, 16, 17, 1535, 2047};
    const auto table = coefficients(length, dimension);
    DeviceBuffer<float> coefficient(table.size());
    DeviceBuffer<std::int32_t> p(rows), status(2);
    upload(context, coefficient, table); upload(context, p, positions);
    for (std::size_t heads : {1, 8, 16}) {
        const auto columns = heads * dimension, stride = columns + 9;
        std::vector<float> input(rows * stride + 2 * guard, guard_value);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < columns; ++c) { input[guard + r * stride + c] = float(int((r + c * 3) % 71) - 35) / 23.0f; }
        }
        DeviceBuffer<float> x(input.size());
        upload(context, x, input);
        reset_status(context, matrix_view(status, 1, 2));
        rope(context, padded(x, rows, columns, stride), read_only(matrix_view(p, rows, 1)),
             read_only(matrix_view(coefficient, length, dimension)), matrix_view(status, 1, 2));
        const auto actual = download(context, x);
        check_guards(actual, rows, columns, stride);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t h = 0; h < heads; ++h) {
                for (std::size_t c = 0; c < dimension / 2; ++c) {
                    const auto index = guard + r * stride + h * dimension + c;
                    const double left = input[index], right = input[index + dimension / 2];
                    const double cosine = table[std::size_t(positions[r]) * dimension + c];
                    const double sine = table[std::size_t(positions[r]) * dimension + c + dimension / 2];
                    near(actual[index], left * cosine - right * sine);
                    near(actual[index + dimension / 2], left * sine + right * cosine);
                }
            }
        }
        check_status(context, status);
    }
}

TEST(ops_rope_invalid_positions_are_masked) {
    CudaContext context;
    DeviceBuffer<float> x(4 * 256), table(16 * 128);
    DeviceBuffer<std::int32_t> positions(4), status(2);
    upload(context, x, std::vector<float>(x.size(), 1.0f));
    upload(context, table, coefficients(16, 128));
    upload(context, positions, std::vector<std::int32_t>{0, -1, 16, 15});
    reset_status(context, matrix_view(status, 1, 2));
    rope(context, matrix_view(x, 4, 256), read_only(matrix_view(positions, 4, 1)),
         read_only(matrix_view(table, 16, 128)), matrix_view(status, 1, 2));
    const auto actual = download(context, x);
    for (std::size_t i = 256; i < 3 * 256; ++i) { CHECK(std::isnan(actual[i])); }
    for (std::size_t i = 0; i < 256; ++i) { CHECK(actual[i] == 1.0f); }
    check_status(context, status, int(DeviceError::invalid_position), 1);
}

TEST(ops_residual_swiglu_strides_tails_and_extreme_gates) {
    CudaContext context;
    for (std::size_t columns : {1, 33, 1024, 3072}) {
        constexpr std::size_t rows = 3;
        const auto ls = columns + 3, rs = columns + 7;
        std::vector<float> left(rows * ls + 2 * guard, guard_value), right(rows * rs, std::numeric_limits<float>::quiet_NaN());
        const std::array<float, 9> values{0, 1, -1, 80, -80, 1e4f, -1e4f, 1e30f, -1e30f};
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < columns; ++c) {
                left[guard + r * ls + c] = values[(r + c) % values.size()];
                right[r * rs + c] = float(int(c % 13) - 6) / 17.0f;
            }
        }
        DeviceBuffer<float> x(left.size()), y(right.size());
        upload(context, y, right);
        for (bool activation : {false, true}) {
            upload(context, x, left);
            if (activation) { swiglu(context, padded(x, rows, columns, ls), read_only(matrix_view(y, rows, columns, rs))); }
            else { residual_add(context, padded(x, rows, columns, ls), read_only(matrix_view(y, rows, columns, rs))); }
            const auto actual = download(context, x);
            check_guards(actual, rows, columns, ls);
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < columns; ++c) {
                    const double a = left[guard + r * ls + c], b = right[r * rs + c];
                    const double expected = activation ? (a / (1.0 + std::exp(-a))) * b : a + b;
                    near(actual[guard + r * ls + c], expected);
                }
            }
        }
    }
}

TEST(ops_argmax_ties_nonfinite_and_real_vocabulary) {
    CudaContext context;
    constexpr std::size_t rows = 6;
    DeviceBuffer<std::int32_t> ids(rows * 3 + 2 * guard), status(2);
    for (std::size_t columns : {1, 31, 32, 33, 257, 151936}) {
        const auto stride = columns + 7;
        std::vector<float> input(rows * stride, std::numeric_limits<float>::quiet_NaN());
        for (std::size_t r = 0; r < rows; ++r) { std::fill_n(input.begin() + r * stride, columns, -7.0f); }
        const auto first = std::min<std::size_t>(1, columns - 1), later = std::min<std::size_t>(256, columns - 1);
        input[stride + first] = input[stride + later] = 3.0f;
        input[2 * stride + columns - 1] = 100.0f;
        input[3 * stride + columns - 1] = std::numeric_limits<float>::quiet_NaN();
        input[4 * stride] = std::numeric_limits<float>::infinity();
        input[5 * stride + columns / 2] = -std::numeric_limits<float>::infinity();
        DeviceBuffer<float> x(input.size());
        upload(context, x, input);
        upload(context, ids, std::vector<std::int32_t>(ids.size(), -559));
        reset_status(context, matrix_view(status, 1, 2));
        argmax(context, read_only(matrix_view(x, rows, columns, stride)), padded(ids, rows, 1, 3), matrix_view(status, 1, 2));
        const auto result = download(context, ids);
        const std::array<std::int32_t, rows> expected{0, std::int32_t(first), std::int32_t(columns - 1), -1, -1, -1};
        for (std::size_t i = 0; i < result.size(); ++i) {
            if (i >= guard && i < guard + rows * 3 && (i - guard) % 3 == 0) { CHECK(result[i] == expected[(i - guard) / 3]); }
            else { CHECK(result[i] == -559); }
        }
        check_status(context, status, int(DeviceError::nonfinite), 3);
    }
}

TEST(ops_preflight_rejects_descriptors_before_any_write) {
    CudaContext context;
    DeviceBuffer<float> x(256), y(256), w(256);
    DeviceBuffer<std::int32_t> ids(32), status(2);
    upload(context, x, std::vector<float>(256, 1.0f));
    upload(context, y, std::vector<float>(256, guard_value));
    upload(context, w, std::vector<float>(256, 1.0f));
    upload(context, ids, std::vector<std::int32_t>(32, 0));
    reset_status(context, matrix_view(status, 1, 2));
    const auto a = read_only(matrix_view(x, 2, 33)), b = read_only(matrix_view(w, 1, 33));
    const auto out = matrix_view(y, 2, 33);
    const auto index = read_only(matrix_view(ids, 2, 1));
    const auto error = matrix_view(status, 1, 2);
    const auto before = allocation_stats();
    for (float epsilon : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        test::throws<std::invalid_argument>([&] { rms_norm(context, a, b, out, epsilon); });
    }
    auto bad = a;
    bad.rows = 0;
    test::throws<std::invalid_argument>([&] { rms_norm(context, bad, b, out, 1e-6f); });
    bad = a; bad.capacity = 1;
    test::throws<std::invalid_argument>([&] { rms_norm(context, bad, b, out, 1e-6f); });
    bad = a; bad.device = 99;
    test::throws<std::invalid_argument>([&] { rms_norm(context, bad, b, out, 1e-6f); });
    bad = a; bad.stride = std::size_t(INT_MAX) + 1;
    test::throws<std::invalid_argument>([&] { rms_norm(context, bad, b, out, 1e-6f); });
    bad = a; bad.data = reinterpret_cast<const float*>(reinterpret_cast<const char*>(a.data) + 1);
    test::throws<std::invalid_argument>([&] { rms_norm(context, bad, b, out, 1e-6f); });
    auto overlap = matrix_view(x, 2, 33);
    ++overlap.data; --overlap.capacity;
    test::throws<std::invalid_argument>([&] { rms_norm(context, a, b, overlap, 1e-6f); });
    test::throws<std::invalid_argument>([&] { rms_norm(context, a, read_only(matrix_view(w, 1, 7)), out, 1e-6f); });
    test::throws<std::invalid_argument>([&] { gather_rows(context, a, index, matrix_view(x, 2, 33), error); });
    test::throws<std::invalid_argument>([&] { gather_rows(context, a, index, out, matrix_view(ids, 1, 2)); });
    test::throws<std::invalid_argument>([&] { rope(context, out, index, b, error); });
    test::throws<std::invalid_argument>([&] { rope(context, matrix_view(x, 2, 32), index, read_only(matrix_view(x, 4, 8)), error); });
    test::throws<std::invalid_argument>([&] { residual_add(context, overlap, a); });
    test::throws<std::invalid_argument>([&] { swiglu(context, out, read_only(matrix_view(x, 1, 33))); });
    test::throws<std::invalid_argument>([&] { argmax(context, a, matrix_view(status, 2, 1), error); });
    test::throws<std::invalid_argument>([&] {
        check_finite(context, {reinterpret_cast<const float*>(status.data()), 1, 2, 2, 2, status.device()}, error);
    });
    test::throws<std::invalid_argument>([&] { reset_status(context, matrix_view(status, 2, 1)); });
    CHECK(download(context, y) == std::vector<float>(256, guard_value));
    check_status(context, status);
    const auto after = allocation_stats();
    CHECK(before.allocation_calls == after.allocation_calls && before.release_calls == after.release_calls);
}

TEST(ops_composed_device_pipeline_reuses_stream_and_buffers) {
    CudaContext context;
    constexpr std::size_t rows = 3, width = 4, q_width = 8;
    std::vector<float> embedding(7 * width), projection(q_width * width), norm(width), residual(rows * q_width, 0.125f);
    for (std::size_t i = 0; i < embedding.size(); ++i) { embedding[i] = float(int(i * 3 % 17) - 8) / 7.0f; }
    for (std::size_t i = 0; i < projection.size(); ++i) { projection[i] = float(int(i * 7 % 23) - 11) / 13.0f; }
    for (std::size_t i = 0; i < norm.size(); ++i) { norm[i] = float(i + 3) / 7.0f; }
    const std::vector<std::int32_t> indices{6, 1, 3}, positions{0, 7, 15};
    const auto table = coefficients(16, width);
    DeviceBuffer<float> emb(embedding.size()), proj(projection.size()), weights(width), coefficients_device(table.size()),
        hidden(rows * width), query(rows * q_width), addend(residual.size());
    DeviceBuffer<std::int32_t> ids(rows), p(rows), tokens(rows), status(2);
    upload(context, emb, embedding); upload(context, proj, projection); upload(context, weights, norm);
    upload(context, coefficients_device, table); upload(context, addend, residual);
    upload(context, ids, indices); upload(context, p, positions);
    const auto h = matrix_view(hidden, rows, width), q = matrix_view(query, rows, q_width);
    const auto n = read_only(matrix_view(weights, 1, width));
    const auto error = matrix_view(status, 1, 2);
    const auto before = allocation_stats();
    reset_status(context, error);
    gather_rows(context, read_only(matrix_view(emb, 7, width)), read_only(matrix_view(ids, rows, 1)), h, error);
    rms_norm(context, read_only(h), n, h, 1e-6f);
    matrix_multiply(context, read_only(h), read_only(matrix_view(proj, q_width, width)), q);
    rms_norm(context, read_only(q), n, q, 1e-6f);
    rope(context, q, read_only(matrix_view(p, rows, 1)), read_only(matrix_view(coefficients_device, 16, width)), error);
    residual_add(context, q, read_only(matrix_view(addend, rows, q_width)));
    swiglu(context, q, read_only(matrix_view(addend, rows, q_width)));
    argmax(context, read_only(q), matrix_view(tokens, rows, 1), error);
    const auto actual = download(context, query);
    const auto selected = download(context, tokens);
    check_status(context, status);
    for (std::size_t r = 0; r < rows; ++r) {
        std::vector<float> host_hidden(width), expected(q_width);
        std::copy_n(embedding.begin() + std::size_t(indices[r]) * width, width, host_hidden.begin());
        reference_norm(host_hidden, norm, 1e-6f);
        for (std::size_t nrow = 0; nrow < q_width; ++nrow) {
            double sum = 0;
            for (std::size_t c = 0; c < width; ++c) { sum += double(host_hidden[c]) * projection[nrow * width + c]; }
            expected[nrow] = float(sum);
        }
        for (std::size_t head = 0; head < 2; ++head) {
            reference_norm({expected.data() + head * width, width}, norm, 1e-6f);
            for (std::size_t i = 0; i < width / 2; ++i) {
                const auto offset = head * width + i;
                const float a = expected[offset], b = expected[offset + width / 2];
                const float cosine = table[std::size_t(positions[r]) * width + i];
                const float sine = table[std::size_t(positions[r]) * width + i + width / 2];
                expected[offset] = a * cosine - b * sine;
                expected[offset + width / 2] = a * sine + b * cosine;
            }
        }
        for (std::size_t i = 0; i < q_width; ++i) {
            const float a = expected[i] + residual[r * q_width + i];
            expected[i] = (a / (1.0f + std::exp(-a))) * residual[r * q_width + i];
            near(actual[r * q_width + i], expected[i]);
        }
        CHECK(selected[r] == std::distance(expected.begin(), std::max_element(expected.begin(), expected.end())));
    }
    const auto after = allocation_stats();
    CHECK(before.allocation_calls == after.allocation_calls && before.release_calls == after.release_calls);
}

int main() { return test::run(); }
