#include "test_support.h"
#include "minillm/cuda/matrix.h"

#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

using namespace minillm::cuda;

namespace {

struct CountingMemory {
    static inline int calls = 0;
    static inline int allocations = 0;
    static inline int frees = 0;
    static inline int fail_call = 0;
    static void reset() { calls = allocations = frees = fail_call = 0; }
    static void* allocate(std::size_t bytes, int device) {
        ++calls;
        if (calls == fail_call) { throw Error("分配故障注入"); }
        auto* pointer = DeviceMemory::allocate(bytes, device);
        ++allocations;
        return pointer;
    }
    static void release(void* pointer, int device) noexcept {
        ++frees;
        DeviceMemory::release(pointer, device);
    }
};

using CountedBuffer = DeviceBuffer<float, CountingMemory>;
static_assert(!std::is_copy_constructible_v<CountedBuffer>);
static_assert(!std::is_copy_assignable_v<CountedBuffer>);
static_assert(std::is_nothrow_move_constructible_v<CountedBuffer>);
static_assert(std::is_nothrow_move_assignable_v<CountedBuffer>);
static_assert(std::is_nothrow_destructible_v<CountedBuffer>);
static_assert(!std::is_move_constructible_v<CudaContext>);
static_assert(std::is_nothrow_destructible_v<CudaContext>);

__global__ void prepare_input(float* data, int count) {
    const auto i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    if (i < count) { data[i] = static_cast<float>(i - 3) * 0.25f; }
}

void upload(const CudaContext& context, DeviceBuffer<float>& buffer, const std::vector<float>& host) {
    CHECK(buffer.size() == host.size());
    check_cuda(cudaMemcpyAsync(buffer.data(), host.data(), buffer.bytes(),
                              cudaMemcpyHostToDevice, context.stream()), "上传测试矩阵");
}

void run_matrix(std::size_t m, std::size_t n, std::size_t k, bool padded) {
    CudaContext context;
    const auto xs = k + (padded ? 3 : 0);
    const auto ws = k + (padded ? 5 : 0);
    const auto ys = n + (padded ? 7 : 0);
    constexpr std::size_t guard = 4;
    constexpr float sentinel = 731.25f;
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x(m * xs, nan), w(n * ws, nan), y(m * ys + 2 * guard, sentinel);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            x[i * xs + j] = static_cast<float>(static_cast<int>((i * 13 + j * 7) % 41) - 20) / 31.0f;
        }
        for (std::size_t j = 0; j < n; ++j) { y[guard + i * ys + j] = nan; }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            w[i * ws + j] = static_cast<float>(static_cast<int>((i * 17 + j * 3) % 37) - 18) / 29.0f;
        }
    }
    DeviceBuffer<float> dx(x.size()), dw(w.size()), dy(y.size());
    upload(context, dx, x);
    upload(context, dw, w);
    upload(context, dy, y);
    auto output = matrix_view(dy, m, n, ys);
    output.data += guard;
    output.capacity -= 2 * guard;
    matrix_multiply(context, matrix_view(std::as_const(dx), m, k, xs),
                    matrix_view(std::as_const(dw), n, k, ws), output);
    check_cuda(cudaMemcpyAsync(y.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost, context.stream()), "下载测试矩阵");
    context.synchronize();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double expected = 0;
            for (std::size_t p = 0; p < k; ++p) { expected += static_cast<double>(x[i * xs + p]) * w[j * ws + p]; }
            const auto actual = y[guard + i * ys + j];
            CHECK(std::isfinite(actual));
            CHECK(std::abs(static_cast<double>(actual) - expected) <= 2e-4 + 2e-4 * std::abs(expected));
        }
        for (std::size_t j = n; j < ys; ++j) { CHECK(y[guard + i * ys + j] == sentinel); }
    }
    for (std::size_t i = 0; i < guard; ++i) {
        CHECK(y[i] == sentinel);
        CHECK(y[y.size() - 1 - i] == sentinel);
    }
}

} // namespace

TEST(buffer_empty_and_overflow) {
    CountingMemory::reset();
    CountedBuffer empty(0);
    CHECK(empty.data() == nullptr && empty.size() == 0 && empty.device() == -1);
    test::throws<std::overflow_error>([] {
        CountedBuffer invalid(std::numeric_limits<std::size_t>::max() / sizeof(float) + 1);
    });
    test::throws<std::invalid_argument>([] { CountedBuffer invalid(16, -1); });
    CHECK(CountingMemory::calls == 0 && CountingMemory::frees == 0);
}

TEST(buffer_move_and_exact_release) {
    CountingMemory::reset();
    {
        CountedBuffer first(17), second(31);
        auto* original = first.data();
        CountedBuffer moved(std::move(first));
        CHECK(first.data() == nullptr && first.bytes() == 0 && first.device() == -1);
        CHECK(moved.data() == original && moved.size() == 17);
        second = std::move(moved);
        CHECK(CountingMemory::allocations == 2 && CountingMemory::frees == 1);
        CHECK(moved.data() == nullptr && second.data() == original);
        auto& same = second;
        second = std::move(same);
        CHECK(second.data() == original);
        second.reset();
        second.reset();
        CHECK(CountingMemory::frees == 2);
    }
    CHECK(CountingMemory::frees == CountingMemory::allocations);
}

TEST(buffer_allocation_failure_unwinds_owners) {
    CountingMemory::reset();
    CountingMemory::fail_call = 2;
    struct Owners { CountedBuffer first{64}; CountedBuffer second{64}; };
    test::throws<Error>([] { Owners owners; });
    CHECK(CountingMemory::calls == 2 && CountingMemory::allocations == 1 && CountingMemory::frees == 1);
    CountingMemory::fail_call = 0;
    CountedBuffer recovered(64);
    CHECK(recovered.bytes() == 256);
}

TEST(context_stream_workspace_and_math_mode) {
    CudaContext context;
    cudaStream_t stream = nullptr;
    cublasMath_t math{};
    cublasPointerMode_t pointer{};
    cublasAtomicsMode_t atomics{};
    check_cublas(cublasGetStream(context.handle(), &stream), "cublasGetStream");
    check_cublas(cublasGetMathMode(context.handle(), &math), "cublasGetMathMode");
    check_cublas(cublasGetPointerMode(context.handle(), &pointer), "cublasGetPointerMode");
    check_cublas(cublasGetAtomicsMode(context.handle(), &atomics), "cublasGetAtomicsMode");
    CHECK(stream != nullptr && stream == context.stream());
    CHECK(math == CUBLAS_PEDANTIC_MATH && pointer == CUBLAS_POINTER_MODE_HOST);
    CHECK(atomics == CUBLAS_ATOMICS_NOT_ALLOWED);
    CHECK(context.workspace_bytes() == CudaContext::default_workspace_bytes);
    const auto memory = context.memory_info();
    CHECK(memory.total_bytes > 0 && memory.free_bytes <= memory.total_bytes);
    context.synchronize();
}

TEST(context_initialization_failure_cleanup) {
    test::throws<std::invalid_argument>([] { CudaContext invalid(0, 0); });
    test::throws<std::invalid_argument>([] { CudaContext invalid(-1); });
    // stream 和 handle 已创建后触发不可表示的分配范围，验证构造异常清理。
    test::throws<std::length_error>([] { CudaContext invalid(0, std::numeric_limits<std::size_t>::max()); });
    CudaContext recovered;
    recovered.synchronize();
}

TEST(matrix_known_asymmetric_values_and_stream_order) {
    CudaContext context;
    DeviceBuffer<float> dx(6), dw(12), dy(8);
    const std::vector<float> w{1, 2, -3, 4, -5, 6, -7, 8, 9, 10, 11, -12};
    upload(context, dw, w);
    prepare_input<<<1, 32, 0, context.stream()>>>(dx.data(), 6);
    check_cuda(cudaGetLastError(), "prepare_input");
    matrix_multiply(context, matrix_view(std::as_const(dx), 2, 3),
                    matrix_view(std::as_const(dw), 4, 3), matrix_view(dy, 2, 4));
    std::array<float, 8> output{};
    check_cuda(cudaMemcpyAsync(output.data(), dy.data(), dy.bytes(), cudaMemcpyDeviceToHost, context.stream()), "下载已知矩阵");
    context.synchronize();
    const std::array<float, 8> expected{-1.0f, -2.0f, -1.0f, -10.0f, -1.0f, 1.75f, 6.5f, -3.25f};
    CHECK(output == expected);
}

TEST(matrix_padded_rectangular_and_guards) { run_matrix(7, 19, 33, true); }
TEST(matrix_non_warp_aligned_shape) { run_matrix(5, 37, 131, false); }
TEST(matrix_projection_shape) { run_matrix(2, 2048, 1024, false); }

TEST(matrix_preflight_rejects_invalid_descriptors) {
    CudaContext context;
    DeviceBuffer<float> dx(16), dw(16), dy(16);
    const auto x = matrix_view(std::as_const(dx), 2, 3);
    const auto w = matrix_view(std::as_const(dw), 4, 3);
    const auto y = matrix_view(dy, 2, 4);
    auto invalid = x;
    invalid.rows = 0;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    invalid = x; invalid.stride = 2;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    invalid = x; invalid.capacity = 5;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    invalid = x; invalid.data = nullptr;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    invalid = x; invalid.device = 99;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    invalid = x; invalid.columns = 2;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    invalid = x; invalid.stride = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, invalid, w, y); });
    auto bad_output = y; bad_output.capacity = 7;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, x, w, bad_output); });
    bad_output = y; bad_output.data = dx.data() + 1;
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, x, w, bad_output); });
    bad_output = y; bad_output.data = dw.data();
    test::throws<std::invalid_argument>([&] { matrix_multiply(context, x, w, bad_output); });
    std::vector<float> zero(16, 0.0f);
    upload(context, dx, zero);
    upload(context, dw, zero);
    matrix_multiply(context, x, w, y);
    context.synchronize();
}

TEST(error_checks_retain_vendor_diagnostic) {
    try { check_cuda(cudaErrorMemoryAllocation, "allocation_fixture"); CHECK(false); }
    catch (const Error& error) { CHECK(std::string(error.what()).find("cudaErrorMemoryAllocation") != std::string::npos); }
    try { check_cublas(CUBLAS_STATUS_ALLOC_FAILED, "cublas_fixture"); CHECK(false); }
    catch (const Error& error) { CHECK(std::string(error.what()).find("CUBLAS_STATUS_ALLOC_FAILED") != std::string::npos); }
}

int main() {
    try {
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "CUDA 单元设备：" << properties.name << "，计算能力 "
                  << properties.major << '.' << properties.minor << '\n';
        return test::run();
    } catch (const std::exception& error) {
        std::cerr << "CUDA 单元环境失败：" << error.what() << '\n';
        return 1;
    }
}
