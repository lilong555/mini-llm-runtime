#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA 错误：%s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void add(const float* a, const float* b, float* output, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) output[i] = a[i] + b[i];
}

int main() {
    // 非整块长度用于验证最后一个线程块的边界。
    constexpr int count = (1 << 20) + 17;
    constexpr int launches = 64;
    const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    std::vector<float> a(count), b(count), output(count);
    for (int i = 0; i < count; ++i) {
        a[i] = static_cast<float>(i % 1024) * 0.25f;
        b[i] = static_cast<float>(i % 127) * 0.5f;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_output = nullptr;
    check(cudaMalloc(&device_a, bytes));
    check(cudaMalloc(&device_b, bytes));
    check(cudaMalloc(&device_output, bytes));
    check(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < launches; ++i) {
        add<<<(count + 255) / 256, 256>>>(device_a, device_b, device_output, count);
        check(cudaGetLastError());
    }
    check(cudaDeviceSynchronize());
    check(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_a));
    check(cudaFree(device_b));
    check(cudaFree(device_output));

    for (int i = 0; i < count; ++i) {
        if (output[i] != a[i] + b[i]) {
            std::fprintf(stderr, "结果不匹配：索引 %d，实际 %g，预期 %g\n",
                         i, output[i], a[i] + b[i]);
            return 2;
        }
    }
    std::printf("GPU=%s，计算能力=%d.%d，%d 次 kernel，%d 个结果校验通过\n",
                properties.name, properties.major, properties.minor, launches, count);
}
