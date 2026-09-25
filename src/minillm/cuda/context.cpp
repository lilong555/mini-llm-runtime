#include "minillm/cuda/context.h"

#include <cstdio>
#include <limits>

namespace minillm::cuda {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw Error(std::string(operation) + ": " + cudaGetErrorName(status) + " / " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw Error(std::string(operation) + ": " + cublasGetStatusName(status) + " / " + cublasGetStatusString(status));
    }
}

void report_cuda(cudaError_t status, const char* operation) noexcept {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA 资源清理失败：%s: %s\n", operation, cudaGetErrorString(status));
    }
}

void report_cublas(cublasStatus_t status, const char* operation) noexcept {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS 资源清理失败：%s: %s\n", operation, cublasGetStatusString(status));
    }
}

std::size_t checked_product(std::size_t count, std::size_t width) {
    if (width != 0 && count > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("CUDA buffer 元素数量或字节数溢出");
    }
    return count * width;
}

DeviceScope::DeviceScope(int device) {
    if (device < 0) { throw std::invalid_argument("CUDA 设备编号不能为负数"); }
    check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device) {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        changed_ = true;
    }
}

DeviceScope::~DeviceScope() {
    if (changed_) { report_cuda(cudaSetDevice(previous_), "cudaSetDevice 恢复设备"); }
}

void* DeviceMemory::allocate(std::size_t bytes, int device) {
    if (bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::length_error("CUDA 分配范围超过 PTRDIFF_MAX");
    }
    DeviceScope scope(device);
    void* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
    return pointer;
}

void DeviceMemory::release(void* pointer, int device) noexcept {
    if (!pointer) { return; }
    try {
        DeviceScope scope(device);
        report_cuda(cudaFree(pointer), "cudaFree");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA buffer 清理无法选择设备：%s\n", error.what());
    }
}

CudaContext::CudaContext(int device, std::size_t workspace_bytes) : device_(device) {
    if (workspace_bytes < 16 * 1024) {
        throw std::invalid_argument("cuBLAS workspace 至少需要 16 KiB");
    }
    DeviceScope scope(device_);
    try {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        check_cublas(cublasSetStream(handle_, stream_), "cublasSetStream");
        check_cublas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        check_cublas(cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
        workspace_ = DeviceBuffer<std::byte>(workspace_bytes, device_);
        // cublasSetStream 会重置 workspace；绑定顺序固定为 stream 后 workspace。
        check_cublas(cublasSetWorkspace(handle_, workspace_.data(), workspace_.bytes()), "cublasSetWorkspace");
    } catch (...) {
        cleanup();
        throw;
    }
}

CudaContext::~CudaContext() { cleanup(); }

void CudaContext::cleanup() noexcept {
    try {
        DeviceScope scope(device_);
        if (stream_) { report_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize 析构"); }
        if (handle_) { report_cublas(cublasDestroy(handle_), "cublasDestroy"); }
        handle_ = nullptr;
        workspace_.reset();
        if (stream_) { report_cuda(cudaStreamDestroy(stream_), "cudaStreamDestroy"); }
        stream_ = nullptr;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA context 清理无法选择设备：%s\n", error.what());
    }
}

void CudaContext::synchronize() const {
    DeviceScope scope(device_);
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

MemoryInfo CudaContext::memory_info() const {
    DeviceScope scope(device_);
    MemoryInfo result{};
    check_cuda(cudaMemGetInfo(&result.free_bytes, &result.total_bytes), "cudaMemGetInfo");
    return result;
}

} // namespace minillm::cuda
