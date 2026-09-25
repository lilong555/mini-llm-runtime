#pragma once

#include "minillm/cuda/device_buffer.h"

namespace minillm::cuda {

struct MemoryInfo {
    std::size_t free_bytes;
    std::size_t total_bytes;
};

// 单调用者、不可重入。外部 buffer 的生命周期必须覆盖所有入队工作。
class CudaContext {
public:
    static constexpr std::size_t default_workspace_bytes = 4 * 1024 * 1024;
    explicit CudaContext(int device = 0, std::size_t workspace_bytes = default_workspace_bytes);
    ~CudaContext();
    CudaContext(const CudaContext&) = delete;
    CudaContext& operator=(const CudaContext&) = delete;
    int device() const noexcept { return device_; }
    cudaStream_t stream() const noexcept { return stream_; }
    // 调用方不得改变 handle 的 stream、workspace、pointer mode 或算术模式。
    cublasHandle_t handle() const noexcept { return handle_; }
    std::size_t workspace_bytes() const noexcept { return workspace_.bytes(); }
    MemoryInfo memory_info() const;
    void synchronize() const;

private:
    void cleanup() noexcept;
    int device_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t handle_ = nullptr;
    DeviceBuffer<std::byte> workspace_;
};

} // namespace minillm::cuda
