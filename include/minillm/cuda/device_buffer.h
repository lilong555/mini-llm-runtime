#pragma once

#include "minillm/cuda/error.h"

#include <type_traits>
#include <utility>

namespace minillm::cuda {

struct DeviceMemory {
    static void* allocate(std::size_t bytes, int device);
    static void release(void* pointer, int device) noexcept;
};

// Memory 参数用于资源故障测试；正式路径固定使用 cudaMalloc/cudaFree。
// reset、移动赋值和析构前，调用方必须已检查所有使用该 buffer 的工作完成。
template<class T, class Memory = DeviceMemory>
class DeviceBuffer {
    static_assert(std::is_trivially_copyable_v<T> && !std::is_const_v<T>);
    static_assert(alignof(T) <= 256);
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t count, int device = 0) {
        const auto bytes = checked_product(count, sizeof(T));
        if (device < 0) { throw std::invalid_argument("CUDA 设备编号不能为负数"); }
        if (count == 0) { return; }
        auto* pointer = static_cast<T*>(Memory::allocate(bytes, device));
        if (!pointer) { throw Error("CUDA 分配器返回空指针"); }
        data_ = pointer;
        size_ = count;
        device_ = device;
    }
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }
    void reset() noexcept {
        if (data_) { Memory::release(data_, device_); }
        data_ = nullptr;
        size_ = 0;
        device_ = -1;
    }
    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(device_, other.device_);
    }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    int device() const noexcept { return device_; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
    int device_ = -1;
};

} // namespace minillm::cuda
