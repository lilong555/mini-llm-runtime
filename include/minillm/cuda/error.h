#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace minillm::cuda {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

void check_cuda(cudaError_t status, const char* operation);
void check_cublas(cublasStatus_t status, const char* operation);
void report_cuda(cudaError_t status, const char* operation) noexcept;
void report_cublas(cublasStatus_t status, const char* operation) noexcept;
std::size_t checked_product(std::size_t count, std::size_t width);

// 仅改变当前调用线程的设备；析构尽力恢复，失败写入诊断而不抛出异常。
class DeviceScope {
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = -1;
    bool changed_ = false;
};

} // namespace minillm::cuda
