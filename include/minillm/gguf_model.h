#pragma once

#include "minillm/kernels.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace minillm {

struct TensorView {
    const std::byte* data;
    WeightType type;
    std::size_t columns;
    std::size_t rows;
    std::size_t stride;
    const std::byte* row(std::size_t index) const;
    std::vector<float> vector() const;
};

class GgufModel {
public:
    explicit GgufModel(const std::string& path);
    ~GgufModel();
    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;
    std::string string_value(const std::string& key) const;
    std::uint64_t integer_value(const std::string& key) const;
    float float_value(const std::string& key) const;
    bool contains(const std::string& key) const;
    bool has_tensor(const std::string& name) const;
    TensorView tensor(const std::string& name) const;
    std::size_t tensor_count() const;
    std::size_t file_bytes() const;
    std::uint32_t version() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace minillm
