#pragma once

#include "attention.h"
#include "storage.h"

namespace minillm::cuda {

// 借用同一 CudaStorage 的权重、scratch、metadata 和 KV；不拥有设备分配。
// hidden/slots/positions 由调用方准备，status 在整个 batch 开始时重置。
class LayerExecutor {
public:
    explicit LayerExecutor(CudaStorage& storage);
    LayerExecutor(const LayerExecutor&) = delete;
    LayerExecutor& operator=(const LayerExecutor&) = delete;
    void enqueue(std::size_t layer, std::size_t tokens, std::size_t max_context);

private:
    struct Weights {
        DeviceTensorView<const float> attention_norm, query_norm, key_norm, ffn_norm;
        DeviceTensorView<const float> query, key, value, output, gate, up, down;
    };
    CudaStorage& storage_;
    ModelDimensions dimensions_;
    KvShape kv_shape_;
    std::vector<Weights> layers_;
};

}
