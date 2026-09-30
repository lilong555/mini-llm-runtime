#pragma once

#include "minillm/gguf_model.h"
#include "minillm/model_types.h"

namespace minillm {

struct Qwen3Layer {
    std::vector<float> attention_norm;
    std::vector<float> query_norm;
    std::vector<float> key_norm;
    std::vector<float> ffn_norm;
    TensorView query;
    TensorView key;
    TensorView value;
    TensorView attention_output;
    TensorView gate;
    TensorView up;
    TensorView down;
};

// 独占只读 GGUF 映射。所有借用视图只能在该 owner 存活期间使用。
// 构造时完成模型绑定与校验；不创建执行线程、KV 或设备资源。
class Qwen3Model {
public:
    explicit Qwen3Model(const std::string& path);
    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    const ModelDimensions& dimensions() const noexcept { return dims_; }
    const GgufModel& source() const noexcept { return weights_; }
    const TensorView& embeddings() const noexcept { return embeddings_; }
    const TensorView& output() const noexcept { return output_; }
    const std::vector<float>& output_norm() const noexcept { return output_norm_; }
    const std::vector<Qwen3Layer>& layers() const noexcept { return layers_; }
    bool tied_output() const noexcept { return output_.data == embeddings_.data; }

private:
    GgufModel weights_;
    ModelDimensions dims_;
    TensorView embeddings_;
    TensorView output_;
    std::vector<float> output_norm_;
    std::vector<Qwen3Layer> layers_;
};

} // namespace minillm
