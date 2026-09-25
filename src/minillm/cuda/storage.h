#pragma once

#include "minillm/cuda/matrix.h"
#include "minillm/qwen3_model.h"

#include <string>
#include <vector>

namespace minillm::cuda {

struct StorageLimits {
    std::size_t max_sequences = 4;
    std::size_t max_model_len = 2048;
    std::size_t max_batch_tokens = 128;
    std::size_t device_budget_bytes = 0;
};

struct WeightRecord {
    std::string name;
    WeightType source_type;
    std::size_t rows, columns, offset, bytes;
    std::string alias_of;
    std::string effective_sha256;
};

enum class Workspace {
    hidden, normalized, query, key, value, attention, projected, gate, up, down,
    selected_hidden, scores, probabilities, logits, tokens, positions, slots,
    selected_rows, pending_lengths, samples, status
};
enum class StorageType { f32, i32 };
struct WorkspaceRegion {
    Workspace id;
    std::string name;
    StorageType type;
    std::size_t rows, columns, offset, bytes;
};

struct MemoryPlan {
    StorageLimits limits;
    std::vector<WeightRecord> weights;
    std::vector<WorkspaceRegion> regions;
    std::size_t weight_payload = 0, weight_bytes = 0;
    std::size_t activation_bytes = 0, attention_bytes = 0, logits_bytes = 0, metadata_bytes = 0;
    std::size_t workspace_bytes = 0, kv_bytes = 0;
    std::size_t cublas_bytes = CudaContext::default_workspace_bytes;
    std::size_t padding_bytes = 0, total_bytes = 0;
    std::string describe() const;
};

constexpr std::size_t weight_staging_bytes = 8 * 1024 * 1024;
MemoryPlan make_memory_plan(const Qwen3Model& model, StorageLimits limits);
std::size_t memory_budget(const MemoryPlan& plan, MemoryInfo available);
void check_memory_budget(const MemoryPlan& plan, MemoryInfo available);

// 内部存储基础，不提供模型 forward。构造后不再分配、转换或上传权重。
// 借用视图有效期截至 owner 析构；使用后必须检查 context().synchronize()。
class CudaStorage {
public:
    explicit CudaStorage(const Qwen3Model& model, StorageLimits limits = {}, int device = 0);
    ~CudaStorage();
    CudaStorage(const CudaStorage&) = delete;
    CudaStorage& operator=(const CudaStorage&) = delete;
    const CudaContext& context() const noexcept { return context_; }
    const MemoryPlan& plan() const noexcept { return plan_; }
    MemoryInfo available_at_gate() const noexcept { return available_; }
    std::size_t uploaded_bytes() const noexcept { return uploaded_bytes_; }
    std::size_t upload_chunks() const noexcept { return upload_chunks_; }
    std::size_t max_upload_chunk_bytes() const noexcept { return max_upload_chunk_bytes_; }
    DeviceTensorView<const float> weight(const std::string& name) const;
    const WorkspaceRegion& region(Workspace id) const;
    template<class T> DeviceTensorView<T> workspace(Workspace id, std::size_t rows) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::int32_t>);
        const auto& r = region(id);
        const auto type = std::is_same_v<T, float> ? StorageType::f32 : StorageType::i32;
        if (r.type != type || rows == 0 || rows > r.rows) {
            throw std::invalid_argument("workspace 类型或行数无效：" + r.name);
        }
        return {reinterpret_cast<T*>(workspace_.data() + r.offset), rows, r.columns,
                r.columns, r.bytes / sizeof(T), context_.device()};
    }
    const void* kv_reservation() const noexcept { return kv_.data(); }

private:
    void upload(const Qwen3Model& model);
    // context 先构造、最后析构；本类析构及构造失败路径先完成在途工作。
    MemoryPlan plan_;
    CudaContext context_;
    MemoryInfo available_{};
    DeviceBuffer<std::byte> weights_, workspace_, kv_;
    std::size_t uploaded_bytes_ = 0, upload_chunks_ = 0, max_upload_chunk_bytes_ = 0;
};

} // namespace minillm::cuda
