#include "storage.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>

namespace minillm::cuda {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("CUDA 内存计划加法溢出");
    }
    return a + b;
}
std::size_t align(std::size_t bytes) { return add(bytes, 255) & ~std::size_t{255}; }
void dimension(std::size_t n) {
    if (!n || n > INT_MAX) { throw std::invalid_argument("CUDA 配置维度必须在 [1, INT_MAX] 内"); }
}
void finish_noexcept(const CudaContext& context) noexcept {
    try { context.synchronize(); }
    catch (const std::exception& error) { std::fprintf(stderr, "CUDA 存储清理同步失败：%s\n", error.what()); }
}
std::string digest(sha256_t& state) {
    unsigned char bytes[32];
    sha256_final(&state, bytes);
    std::ostringstream output;
    for (const auto b : bytes) { output << std::hex << std::setw(2) << std::setfill('0') << unsigned(b); }
    return output.str();
}
}

MemoryPlan make_memory_plan(const Qwen3Model& model, StorageLimits limits) {
    dimension(limits.max_sequences); dimension(limits.max_model_len); dimension(limits.max_batch_tokens);
    const auto& d = model.dimensions();
    if (limits.max_model_len > d.trained_context) {
        throw std::invalid_argument("CUDA context 上限超过模型训练范围");
    }
    MemoryPlan p;
    p.limits = limits;
    const auto weight = [&](const std::string& name) {
        const auto t = model.source().tensor(name);
        dimension(t.rows); dimension(t.columns);
        if (checked_product(t.columns, sizeof(float)) > weight_staging_bytes) {
            throw std::length_error("单行权重超过 8 MiB staging 上限：" + name);
        }
        const auto bytes = checked_product(checked_product(t.rows, t.columns), sizeof(float));
        const auto offset = align(p.weight_bytes);
        p.weights.push_back({name, t.type, t.rows, t.columns, offset, bytes, {}, {}});
        p.weight_payload = add(p.weight_payload, bytes);
        p.weight_bytes = add(offset, bytes);
    };
    weight("token_embd.weight");
    weight("output_norm.weight");
    if (model.tied_output()) {
        auto alias = p.weights.front();
        alias.name = "output.weight";
        alias.alias_of = "token_embd.weight";
        p.weights.push_back(std::move(alias));
    } else { weight("output.weight"); }
    for (std::size_t i = 0; i < d.layers; ++i) {
        const auto prefix = "blk." + std::to_string(i) + ".";
        for (const auto* name : {"attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm", "attn_q",
                                 "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}) {
            weight(prefix + name + ".weight");
        }
    }
    p.weight_bytes = align(p.weight_bytes);
    const auto region = [&](Workspace id, const char* name, std::size_t rows, std::size_t columns,
                            StorageType type, std::size_t& category) {
        dimension(columns);
        const auto bytes = checked_product(checked_product(rows, columns), std::size_t{4});
        const auto offset = align(p.workspace_bytes);
        p.regions.push_back({id, name, type, rows, columns, offset, bytes});
        p.workspace_bytes = add(offset, bytes);
        category = add(category, bytes);
    };
    const auto b = limits.max_batch_tokens;
    const auto q = checked_product(d.heads, d.head_dim);
    const auto kv = checked_product(d.kv_heads, d.head_dim);
    const auto activation = [&](Workspace id, const char* name, std::size_t columns) {
        region(id, name, b, columns, StorageType::f32, p.activation_bytes);
    };
    activation(Workspace::hidden, "hidden", d.embedding);
    activation(Workspace::normalized, "normalized", d.embedding);
    activation(Workspace::query, "query", q);
    activation(Workspace::key, "key", kv);
    activation(Workspace::value, "value", kv);
    activation(Workspace::attention, "attention", q);
    activation(Workspace::projected, "projected", d.embedding);
    activation(Workspace::gate, "gate", d.feed_forward);
    activation(Workspace::up, "up", d.feed_forward);
    activation(Workspace::down, "down", d.embedding);
    activation(Workspace::selected_hidden, "selected_hidden", d.embedding);
    region(Workspace::scores, "scores", b, checked_product(d.heads, limits.max_model_len), StorageType::f32, p.attention_bytes);
    region(Workspace::probabilities, "probabilities", b, checked_product(d.heads, limits.max_model_len), StorageType::f32, p.attention_bytes);
    region(Workspace::logits, "logits", b, d.vocabulary, StorageType::f32, p.logits_bytes);
    region(Workspace::tokens, "tokens", b, 1, StorageType::i32, p.metadata_bytes);
    region(Workspace::positions, "positions", b, 1, StorageType::i32, p.metadata_bytes);
    region(Workspace::slots, "slots", b, 1, StorageType::i32, p.metadata_bytes);
    region(Workspace::selected_rows, "selected_rows", b, 1, StorageType::i32, p.metadata_bytes);
    region(Workspace::pending_lengths, "pending_lengths", limits.max_sequences, 1, StorageType::i32, p.metadata_bytes);
    // samples 每行依次存 sequence、input_index、token；status 为 nonfinite、first_bad_row。
    region(Workspace::samples, "samples", b, 3, StorageType::i32, p.metadata_bytes);
    region(Workspace::status, "status", 1, 2, StorageType::i32, p.metadata_bytes);
    p.workspace_bytes = align(p.workspace_bytes);
    // [sequence][layer][K_or_V][position][kv_head * head_dim]，物理元素为 FP16。
    p.kv_bytes = checked_product(checked_product(checked_product(limits.max_sequences, d.layers), 2),
                                checked_product(checked_product(limits.max_model_len, kv), 2));
    const auto payload = add(add(p.activation_bytes, p.attention_bytes), add(p.logits_bytes, p.metadata_bytes));
    p.padding_bytes = add(p.weight_bytes - p.weight_payload, p.workspace_bytes - payload);
    p.total_bytes = add(add(p.weight_bytes, p.workspace_bytes), add(p.kv_bytes, p.cublas_bytes));
    if (p.total_bytes > static_cast<std::size_t>(PTRDIFF_MAX)) { throw std::length_error("CUDA 内存计划超过 PTRDIFF_MAX"); }
    return p;
}

std::string MemoryPlan::describe() const {
    std::ostringstream s;
    s << "S=" << limits.max_sequences << " L=" << limits.max_model_len << " B=" << limits.max_batch_tokens
      << " user_budget=" << limits.device_budget_bytes << " weight_payload=" << weight_payload
      << " weight_arena=" << weight_bytes << " activation=" << activation_bytes << " attention=" << attention_bytes
      << " logits=" << logits_bytes << " metadata=" << metadata_bytes << " padding=" << padding_bytes
      << " workspace_arena=" << workspace_bytes << " KV=" << kv_bytes << " cuBLAS=" << cublas_bytes << " total=" << total_bytes;
    for (const auto& w : weights) { s << "\nweight " << w.name << " " << w.rows << "x" << w.columns << " offset=" << w.offset << " bytes=" << w.bytes << " alias=" << w.alias_of; }
    for (const auto& r : regions) { s << "\nworkspace " << r.name << " " << r.rows << "x" << r.columns << " offset=" << r.offset << " bytes=" << r.bytes; }
    return s.str();
}

std::size_t memory_budget(const MemoryPlan& p, MemoryInfo available) {
    constexpr std::size_t reserve = 512 * 1024 * 1024;
    const auto f = available.free_bytes;
    const auto eighty_percent = f - (f / 5 + (f % 5 != 0));
    auto result = std::min(eighty_percent, f > reserve ? f - reserve : 0);
    if (p.limits.device_budget_bytes) { result = std::min(result, p.limits.device_budget_bytes); }
    return result;
}
void check_memory_budget(const MemoryPlan& p, MemoryInfo available) {
    if (p.total_bytes > memory_budget(p, available)) {
        throw Error("CUDA 显存预算不足：free=" + std::to_string(available.free_bytes) +
                    " allowed=" + std::to_string(memory_budget(p, available)) + "\n" + p.describe());
    }
}

CudaStorage::CudaStorage(const Qwen3Model& model, StorageLimits limits, int device)
    : plan_(make_memory_plan(model, limits)), context_(device, plan_.cublas_bytes) {
    available_ = context_.memory_info();
    // free 已扣除 context/cuBLAS，仍用含显式 cuBLAS workspace 的完整计划比较，保守预留。
    check_memory_budget(plan_, available_);
    DeviceScope scope(device);
    try {
        weights_ = DeviceBuffer<std::byte>(plan_.weight_bytes, device);
        workspace_ = DeviceBuffer<std::byte>(plan_.workspace_bytes, device);
        kv_ = DeviceBuffer<std::byte>(plan_.kv_bytes, device);
        check_cuda(cudaMemsetAsync(workspace_.data(), 0, workspace_.bytes(), context_.stream()), "初始化 workspace");
        // 未使用 KV 填充为 FP16 NaN，后续 attention 必须先检查有效位置。
        check_cuda(cudaMemsetAsync(kv_.data(), 0xff, kv_.bytes(), context_.stream()), "初始化 KV 预留");
        upload(model);
        context_.synchronize();
    } catch (const Error& error) {
        finish_noexcept(context_);
        throw Error(std::string(error.what()) + "\n" + plan_.describe());
    } catch (...) { finish_noexcept(context_); throw; }
}
CudaStorage::~CudaStorage() { finish_noexcept(context_); }

void CudaStorage::upload(const Qwen3Model& model) {
    std::vector<float> staging(weight_staging_bytes / sizeof(float));
    try {
        for (auto& w : plan_.weights) {
            if (!w.alias_of.empty()) {
                w.effective_sha256 = plan_.weights.front().effective_sha256;
                continue;
            }
            const auto source = model.source().tensor(w.name);
            const auto rows_per_chunk = staging.size() / w.columns;
            sha256_t state;
            sha256_init(&state);
            for (std::size_t first = 0; first < w.rows; first += rows_per_chunk) {
                const auto count = std::min(rows_per_chunk, w.rows - first);
                for (std::size_t row = 0; row < count; ++row) {
                    decode_row(source.type, source.row(first + row), staging.data() + row * w.columns, w.columns);
                }
                const auto elements = count * w.columns;
                if (!std::all_of(staging.begin(), staging.begin() + static_cast<std::ptrdiff_t>(elements),
                                 [](float v) { return std::isfinite(v); })) {
                    throw Error("权重包含非有限有效值：" + w.name);
                }
                const auto bytes = elements * sizeof(float);
                sha256_update(&state, reinterpret_cast<const unsigned char*>(staging.data()), bytes);
                check_cuda(cudaMemcpyAsync(weights_.data() + w.offset + first * w.columns * sizeof(float),
                                          staging.data(), bytes, cudaMemcpyHostToDevice, context_.stream()), "上传有效权重");
                context_.synchronize();
                uploaded_bytes_ += bytes;
                ++upload_chunks_;
                max_upload_chunk_bytes_ = std::max(max_upload_chunk_bytes_, bytes);
            }
            w.effective_sha256 = digest(state);
        }
    } catch (...) { finish_noexcept(context_); throw; }
}

DeviceTensorView<const float> CudaStorage::weight(const std::string& name) const {
    for (const auto& w : plan_.weights) {
        if (w.name == name) {
            return {reinterpret_cast<const float*>(weights_.data() + w.offset), w.rows, w.columns,
                    w.columns, w.bytes / sizeof(float), context_.device()};
        }
    }
    throw std::out_of_range("未知设备权重：" + name);
}
const WorkspaceRegion& CudaStorage::region(Workspace id) const {
    for (const auto& r : plan_.regions) { if (r.id == id) { return r; } }
    throw std::out_of_range("未知 workspace 区域");
}

} // namespace minillm::cuda
