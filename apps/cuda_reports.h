#pragma once

#include "minillm/cuda/runtime.h"
#include <nlohmann/json.hpp>
extern "C" {
#include "hash/sha256/sha256.h"
}

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cuda_reports {
using json = nlohmann::ordered_json;

inline std::string file_hash(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary);
    if (!file) { throw std::runtime_error("无法读取文件：" + path.string()); }
    sha256_t state; sha256_init(&state);
    std::array<char,65536> buffer{};
    while (file.read(buffer.data(),buffer.size()) || file.gcount()) {
        sha256_update(&state,reinterpret_cast<const unsigned char*>(buffer.data()),std::size_t(file.gcount()));
    }
    if (!file.eof()) { throw std::runtime_error("文件读取失败：" + path.string()); }
    unsigned char digest[32]; sha256_final(&state,digest);
    std::ostringstream out;
    for (auto b : digest) { out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b); }
    return out.str();
}
inline void write(const std::filesystem::path& path, const json& value) {
    std::ofstream file(path,std::ios::binary);
    file << value.dump(2) << '\n'; file.close();
    if (!file) { throw std::runtime_error("报告写入失败：" + path.string()); }
}
inline json memory(const minillm::cuda::CudaMemoryPlan& p) {
    json result = {{"weights_bytes",p.weights_bytes},{"workspace_bytes",p.workspace_bytes},{"kv_bytes",p.kv_bytes},
        {"library_workspace_bytes",p.library_workspace_bytes},{"activations_bytes",p.activations_bytes},
        {"attention_scratch_bytes",p.attention_scratch_bytes},{"logits_bytes",p.logits_bytes},
        {"metadata_bytes",p.metadata_bytes},{"rope_bytes",p.rope_bytes},{"padding_bytes",p.padding_bytes},
        {"total_owned_bytes",p.total_owned_bytes}};
    if (p.kv_table_bytes) { result["kv_table_bytes"] = p.kv_table_bytes; }
    return result;
}
inline json diagnostics(const minillm::cuda::CudaDiagnostics& d) {
    json result = {{"state",d.state == minillm::cuda::CudaRuntimeState::ready ? "ready" : "poisoned"},
        {"sequence_lengths",d.sequence_lengths},{"live_sequences",d.live_sequences},{"live_kv_tokens",d.live_kv_tokens},
        {"kv_capacity_tokens",d.kv_capacity_tokens},{"resident",memory(d.resident)},{"weight_h2d_bytes",d.weight_h2d_bytes},
        {"rope_h2d_bytes",d.rope_h2d_bytes},{"metadata_h2d_bytes",d.metadata_h2d_bytes},{"token_d2h_bytes",d.token_d2h_bytes},
        {"status_d2h_bytes",d.status_d2h_bytes},{"debug_d2h_bytes",d.debug_d2h_bytes},
        {"intermediate_h2d_bytes",d.intermediate_h2d_bytes},{"intermediate_d2h_bytes",d.intermediate_d2h_bytes},
        {"owned_device_allocations",d.owned_device_allocations},{"owned_device_bytes",d.owned_device_bytes},
        {"completed_forwards",d.completed_forwards},{"post_launch_failures",d.post_launch_failures},
        {"model_load_ns",d.model_load_ns},{"storage_initialization_ns",d.storage_initialization_ns},
        {"weight_decode_upload_ns",d.weight_decode_upload_ns}};
    if (d.kv_layout == minillm::cuda::CudaKvLayout::paged) {
        result["kv_layout"] = "paged";
        result["page_size_tokens"] = d.page_size_tokens ? json(*d.page_size_tokens) : json(nullptr);
        result["capacity_pages"] = d.capacity_pages ? json(*d.capacity_pages) : json(nullptr);
        result["live_kv_pages"] = d.live_kv_pages ? json(*d.live_kv_pages) : json(nullptr);
        result["page_table_h2d_bytes"] = d.page_table_h2d_bytes;
    }
    return result;
}
inline json device(const minillm::cuda::CudaDeviceInfo& d) {
    return {{"name",d.name},{"uuid",d.uuid},{"compute_capability",{d.compute_major,d.compute_minor}},
        {"driver_version",d.driver_version},{"runtime_version",d.runtime_version},{"cublas_version",d.cublas_version}};
}
inline json weights(const std::vector<minillm::cuda::CudaWeightInfo>& records, bool precision_study = false) {
    json result = json::array();
    for (const auto& w : records) {
        result.push_back({{"name",w.name},{"source_dtype",w.source_dtype},{"device_dtype",w.device_dtype},
            {"shape",{w.rows,w.columns}},{"offset",w.offset},{"bytes",w.bytes},{"alias_of",w.alias_of},
            {"effective_sha256",w.effective_sha256}});
        if (precision_study || w.device_dtype != "F32") {
            result.back()["device_payload_sha256"] = w.device_payload_sha256;
        }
    }
    return result;
}
inline json arithmetic(const minillm::cuda::CudaRuntime& runtime, bool precision_study = false) {
    json types = json::object();
    const auto records = runtime.weight_manifest();
    for (const auto& w : records) {
        if (w.alias_of.empty()) { types[w.source_dtype] = types.value(w.source_dtype,0) + 1; }
    }
    json result = {{"source_weight_dtype",records.at(0).source_dtype},{"source_tensor_counts",types},{"device_weight_dtype","F32"},
        {"activation_dtype","F32"},{"kv_dtype","F16"},{"kv_rounding","nearest_even"},{"qk_pv_accumulation_dtype","F32"},
        {"softmax_exponential_dtype","F32"},{"softmax_denominator_dtype","F64"},
        {"gemm_compute","CUBLAS_COMPUTE_32F_PEDANTIC"},{"fast_math",false}};
    const bool half = runtime.config().precision_mode == minillm::cuda::PrecisionMode::f16_matrix_f32acc;
    if (half || precision_study) {
        result["precision_mode"] = minillm::cuda::precision_mode_name(runtime.config().precision_mode);
        result["device_weight_dtype"] = half ? "F16_matrices_F32_norms" : "F32";
        result["matrix_operand_dtype"] = half ? "F16" : "F32";
        result["matrix_accumulation_dtype"] = "F32";
        result["matrix_output_dtype"] = "F32";
        result["norm_weight_dtype"] = "F32";
        result["gemm_compute"] = half ? "CUBLAS_COMPUTE_32F" : "CUBLAS_COMPUTE_32F_PEDANTIC";
        result["math_mode"] = half ? "CUBLAS_DEFAULT_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION" :
                                    "CUBLAS_PEDANTIC_MATH";
    }
    return result;
}
}
