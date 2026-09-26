#include "minillm/cuda/runtime.h"

#include "batch_state.h"
#include "layer.h"
#include "tensor_validation.h"
#include "minillm/tokenizer.h"

#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace minillm::cuda {
namespace {

using Clock = std::chrono::steady_clock;
std::uint64_t elapsed(Clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());
}
CudaRuntimeConfig checked(CudaRuntimeConfig config) {
    if (config.model_path.empty() || config.device < 0 || config.max_sequences == 0 ||
        config.max_sequences > CudaRuntimeConfig::max_supported_sequences) {
        throw std::invalid_argument("CUDA 模型路径、设备或 sequence 上限无效，最多支持 4 个序列");
    }
    if (config.batch_tokens == 0 || config.batch_tokens > CudaRuntimeConfig::max_supported_batch_tokens) {
        throw std::invalid_argument("CUDA 模型 batch_tokens 必须位于 [1, 128]");
    }
    detail::as_int(config.max_model_len); detail::as_int(config.batch_tokens);
    return config;
}
StorageLimits limits(const CudaRuntimeConfig& config) {
    return {config.max_sequences,config.max_model_len,config.batch_tokens,config.device_budget_bytes};
}
void finish_noexcept(const CudaContext& context) noexcept {
    try { context.synchronize(); }
    catch (const std::exception& error) { std::fprintf(stderr,"CUDA Runtime 清理同步失败：%s\n",error.what()); }
}

class Events {
public:
    explicit Events(int device) : device_(device) {
        DeviceScope scope(device_);
        try {
            check_cuda(cudaEventCreate(&start),"创建 CUDA 起始事件");
            check_cuda(cudaEventCreate(&stop),"创建 CUDA 结束事件");
        } catch (...) { cleanup(); throw; }
    }
    ~Events() { cleanup(); }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
    cudaEvent_t start = nullptr, stop = nullptr;
private:
    void cleanup() noexcept {
        try {
            DeviceScope scope(device_);
            if (start) { report_cuda(cudaEventDestroy(start),"销毁 CUDA 起始事件"); }
            if (stop) { report_cuda(cudaEventDestroy(stop),"销毁 CUDA 结束事件"); }
            start = stop = nullptr;
        } catch (const std::exception& error) { std::fprintf(stderr,"CUDA 事件清理失败：%s\n",error.what()); }
    }
    int device_;
};

CudaDeviceInfo read_device(const CudaContext& context) {
    DeviceScope scope(context.device());
    cudaDeviceProp properties{};
    CudaDeviceInfo info;
    check_cuda(cudaGetDeviceProperties(&properties,context.device()),"查询 CUDA 设备");
    info.name = properties.name; info.compute_major = properties.major; info.compute_minor = properties.minor;
    std::ostringstream uuid;
    for (unsigned i = 0; i < sizeof(properties.uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { uuid << '-'; }
        uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    info.uuid = uuid.str();
    check_cuda(cudaDriverGetVersion(&info.driver_version),"查询 CUDA driver 版本");
    check_cuda(cudaRuntimeGetVersion(&info.runtime_version),"查询 CUDA runtime 版本");
    check_cublas(cublasGetVersion(context.handle(),&info.cublas_version),"查询 cuBLAS 版本");
    return info;
}
CudaMemoryPlan public_plan(const MemoryPlan& p) {
    return {p.weight_bytes,p.workspace_bytes,p.kv_bytes,p.cublas_bytes,p.activation_bytes,p.attention_bytes,
            p.logits_bytes,p.metadata_bytes,p.rope_bytes,p.padding_bytes,p.total_bytes};
}
}

struct CudaRuntime::Impl {
    CudaRuntimeConfig config;
    Clock::time_point load_started = Clock::now();
    Qwen3Model model;
    Tokenizer tokenizer;
    std::uint64_t model_load_ns;
    BatchState state;
    std::vector<std::int32_t> ids, positions, slots, selected, output_ids;
    std::array<std::int32_t,2> device_status{};
    Clock::time_point storage_started = Clock::now();
    // host metadata 先构造、后销毁；所有异步 copy 完成后才允许复用。
    CudaStorage storage;
    std::uint64_t storage_initialization_ns;
    LayerExecutor layers;
    Events events;
    CudaDeviceInfo device;
    DeviceTensorView<const float> embedding_weight, output_weight, output_norm;
    std::uint64_t metadata_h2d = 0, token_d2h = 0, status_d2h = 0, debug_d2h = 0;
    std::uint64_t completed = 0, failed = 0;

    explicit Impl(CudaRuntimeConfig cfg)
        : config(checked(std::move(cfg))), model(config.model_path),
          tokenizer(config.model_path,model.dimensions().vocabulary), model_load_ns(elapsed(load_started)),
          state(limits(config),model.dimensions().vocabulary), ids(config.batch_tokens), positions(config.batch_tokens),
          slots(config.batch_tokens), selected(config.batch_tokens), output_ids(config.batch_tokens),
          storage(model,limits(config),config.device), storage_initialization_ns(elapsed(storage_started)),
          layers(storage), events(config.device), device(read_device(storage.context())),
          embedding_weight(storage.weight("token_embd.weight")), output_weight(storage.weight("output.weight")),
          output_norm(storage.weight("output_norm.weight")) {
        detail::as_int(storage.kv_view().rows);
        if (storage.allocated_bytes() != storage.plan().total_bytes) { throw Error("CUDA 实分配与内存计划不符"); }
    }
    ~Impl() { finish_noexcept(storage.context()); }

    void upload(Workspace id, const std::vector<std::int32_t>& values, std::size_t count) {
        auto view = storage.workspace<std::int32_t>(id,count);
        const auto bytes = count*sizeof(std::int32_t);
        check_cuda(cudaMemcpyAsync(view.data,values.data(),bytes,cudaMemcpyHostToDevice,storage.context().stream()),
                   "上传 CUDA token metadata");
        metadata_h2d += bytes;
    }

    CudaForwardResult forward(std::span<const InputToken> tokens, CudaOutputMode mode, bool timing) {
        const auto started = Clock::now();
        if (mode != CudaOutputMode::greedy && mode != CudaOutputMode::debug_logits) {
            throw std::invalid_argument("CUDA output mode 无效");
        }
        const auto summary = state.prepare(tokens);
        CudaForwardResult result;
        const auto& context = storage.context();
        try {
            const auto& d = model.dimensions();
            detail::as_int(checked_product(checked_product(tokens.size(),d.heads),(summary.max_context+7)/8));
            result.samples.resize(summary.logits);
            if (mode == CudaOutputMode::debug_logits) {
                result.logits.resize(summary.logits);
                for (auto& row : result.logits) { row.values.resize(d.vocabulary); }
            }
            std::size_t chosen = 0;
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                ids[i] = tokens[i].token; positions[i] = tokens[i].position; slots[i] = tokens[i].sequence;
                if (tokens[i].logits) {
                    selected[chosen] = static_cast<std::int32_t>(i);
                    result.samples[chosen] = {tokens[i].sequence,i,-1};
                    if (!result.logits.empty()) { result.logits[chosen].sequence = tokens[i].sequence; }
                    ++chosen;
                }
            }
            DeviceScope scope(config.device);
            state.start();
            if (timing) { check_cuda(cudaEventRecord(events.start,context.stream()),"记录 CUDA 起始事件"); }
            const auto status = storage.workspace<std::int32_t>(Workspace::status,1);
            reset_status(context,status);
            upload(Workspace::tokens,ids,tokens.size());
            upload(Workspace::positions,positions,tokens.size());
            upload(Workspace::slots,slots,tokens.size());
            if (chosen) { upload(Workspace::selected_rows,selected,chosen); }
            const auto hidden = storage.workspace<float>(Workspace::hidden,tokens.size());
            gather_rows(context,embedding_weight,
                        detail::read_only(storage.workspace<std::int32_t>(Workspace::tokens,tokens.size())),hidden,status);
            for (std::size_t layer = 0; layer < d.layers; ++layer) { layers.enqueue(layer,tokens.size(),summary.max_context); }
            if (chosen) {
                const auto normalized = storage.workspace<float>(Workspace::normalized,tokens.size());
                const auto selected_hidden = storage.workspace<float>(Workspace::selected_hidden,chosen);
                const auto logits = storage.workspace<float>(Workspace::logits,chosen);
                rms_norm(context,detail::read_only(hidden),output_norm,normalized,d.rms_epsilon);
                gather_rows(context,detail::read_only(normalized),
                    detail::read_only(storage.workspace<std::int32_t>(Workspace::selected_rows,chosen)),selected_hidden,status);
                matrix_multiply(context,detail::read_only(selected_hidden),output_weight,logits);
                // embedding gather 已完成，输入 ID 区域可复用为紧凑的输出 ID。
                const auto output = storage.workspace<std::int32_t>(Workspace::tokens,chosen);
                argmax(context,detail::read_only(logits),output,status);
                const auto bytes = chosen*sizeof(std::int32_t);
                check_cuda(cudaMemcpyAsync(output_ids.data(),output.data,bytes,cudaMemcpyDeviceToHost,context.stream()),
                           "下载 CUDA greedy token");
                token_d2h += bytes;
                for (std::size_t i = 0; i < result.logits.size(); ++i) {
                    const auto row_bytes = d.vocabulary*sizeof(float);
                    check_cuda(cudaMemcpyAsync(result.logits[i].values.data(),logits.data+i*logits.stride,
                        row_bytes,cudaMemcpyDeviceToHost,context.stream()),"下载显式 CUDA debug logits");
                    debug_d2h += row_bytes;
                }
            }
            check_cuda(cudaMemcpyAsync(device_status.data(),status.data,sizeof(device_status),cudaMemcpyDeviceToHost,
                context.stream()),"下载 CUDA status");
            status_d2h += sizeof(device_status);
            if (timing) { check_cuda(cudaEventRecord(events.stop,context.stream()),"记录 CUDA 结束事件"); }
            context.synchronize();
            if (device_status[0] != 0 || device_status[1] != INT_MAX) {
                throw Error("CUDA forward 设备状态失败：bits="+std::to_string(device_status[0])+
                            " first_bad_row="+std::to_string(device_status[1]));
            }
            for (std::size_t i = 0; i < chosen; ++i) {
                if (output_ids[i] < 0 || std::size_t(output_ids[i]) >= d.vocabulary) { throw Error("CUDA greedy token 越界"); }
                result.samples[i].token = output_ids[i];
            }
            if (timing) {
                float milliseconds = 0;
                check_cuda(cudaEventElapsedTime(&milliseconds,events.start,events.stop),"读取 CUDA 事件时间");
                if (!std::isfinite(milliseconds) || milliseconds < 0) { throw Error("CUDA 事件时间无效"); }
                result.device_elapsed_ms = double(milliseconds);
            }
            state.commit();
            ++completed;
            result.host_forward_to_token_ns = elapsed(started);
            return result;
        } catch (...) {
            if (state.phase() == BatchPhase::executing) {
                state.poison(); ++failed;
                // local debug 输出可能仍是异步 D2H 的目标，必须在销毁前终结在途工作。
                finish_noexcept(context);
            } else if (state.phase() == BatchPhase::prepared) { state.discard_prepared(); }
            throw;
        }
    }
};

CudaRuntime::CudaRuntime(CudaRuntimeConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}
CudaRuntime::~CudaRuntime() = default;
const CudaRuntimeConfig& CudaRuntime::config() const noexcept { return impl_->config; }
const ModelDimensions& CudaRuntime::dimensions() const noexcept { return impl_->model.dimensions(); }
const CudaDeviceInfo& CudaRuntime::device_info() const noexcept { return impl_->device; }
CudaRuntimeState CudaRuntime::state() const noexcept {
    return impl_->state.phase() == BatchPhase::poisoned ? CudaRuntimeState::poisoned : CudaRuntimeState::ready;
}
std::size_t CudaRuntime::live_kv_tokens() const noexcept { return impl_->state.live_tokens(); }
std::vector<CudaWeightInfo> CudaRuntime::weight_manifest() const {
    std::vector<CudaWeightInfo> result;
    result.reserve(impl_->storage.plan().weights.size());
    for (const auto& w : impl_->storage.plan().weights) {
        const char* type = w.source_type == WeightType::q8_0 ? "Q8_0" : w.source_type == WeightType::f16 ? "F16" : "F32";
        result.push_back({w.name,type,w.alias_of,w.effective_sha256,w.rows,w.columns,w.offset,w.bytes});
    }
    return result;
}
std::vector<std::int32_t> CudaRuntime::tokenize(std::string_view text) const { return impl_->tokenizer.tokenize(text); }
std::string CudaRuntime::token_piece(std::int32_t token) const {
    if (token < 0 || std::size_t(token) >= dimensions().vocabulary) { throw std::invalid_argument("CUDA tokenizer token 越界"); }
    return impl_->tokenizer.token_piece(token);
}
bool CudaRuntime::is_eog(std::int32_t token) const {
    if (token < 0 || std::size_t(token) >= dimensions().vocabulary) { throw std::invalid_argument("CUDA tokenizer token 越界"); }
    return impl_->tokenizer.is_eog(token);
}
CudaForwardResult CudaRuntime::forward(std::span<const InputToken> tokens, CudaOutputMode mode, bool device_timing) {
    return impl_->forward(tokens,mode,device_timing);
}
void CudaRuntime::clear_sequence(std::int32_t sequence) { impl_->state.clear(sequence); }
CudaDiagnostics CudaRuntime::diagnostics() const {
    CudaDiagnostics result;
    const auto& p = *impl_;
    result.state = state();
    result.sequence_lengths.assign(p.state.lengths().begin(),p.state.lengths().end());
    result.live_sequences = p.state.live_sequences(); result.live_kv_tokens = p.state.live_tokens();
    result.kv_capacity_tokens = p.config.max_sequences*p.config.max_model_len;
    result.resident = public_plan(p.storage.plan());
    result.weight_h2d_bytes = p.storage.uploaded_bytes(); result.rope_h2d_bytes = p.storage.rope_uploaded_bytes();
    result.metadata_h2d_bytes = p.metadata_h2d; result.token_d2h_bytes = p.token_d2h;
    result.status_d2h_bytes = p.status_d2h; result.debug_d2h_bytes = p.debug_d2h;
    result.owned_device_allocations = p.storage.allocations(); result.owned_device_bytes = p.storage.allocated_bytes();
    result.completed_forwards = p.completed; result.post_launch_failures = p.failed;
    result.model_load_ns = p.model_load_ns; result.storage_initialization_ns = p.storage_initialization_ns;
    result.weight_decode_upload_ns = p.storage.weight_decode_upload_ns();
    return result;
}

}
