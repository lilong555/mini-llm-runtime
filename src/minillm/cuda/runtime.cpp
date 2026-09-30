#include "minillm/cuda/runtime.h"

#include "batch_state.h"
#include "page_table.h"
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
    precision_mode_name(config.precision_mode);
    checked_kv_capacity(config.kv_layout, config.max_sequences, config.max_model_len,
                        config.kv_capacity_tokens, config.page_tokens);
    if (config.kv_layout == CudaKvLayout::paged && config.precision_mode != PrecisionMode::f32_pedantic) {
        throw std::invalid_argument("分页 CUDA 模型只支持 F32 矩阵精度");
    }
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
    return {config.max_sequences,config.max_model_len,config.batch_tokens,config.device_budget_bytes,
            config.precision_mode,config.kv_layout,config.kv_capacity_tokens,config.page_tokens};
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
            p.logits_bytes,p.metadata_bytes,p.rope_bytes,p.padding_bytes,p.total_bytes,p.kv_table_bytes};
}
}

struct CudaRuntime::Impl {
    CudaRuntimeConfig config;
    Clock::time_point load_started = Clock::now();
    Qwen3Model model;
    Tokenizer tokenizer;
    std::uint64_t model_load_ns;
    BatchState state;
    std::unique_ptr<PageTableState> pages;
    std::vector<std::int32_t> ids, positions, slots, selected, output_ids;
    std::array<std::int32_t,2> device_status{};
    Clock::time_point storage_started = Clock::now();
    // host metadata 先构造、后销毁；所有异步 copy 完成后才允许复用。
    CudaStorage storage;
    std::uint64_t storage_initialization_ns;
    LayerExecutor layers;
    Events events;
    CudaDeviceInfo device;
    MatrixWeightView embedding_weight, output_weight;
    DeviceTensorView<const float> output_norm;
    std::uint64_t metadata_h2d = 0, token_d2h = 0, status_d2h = 0, debug_d2h = 0;
    std::uint64_t completed = 0, failed = 0;

    explicit Impl(CudaRuntimeConfig cfg)
        : config(checked(std::move(cfg))), model(config.model_path),
          tokenizer(config.model_path,model.dimensions().vocabulary), model_load_ns(elapsed(load_started)),
          state(limits(config),model.dimensions().vocabulary),
          pages(config.kv_layout == CudaKvLayout::paged
              ? std::make_unique<PageTableState>(PageTableLimits{config.max_sequences,config.max_model_len,
                    config.page_tokens,config.kv_capacity_tokens/config.page_tokens}) : nullptr),
          ids(config.batch_tokens), positions(config.batch_tokens),
          slots(config.batch_tokens), selected(config.batch_tokens), output_ids(config.batch_tokens),
          storage(model,limits(config),config.device), storage_initialization_ns(elapsed(storage_started)),
          layers(storage), events(config.device), device(read_device(storage.context())),
          embedding_weight(storage.matrix_weight("token_embd.weight")), output_weight(storage.matrix_weight("output.weight")),
          output_norm(storage.norm_weight("output_norm.weight")) {
        detail::as_int(pages ? storage.paged_kv_view().rows : storage.kv_view().rows);
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
            if (pages) { pages->prepare(state.lengths(),state.pending_lengths()); }
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
            if (pages) { pages->begin_execution(); }
            if (timing) { check_cuda(cudaEventRecord(events.start,context.stream()),"记录 CUDA 起始事件"); }
            if (pages && pages->upload_required()) { storage.upload_page_table(pages->device_table_for_upload()); }
            const auto status = storage.workspace<std::int32_t>(Workspace::status,1);
            reset_status(context,status);
            upload(Workspace::tokens,ids,tokens.size());
            upload(Workspace::positions,positions,tokens.size());
            upload(Workspace::slots,slots,tokens.size());
            if (chosen) { upload(Workspace::selected_rows,selected,chosen); }
            const auto hidden = storage.workspace<float>(Workspace::hidden,tokens.size());
            std::visit([&](auto weight) {
                gather_rows(context,weight,
                    detail::read_only(storage.workspace<std::int32_t>(Workspace::tokens,tokens.size())),hidden,status);
            },embedding_weight);
            for (std::size_t layer = 0; layer < d.layers; ++layer) { layers.enqueue(layer,tokens.size(),summary.max_context); }
            if (chosen) {
                const auto normalized = storage.workspace<float>(Workspace::normalized,tokens.size());
                const auto selected_hidden = storage.workspace<float>(Workspace::selected_hidden,chosen);
                const auto logits = storage.workspace<float>(Workspace::logits,chosen);
                rms_norm(context,detail::read_only(hidden),output_norm,normalized,d.rms_epsilon);
                gather_rows(context,detail::read_only(normalized),
                    detail::read_only(storage.workspace<std::int32_t>(Workspace::selected_rows,chosen)),selected_hidden,status);
                matrix_multiply(context,storage.prepare_matrix_input(detail::read_only(selected_hidden)),output_weight,logits);
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
            if (state.phase() != BatchPhase::executing ||
                (pages && pages->phase() != PageTablePhase::executing)) {
                throw Error("CUDA 页与长度的提交阶段不一致");
            }
            // 单调用者下的无分配发布区；从此处到返回不再调用 CUDA API。
            if (pages) { pages->commit(); }
            state.commit();
            ++completed;
            result.host_forward_to_token_ns = elapsed(started);
            return result;
        } catch (...) {
            if (state.phase() == BatchPhase::executing) {
                state.poison(); ++failed;
                if (pages) { pages->poison(); }
                // local debug 输出可能仍是异步 D2H 的目标，必须在销毁前终结在途工作。
                finish_noexcept(context);
            } else if (state.phase() == BatchPhase::prepared) {
                if (pages && pages->phase() == PageTablePhase::prepared) { pages->discard_prepared(); }
                if (pages && pages->phase() == PageTablePhase::poisoned) { state.poison(); }
                else { state.discard_prepared(); }
            }
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
    return impl_->state.phase() == BatchPhase::poisoned ||
           (impl_->pages && impl_->pages->phase() == PageTablePhase::poisoned)
        ? CudaRuntimeState::poisoned : CudaRuntimeState::ready;
}
std::size_t CudaRuntime::live_kv_tokens() const noexcept { return impl_->state.live_tokens(); }
std::optional<std::size_t> CudaRuntime::live_kv_pages() const noexcept {
    if (!impl_->pages || state() != CudaRuntimeState::ready ||
        impl_->pages->phase() != PageTablePhase::ready) { return std::nullopt; }
    return impl_->pages->assigned_pages();
}
std::vector<CudaWeightInfo> CudaRuntime::weight_manifest() const {
    std::vector<CudaWeightInfo> result;
    result.reserve(impl_->storage.plan().weights.size());
    for (const auto& w : impl_->storage.plan().weights) {
        const char* type = w.source_type == WeightType::q8_0 ? "Q8_0" : w.source_type == WeightType::f16 ? "F16" : "F32";
        result.push_back({w.name,type,w.alias_of,w.effective_sha256,w.rows,w.columns,w.offset,w.bytes,
                          w.device_storage_type == StorageType::f16 ? "F16" : "F32",w.device_payload_sha256});
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
void CudaRuntime::clear_sequence(std::int32_t sequence) {
    auto& p = *impl_;
    if (p.state.phase() != BatchPhase::ready || (p.pages && p.pages->phase() != PageTablePhase::ready)) {
        throw Error("CUDA KV 只能在 ready 完成点清理，不能恢复 poisoned");
    }
    if (sequence < 0 || std::size_t(sequence) >= p.config.max_sequences) {
        throw std::invalid_argument("CUDA sequence 越界");
    }
    try {
        if (p.pages) { p.pages->clear(std::size_t(sequence)); }
        p.state.clear(sequence);
    } catch (...) {
        p.state.poison();
        if (p.pages) { p.pages->poison(); }
        throw;
    }
}
CudaDiagnostics CudaRuntime::diagnostics() const {
    CudaDiagnostics result;
    const auto& p = *impl_;
    result.state = state();
    result.sequence_lengths.assign(p.state.lengths().begin(),p.state.lengths().end());
    result.live_sequences = p.state.live_sequences(); result.live_kv_tokens = p.state.live_tokens();
    result.kv_capacity_tokens = p.storage.plan().kv_capacity_tokens;
    result.kv_layout = p.config.kv_layout;
    if (p.pages) {
        result.page_size_tokens = p.config.page_tokens;
        result.capacity_pages = p.storage.plan().physical_pages;
        result.live_kv_pages = live_kv_pages();
    }
    result.page_table_h2d_bytes = p.storage.page_table_h2d_bytes();
    result.resident = public_plan(p.storage.plan());
    result.matrix_cast_calls = p.storage.matrix_cast_calls();
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
