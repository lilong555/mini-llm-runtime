#include "cuda_micro_protocol.h"
#include "cuda_reports.h"
#include "options.h"
#include "attention.h"
#include "storage.h"
#include "tensor_validation.h"
#include "hash/hash.h"

#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace minillm;
namespace gpu = minillm::cuda;
using namespace cuda_micro;
using gpu::detail::read_only;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}
json read(const std::string& path) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("无法读取 micro 协议：" + path); }
    return json::parse(file);
}
json allocations() {
    const auto s = gpu::allocation_stats();
    return {{"allocation_calls", s.allocation_calls}, {"allocations", s.allocations},
        {"release_calls", s.release_calls}, {"releases", s.releases}, {"allocated_bytes", s.allocated_bytes}};
}
void finish(const gpu::CudaContext& context) noexcept {
    try { context.synchronize(); }
    catch (const std::exception& e) { std::cerr << "micro 清理同步失败：" << e.what() << '\n'; }
}
struct Transfers {
    std::size_t h2d_bytes = 0, d2h_bytes = 0, h2d_calls = 0, d2h_calls = 0;
    json since(const Transfers& before) const {
        return {{"h2d_bytes", h2d_bytes-before.h2d_bytes}, {"d2h_bytes", d2h_bytes-before.d2h_bytes},
            {"h2d_calls", h2d_calls-before.h2d_calls}, {"d2h_calls", d2h_calls-before.d2h_calls}};
    }
};
template<class T>
void upload(const gpu::CudaContext& context, gpu::DeviceTensorView<T> target, const std::vector<T>& source,
            Transfers& transfers) {
    if (target.stride != target.columns || target.rows * target.columns != source.size()) {
        throw std::invalid_argument("micro 上传布局不符");
    }
    gpu::check_cuda(cudaMemcpyAsync(target.data, source.data(), source.size()*sizeof(T),
        cudaMemcpyHostToDevice, context.stream()), "micro 输入上传");
    transfers.h2d_bytes += source.size()*sizeof(T); ++transfers.h2d_calls;
    context.synchronize();
}
template<class T>
void download(const gpu::CudaContext& context, gpu::DeviceTensorView<T> source, std::vector<T>& target,
              Transfers& transfers) {
    if (source.stride != source.columns || source.rows * source.columns != target.size()) {
        throw std::invalid_argument("micro 下载布局不符");
    }
    gpu::check_cuda(cudaMemcpyAsync(target.data(), source.data, target.size()*sizeof(T),
        cudaMemcpyDeviceToHost, context.stream()), "micro 验证下载");
    transfers.d2h_bytes += target.size()*sizeof(T); ++transfers.d2h_calls;
    context.synchronize();
}
json memory(const gpu::MemoryPlan& p) {
    return {{"weights_bytes", p.weight_bytes}, {"workspace_bytes", p.workspace_bytes}, {"kv_bytes", p.kv_bytes},
        {"library_workspace_bytes", p.cublas_bytes}, {"activations_bytes", p.activation_bytes},
        {"attention_scratch_bytes", p.attention_bytes}, {"logits_bytes", p.logits_bytes},
        {"metadata_bytes", p.metadata_bytes}, {"rope_bytes", p.rope_bytes}, {"padding_bytes", p.padding_bytes},
        {"total_owned_bytes", p.total_bytes}};
}
json device(const gpu::CudaContext& context) {
    cudaDeviceProp properties{};
    gpu::check_cuda(cudaGetDeviceProperties(&properties, context.device()), "micro 设备查询");
    gpu::CudaDeviceInfo info;
    info.name = properties.name; info.compute_major = properties.major; info.compute_minor = properties.minor;
    std::ostringstream uuid;
    for (unsigned i = 0; i < sizeof(properties.uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { uuid << '-'; }
        uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    info.uuid = uuid.str();
    gpu::check_cuda(cudaDriverGetVersion(&info.driver_version), "micro driver 版本");
    gpu::check_cuda(cudaRuntimeGetVersion(&info.runtime_version), "micro runtime 版本");
    gpu::check_cublas(cublasGetVersion(context.handle(), &info.cublas_version), "micro cuBLAS 版本");
    return cuda_reports::device(info);
}
json weights(const gpu::CudaStorage& storage) {
    std::vector<gpu::CudaWeightInfo> result;
    for (const auto& w : storage.plan().weights) {
        result.push_back({w.name, w.source_type == WeightType::q8_0 ? "Q8_0" : w.source_type == WeightType::f32 ? "F32" : "F16",
                          w.alias_of, w.effective_sha256, w.rows, w.columns, w.offset, w.bytes});
    }
    return cuda_reports::weights(result);
}

class Events {
public:
    explicit Events(const gpu::CudaContext& context) : context_(context) {
        try {
            gpu::check_cuda(cudaEventCreate(&start), "micro 起始 event");
            gpu::check_cuda(cudaEventCreate(&stop), "micro 结束 event");
        } catch (...) { destroy(); throw; }
    }
    ~Events() { finish(context_); destroy(); }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
    cudaEvent_t start = nullptr, stop = nullptr;
private:
    void destroy() noexcept {
        if (start) { gpu::report_cuda(cudaEventDestroy(start), "micro 销毁起始 event"); }
        if (stop) { gpu::report_cuda(cudaEventDestroy(stop), "micro 销毁结束 event"); }
        start = stop = nullptr;
    }
    const gpu::CudaContext& context_;
};

json initialize_kv(gpu::CudaStorage& storage) {
    Transfers transfers;
    const auto& context = storage.context();
    const auto& d = storage.plan().dimensions;
    const gpu::KvShape shape{4,d.layers,max_length,d.kv_heads,d.head_dim};
    std::vector<float> key(max_batch*d.kv_heads*d.head_dim), value(key.size());
    std::vector<std::int32_t> slots(max_batch), positions(max_batch);
    const auto status = storage.workspace<std::int32_t>(gpu::Workspace::status, 1);
    gpu::reset_status(context, status);
    for (std::size_t slot = 0; slot < 4; ++slot) {
        std::fill(slots.begin(), slots.end(), static_cast<std::int32_t>(slot));
        for (std::size_t first = 0; first < max_length; first += max_batch) {
            for (std::size_t row = 0; row < max_batch; ++row) {
                positions[row] = static_cast<std::int32_t>(first + row);
                for (std::size_t col = 0; col < d.kv_heads*d.head_dim; ++col) {
                    key[row*d.kv_heads*d.head_dim+col] = kv_value(slot, 0, first+row, col);
                    value[row*d.kv_heads*d.head_dim+col] = kv_value(slot, 1, first+row, col);
                }
            }
            const auto k = storage.workspace<float>(gpu::Workspace::key, max_batch);
            const auto v = storage.workspace<float>(gpu::Workspace::value, max_batch);
            const auto s = storage.workspace<std::int32_t>(gpu::Workspace::slots, max_batch);
            const auto p = storage.workspace<std::int32_t>(gpu::Workspace::positions, max_batch);
            upload(context, k, key, transfers); upload(context, v, value, transfers);
            upload(context, s, slots, transfers); upload(context, p, positions, transfers);
            gpu::store_kv(context, storage.kv_view(), shape, 0, read_only(k), read_only(v), read_only(s), read_only(p), status);
            context.synchronize();
        }
    }
    std::vector<std::int32_t> state(2);
    download(context, status, state, transfers);
    if (state[0] != 0 || state[1] != INT_MAX) { throw std::runtime_error("micro KV 初始化失败"); }
    return transfers.since({});
}

struct Prepared {
    const gpu::CudaContext& context;
    gpu::DeviceTensorView<float> x{}, y{};
    std::vector<float> input, output, norm, coefficients, probability_initial;
    std::vector<double> expected;
    std::vector<std::pair<std::size_t,double>> points;
    std::vector<std::int32_t> status{0,0};
    Transfers transfers;
    explicit Prepared(const gpu::CudaContext& c) : context(c) {}
    ~Prepared() { finish(context); }
};

void attention_reference(Prepared& p, const Case& c, const ModelDimensions& d) {
    for (auto row : sample_rows(c.m)) {
        const auto length = std::size_t(c.positions[row]) + 1, slot = std::size_t(c.slots[row]);
        for (auto head : {std::size_t{0}, std::size_t{7}, std::size_t{8}, std::size_t{15}}) {
            std::vector<double> probabilities(length);
            const auto kh = head / (d.heads/d.kv_heads);
            for (std::size_t pos = 0; pos < length; ++pos) {
                double dot = 0;
                for (std::size_t col = 0; col < d.head_dim; ++col) {
                    dot += double(p.input[row*c.n+head*d.head_dim+col]) *
                        half_to_float(float_to_half(kv_value(slot, 0, pos, kh*d.head_dim+col)));
                }
                probabilities[pos] = dot / std::sqrt(double(d.head_dim));
            }
            const auto maximum = *std::max_element(probabilities.begin(), probabilities.end());
            double sum = 0;
            for (auto& value : probabilities) { value = std::exp(value-maximum); sum += value; }
            for (auto& value : probabilities) { value /= sum; }
            for (auto col : {std::size_t{0}, std::size_t{1}, std::size_t{63}, std::size_t{127}}) {
                double result = 0;
                for (std::size_t pos = 0; pos < length; ++pos) {
                    result += probabilities[pos] * half_to_float(float_to_half(kv_value(slot, 1, pos, kh*d.head_dim+col)));
                }
                p.points.emplace_back(row*c.n+head*d.head_dim+col, result);
            }
        }
    }
}

void prepare(Prepared& p, const Case& c, const Qwen3Model& model, gpu::CudaStorage& storage) {
    using W = gpu::Workspace;
    const auto& d = model.dimensions();
    W input = W::hidden, output = W::normalized;
    if (c.operation == "matrix") {
        if (c.role == "Q") { input = W::normalized; output = W::query; }
        if (c.role == "K") { input = W::normalized; output = W::key; }
        if (c.role == "V") { input = W::normalized; output = W::value; }
        if (c.role == "attention_output") { input = W::attention; output = W::projected; }
        if (c.role == "gate") { input = W::normalized; output = W::gate; }
        if (c.role == "up") { input = W::normalized; output = W::up; }
        if (c.role == "down") { input = W::gate; output = W::down; }
        if (c.role == "LM_head") { input = W::selected_hidden; output = W::logits; }
    } else if (c.operation == "rms_norm" || c.operation == "rope") {
        if (c.role == "query") { input = W::query; output = W::attention; }
        if (c.role == "key") { input = W::key; output = W::value; }
        if (c.operation == "rope") { output = input; }
    } else if (c.operation == "softmax") { input = W::scores; output = W::probabilities; }
    else { input = W::query; output = W::attention; }
    p.x = storage.workspace<float>(input, c.m);
    p.y = storage.workspace<float>(output, c.m);
    if (p.x.columns != (c.operation == "matrix" ? c.k : c.n) || p.y.columns != c.n) {
        throw std::runtime_error("micro 形状与实际 workspace 不一致");
    }
    p.input.resize(p.x.rows*p.x.columns);
    p.output.resize(p.y.rows*p.y.columns);
    for (std::size_t i = 0; i < p.input.size(); ++i) { p.input[i] = input_value(i, c.seed); }
    if (c.operation == "matrix") {
        const auto w = storage.weight(c.tensor);
        if (w.rows != c.n || w.columns != c.k) { throw std::runtime_error("micro 矩阵形状与实际权重不一致"); }
        const auto source = model.source().tensor(c.tensor == "output.weight" && model.tied_output() ? "token_embd.weight" : c.tensor);
        std::vector<float> weight(c.k);
        for (auto col : sample_columns(c.n)) {
            decode_row(source.type, source.row(col), weight.data(), c.k);
            for (auto row : sample_rows(c.m)) {
                double sum = 0;
                for (std::size_t k = 0; k < c.k; ++k) { sum += double(p.input[row*c.k+k])*weight[k]; }
                p.points.emplace_back(row*c.n+col, sum);
            }
        }
    } else if (c.operation == "rms_norm") {
        const auto source = model.source().tensor(c.tensor);
        p.norm.resize(c.group_width);
        decode_row(source.type, source.row(0), p.norm.data(), p.norm.size());
        p.expected.resize(p.output.size());
        for (std::size_t first = 0; first < p.input.size(); first += c.group_width) {
            double square = 0;
            for (std::size_t i = 0; i < c.group_width; ++i) { square += double(p.input[first+i])*p.input[first+i]; }
            const auto scale = 1.0/std::sqrt(square/double(c.group_width)+d.rms_epsilon);
            for (std::size_t i = 0; i < c.group_width; ++i) { p.expected[first+i] = p.input[first+i]*scale*p.norm[i]; }
        }
    } else if (c.operation == "rope") {
        p.coefficients.resize(max_length*d.head_dim);
        download(p.context, storage.workspace<float>(W::rope_coefficients, max_length), p.coefficients, p.transfers);
        p.expected.assign(p.input.begin(), p.input.end());
        for (std::size_t repeat = 0; repeat < calls_per_sample; ++repeat) {
            for (std::size_t row = 0; row < c.m; ++row) {
                for (std::size_t group = 0; group < c.n; group += d.head_dim) {
                    for (std::size_t i = 0; i < d.head_dim/2; ++i) {
                        const auto left = row*c.n+group+i, right = left+d.head_dim/2;
                        const auto offset = std::size_t(c.positions[row])*d.head_dim+i;
                        const double cosine = p.coefficients[offset], sine = p.coefficients[offset+d.head_dim/2];
                        const auto a = p.expected[left], b = p.expected[right];
                        p.expected[left] = a*cosine-b*sine; p.expected[right] = a*sine+b*cosine;
                    }
                }
            }
        }
    } else if (c.operation == "softmax") {
        std::fill(p.input.begin(), p.input.end(), std::numeric_limits<float>::quiet_NaN());
        p.probability_initial.assign(p.output.size(), -719.25f);
        p.expected.assign(p.output.size(), -719.25);
        for (std::size_t row = 0; row < c.m; ++row) {
            for (std::size_t head = 0; head < d.heads; ++head) {
                const auto start = row*c.n+head*max_length, length = std::size_t(c.positions[row])+1;
                for (std::size_t pos = 0; pos < length; ++pos) {
                    p.input[start+pos] = float(int((pos*7+head*3+row*11)%127)-63)/16.0f;
                }
                const auto maximum = *std::max_element(p.input.begin()+start, p.input.begin()+start+length);
                double total = 0;
                for (std::size_t pos = 0; pos < length; ++pos) { total += std::exp(double(p.input[start+pos])-maximum); }
                for (std::size_t pos = 0; pos < c.max_context; ++pos) {
                    p.expected[start+pos] = pos < length ? std::exp(double(p.input[start+pos])-maximum)/total : 0.0;
                }
            }
        }
    } else { attention_reference(p, c, d); }
}

json verify(Prepared& p) {
    for (auto value : p.output) {
        if (!std::isfinite(value)) { throw std::runtime_error("micro 输出含非有限值"); }
    }
    double maximum = 0, ratio = 0;
    json points = json::array();
    const auto compare = [&](std::size_t index, double expected) {
        const auto error = std::abs(double(p.output.at(index))-expected), limit = 2e-4+2e-4*std::abs(expected);
        maximum = std::max(maximum,error); ratio = std::max(ratio,error/limit);
        if (!std::isfinite(expected) || error > limit) {
            throw std::runtime_error("micro FP64 对照失败，index="+std::to_string(index)+" absolute="+std::to_string(error));
        }
    };
    for (std::size_t i = 0; i < p.expected.size(); ++i) { compare(i, p.expected[i]); }
    for (const auto& [i, value] : p.points) {
        compare(i, value);
        points.push_back({{"index",i},{"actual",p.output[i]},{"reference",value}});
    }
    return {{"all_finite", true}, {"checked_elements", p.expected.size()+p.points.size()},
        {"max_absolute", maximum}, {"max_tolerance_ratio", ratio},
        {"points",points},{"reference_sha256",p.expected.empty() ? json(nullptr) :
            json(hash_sha256_hex(p.expected.data(), p.expected.size()*sizeof(double)))},
        {"output_sha256", hash_sha256_hex(p.output.data(), p.output.size()*sizeof(float))}};
}

void run_case(json& report, const Case& c, const Qwen3Model& model, gpu::CudaStorage& storage, Events& events) {
    report = describe(c, model.dimensions());
    report["status"] = "running"; report["samples"] = json::array();
    const auto& context = storage.context();
    const auto& d = model.dimensions();
    const gpu::KvShape shape{4,d.layers,max_length,d.kv_heads,d.head_dim};
    const auto status = storage.workspace<std::int32_t>(gpu::Workspace::status,1);
    const auto slots = storage.workspace<std::int32_t>(gpu::Workspace::slots,c.m);
    const auto positions = storage.workspace<std::int32_t>(gpu::Workspace::positions,c.m);
    const auto scores = storage.workspace<float>(gpu::Workspace::scores,c.m);
    const auto probabilities = storage.workspace<float>(gpu::Workspace::probabilities,c.m);
    Prepared p(context);
    const auto preparation = Clock::now();
    prepare(p, c, model, storage);
    report["preparation_host_ns"] = elapsed(preparation);
    report["preparation_transfers"] = p.transfers.since({});
    report["input_sha256"] = hash_sha256_hex(p.input.data(),p.input.size()*sizeof(float));
    const auto operand = c.tensor.empty() ? gpu::DeviceTensorView<const float>{} : storage.weight(c.tensor);
    const auto coefficients = storage.workspace<float>(gpu::Workspace::rope_coefficients,max_length);
    const auto before = allocations();
    report["before_allocations"] = before;
    for (std::size_t iteration = 0; iteration < repetitions; ++iteration) {
        const auto setup = Clock::now();
        const auto transfers_before = p.transfers;
        upload(context, p.x, p.input, p.transfers);
        if (!c.slots.empty()) { upload(context, slots, c.slots, p.transfers); }
        if (!c.positions.empty()) { upload(context, positions, c.positions, p.transfers); }
        if (!p.probability_initial.empty()) { upload(context, p.y, p.probability_initial, p.transfers); }
        gpu::reset_status(context,status); context.synchronize();
        const auto setup_ns = elapsed(setup);
        const auto transfers_ready = p.transfers;
        const auto started = Clock::now();
        gpu::check_cuda(cudaEventRecord(events.start,context.stream()), "micro 开始计时");
        for (std::size_t call = 0; call < calls_per_sample; ++call) {
            if (c.operation == "matrix") { gpu::matrix_multiply(context,read_only(p.x),operand,p.y); }
            else if (c.operation == "rms_norm") { gpu::rms_norm(context,read_only(p.x),operand,p.y,d.rms_epsilon); }
            else if (c.operation == "rope") { gpu::rope(context,p.x,read_only(positions),read_only(coefficients),status); }
            else if (c.operation == "softmax") {
                gpu::causal_softmax(context,shape,d.heads,read_only(slots),read_only(positions),c.max_context,p.x,p.y,status);
            } else {
                gpu::causal_attention(context,read_only(storage.kv_view()),shape,0,read_only(p.x),d.heads,
                    read_only(slots),read_only(positions),c.max_context,scores,probabilities,p.y,status);
            }
        }
        gpu::check_cuda(cudaEventRecord(events.stop,context.stream()), "micro 停止计时");
        context.synchronize();
        const auto host_ns = elapsed(started);
        const auto transfers_done = p.transfers;
        const auto validation = Clock::now();
        download(context,status,p.status,p.transfers);
        float device_ms = 0;
        gpu::check_cuda(cudaEventElapsedTime(&device_ms,events.start,events.stop), "micro 读取 events");
        if (!std::isfinite(device_ms) || device_ms <= 0 || p.status[0] != 0 || p.status[1] != INT_MAX) {
            throw std::runtime_error("micro 设备状态或时钟无效");
        }
        download(context,p.y,p.output,p.transfers);
        const auto verification = verify(p);
        report["samples"].push_back({{"iteration",iteration},{"phase",iteration < 2 ? "warmup" : "measured"},
            {"calls",calls_per_sample},{"setup_host_ns",setup_ns},{"host_enqueue_to_completion_ns",host_ns},
            {"device_interval_ms",device_ms},{"validation_host_ns",elapsed(validation)},{"verification",verification},
            {"setup_transfers",transfers_ready.since(transfers_before)},
            {"measured_transfers",transfers_done.since(transfers_ready)},
            {"validation_transfers",p.transfers.since(transfers_done)}});
        if (before != allocations()) { throw std::runtime_error("micro 稳态项目设备分配或释放发生变化"); }
    }
    report["after_allocations"] = allocations();
    report["status"] = "passed";
}
}

int main(int argc, char** argv) {
    json report = {{"schema_version",1},{"benchmark","minillm-cuda-micro"},{"status","failed"},{"cases",json::array()}};
    std::string output;
    try {
        Options options(argc,argv,{"--model","--input","--output","--trial","--manifest"});
        if (options.has("--help")) {
            std::cout << "mini-cuda-kernel-bench --model MODEL.gguf --input INPUT.json --output NEW_REPORT.json\n"
                         "                       [--trial 0..4] [--manifest MANIFEST.json]\n";
            return 0;
        }
        const auto candidate = options.get("--output");
        if (candidate.empty() || std::filesystem::exists(candidate)) { throw std::invalid_argument("micro 报告必须是新文件"); }
        const auto parent = std::filesystem::path(candidate).parent_path();
        if (!parent.empty()) { std::filesystem::create_directories(parent); }
        output = candidate;
        const auto input_hash = cuda_reports::file_hash(options.get("--input"));
        if (input_hash != "6aa2bc8af5de001ab3a8baedd305d0c77822a48b5baddc8f5708e79f496e706c") {
            throw std::invalid_argument("micro 冻结输入摘要不符");
        }
        const auto recipe = read(options.get("--input"));
        const auto model_hash = cuda_reports::file_hash(options.get("--model"));
        if (recipe.at("protocol_id") != "qwen3-cuda-micro-v0" ||
            model_hash != "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031") {
            throw std::invalid_argument("micro 输入或模型身份不符");
        }
        const auto trial = options.integer("--trial",0,0,4);
        report["model_sha256"] = model_hash; report["input_sha256"] = input_hash; report["trial"] = trial;
        report["protocol"] = recipe.at("measurement"); report["run_identity"] = nullptr;
        if (options.has("--manifest")) {
            const auto manifest = read(options.get("--manifest"));
            const auto binary = cuda_reports::file_hash(std::filesystem::canonical(argv[0]));
            if (manifest.at("benchmark") != "minillm-cuda-micro" || manifest.at("schema_version") != 1 ||
                manifest.at("binary").at("sha256") != binary || manifest.at("model").at("sha256") != model_hash ||
                manifest.at("input").at("sha256") != input_hash ||
                manifest.at("reports").at(trial).at("trial") != trial ||
                manifest.at("reports").at(trial).at("file") != std::filesystem::path(output).filename().string()) {
                throw std::invalid_argument("micro manifest 身份不符");
            }
            report["run_identity"] = {{"run_id",manifest.at("run_id")},
                {"manifest_sha256",cuda_reports::file_hash(options.get("--manifest"))},{"binary_sha256",binary},
                {"source_state_sha256",manifest.at("source").at("worktree_state_sha256")}};
        }
        report["before_initialization_allocations"] = allocations();
        {
            const auto load_start = Clock::now();
            Qwen3Model model(options.get("--model"));
            report["model_load_ns"] = elapsed(load_start);
            const auto& d = model.dimensions();
            if (d.heads != 16 || d.kv_heads != 8 || d.head_dim != 128) { throw std::invalid_argument("micro 模型 head 配置不符"); }
            auto cases = make_cases(recipe,d);
            report["dimensions"] = {{"embedding",d.embedding},{"layers",d.layers},{"heads",d.heads},{"kv_heads",d.kv_heads},
                {"head_dim",d.head_dim},{"feed_forward",d.feed_forward},{"vocabulary",d.vocabulary},
                {"rms_epsilon",d.rms_epsilon},{"rope_base",d.rope_base}};
            report["memory_plan"] = memory(gpu::make_memory_plan(model,{}));
            const auto storage_start = Clock::now();
            gpu::CudaStorage storage(model);
            report["storage_initialization_ns"] = elapsed(storage_start);
            report["weight_decode_upload_ns"] = storage.weight_decode_upload_ns();
            report["weight_h2d_bytes"] = storage.uploaded_bytes();
            report["rope_h2d_bytes"] = storage.rope_uploaded_bytes();
            report["owned_device_bytes"] = storage.allocated_bytes();
            if (storage.allocated_bytes() != storage.plan().total_bytes) { throw std::runtime_error("micro 内存计划与分配不符"); }
            report["weights"] = weights(storage); report["device"] = device(storage.context());
            report["arithmetic"] = {{"source_weight_dtype","Q8_0"},{"device_weight_dtype","F32"},{"activation_dtype","F32"},
                {"kv_dtype","F16"},{"kv_rounding","nearest_even"},{"qk_pv_accumulation_dtype","F32"},
                {"softmax_exponential_dtype","F32"},{"softmax_denominator_dtype","F64"},
                {"gemm_compute","CUBLAS_COMPUTE_32F_PEDANTIC"},{"fast_math",false}};
            Events events(storage.context());
            const auto kv_start = Clock::now();
            report["kv_initialization_transfers"] = initialize_kv(storage);
            report["kv_initialization_ns"] = elapsed(kv_start);
            report["before_cases_allocations"] = allocations();
            if (trial % 2) { std::reverse(cases.begin(),cases.end()); }
            for (const auto& c : cases) {
                report["cases"].push_back(json::object());
                run_case(report["cases"].back(),c,model,storage,events);
                std::cout << c.name << " 通过\n" << std::flush;
            }
            report["after_cases_allocations"] = allocations();
            if (report["before_cases_allocations"] != report["after_cases_allocations"]) {
                throw std::runtime_error("micro 测量区间出现项目设备分配或释放");
            }
        }
        report["after_destruction_allocations"] = allocations();
        report["status"] = "passed";
        cuda_reports::write(output,report);
        return 0;
    } catch (const std::exception& e) {
        report["error"] = e.what();
        if (!output.empty()) {
            try { cuda_reports::write(output,report); }
            catch (const std::exception& error) { std::cerr << "micro 报告写入失败：" << error.what() << '\n'; }
        }
        std::cerr << "mini-cuda-kernel-bench: " << e.what() << '\n';
        return 1;
    }
}
