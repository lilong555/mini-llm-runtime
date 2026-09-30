#include "qwen3_fixture.h"
#include "storage.h"
#include "page_table.h"
#include "layer.h"
#include "ops.h"
#include "tensor_validation.h"
#include "nlohmann/json.hpp"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace minillm;
using namespace minillm::cuda;
using json = nlohmann::ordered_json;

#ifdef MINILLM_TEST_CUDA_ALLOC_FAILURE
namespace allocation_failure {
int calls = 0, fail_call = 0;
struct Injection {
    explicit Injection(int fail) { calls = 0; fail_call = fail; }
    ~Injection() { fail_call = 0; }
};
}
// 链接器仅拦截项目中的直接调用，不干预动态库内部的 CUDA 分配。
extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
    if (++allocation_failure::calls == allocation_failure::fail_call) {
        *pointer = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaMalloc(pointer, bytes);
}
#endif

namespace {
constexpr StorageLimits small{2, 16, 8, 0};
constexpr double unit_atol = 2e-4, unit_rtol = 2e-4;
std::string hash(sha256_t& state) {
    unsigned char bytes[32];
    sha256_final(&state, bytes);
    std::ostringstream out;
    for (auto b : bytes) { out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b); }
    return out.str();
}
std::string file_hash(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("无法读取文件：" + path.string()); }
    sha256_t state;
    sha256_init(&state);
    std::array<char, 65536> buffer{};
    while (file.read(buffer.data(), buffer.size()) || file.gcount()) {
        sha256_update(&state, reinterpret_cast<const unsigned char*>(buffer.data()),
                      static_cast<std::size_t>(file.gcount()));
    }
    if (!file.eof()) { throw std::runtime_error("文件读取失败：" + path.string()); }
    return hash(state);
}
std::string check_model_identity(const std::filesystem::path& path, const std::string& expected) {
    const auto actual = file_hash(path);
    if (actual != expected) {
        throw std::runtime_error("模型 SHA-256 不符合固定契约：expected=" + expected + " actual=" + actual);
    }
    return actual;
}
void prepare_output(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    if (!std::filesystem::create_directory(path)) {
        throw std::runtime_error("报告目录已存在，必须使用新目录：" + path.string());
    }
}
void write_json(const std::filesystem::path& path, const json& value) {
    std::ofstream file(path);
    file << value.dump(2) << '\n';
    file.close();
    if (!file) { throw std::runtime_error("写入报告失败：" + path.string()); }
}
const char* dtype(WeightType t) {
    switch (t) {
        case WeightType::f32: return "F32";
        case WeightType::f16: return "F16";
        case WeightType::q8_0: return "Q8_0";
    }
    throw std::runtime_error("未知 dtype");
}
void same_allocations(AllocationStats a, AllocationStats b) {
    CHECK(a.allocation_calls == b.allocation_calls);
    CHECK(a.allocations == b.allocations);
    CHECK(a.release_calls == b.release_calls);
    CHECK(a.releases == b.releases);
    CHECK(a.allocated_bytes == b.allocated_bytes);
}
void download(const CudaContext& c, void* host, const void* device, std::size_t bytes) {
    DeviceScope scope(c.device());
    check_cuda(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, c.stream()), "验证下载");
    c.synchronize();
}
void upload(const CudaContext& c, void* device, const void* host, std::size_t bytes) {
    DeviceScope scope(c.device());
    check_cuda(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, c.stream()), "验证输入上传");
    c.synchronize();
}

json verify_weights(const Qwen3Model& model, const CudaStorage& storage) {
    json records = json::array();
    std::vector<float> expected(weight_staging_bytes / sizeof(float)), actual(expected.size());
    std::vector<std::uint16_t> rounded(expected.size());
    const auto pointer = [&](const WeightRecord& w) -> const void* {
        if (w.device_storage_type == StorageType::f16) {
            return std::get<DeviceTensorView<const std::uint16_t>>(storage.matrix_weight(w.name)).data;
        }
        return storage.weight(w.name).data;
    };
    std::size_t bytes_verified = 0;
    for (const auto& w : storage.plan().weights) {
        const auto* data = static_cast<const std::byte*>(pointer(w));
        const bool half = w.device_storage_type == StorageType::f16;
        const auto width = half ? sizeof(std::uint16_t) : sizeof(float);
        CHECK(reinterpret_cast<std::uintptr_t>(data) % 256 == 0);
        if (!w.alias_of.empty()) {
            CHECK(data == pointer(storage.plan().weights.front()));
            CHECK(w.effective_sha256 == storage.plan().weights.front().effective_sha256);
            CHECK(w.device_payload_sha256 == storage.plan().weights.front().device_payload_sha256);
        } else {
            const auto source = model.source().tensor(w.name);
            const auto chunk = expected.size() / w.columns;
            sha256_t state, payload;
            sha256_init(&state);
            sha256_init(&payload);
            for (std::size_t first = 0; first < w.rows; first += chunk) {
                const auto count = std::min(chunk, w.rows - first);
                const auto elements = count * w.columns, bytes = elements * width;
                for (std::size_t row = 0; row < count; ++row) {
                    decode_row(source.type, source.row(first + row), expected.data() + row * w.columns, w.columns);
                }
                sha256_update(&state, reinterpret_cast<const unsigned char*>(expected.data()), elements * sizeof(float));
                if (half) {
                    for (std::size_t i = 0; i < elements; ++i) { rounded[i] = float_to_half(expected[i]); }
                }
                download(storage.context(), actual.data(), data + first * w.columns * width, bytes);
                const void* reference = half ? static_cast<const void*>(rounded.data()) : expected.data();
                CHECK(std::memcmp(reference, actual.data(), bytes) == 0);
                sha256_update(&payload, reinterpret_cast<const unsigned char*>(actual.data()), bytes);
                bytes_verified += bytes;
            }
            CHECK(hash(state) == w.effective_sha256);
            CHECK(hash(payload) == w.device_payload_sha256);
        }
        records.push_back({{"name", w.name}, {"shape", {w.rows, w.columns}}, {"source_dtype", dtype(w.source_type)},
                           {"device_dtype", half ? "F16" : "F32"}, {"offset", w.offset}, {"bytes", w.bytes},
                           {"alias_of", w.alias_of}, {"effective_sha256", w.effective_sha256},
                           {"device_payload_sha256", w.device_payload_sha256}});
    }
    CHECK(bytes_verified == storage.plan().weight_payload);
    CHECK(storage.uploaded_bytes() == bytes_verified);
    CHECK(storage.max_upload_chunk_bytes() > 0);
    CHECK(storage.max_upload_chunk_bytes() <= weight_staging_bytes);
    CHECK(storage.weight_staging_peak_bytes() <= weight_staging_bytes);
    return {{"verified_unique_bytes", bytes_verified}, {"records", records}};
}

void check_layout(const MemoryPlan& p) {
    std::size_t end = 0, payload = 0;
    for (const auto& w : p.weights) {
        if (!w.alias_of.empty()) { continue; }
        CHECK(w.offset % 256 == 0 && w.offset >= end);
        CHECK(w.bytes == w.rows * w.columns * (w.device_storage_type == StorageType::f16 ? 2 : 4));
        end = w.offset + w.bytes;
        payload += w.bytes;
    }
    CHECK(end <= p.weight_bytes && p.weight_bytes % 256 == 0);
    CHECK(payload == p.weight_payload);
    end = 0;
    std::size_t workspace_payload = 0;
    for (const auto& r : p.regions) {
        CHECK(r.offset % 256 == 0 && r.offset >= end);
        CHECK(r.bytes == r.rows * r.columns * (r.type == StorageType::f16 ? 2 : 4));
        end = r.offset + r.bytes;
        workspace_payload += r.bytes;
    }
    CHECK(end <= p.workspace_bytes && p.workspace_bytes % 256 == 0);
    CHECK(workspace_payload == p.activation_bytes + p.attention_bytes + p.logits_bytes + p.metadata_bytes + p.rope_bytes);
    CHECK(p.padding_bytes == p.weight_bytes - payload + p.workspace_bytes - workspace_payload);
    CHECK(p.total_bytes == payload + workspace_payload + p.padding_bytes + p.kv_bytes + p.cublas_bytes);
}
}

TEST(storage_report_rejects_reuse_without_modification) {
    Qwen3Fixture fixture;
    const auto output = std::filesystem::path(fixture.write()).parent_path() / "reports" / "trial";
    prepare_output(output);
    const auto summary = output / "validation-summary.json";
    write_json(summary, {{"passed", true}});
    const auto before = file_hash(summary);
    test::throws<std::runtime_error>([&] { prepare_output(output); });
    CHECK(file_hash(summary) == before);
    test::throws<std::filesystem::filesystem_error>([&] { prepare_output(summary / "child"); });
    CHECK(file_hash(summary) == before);
}

TEST(storage_model_identity_precedes_device_allocation) {
    Qwen3Fixture fixture;
    const auto path = std::filesystem::path(fixture.write()).parent_path() / "identity";
    { std::ofstream file(path, std::ios::binary); file << "abc"; CHECK(file); }
    const std::string expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const auto before = allocation_stats();
    CHECK(check_model_identity(path, expected) == expected);
    test::throws<std::runtime_error>([&] { check_model_identity(path, std::string(64, '0')); });
    test::throws<std::runtime_error>([&] { check_model_identity(path.string() + ".missing", expected); });
    same_allocations(before, allocation_stats());
}

TEST(storage_plan_alignment_alias_and_untied) {
    Qwen3Fixture fixture;
    Qwen3Model tied(fixture.write()), untied(fixture.write(false));
    const auto a = make_memory_plan(tied, small), b = make_memory_plan(untied, small);
    check_layout(a); check_layout(b);
    CHECK(a.weights.size() == 25 && b.weights.size() == 25);
    CHECK(a.weights[2].alias_of == "token_embd.weight" && a.weights[2].offset == a.weights[0].offset);
    CHECK(b.weights[2].alias_of.empty() && b.weights[2].offset != b.weights[0].offset);
    CHECK(b.weight_payload - a.weight_payload == 9 * 4 * sizeof(float));
    CHECK(a.kv_bytes == 2 * 2 * 2 * 16 * 4 * 2);
}

TEST(storage_limits_and_budget_boundaries) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    for (const auto bad : {StorageLimits{0,16,8,0}, {2,0,8,0}, {2,16,0,0}, {2,17,8,0},
                          {std::size_t(INT_MAX) + 1,16,8,0}, {2,16,std::size_t(INT_MAX) + 1,0}}) {
        test::throws<std::invalid_argument>([&] { make_memory_plan(model, bad); });
    }
    test::throws<std::overflow_error>([] { checked_product(SIZE_MAX, 4); });
    auto p = make_memory_plan(model, small);
    constexpr std::size_t gib = 1024ULL * 1024 * 1024;
    const MemoryInfo roomy{4 * gib, 8 * gib};
    CHECK(memory_budget(p, roomy) == (roomy.free_bytes / 5) * 4 + (roomy.free_bytes % 5) * 4 / 5);
    CHECK(memory_budget(p, {gib, 8 * gib}) == gib / 2);
    CHECK(memory_budget(p, {gib / 2 - 1, gib}) == 0);
    CHECK(memory_budget(p, {SIZE_MAX, SIZE_MAX}) == (SIZE_MAX / 5) * 4);
    p.limits.device_budget_bytes = p.total_bytes;
    check_memory_budget(p, roomy);
    --p.limits.device_budget_bytes;
    try { check_memory_budget(p, roomy); CHECK(false); }
    catch (const Error& e) {
        CHECK(std::string(e.what()).find("workspace logits") != std::string::npos);
        CHECK(std::string(e.what()).find("weight token_embd.weight") != std::string::npos);
    }
    p.limits.device_budget_bytes = 0;
    p.total_bytes = memory_budget(p, roomy);
    check_memory_budget(p, roomy);
    ++p.total_bytes;
    test::throws<Error>([&] { check_memory_budget(p, roomy); });
}

TEST(storage_invalid_precision_is_rejected_before_device_allocation) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    auto limits = small;
    CHECK(limits.precision_mode == PrecisionMode::f32_pedantic);
    CHECK(make_memory_plan(model, limits).limits.precision_mode == limits.precision_mode);
    const auto before = allocation_stats();
    limits.precision_mode = static_cast<PrecisionMode>(99);
    test::throws<std::invalid_argument>([&] { make_memory_plan(model, limits); });
    test::throws<std::invalid_argument>([&] { CudaStorage storage(model, limits); });
    same_allocations(before, allocation_stats());
}

TEST(storage_kv_layout_contract_and_explicit_contiguous_capacity) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    auto limits = small;
    const auto original = make_memory_plan(model,limits);
    limits.kv_capacity_tokens = limits.max_sequences * limits.max_model_len;
    CHECK(make_memory_plan(model,limits).total_bytes == original.total_bytes);
    const auto before = allocation_stats();
    --limits.kv_capacity_tokens;
    test::throws<std::invalid_argument>([&] { CudaStorage storage(model,limits); });
    limits.kv_layout = CudaKvLayout::paged;
    limits.kv_capacity_tokens = 16;
    limits.precision_mode = PrecisionMode::f16_matrix_f32acc;
    try { CudaStorage storage(model,limits); CHECK(false); }
    catch (const std::invalid_argument& error) {
        CHECK(std::string(error.what()).find("F32") != std::string::npos);
    }
    same_allocations(before,allocation_stats());
}

TEST(storage_paged_pool_and_table_are_counted_once) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write(true,[](gguf_context* info) {
        gguf_set_val_u32(info,"qwen3.context_length",65);
    }));
    StorageLimits limits{4,65,8,0};
    const auto baseline = make_memory_plan(model,limits);
    CHECK(baseline.kv_capacity_tokens == 260 && baseline.physical_pages == 0 && baseline.kv_table_bytes == 0);
    limits.kv_layout = CudaKvLayout::paged; limits.kv_capacity_tokens = 80;
    const auto plan = make_memory_plan(model,limits);
    check_layout(plan);
    CHECK(plan.physical_pages == 5 && plan.kv_capacity_tokens == 80 && plan.kv_table_bytes == 80);
    CHECK(plan.kv_bytes == 80 * 2 * 2 * 4 * sizeof(std::uint16_t));
    CHECK(plan.metadata_bytes == baseline.metadata_bytes + plan.kv_table_bytes);
    CHECK(plan.total_bytes == baseline.total_bytes - baseline.kv_bytes + plan.kv_bytes +
                             plan.workspace_bytes - baseline.workspace_bytes);
    {
        CudaStorage storage(model,limits);
        CHECK(storage.allocated_bytes() == plan.total_bytes && storage.allocations() == 4);
        CHECK(storage.page_table_h2d_bytes() == 0);
        const auto table = storage.kv_block_table();
        CHECK(table.rows == 4 && table.columns == 5 && table.capacity == 20);
        std::vector<std::int32_t> table_data(20,0);
        download(storage.context(),table_data.data(),table.data,plan.kv_table_bytes);
        CHECK(table_data == std::vector<std::int32_t>(20,-1));
        const auto slab = storage.paged_kv_view();
        CHECK(slab.rows == 2 * 2 * 5 * 16 && slab.columns == 4);
        std::vector<std::uint16_t> data(slab.capacity);
        download(storage.context(),data.data(),slab.data,plan.kv_bytes);
        CHECK(std::all_of(data.begin(),data.end(),[](auto value) { return value == 0xffff; }));
        test::throws<std::invalid_argument>([&] { storage.kv_view(); });
        LayerExecutor layer(storage);
    }
    {
        CudaStorage storage(model,{4,65,8,0});
        test::throws<std::invalid_argument>([&] { storage.paged_kv_view(); });
        test::throws<std::invalid_argument>([&] { storage.kv_block_table(); });
        test::throws<std::invalid_argument>([&] { storage.upload_page_table({}); });
    }
    limits.device_budget_bytes = plan.total_bytes - 1;
    const auto before = allocation_stats();
    test::throws<Error>([&] { CudaStorage storage(model,limits); });
    const auto after = allocation_stats();
    CHECK(after.allocations - before.allocations == 1 && after.releases - before.releases == 1);
}

TEST(storage_paged_table_store_clear_reuse_and_upload_accounting) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write(true,[](gguf_context* info) {
        gguf_set_val_u32(info,"qwen3.context_length",33);
    }));
    StorageLimits limits{2,33,8,0};
    limits.kv_layout = CudaKvLayout::paged; limits.kv_capacity_tokens = 64;
    PageTableState pages({2,33,16,4});
    // host 页表先创建、后销毁，异常路径上的存储析构仍可先等待 H2D 完成。
    CudaStorage storage(model,limits);
    const auto& context = storage.context();
    const auto slab = storage.paged_kv_view();
    const auto table = storage.kv_block_table();
    const PagedKvMapping mapping{table,4,16};
    const KvShape shape{2,2,33,1,4};
    std::vector<std::uint16_t> expected(slab.capacity,0xffff), actual(slab.capacity);
    std::array<std::size_t,2> lengths{};
    const auto allocations = allocation_stats();
    const auto run = [&](std::array<std::size_t,2> next, std::int32_t slot, std::int32_t position, float base) {
        pages.prepare(lengths,next);
        const bool changed = pages.upload_required();
        const auto before = storage.page_table_h2d_bytes();
        pages.begin_execution();
        if (changed) { storage.upload_page_table(pages.device_table_for_upload()); }
        const auto kv = std::vector<float>{base,base+1,base+2,base+3};
        const std::int32_t slots[]{slot}, positions[]{position};
        const auto key = storage.workspace<float>(Workspace::key,1), value = storage.workspace<float>(Workspace::value,1);
        const auto s = storage.workspace<std::int32_t>(Workspace::slots,1);
        const auto p = storage.workspace<std::int32_t>(Workspace::positions,1);
        const auto status = storage.workspace<std::int32_t>(Workspace::status,1);
        upload(context,key.data,kv.data(),kv.size()*sizeof(float));
        upload(context,value.data,kv.data(),kv.size()*sizeof(float));
        upload(context,s.data,slots,sizeof(slots)); upload(context,p.data,positions,sizeof(positions));
        reset_status(context,status);
        for (std::size_t layer = 0; layer < shape.layers; ++layer) {
            store_kv(context,slab,shape,mapping,layer,
                minillm::cuda::detail::read_only(key),minillm::cuda::detail::read_only(value),
                minillm::cuda::detail::read_only(s),minillm::cuda::detail::read_only(p),status);
        }
        std::array<std::int32_t,2> result{};
        download(context,result.data(),status.data,sizeof(result));
        CHECK(result[0] == 0);
        pages.commit(); lengths = next;
        CHECK(storage.page_table_h2d_bytes() - before == (changed ? table.capacity*sizeof(std::int32_t) : 0));
        std::vector<std::int32_t> device_table(table.capacity);
        download(context,device_table.data(),table.data,device_table.size()*sizeof(std::int32_t));
        CHECK(std::equal(device_table.begin(),device_table.end(),pages.committed_table().begin()));
        const auto page = std::size_t(device_table[std::size_t(slot)*3+std::size_t(position)/16]);
        for (std::size_t layer = 0; layer < 2; ++layer) {
            for (std::size_t kind = 0; kind < 2; ++kind) {
                const auto row = ((layer*2+kind)*4+page)*16+std::size_t(position)%16;
                for (std::size_t c = 0; c < 4; ++c) { expected[row*4+c] = float_to_half(kv[c]); }
            }
        }
        download(context,actual.data(),slab.data,storage.plan().kv_bytes);
        CHECK(actual == expected);
    };
    run({17,1},0,16,1.0f);
    run({17,2},1,1,11.0f);
    CHECK(storage.page_table_h2d_bytes() == table.capacity*sizeof(std::int32_t));
    pages.clear(0); lengths[0] = 0;
    download(context,actual.data(),slab.data,storage.plan().kv_bytes);
    CHECK(actual == expected);
    run({1,2},0,0,21.0f);
    CHECK(storage.page_table_h2d_bytes() == 2*table.capacity*sizeof(std::int32_t));
    const auto uploads = storage.page_table_h2d_bytes();
    std::vector<std::int32_t> invalid(table.capacity,-1);
    test::throws<std::invalid_argument>([&] { storage.upload_page_table({invalid.data(),invalid.size()-1}); });
    for (const auto id : {-2,4,INT_MAX}) {
        invalid[0] = id;
        test::throws<std::invalid_argument>([&] { storage.upload_page_table(invalid); });
    }
    CHECK(storage.page_table_h2d_bytes() == uploads);
    same_allocations(allocations,allocation_stats());
}

TEST(storage_budget_failure_has_no_arena_allocation) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    const auto before = allocation_stats();
    test::throws<Error>([&] { CudaStorage storage(model, {2,16,8,1}); });
    const auto after = allocation_stats();
    CHECK(after.allocations - before.allocations == 1);
    CHECK(after.releases - before.releases == 1);
    CHECK(after.allocated_bytes - before.allocated_bytes == CudaContext::default_workspace_bytes);
}

#ifdef MINILLM_TEST_CUDA_ALLOC_FAILURE
TEST(storage_allocation_failures_release_partial_owners) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    for (int fail = 1; fail <= 4; ++fail) {
        const auto before = allocation_stats();
        {
            allocation_failure::Injection injection(fail);
            try { CudaStorage storage(model, small); CHECK(false); }
            catch (const Error& error) {
                CHECK(std::string(error.what()).find("cudaErrorMemoryAllocation") != std::string::npos);
                if (fail > 1) { CHECK(std::string(error.what()).find("weight_arena=") != std::string::npos); }
            }
        }
        const auto after = allocation_stats();
        CHECK(after.allocation_calls - before.allocation_calls == std::size_t(fail));
        CHECK(after.allocations - before.allocations == std::size_t(fail - 1));
        CHECK(after.releases - before.releases == std::size_t(fail - 1));
        CHECK(after.release_calls - before.release_calls == after.releases - before.releases);
        CudaStorage recovered(model, small);
        verify_weights(model, recovered);
    }
}

TEST(storage_half_allocation_failures_release_partial_owners) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    auto limits = small;
    limits.precision_mode = PrecisionMode::f16_matrix_f32acc;
    for (int fail = 1; fail <= 4; ++fail) {
        const auto before = allocation_stats();
        {
            allocation_failure::Injection injection(fail);
            test::throws<Error>([&] { CudaStorage storage(model, limits); });
        }
        const auto after = allocation_stats();
        CHECK(after.allocation_calls - before.allocation_calls == std::size_t(fail));
        CHECK(after.allocations - before.allocations == std::size_t(fail - 1));
        CHECK(after.releases - before.releases == std::size_t(fail - 1));
    }
    CudaStorage recovered(model, limits);
    verify_weights(model, recovered);
}

TEST(storage_paged_allocation_failures_release_partial_owners) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    auto limits = small;
    limits.kv_layout = CudaKvLayout::paged; limits.kv_capacity_tokens = 16;
    for (int fail = 1; fail <= 4; ++fail) {
        const auto before = allocation_stats();
        {
            allocation_failure::Injection injection(fail);
            test::throws<Error>([&] { CudaStorage storage(model,limits); });
        }
        const auto after = allocation_stats();
        CHECK(after.allocations - before.allocations == std::size_t(fail-1));
        CHECK(after.releases - before.releases == std::size_t(fail-1));
    }
    CudaStorage recovered(model,limits);
    CHECK(recovered.plan().physical_pages == 1);
}
#endif

TEST(storage_f32_f16_tied_and_untied_upload) {
    Qwen3Fixture fixture;
    for (const auto dtype : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (bool tied : {true, false}) {
            Qwen3Model model(fixture.write(tied, {}, {}, {}, dtype));
            CudaStorage storage(model, small);
            verify_weights(model, storage);
            test::throws<std::out_of_range>([&] { storage.weight("missing"); });
            auto limits = small;
            limits.precision_mode = PrecisionMode::f16_matrix_f32acc;
            CudaStorage half(model, limits);
            check_layout(half.plan());
            verify_weights(model, half);
            for (std::size_t i = 0; i < half.plan().weights.size(); ++i) {
                const auto& w = half.plan().weights[i];
                const bool norm = w.name.ends_with("_norm.weight");
                CHECK(w.device_storage_type == (norm ? StorageType::f32 : StorageType::f16));
                CHECK(w.effective_sha256 == storage.plan().weights[i].effective_sha256);
                if (norm) {
                    CHECK(half.norm_weight(w.name).data == half.weight(w.name).data);
                    test::throws<std::invalid_argument>([&] { half.matrix_weight(w.name); });
                } else {
                    test::throws<std::invalid_argument>([&] { half.weight(w.name); });
                    test::throws<std::invalid_argument>([&] { half.norm_weight(w.name); });
                }
            }
            CHECK(half.allocated_bytes() == half.plan().total_bytes && half.allocations() == 4);
            const auto scratch = half.matrix_input(8, 8);
            const auto smaller = half.matrix_input(2, 4);
            CHECK(scratch.data == smaller.data && smaller.stride == 4 && smaller.columns == 4);
            CHECK(scratch.capacity == smaller.capacity && scratch.capacity == 64);
            CHECK(reinterpret_cast<std::uintptr_t>(smaller.data) % 256 == 0);
            test::throws<std::invalid_argument>([&] { half.matrix_input(9, 4); });
            test::throws<std::invalid_argument>([&] { half.matrix_input(1, 9); });
            test::throws<std::invalid_argument>([&] { half.matrix_input(0, 4); });
            test::throws<std::invalid_argument>([&] { half.matrix_input(1, 0); });
            test::throws<std::out_of_range>([&] { storage.matrix_input(1, 4); });
        }
    }
}

TEST(storage_half_overflow_and_budget_release_owners) {
    Qwen3Fixture fixture;
    auto limits = small;
    limits.precision_mode = PrecisionMode::f16_matrix_f32acc;
    for (float value : {65520.0f, -65520.0f, std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN()}) {
        Qwen3Model model(fixture.write(true, {}, {}, {}, GGML_TYPE_F32, true, value));
        const auto before = allocation_stats();
        test::throws<Error>([&] { CudaStorage storage(model, limits); });
        const auto after = allocation_stats();
        CHECK(after.allocations - before.allocations == 4);
        CHECK(after.releases - before.releases == 4);
    }
    Qwen3Model model(fixture.write());
    limits.device_budget_bytes = make_memory_plan(model, limits).total_bytes - 1;
    const auto before = allocation_stats();
    test::throws<Error>([&] { CudaStorage storage(model, limits); });
    const auto after = allocation_stats();
    CHECK(after.allocations - before.allocations == 1 && after.releases - before.releases == 1);
}

TEST(storage_half_matrix_groups_reuse_actual_shape_scratch) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    auto limits = small;
    limits.precision_mode = PrecisionMode::f16_matrix_f32acc;
    CudaStorage storage(model, limits);
    const auto& context = storage.context();
    const auto status = storage.workspace<std::int32_t>(Workspace::status, 1);
    struct Group { std::size_t columns; std::vector<std::string> weights; };
    const std::array<Group, 4> groups{{
        {4, {"attn_q", "attn_k", "attn_v"}}, {8, {"attn_output"}},
        {4, {"ffn_gate", "ffn_up"}}, {6, {"ffn_down"}}
    }};
    constexpr std::size_t m = 2;
    std::vector<DeviceBuffer<float>> inputs, outputs;
    std::vector<std::vector<float>> host_inputs;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i];
        host_inputs.emplace_back(m * group.columns);
        auto& values = host_inputs.back();
        for (std::size_t j = 0; j < values.size(); ++j) { values[j] = float(int(i * 3 + j) - 7) / 31; }
        inputs.emplace_back(values.size());
        upload(context, inputs.back().data(), values.data(), inputs.back().bytes());
        for (const auto& name : group.weights) {
            const auto weight = std::get<DeviceTensorView<const std::uint16_t>>(
                storage.matrix_weight("blk.0." + name + ".weight"));
            outputs.emplace_back(m * weight.rows);
        }
    }
    const auto before = allocation_stats();
    reset_status(context, status);
    std::size_t output_index = 0;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i];
        const auto scratch = storage.matrix_input(m, group.columns);
        cast_matrix_input(context, matrix_view(std::as_const(inputs[i]), m, group.columns), scratch, status);
        for (const auto& name : group.weights) {
            const auto w = std::get<DeviceTensorView<const std::uint16_t>>(
                storage.matrix_weight("blk.0." + name + ".weight"));
            matrix_multiply(context, minillm::cuda::detail::read_only(scratch), w,
                             matrix_view(outputs[output_index++], m, w.rows));
        }
    }
    context.synchronize();
    same_allocations(before, allocation_stats());
    std::array<std::int32_t, 2> errors{};
    download(context, errors.data(), status.data, sizeof(errors));
    CHECK(errors[0] == 0 && errors[1] == INT_MAX);
    output_index = 0;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        for (const auto& name : groups[i].weights) {
            const auto w = model.source().tensor("blk.0." + name + ".weight");
            const auto& buffer = outputs[output_index++];
            std::vector<float> actual(buffer.size()), row(w.columns);
            download(context, actual.data(), buffer.data(), buffer.bytes());
            for (std::size_t n = 0; n < w.rows; ++n) {
                decode_row(w.type, w.row(n), row.data(), w.columns);
                for (std::size_t r = 0; r < m; ++r) {
                    double expected = 0;
                    for (std::size_t k = 0; k < w.columns; ++k) {
                        expected += double(half_to_float(float_to_half(host_inputs[i][r * w.columns + k]))) *
                                    half_to_float(float_to_half(row[k]));
                    }
                    CHECK(std::abs(double(actual[r * w.rows + n]) - expected) <= unit_atol + unit_rtol * std::abs(expected));
                }
            }
        }
    }
}

TEST(storage_nonfinite_failure_cleanup_and_recovery) {
    Qwen3Fixture fixture;
    const auto infinity = std::numeric_limits<float>::infinity();
    for (auto dtype : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (float value : {infinity, -infinity, std::numeric_limits<float>::quiet_NaN()}) {
            Qwen3Model bad(fixture.write(true, {}, {}, {}, dtype, true, value));
            const auto before = allocation_stats();
            test::throws<Error>([&] { CudaStorage storage(bad, small); });
            const auto after = allocation_stats();
            CHECK(after.allocations - before.allocations == 4);
            CHECK(after.allocations - before.allocations == after.releases - before.releases);
            CHECK(after.release_calls - before.release_calls == after.releases - before.releases);
        }
    }
    Qwen3Model good(fixture.write());
    CudaStorage storage(good, small);
    verify_weights(good, storage);
}

TEST(storage_workspace_types_bounds_initialization_and_owner) {
    Qwen3Fixture fixture;
    std::unique_ptr<CudaStorage> storage;
    {
        Qwen3Model model(fixture.write());
        storage = std::make_unique<CudaStorage>(model, small);
    }
    auto& s = *storage;
    const auto h = s.workspace<float>(Workspace::hidden, 8);
    CHECK(s.workspace<float>(Workspace::hidden, 1).data == h.data);
    std::vector<float> values(h.capacity);
    download(s.context(), values.data(), h.data, values.size() * sizeof(float));
    CHECK(std::all_of(values.begin(), values.end(), [](float v) { return v == 0; }));
    const auto status = s.workspace<std::int32_t>(Workspace::status, 1);
    CHECK(status.columns == 2 && status.capacity == 2);
    test::throws<std::invalid_argument>([&] { s.workspace<float>(Workspace::status, 1); });
    test::throws<std::invalid_argument>([&] { s.workspace<std::int32_t>(Workspace::hidden, 1); });
    test::throws<std::invalid_argument>([&] { s.workspace<float>(Workspace::hidden, 9); });
    test::throws<std::invalid_argument>([&] { s.workspace<float>(Workspace::hidden, 0); });
    test::throws<std::out_of_range>([&] { s.region(static_cast<Workspace>(100)); });
    std::vector<std::uint16_t> kv(s.plan().kv_bytes / 2);
    download(s.context(), kv.data(), s.kv_reservation(), s.plan().kv_bytes);
    CHECK(std::all_of(kv.begin(), kv.end(), [](auto v) { return v == 0xffff; }));
    float first = 0;
    download(s.context(), &first, s.weight("token_embd.weight").data, sizeof(float));
    CHECK(first == 1.0f / 16.0f);
}

TEST(storage_rope_coefficients_match_host_formula) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    CudaStorage storage(model, small);
    const auto& d = model.dimensions();
    const auto view = storage.workspace<float>(Workspace::rope_coefficients, small.max_model_len);
    std::vector<float> actual(view.capacity);
    download(storage.context(), actual.data(), view.data, actual.size() * sizeof(float));
    for (std::size_t p = 0; p < small.max_model_len; ++p) {
        for (std::size_t i = 0; i < d.head_dim / 2; ++i) {
            const float angle = float(p) * std::pow(d.rope_base, -2.0f * float(i) / float(d.head_dim));
            CHECK(actual[p * d.head_dim + i] == std::cos(angle));
            CHECK(actual[p * d.head_dim + i + d.head_dim / 2] == std::sin(angle));
        }
    }
    CHECK(storage.rope_uploaded_bytes() == storage.plan().rope_bytes);
    CHECK(storage.plan().rope_bytes == small.max_model_len * d.head_dim * sizeof(float));
    CHECK(storage.kv_view().capacity * sizeof(std::uint16_t) == storage.plan().kv_bytes);
}

namespace {
struct Role { const char* name; const char* tensor; Workspace input, output; };
const std::array<Role, 8> roles{{
    {"Q", "blk.0.attn_q.weight", Workspace::normalized, Workspace::query},
    {"K", "blk.0.attn_k.weight", Workspace::normalized, Workspace::key},
    {"V", "blk.0.attn_v.weight", Workspace::normalized, Workspace::value},
    {"attention_output", "blk.0.attn_output.weight", Workspace::attention, Workspace::projected},
    {"gate", "blk.0.ffn_gate.weight", Workspace::normalized, Workspace::gate},
    {"up", "blk.0.ffn_up.weight", Workspace::normalized, Workspace::up},
    {"down", "blk.0.ffn_down.weight", Workspace::gate, Workspace::down},
    {"LM_head", "output.weight", Workspace::selected_hidden, Workspace::logits}}};

json gemm(const Qwen3Model& model, CudaStorage& s, Role role, std::size_t m, bool dense) {
    const auto w = s.weight(role.tensor);
    const auto source = model.source().tensor(std::string(role.tensor) == "output.weight" && model.tied_output()
                                             ? "token_embd.weight" : role.tensor);
    auto x = s.workspace<float>(role.input, m), y = s.workspace<float>(role.output, m);
    CHECK(x.columns == w.columns && y.columns == w.rows);
    const auto k = w.columns, n = w.rows;
    std::vector<float> hx(m * k, 0), hy(m * n), row(k);
    for (std::size_t i = 0; i < m; ++i) {
        if (dense) {
            for (std::size_t j = 0; j < k; ++j) { hx[i * k + j] = float(int((i * 13 + j * 7) % 41) - 20) / 127.0f; }
        } else {
            for (std::size_t j = 0; j < 4; ++j) { hx[i * k + (i * 17 + j * 127) % k] = float(int(j) - 2) / 7.0f; }
        }
    }
    upload(s.context(), x.data, hx.data(), hx.size() * sizeof(float));
    const auto before = allocation_stats();
    // 重复入队使用同一已分配 workspace；beta=0 必须覆盖前次输出。
    for (int repeat = 0; repeat < 2; ++repeat) {
        matrix_multiply(s.context(), {x.data, x.rows, x.columns, x.stride, x.capacity, x.device}, w, y);
    }
    download(s.context(), hy.data(), y.data, hy.size() * sizeof(float));
    same_allocations(before, allocation_stats());
    double max_abs = 0, sum_sq = 0;
    for (std::size_t j = 0; j < n; ++j) {
        decode_row(source.type, source.row(j), row.data(), k);
        for (std::size_t i = 0; i < m; ++i) {
            double expected = 0;
            if (dense) {
                for (std::size_t p = 0; p < k; ++p) { expected += double(hx[i * k + p]) * row[p]; }
            } else {
                for (std::size_t p = 0; p < 4; ++p) {
                    const auto index = (i * 17 + p * 127) % k;
                    expected += double(hx[i * k + index]) * row[index];
                }
            }
            const double actual = hy[i * n + j], error = std::abs(actual - expected);
            CHECK(std::isfinite(actual));
            CHECK(error <= unit_atol + unit_rtol * std::abs(expected));
            max_abs = std::max(max_abs, error);
            sum_sq += error * error;
        }
    }
    return {{"role", role.name}, {"M", m}, {"N", n}, {"K", k}, {"input", dense ? "dense" : "sparse4"},
            {"checked_elements", m * n}, {"max_absolute", max_abs}, {"rmse", std::sqrt(sum_sq / double(m * n))},
            {"project_allocation_calls", 0}, {"project_release_calls", 0}, {"passed", true}};
}
}

TEST(storage_repeated_gemm_reuses_workspace) {
    Qwen3Fixture fixture;
    Qwen3Model model(fixture.write());
    CudaStorage storage(model, small);
    for (const auto& role : roles) { gemm(model, storage, role, 2, true); }
}

namespace {
void real_paged_storage(const std::string& path, const std::string& contract_path,
                        const std::filesystem::path& output) {
    std::ifstream contract_file(contract_path);
    const auto contract = json::parse(contract_file);
    CHECK(contract.at("schema_version") == 1 && contract.at("contract_id") == "qwen3-cuda-numerical-v1");
    const auto model_sha = check_model_identity(path,contract.at("model").at("sha256"));
    std::filesystem::copy_file(contract_path,output/"validation-contract.json");
    Qwen3Model model(path);
    const auto& d = model.dimensions();
    CHECK(d.layers == 28 && d.kv_heads == 8 && d.head_dim == 128);
    const auto baseline = make_memory_plan(model,{});
    StorageLimits limits;
    limits.kv_layout = CudaKvLayout::paged; limits.kv_capacity_tokens = 2560;
    const auto before = allocation_stats();
    {
        PageTableState pages({4,2048,16,160});
        CudaStorage storage(model,limits);
        const auto& plan = storage.plan();
        check_layout(plan);
        CHECK(plan.kv_bytes == 280ULL*1024*1024 && plan.kv_table_bytes == 2048);
        CHECK(plan.kv_bytes+plan.kv_table_bytes <= 288ULL*1024*1024);
        CHECK(storage.allocated_bytes() == plan.total_bytes && storage.allocations() == 4);
        CHECK(allocation_stats().allocated_bytes-before.allocated_bytes == plan.total_bytes);
        const auto& context = storage.context();
        const auto slab = storage.paged_kv_view();
        const auto table = storage.kv_block_table();
        const PagedKvMapping mapping{table,160,16};
        const KvShape shape{4,d.layers,2048,d.kv_heads,d.head_dim};
        const auto width = d.kv_heads*d.head_dim;
        std::vector<float> input(4*width);
        for (std::size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i%97)-48)/32.0f; }
        const std::array<std::int32_t,4> slots{0,1,2,3}, positions{1567,159,159,159};
        const auto key = storage.workspace<float>(Workspace::key,4), value = storage.workspace<float>(Workspace::value,4);
        const auto s = storage.workspace<std::int32_t>(Workspace::slots,4);
        const auto p = storage.workspace<std::int32_t>(Workspace::positions,4);
        const auto status = storage.workspace<std::int32_t>(Workspace::status,1);
        upload(context,key.data,input.data(),input.size()*sizeof(float));
        upload(context,value.data,input.data(),input.size()*sizeof(float));
        upload(context,s.data,slots.data(),sizeof(slots)); upload(context,p.data,positions.data(),sizeof(positions));
        const auto allocated = allocation_stats();
        pages.prepare(std::array<std::size_t,4>{},std::array<std::size_t,4>{1568,160,160,160});
        pages.begin_execution();
        storage.upload_page_table(pages.device_table_for_upload());
        reset_status(context,status);
        for (const auto layer : {std::size_t{0},d.layers-1}) {
            store_kv(context,slab,shape,mapping,layer,minillm::cuda::detail::read_only(key),
                minillm::cuda::detail::read_only(value),minillm::cuda::detail::read_only(s),
                minillm::cuda::detail::read_only(p),status);
        }
        std::array<std::int32_t,2> result{};
        download(context,result.data(),status.data,sizeof(result));
        CHECK(result[0] == 0);
        pages.commit();
        CHECK(pages.assigned_pages() == 128 && pages.free_pages() == 32);
        std::vector<std::int32_t> table_data(table.capacity);
        download(context,table_data.data(),table.data,plan.kv_table_bytes);
        CHECK(std::equal(table_data.begin(),table_data.end(),pages.committed_table().begin()));
        std::vector<std::uint16_t> actual(width);
        for (const auto layer : {std::size_t{0},d.layers-1}) {
            for (std::size_t sequence = 0; sequence < 4; ++sequence) {
                const auto position = std::size_t(positions[sequence]);
                const auto page = std::size_t(table_data[sequence*128+position/16]);
                for (std::size_t kind = 0; kind < 2; ++kind) {
                    const auto row = ((layer*2+kind)*160+page)*16+position%16;
                    download(context,actual.data(),slab.data+row*width,width*sizeof(std::uint16_t));
                    for (std::size_t c = 0; c < width; ++c) {
                        CHECK(actual[c] == float_to_half(input[sequence*width+c]));
                    }
                }
            }
        }
        same_allocations(allocated,allocation_stats());
        write_json(output/"memory-plan.json",{
            {"kv_layout","paged"},{"capacity_tokens",plan.kv_capacity_tokens},{"page_tokens",16},
            {"physical_pages",plan.physical_pages},{"kv_payload_bytes",plan.kv_bytes},
            {"kv_table_bytes",plan.kv_table_bytes},{"metadata_bytes",plan.metadata_bytes},
            {"workspace_bytes",plan.workspace_bytes},{"weight_bytes",plan.weight_bytes},
            {"owned_device_bytes",storage.allocated_bytes()},{"baseline_owned_bytes",baseline.total_bytes},
            {"baseline_kv_bytes",baseline.kv_bytes},{"kv_budget_bytes",288ULL*1024*1024},
            {"project_allocations",storage.allocations()},{"page_table_h2d_bytes",storage.page_table_h2d_bytes()},
            {"assigned_pages",pages.assigned_pages()},{"free_pages",pages.free_pages()},
            {"checked_fp16_values",2*4*2*width},{"steady_project_allocations",0},
            {"scope","真实模型尺寸的 slab/table/store 检查；不是完整 forward、admission 或性能结果"}});
    }
    const auto after = allocation_stats();
    CHECK(after.allocations-before.allocations == 4 && after.releases-before.releases == 4);
    write_json(output/"validation-summary.json",{{"schema_version",1},{"status","passed"},{"passed",true},
        {"model_sha256",model_sha},{"complete_gpu_model",false},{"kv_layout","paged"},
        {"released_project_allocations",after.releases-before.releases}});
}

void real_model(const std::string& path, const std::string& contract_path, const std::filesystem::path& output) {
    std::ifstream contract_file(contract_path);
    const auto contract = json::parse(contract_file);
    CHECK(contract.at("schema_version") == 1);
    CHECK(contract.at("contract_id") == "qwen3-cuda-numerical-v1");
    CHECK(contract.at("thresholds").at("unit_atol") == unit_atol);
    CHECK(contract.at("thresholds").at("unit_rtol") == unit_rtol);
    const auto model_sha = check_model_identity(path, contract.at("model").at("sha256"));
    std::filesystem::copy_file(contract_path, output / "validation-contract.json");
    Qwen3Model model(path);
    CHECK(model.dimensions().vocabulary == contract.at("model").at("vocabulary"));
    const auto before = allocation_stats();
    std::size_t unique_tensors = 0, gemm_cases = 0;
    {
        CudaStorage storage(model);
        const auto& p = storage.plan();
        check_layout(p);
        CHECK(p.kv_bytes == 896ULL * 1024 * 1024);
        CHECK(model.source().tensor_count() == 310 && model.tied_output());
        const auto allocated = allocation_stats();
        CHECK(allocated.allocations - before.allocations == 4);
        CHECK(allocated.allocated_bytes - before.allocated_bytes == p.total_bytes);
        const auto weights = verify_weights(model, storage);
        CHECK(weights["records"].size() == 311);
        json source_counts = json::object();
        for (const auto& weight : p.weights) {
            if (weight.alias_of.empty()) {
                const auto name = dtype(weight.source_type);
                source_counts[name] = source_counts.value(name, std::size_t{0}) + 1;
                ++unique_tensors;
            }
        }
        CHECK(source_counts.size() == contract.at("arithmetic").at("source_tensor_counts").size());
        for (const auto& [name, count] : source_counts.items()) {
            CHECK(count == contract.at("arithmetic").at("source_tensor_counts").at(name));
        }
        write_json(output / "weight-plan.json", weights);
        json regions = json::array();
        for (const auto& r : p.regions) {
            regions.push_back({{"name", r.name}, {"dtype", r.type == StorageType::f32 ? "F32" : "I32"},
                               {"shape", {r.rows, r.columns}}, {"offset", r.offset}, {"bytes", r.bytes}});
        }
        write_json(output / "memory-plan.json", {
            {"S", p.limits.max_sequences}, {"Lmax", p.limits.max_model_len}, {"B", p.limits.max_batch_tokens},
            {"user_budget_bytes", p.limits.device_budget_bytes}, {"weight_payload_bytes", p.weight_payload},
            {"weight_arena_bytes", p.weight_bytes}, {"activation_bytes", p.activation_bytes},
            {"attention_bytes", p.attention_bytes}, {"logits_bytes", p.logits_bytes}, {"metadata_bytes", p.metadata_bytes},
            {"rope_bytes", p.rope_bytes},
            {"padding_bytes", p.padding_bytes}, {"workspace_arena_bytes", p.workspace_bytes},
            {"kv_reservation_bytes", p.kv_bytes}, {"cublas_workspace_bytes", p.cublas_bytes}, {"total_bytes", p.total_bytes},
            {"free_at_gate_bytes", storage.available_at_gate().free_bytes},
            {"allowed_bytes", memory_budget(p, storage.available_at_gate())}, {"regions", regions}});
        json cases = json::array();
        for (const auto& role : roles) {
            for (std::size_t m : {1,2,4,8,16,18,32,64,128}) { cases.push_back(gemm(model, storage, role, m, false)); }
            for (std::size_t m : {1,2}) { cases.push_back(gemm(model, storage, role, m, true)); }
            std::cout << "[PASS] real GEMM " << role.name << " 11 cases\n";
        }
        same_allocations(allocated, allocation_stats());
        gemm_cases = cases.size();
        write_json(output / "matrix-validation.json", {{"atol", unit_atol}, {"rtol", unit_rtol}, {"oracle", "CPU FP64"},
                   {"scope", "真实权重矩阵正确性；不是完整模型或性能基线"}, {"cases", cases}});
        write_json(output / "copy-allocation-summary.json", {
            {"scope", "项目 DeviceMemory 包装器；不统计 CUDA/cuBLAS 内部资源"},
            {"initialization_allocation_count", 4}, {"initialization_allocated_bytes", p.total_bytes},
            {"weight_upload_bytes", storage.uploaded_bytes()}, {"weight_upload_chunks", storage.upload_chunks()},
            {"rope_upload_bytes", storage.rope_uploaded_bytes()},
            {"host_upload_staging_limit_bytes", weight_staging_bytes},
            {"max_upload_chunk_bytes", storage.max_upload_chunk_bytes()},
            {"matrix_case_count", cases.size()}, {"gemm_calls", cases.size() * 2},
            {"matrix_phase_allocation_calls", 0}, {"matrix_phase_release_calls", 0}});
        cudaDeviceProp device{};
        int driver = 0, runtime = 0, cublas = 0;
        check_cuda(cudaGetDeviceProperties(&device, storage.context().device()), "cudaGetDeviceProperties");
        check_cuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        check_cuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        check_cublas(cublasGetVersion(storage.context().handle(), &cublas), "cublasGetVersion");
        std::ostringstream uuid;
        for (auto byte : device.uuid.bytes) {
            uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(static_cast<unsigned char>(byte));
        }
        write_json(output / "environment.json", {{"device", storage.context().device()}, {"name", device.name},
            {"uuid", uuid.str()}, {"compute_capability", {device.major, device.minor}},
            {"driver_api_version", driver}, {"runtime_version", runtime}, {"cublas_version", cublas},
            {"model_sha256", model_sha}, {"contract_sha256", file_hash(contract_path)},
            {"source_weight_dtype", "Q8_0"}, {"source_tensor_counts", source_counts}, {"device_weight_dtype", "F32"},
            {"matrix_input_output_dtype", "F32"}, {"gemm_compute", "CUBLAS_COMPUTE_32F_PEDANTIC"},
            {"kv_reservation_dtype", "F16"}, {"stream_count", 1}, {"complete_gpu_model", false}});
        storage.context().synchronize();
    }
    const auto after = allocation_stats();
    CHECK(after.releases - before.releases == 4);
    CHECK(after.release_calls - before.release_calls == 4);
    write_json(output / "validation-summary.json", {{"schema_version", 1}, {"status", "passed"},
               {"passed", true}, {"unique_tensors", unique_tensors}, {"model_sha256", model_sha},
               {"gemm_cases", gemm_cases}, {"released_project_allocations", after.releases - before.releases},
               {"complete_gpu_model", false}});
}
}

int main(int argc, char** argv) {
    if (argc == 1) { return test::run(); }
    const bool paged = argc == 8 && std::string_view(argv[7]) == "--paged-kv";
    if ((argc != 7 && !paged) || std::string_view(argv[1]) != "--model" || std::string_view(argv[3]) != "--contract" ||
        std::string_view(argv[5]) != "--output") {
        std::cerr << "用法：minillm-cuda-storage-tests [--model MODEL.gguf --contract CONTRACT.json --output NEW_DIRECTORY [--paged-kv]]\n";
        return 1;
    }
    try {
        const std::string path = argv[2], contract = argv[4];
        const std::filesystem::path output = argv[6];
        prepare_output(output);
        write_json(output / "validation-summary.json",
                   {{"schema_version", 1}, {"status", "incomplete"}, {"passed", false}, {"complete_gpu_model", false}});
        test::cases().push_back({paged ? "storage_real_qwen3_paged_slab" : "storage_real_qwen3_weights_and_matrices",
            [path, contract, output, paged] {
                if (paged) { real_paged_storage(path,contract,output); }
                else { real_model(path,contract,output); }
            }});
        const int result = test::run();
        if (result != 0) {
            write_json(output / "validation-summary.json",
                       {{"schema_version", 1}, {"status", "failed"}, {"passed", false}, {"complete_gpu_model", false}});
        }
        return result;
    } catch (const std::exception& error) {
        std::cerr << "CUDA 存储验证失败：" << error.what() << '\n';
        return 1;
    }
}
