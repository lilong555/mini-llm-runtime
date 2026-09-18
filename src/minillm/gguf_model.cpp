#include "minillm/gguf_model.h"

#include "gguf.h"

#include <bit>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace minillm {
namespace {

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
#ifdef _WIN32
        file_ = CreateFileW(std::filesystem::path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("cannot open GGUF file: " + path);
        }
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart <= 0) {
            close();
            throw std::runtime_error("invalid GGUF file size");
        }
        size_ = static_cast<std::size_t>(size.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_) {
            data_ = static_cast<const std::byte*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        }
#else
        file_ = ::open(path.c_str(), O_RDONLY);
        if (file_ < 0) {
            throw std::runtime_error("cannot open GGUF file: " + path);
        }
        struct stat stat{};
        if (::fstat(file_, &stat) != 0 || stat.st_size <= 0) {
            close();
            throw std::runtime_error("invalid GGUF file size");
        }
        size_ = static_cast<std::size_t>(stat.st_size);
        const auto* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, file_, 0);
        if (mapping != MAP_FAILED) {
            data_ = static_cast<const std::byte*>(mapping);
        }
#endif
        if (!data_) {
            close();
            throw std::runtime_error("cannot memory-map GGUF file");
        }
    }
    ~MappedFile() { close(); }
    const std::byte* data() const { return data_; }
    std::size_t size() const { return size_; }

private:
    void close() noexcept {
#ifdef _WIN32
        if (data_) {
            UnmapViewOfFile(data_);
        }
        if (mapping_) {
            CloseHandle(mapping_);
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_);
        }
        mapping_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (data_) {
            ::munmap(const_cast<std::byte*>(data_), size_);
        }
        if (file_ >= 0) {
            ::close(file_);
        }
        file_ = -1;
#endif
        data_ = nullptr;
    }
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_ = -1;
#endif
};

} // namespace

struct GgufModel::Impl {
    MappedFile file;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata{nullptr, gguf_free};

    explicit Impl(const std::string& path) : file(path) {
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error("MiniLLM currently requires a little-endian host");
        }
        metadata.reset(gguf_init_from_buffer(file.data(), file.size(), {true, nullptr}));
        if (!metadata || gguf_get_version(metadata.get()) != 3) {
            throw std::runtime_error("invalid or unsupported GGUF; version 3 is required");
        }
        const auto base = gguf_get_data_offset(metadata.get());
        for (std::int64_t i = 0; i < gguf_get_n_tensors(metadata.get()); ++i) {
            const auto offset = gguf_get_tensor_offset(metadata.get(), i);
            const auto bytes = gguf_get_tensor_size(metadata.get(), i);
            if (base > file.size() || offset > file.size() - base ||
                bytes > file.size() - base - offset) {
                throw std::runtime_error("GGUF tensor extends beyond the mapped file");
            }
        }
    }

    std::int64_t key(const std::string& name) const {
        const auto id = gguf_find_key(metadata.get(), name.c_str());
        if (id < 0) {
            throw std::runtime_error("missing GGUF metadata: " + name);
        }
        return id;
    }
};

GgufModel::GgufModel(const std::string& path) : impl_(std::make_unique<Impl>(path)) {}
GgufModel::~GgufModel() = default;

std::string GgufModel::string_value(const std::string& key) const {
    const auto id = impl_->key(key);
    if (gguf_get_kv_type(impl_->metadata.get(), id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("GGUF metadata is not a string: " + key);
    }
    return gguf_get_val_str(impl_->metadata.get(), id);
}

std::uint64_t GgufModel::integer_value(const std::string& key) const {
    const auto id = impl_->key(key);
    const auto* ctx = impl_->metadata.get();
    switch (gguf_get_kv_type(ctx, id)) {
    case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
    case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
    case GGUF_TYPE_INT32: {
        const auto value = gguf_get_val_i32(ctx, id);
        if (value >= 0) {
            return static_cast<std::uint64_t>(value);
        }
        break;
    }
    default: break;
    }
    throw std::runtime_error("GGUF metadata is not a nonnegative integer: " + key);
}

float GgufModel::float_value(const std::string& key) const {
    const auto id = impl_->key(key);
    if (gguf_get_kv_type(impl_->metadata.get(), id) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error("GGUF metadata is not float32: " + key);
    }
    return gguf_get_val_f32(impl_->metadata.get(), id);
}

bool GgufModel::contains(const std::string& key) const {
    return gguf_find_key(impl_->metadata.get(), key.c_str()) >= 0;
}
bool GgufModel::has_tensor(const std::string& name) const {
    return gguf_find_tensor(impl_->metadata.get(), name.c_str()) >= 0;
}

TensorView GgufModel::tensor(const std::string& name) const {
    const auto* ctx = impl_->metadata.get();
    const auto id = gguf_find_tensor(ctx, name.c_str());
    if (id < 0) {
        throw std::runtime_error("missing GGUF tensor: " + name);
    }
    WeightType type;
    switch (gguf_get_tensor_type(ctx, id)) {
    case GGML_TYPE_F32: type = WeightType::f32; break;
    case GGML_TYPE_F16: type = WeightType::f16; break;
    case GGML_TYPE_Q8_0: type = WeightType::q8_0; break;
    default: throw std::runtime_error("unsupported tensor type for " + name + "; use F32, F16, or Q8_0");
    }
    const auto* dims = gguf_get_tensor_ne(ctx, id);
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] != 1 || dims[3] != 1) {
        throw std::runtime_error("only nonempty vectors and matrices are supported: " + name);
    }
    const auto columns = static_cast<std::size_t>(dims[0]);
    const auto rows = static_cast<std::size_t>(dims[1]);
    const auto stride = row_bytes(type, columns);
    if (stride == 0 || rows > std::numeric_limits<std::size_t>::max() / stride ||
        rows * stride != gguf_get_tensor_size(ctx, id)) {
        throw std::runtime_error("invalid GGUF row stride: " + name);
    }
    return {impl_->file.data() + gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id),
            type, columns, rows, stride};
}

std::size_t GgufModel::tensor_count() const {
    return static_cast<std::size_t>(gguf_get_n_tensors(impl_->metadata.get()));
}
std::size_t GgufModel::file_bytes() const { return impl_->file.size(); }
std::uint32_t GgufModel::version() const { return gguf_get_version(impl_->metadata.get()); }

const std::byte* TensorView::row(std::size_t index) const {
    if (index >= rows) {
        throw std::out_of_range("tensor row index");
    }
    return data + index * stride;
}

std::vector<float> TensorView::vector() const {
    if (rows != 1) {
        throw std::runtime_error("expected a one-dimensional tensor");
    }
    std::vector<float> values(columns);
    decode_row(type, data, values.data(), columns);
    return values;
}

} // namespace minillm
