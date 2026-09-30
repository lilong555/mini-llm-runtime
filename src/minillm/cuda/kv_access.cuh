#pragma once

#include "attention.h"
#include "device_helpers.cuh"

namespace minillm::cuda::detail {

struct ContiguousKvAccess {
    KvShape shape;

    __device__ bool row(std::size_t slot, std::size_t layer, std::size_t kind, std::size_t position,
                        std::size_t& result, std::int32_t*, int) const {
        result = ((slot * shape.layers + layer) * 2 + kind) * shape.max_length + position;
        return true;
    }
};

struct PagedKvAccess {
    KvShape shape;
    PagedKvMapping mapping;

    __device__ bool row(std::size_t slot, std::size_t layer, std::size_t kind, std::size_t position,
                        std::size_t& result, std::int32_t* status, int input_row) const {
        // Host 入口只接受 P=16，设备寻址使用相同的编译期合同。
        const auto block = position / kv_page_tokens;
        if (slot >= shape.sequences || position >= shape.max_length ||
            slot >= mapping.block_table.rows || block >= mapping.block_table.columns) {
            record_error(status, DeviceError::invalid_index, input_row);
            return false;
        }
        const auto page = mapping.block_table.data[slot * mapping.block_table.stride + block];
        if (page < 0 || std::size_t(page) >= mapping.physical_pages) {
            record_error(status, DeviceError::invalid_index, input_row);
            return false;
        }
        result = ((layer * 2 + kind) * mapping.physical_pages + std::size_t(page)) * kv_page_tokens
                 + position % kv_page_tokens;
        return true;
    }
};

} // namespace minillm::cuda::detail
