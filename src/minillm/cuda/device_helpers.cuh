#pragma once

#include "ops.h"

#include <cuda_runtime.h>

namespace minillm::cuda::detail {

__device__ inline void record_error(std::int32_t* status, DeviceError code, int row) {
    atomicOr(status, static_cast<int>(code));
    atomicMin(status + 1, row);
}

}
