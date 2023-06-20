#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace pcg_constants{
    uint32_t DEFAULT_MAX_PCG_ITER = 15;
    float DEFAULT_EPSILON = 1e-6;
    dim3 DEFAULT_GRID(128);
    dim3 DEFAULT_BLOCK(64);
}