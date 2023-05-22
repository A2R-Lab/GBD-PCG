#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pcg_constants{
    extern uint32_t DEFAULT_MAX_PCG_ITER;
    extern float DEFAULT_EPSILON;
    extern dim3 DEFAULT_GRID;
    extern dim3 DEFAULT_BLOCK;
}