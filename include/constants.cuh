#pragma once
#include <cstdint>
#include <cuda_runtime.h>


#ifndef STATE_SIZE
#define STATE_SIZE  1
#endif

#ifndef KNOT_POINTS
#define KNOT_POINTS  2
#endif

namespace pcg_constants{
    uint32_t DEFAULT_MAX_PCG_ITER = 25;
    float DEFAULT_EPSILON = 1e-6;
    dim3 DEFAULT_GRID(128);
    dim3 DEFAULT_BLOCK(64);
}
