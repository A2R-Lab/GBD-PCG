#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#define NUM_THREADS 64
#define KNOT_POINTS 1
#define NX 3
#define NC 5
#define STATE_SIZE NX/KNOT_POINTS

namespace pcg_constants{
    uint32_t DEFAULT_MAX_PCG_ITER = 25;
    float DEFAULT_EPSILON = 1e-6;
    dim3 DEFAULT_GRID(128);
    dim3 DEFAULT_BLOCK(64);
}
