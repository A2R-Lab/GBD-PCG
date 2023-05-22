#include "constants.cuh"

namespace pcg_constants{
    uint32_t DEFAULT_MAX_PCG_ITER = 500;
    float DEFAULT_EPSILON = 1e-6;
    dim3 DEFAULT_GRID(128);
    dim3 DEFAULT_BLOCK(64);
}