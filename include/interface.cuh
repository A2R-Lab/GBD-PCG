#include <stdint.h>
#include "types.cuh"

template <typename T>
uint32_t solvePCG(
    const uint32_t state_size,
    const uint32_t knot_points,
    T *d_S,
    T *d_Pinv,
    T *d_gamma,
    T *d_lambda,
    pcg_config *config);