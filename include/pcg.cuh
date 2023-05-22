#pragma once
#include <stdint.h>
#include "types.cuh"

template <typename T>
uint32_t pcg(uint32_t state_size,
                   uint32_t knot_points,
                   float  *d_S, 
                   float  *d_Pinv, 
                   float  *d_gamma, 
                   float *d_lambda, 
                   float *d_r, 
                   float *d_p, 
                   float *d_v_temp, 
                   float *d_eta_new_temp, 
                   float *d_r_tilde, 
                   float *d_upsilon, 
                   unsigned *d_iters, 
                   pcg_config *config);

template <typename T>
bool checkPcgOccupancy(void* kernel, 
                       dim3 block, 
                       uint32_t state_size, 
                       uint32_t knot_points);

template <typename T>
uint32_t pcgSharedMemSize(uint32_t state_size, 
                          uint32_t knot_points);