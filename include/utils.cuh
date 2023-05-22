#pragma once
#include <stdint.h>
#include <cooperative_groups.h>
#include "types.cuh"

namespace cgrps = cooperative_groups;

template <typename T>
__device__
void loadbdVec(T *s_var, 
               uint32_t block_dim, 
               uint32_t max_block_id, 
               T *d_var_b, 
               cgrps::thread_block b);


template <typename T>
__device__ 
void bdmv(T *s_dst, 
          uint32_t block_dim, 
          uint32_t max_block_id, 
          T *s_mat, 
          T *s_vec, 
          cgrps::thread_block b);
