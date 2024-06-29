#pragma once

#include <stdint.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;


// 
// block-diagonal matrix-vector product for pcg_block.cuh
// 
template<typename T>
__device__
void bdmv_block(T *s_dst, // size = b_dim
                T *s_matdb, // always diagonal, size = b_dim
                T *s_matod, // always two blocks, size = 2 * b_dim * b_dim
                T *s_vec, // size = 3 * b_dim
                uint32_t b_dim,
                uint32_t max_block_id,
                uint32_t block_id) {

    T val;

    if (block_id == 0) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < b_dim; c++) {
                // only
                // right off-diagonal block times 3rd part of vector
                val += s_matod[b_dim * b_dim + b_dim * c + r] * s_vec[c + 2 * b_dim]; // var and var+1
            }
            // diagonal block times 2nd part of vector
            val += s_matdb[r] * s_vec[b_dim + r];
            s_dst[r] = val;
        }
    } else if (block_id == max_block_id) {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < b_dim; c++) {
                // only
                // left off-diagonal block times 1st part of vector
                val += s_matod[b_dim * c + r] * s_vec[c];
            }
            // diagonal block times 2nd part of vector
            val += s_matdb[r] * s_vec[b_dim + r];
            s_dst[r] = val;
        }
    } else {
        for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x) {
            val = static_cast<T>(0);
            for (unsigned c = 0; c < b_dim; c++) {
                // left off-diagonal block times 1st part of vector
                val += s_matod[b_dim * c + r] * s_vec[c];
                // right off-diagonal block times 3rd part of vector
                val += s_matod[b_dim * b_dim + b_dim * c + r] * s_vec[c + 2 * b_dim];
            }
            // diagonal block times 2nd part of vector
            val += s_matdb[r] * s_vec[b_dim + r];
            s_dst[r] = val;
        }
    }
}


