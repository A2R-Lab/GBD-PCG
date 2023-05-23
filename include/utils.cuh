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
               cgrps::thread_block b)
{

    auto block_id = b.group_index().x;

    cgrps::thread_group tile = cgrps::tiled_partition(b, 32);
    uint32_t tileId = b.thread_rank() / 32;

    if(tileId == 0){
        // Need to load b also now
        for (unsigned ind = tile.thread_rank(); ind < block_dim; ind += tile.size()){
            s_var[ind + block_dim] = *(d_var_b + ind); 
        }
    }
    if(tileId == 1 || b.size() < 33){
        if(block_id == 0){
            for (unsigned ind = tile.thread_rank(); ind < block_dim; ind += tile.size()){
                s_var[ind + 2*block_dim] = *(d_var_b + block_dim + ind); 
            }
        }
        else if (block_id == max_block_id){
            for (unsigned ind = tile.thread_rank(); ind < block_dim; ind += tile.size()){
                s_var[ind] = *(d_var_b - block_dim + ind);
            }
        }
        else{
            T *dst, *src;
            for (unsigned ind = tile.thread_rank(); ind < 2*block_dim; ind += tile.size()){
                dst = s_var + ind + (ind >= block_dim) * block_dim;
                src = d_var_b + ind - (ind < block_dim) * block_dim;
                *dst = *src;
            }
        }
    }
}


// 
// block-diagonal matrix-vector product
// 
template <typename T>
__device__ 
void bdmv(T *s_dst, 
          uint32_t block_dim, 
          uint32_t max_block_id, 
          T *s_mat, 
          T *s_vec,
          int block_id, 
          cgrps::thread_block b)
{
    
    T val;

    if(block_id == 0){
        for (unsigned r = b.thread_rank(); r < block_dim; r += b.size()){
            val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*block_dim; c++){
                val += s_mat[block_dim*block_dim + block_dim * c + r] * s_vec[c + block_dim]; // var and var+1
            }
            s_dst[r] = val;
        }
    }
    else if (block_id == max_block_id){
        for (unsigned r = b.thread_rank(); r < block_dim; r += b.size()){
            val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*block_dim; c++){
                val += s_mat[block_dim * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
    else{
        for (unsigned r = b.thread_rank(); r < block_dim; r += b.size()){
            val = static_cast<T>(0);
            for(unsigned c = 0; c < 3*block_dim; c++){
                val += s_mat[block_dim * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}