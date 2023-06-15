#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "gpuassert.cuh"
#include "utils.cuh"
#include "glass.cuh"


namespace cgrps = cooperative_groups;

template <typename T>
size_t pcgSharedMemSize(uint32_t state_size, uint32_t knot_points){
    return sizeof(T) * (2*3*state_size*state_size + 
                        10 * state_size + 
                        2*max(state_size, knot_points));
}


template <typename T>
bool checkPcgOccupancy(void* kernel, dim3 block, uint32_t state_size, uint32_t knot_points){
    
    const uint32_t smem_size = pcgSharedMemSize<T>(state_size, knot_points);
    int dev = 0;
    
    cudaDeviceProp deviceProp; 
    cudaGetDeviceProperties(&deviceProp, dev);
    
    int supportsCoopLaunch = 0; 
    gpuErrchk(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if(!supportsCoopLaunch){
        printf("[Error] Device does not support Cooperative Threads\n");
        exit(5);
    }
    
    int numProcs = static_cast<T>(deviceProp.multiProcessorCount); 
    int numBlocksPerSm;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x*block.y*block.z, smem_size));

    if(knot_points > numProcs*numBlocksPerSm){
        printf("Too many knot points ([%d]). Device supports [%d] active blocks, over [%d] SMs.\n", knot_points, numProcs*numBlocksPerSm, numProcs);
        exit(6);
    }

    return true;
}



template <typename T>
__global__
void pcg(uint32_t state_size,
         uint32_t knot_points,
         T *d_S, 
         T *d_Pinv, 
         T *d_gamma,  				
         T *d_lambda, 
         T  *d_r, 
         T  *d_p, 
         T *d_v_temp, 
         T *d_eta_new_temp,
         uint32_t *d_iters, 
         uint32_t max_iter, 
         float exit_tol)
{   

    const cgrps::thread_block block = cgrps::this_thread_block();	 
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    extern __shared__ T s_temp[];

    T  *s_S = s_temp;
    T  *s_Pinv = s_S +3*states_sq;
    T  *s_gamma = s_Pinv + 3*states_sq;
    T  *s_scratch = s_gamma + state_size;
    T *s_lambda = s_scratch;
    T *s_r_tilde = s_lambda + 3*state_size;
    T  *s_upsilon = s_r_tilde + state_size;
    T  *s_v_b = s_upsilon + state_size;
    T  *s_eta_new_b = s_v_b + max(knot_points, state_size);
    T  *s_r = s_eta_new_b + max(knot_points, state_size);
    T  *s_p = s_r + 3*state_size;
    T  *s_r_b = s_r + state_size;
    T  *s_p_b = s_p + state_size;
    T *s_lambda_b = s_lambda + state_size;

    uint32_t iter;
    T alpha, beta, eta, eta_new;
    
    // populate shared memory
    for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim){
        if(block_id == 0 && ind < states_sq){ continue; }
        if(block_id == knot_points-1 && ind >= 2*states_sq){ continue; }

        s_S[ind] = d_S[block_id*states_sq*3 + ind];
        s_Pinv[ind] = d_Pinv[block_id*states_sq*3 + ind];
    }
    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma, block);


    //
    // PCG
    //

    // r = gamma - S * lambda
    loadbdVec<T>(s_lambda, state_size, knot_points-1, &d_lambda[block_x_statesize], block);
    __syncthreads();
    bdmv<T>(s_r_b, state_size, knot_points-1, s_S, s_lambda, block_id, block);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind]; 
    }
    
    grid.sync(); //-------------------------------------

    // r_tilde = Pinv * r
    loadbdVec<T>(s_r, state_size, knot_points-1, &d_r[block_x_statesize], block);
    __syncthreads();
    bdmv<T>(s_r_tilde, state_size, knot_points-1, s_Pinv, s_r, block_id, block);
    __syncthreads();
    
    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind]; 
    }


    // eta = r * r_tilde
    glass::dot<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde, block);
    if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    grid.sync(); //-------------------------------------
    glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp, block);
    __syncthreads();
    eta = s_eta_new_b[0];
    

    // MAIN PCG LOOP
    for(iter = 0; iter < max_iter; iter++){
        // upsilon = S * p
        loadbdVec<T>(s_p, state_size, knot_points-1, &d_p[block_x_statesize], block);
        __syncthreads();
        bdmv<T>(s_upsilon, state_size, knot_points-1, s_S, s_p, block_id, block);
        __syncthreads();

        // alpha = eta / p * upsilon
        glass::dot<T>(s_v_b, state_size, s_p_b, s_upsilon, block);
        __syncthreads();
        if(thread_id == 0){ d_v_temp[block_id] = s_v_b[0]; }
        grid.sync(); //-------------------------------------
        glass::reduce(s_v_b, gridDim.x, d_v_temp, block);
        __syncthreads();
        alpha = eta / s_v_b[0];
        // lambda = lambda + alpha * p
        // r = r - alpha * upsilon
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[block_x_statesize + ind] = s_r_b[ind];
        }

        grid.sync(); //-------------------------------------

        // r_tilde = Pinv * r
        loadbdVec<T>(s_r, state_size, knot_points-1, &d_r[block_x_statesize], block);
        __syncthreads();
        bdmv<T>(s_r_tilde, state_size, knot_points-1, s_Pinv, s_r, block_id, block);
        __syncthreads();

        // eta = r * r_tilde
        glass::dot<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde, block);
        __syncthreads();
        if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        grid.sync(); //-------------------------------------
        glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp, block);
        __syncthreads();
        eta_new = s_eta_new_b[0];

        if(abs(eta_new) < exit_tol){ iter++; break; }

        // beta = eta_new / eta
        // eta = eta_new
        beta = eta_new / eta;
        eta = eta_new;

        // p = r_tilde + beta*p
        for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
            s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
            d_p[block_x_statesize + ind] = s_p_b[ind];
        }
        grid.sync(); //-------------------------------------
    }

    // save output
    if(block_id == 0 && thread_id == 0){ *d_iters = iter; }
    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize], block);


}