#pragma once

#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "gpuassert.cuh"
#include "utils.cuh"
#include "utils_block.cuh"
#include "glass.cuh"

// this file is originated from pcg.cuh
// by Shaohui Yang

namespace cgrps = cooperative_groups;

template<typename T>
size_t pcgBlockSharedMemSize(uint32_t state_size, uint32_t knot_points) {
    return sizeof(T) * max(
            (2 * 2 * state_size * state_size + // off-diagonal blocks of S & Pinv
             2 * state_size + // diagonal blocks of S & Pinv
             12 * state_size +
             2 * max(state_size, knot_points)),
            (9 * state_size * state_size)); // don't get it, but leave it here
}


template<typename T>
bool checkPcgBlockOccupancy(void *kernel, dim3 block, uint32_t state_size, uint32_t knot_points) {

    const uint32_t smem_size = pcgBlockSharedMemSize<T>(state_size, knot_points);
    int dev = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    int supportsCoopLaunch = 0;
    gpuErrchk(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if (!supportsCoopLaunch) {
        printf("[Error] Device does not support Cooperative Threads\n");
        exit(5);
    }

    int numProcs = deviceProp.multiProcessorCount;
    int numBlocksPerSm;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x * block.y * block.z,
                                                            smem_size));

    if ((int) knot_points > numProcs * numBlocksPerSm) {
        printf("Too many knot points ([%d]). Device supports [%d] active blocks, over [%d] SMs.\n", knot_points,
               numProcs * numBlocksPerSm, numProcs);
        exit(6);
    }

    return true;
}


template<typename T, uint32_t state_size, uint32_t knot_points>
__global__
void pcgBlock(
        T *d_Sdb, // diagonal blocks of S, size = stateSize * knotPoints
        T *d_Sob,  // off-diagonal blocks of S, size = 2* stateSize * stateSize * knotPoints
        T *d_Pinvdb, // diagonal blocks of Pinv, size = stateSize * knotPoints
        T *d_Pinvob,  // off-diagonal blocks of Pinv, size = 2* stateSize * stateSize * knotPoints
        T *d_gamma,  // size = stateSize * knotPoints
        T *d_lambda, // size = stateSize * knotPoints
        T *d_r, // size = stateSize * knotPoints
        T *d_p, // size = stateSize * knotPoints
        T *d_v_temp,  // size = knotPoints
        T *d_eta_new_temp, // size = knotPoints
        uint32_t *d_iters,
        bool *d_max_iter_exit,
        uint32_t max_iter,
        T exit_tol) {

    const cgrps::thread_block block = cgrps::this_thread_block();
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    extern __shared__ T s_temp[];

    // this part is different from pcg.cuh because memory length changes
    T *s_Sdb = s_temp;
    T *s_Sob = s_Sdb + state_size;
    T *s_Pinvdb = s_Sob + 2 * states_sq;
    T *s_Pinvob = s_Pinvdb + state_size;
    //

    T *s_gamma = s_Pinvob + 2 * states_sq;
    T *s_scratch = s_gamma + state_size;
    T *s_lambda = s_scratch;
    T *s_r_tilde = s_lambda + 3 * state_size;
    T *s_upsilon = s_r_tilde + state_size;
    T *s_v_b = s_upsilon + state_size;
    T *s_eta_new_b = s_v_b + max(knot_points, state_size);
    T *s_r = s_eta_new_b + max(knot_points, state_size);
    T *s_p = s_r + 3 * state_size;
    T *s_r_b = s_r + state_size;
    T *s_p_b = s_p + state_size;
    T *s_lambda_b = s_lambda + state_size;

    uint32_t iter;
    T alpha, beta, eta, eta_new;

    bool max_iter_exit = true;

    // populate shared memory into s_Sob & s_Pinvob
    for (unsigned ind = thread_id; ind < 2 * states_sq; ind += block_dim) {
        if (block_id == 0 && ind < states_sq) { continue; }
        if (block_id == knot_points - 1 && ind >= states_sq) { continue; }

        s_Sob[ind] = d_Sob[block_id * states_sq * 2 + ind];
        s_Pinvob[ind] = d_Pinvob[block_id * states_sq * 2 + ind];
    }
    // populate shared memory into s_Sdb & s_Pinvdb
    glass::copy<T>(state_size, &d_Sdb[block_x_statesize], s_Sdb);
    glass::copy<T>(state_size, &d_Pinvdb[block_x_statesize], s_Pinvdb);

    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);


    //
    // PCG
    //

    // r = gamma - S * lambda
    loadbdVec<T, state_size, knot_points - 1>(s_lambda, block_id, &d_lambda[block_x_statesize]);
    __syncthreads();
    bdmv_block<T>(s_r_b, s_Sdb, s_Sob, s_lambda, state_size, knot_points - 1, block_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind];
    }

    grid.sync(); //-------------------------------------

    // r_tilde = Pinv * r
    loadbdVec<T, state_size, knot_points - 1>(s_r, block_id, &d_r[block_x_statesize]);
    __syncthreads();
    bdmv_block<T>(s_r_tilde, s_Pinvdb, s_Pinvob, s_r, state_size, knot_points - 1, block_id);
    __syncthreads();

    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim) {
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind];
    }


    // eta = r * r_tilde
    glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_tilde);
    if (thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    grid.sync(); //-------------------------------------
    glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
    __syncthreads();
    eta = s_eta_new_b[0];


    // MAIN PCG LOOP

    for (iter = 0; iter < max_iter; iter++) {

        // upsilon = S * p
        loadbdVec<T, state_size, knot_points - 1>(s_p, block_id, &d_p[block_x_statesize]);
        __syncthreads();
        bdmv_block<T>(s_upsilon, s_Sdb, s_Sob, s_p, state_size, knot_points - 1, block_id);
        __syncthreads();

        // alpha = eta / p * upsilon
        glass::dot<T, state_size>(s_v_b, s_p_b, s_upsilon);
        __syncthreads();
        if (thread_id == 0) { d_v_temp[block_id] = s_v_b[0]; }
        grid.sync(); //-------------------------------------
        glass::reduce<T>(s_v_b, knot_points, d_v_temp);
        __syncthreads();
        alpha = eta / s_v_b[0];
        // lambda = lambda + alpha * p
        // r = r - alpha * upsilon
        for (uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[block_x_statesize + ind] = s_r_b[ind];
        }

        grid.sync(); //-------------------------------------

        // r_tilde = Pinv * r
        loadbdVec<T, state_size, knot_points - 1>(s_r, block_id, &d_r[block_x_statesize]);
        __syncthreads();
        bdmv_block<T>(s_r_tilde, s_Pinvdb, s_Pinvob, s_r, state_size, knot_points - 1, block_id);
        __syncthreads();

        // eta = r * r_tilde
        glass::dot<T, state_size>(s_eta_new_b, s_r_b, s_r_tilde);
        __syncthreads();
        if (thread_id == 0) { d_eta_new_temp[block_id] = s_eta_new_b[0]; }
        grid.sync(); //-------------------------------------
        glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
        __syncthreads();
        eta_new = s_eta_new_b[0];

        if (abs(eta_new) < exit_tol) {
            iter++;
            max_iter_exit = false;
            break;
        }

        // beta = eta_new / eta
        // eta = eta_new
        beta = eta_new / eta;
        eta = eta_new;

        // p = r_tilde + beta*p
        for (uint32_t ind = thread_id; ind < state_size; ind += block_dim) {
            s_p_b[ind] = s_r_tilde[ind] + beta * s_p_b[ind];
            d_p[block_x_statesize + ind] = s_p_b[ind];
        }
        grid.sync(); //-------------------------------------
    }


    // save output
    if (block_id == 0 && thread_id == 0) {
        d_iters[0] = iter;
        d_max_iter_exit[0] = max_iter_exit;
    }

    __syncthreads();
    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);

    grid.sync();
}