#pragma once
#include <iostream>
#include <stdint.h>
#include "gpuassert.cuh"
#include "types.cuh"
#include "pcg.cuh"

template <typename T>
uint32_t solvePCG(
    csr_t<T> *h_S, 
    csr_t<T> *h_Pinv, 
    T *h_gamma, 
    T *h_lambda, 
    unsigned stateSize, 
    unsigned knotPoints, 
    pcg_config *config)
{
    std::cout << "NOT IMPLEMENTED" << std::endl;
    exit(12);
}


template <typename T>
uint32_t solvePCG(const uint32_t state_size,
                  const uint32_t knot_points,
                  T *d_S,
                  T *d_Pinv,
                  T *d_gamma,
                  T *d_lambda,
                  T *d_r,
                  T *d_p,
                  T *d_v_temp,
                  T *d_eta_new_temp,
                  pcg_config *config)
{

    
    uint32_t *d_iters;
    gpuErrchk(cudaMalloc(&d_iters, sizeof(uint32_t)));


    void *pcg_kernel = (void *) pcg<T, STATE_SIZE, KNOT_POINTS>;

    // checkPcgOccupancy<T>(pcg_kernel, config->pcg_block, state_size, knot_points);
    void *kernelArgs[] = {
        (void *)&d_S,
        (void *)&d_Pinv,
        (void *)&d_gamma, 
        (void *)&d_lambda,
        (void *)&d_r,
        (void *)&d_p,
        (void *)&d_v_temp,
        (void *)&d_eta_new_temp,
        (void *)&d_iters,
        (void *)&config->pcg_max_iter,
        (void *)&config->pcg_exit_tol
    };
    uint32_t h_iters;

    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);


    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, kernelArgs, ppcg_kernel_smem_size));    
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(&h_iters, d_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));



    gpuErrchk(cudaFree(d_iters));

    return h_iters;
}