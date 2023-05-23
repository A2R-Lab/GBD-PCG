#pragma once
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
    return 1;
}


template <typename T>
uint32_t solvePCG(uint32_t state_size,
                  uint32_t knot_points,
                  T *d_S,
                  T *d_Pinv,
                  T *d_gamma,
                  T *d_lambda,
                  pcg_config *config)
{

    cudaStream_t streams[1];
    gpuErrchk(cudaStreamCreate(&streams[0]));
    
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    uint32_t *d_iters;
    gpuErrchk(cudaMallocAsync(&d_r, state_size*knot_points*sizeof(T), streams[0]));
    gpuErrchk(cudaMallocAsync(&d_p, state_size*knot_points*sizeof(T), streams[0]));
    gpuErrchk(cudaMallocAsync(&d_v_temp, knot_points*sizeof(T), streams[0]));
    gpuErrchk(cudaMallocAsync(&d_eta_new_temp, knot_points*sizeof(T), streams[0]));
    gpuErrchk(cudaMallocAsync(&d_iters, sizeof(uint32_t), streams[0]));



    void *pcg_kernel = (void *) pcg<T>;

    checkPcgOccupancy<T>(pcg_kernel, config->pcg_block, state_size, knot_points);
    void *kernelArgs[] = {
        (void *)&state_size,
        (void *)&knot_points,
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
    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, config->pcg_block, kernelArgs, pcgSharedMemSize<T>(state_size, knot_points), streams[0]));    // EMRE
    
    uint32_t h_iters;
    gpuErrchk(cudaMemcpy(&h_iters, d_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "kernel returned\n";
    gpuErrchk(cudaFreeAsync(d_r, streams[0]));
    gpuErrchk(cudaFreeAsync(d_p, streams[0]));
    gpuErrchk(cudaFreeAsync(d_v_temp, streams[0]));
    gpuErrchk(cudaFreeAsync(d_eta_new_temp, streams[0]));
    gpuErrchk(cudaFreeAsync(d_iters, streams[0]));

    gpuErrchk(cudaStreamDestroy(streams[0]));

    return h_iters;
}