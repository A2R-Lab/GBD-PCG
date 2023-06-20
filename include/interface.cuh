#pragma once
#include <iostream>
#include <stdint.h>
#include "gpuassert.cuh"
#include "types.cuh"
#include "pcg.cuh"
// #include "old_pcg.cuh"

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
                  pcg_config *config)
{

    // cudaStream_t streams[1];
    // gpuErrchk(cudaStreamCreate(&streams[0]));
    
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    uint32_t *d_iters;
    gpuErrchk(cudaMalloc(&d_r, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_iters, sizeof(uint32_t)));
    gpuErrchk(cudaPeekAtLastError());

    struct timespec pcgkernelstart, pcgkernelend;

    void *pcg_kernel = (void *) pcg<T, 14, 50>;
    // void *pcg_kernel = (void *) oldpcg::parallelPCG<T>;

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

    gpuErrchk(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC,&pcgkernelstart);

    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, 50, 64, kernelArgs, ppcg_kernel_smem_size));    
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    clock_gettime(CLOCK_MONOTONIC,&pcgkernelend);
    
    double time = time_delta_us_timespec(pcgkernelstart,pcgkernelend);
    std::cout << "pcg time minus setup " << time << std::endl;
    gpuErrchk(cudaMemcpy(&h_iters, d_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));


    // float h_temp[14];
    // gpuErrchk(cudaMemcpy(h_temp, d_lambda, 14*sizeof(float), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < 14; i++){ std::cout << h_temp[i] << " ";}
    // std::cout << std::endl;

    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(d_p));
    gpuErrchk(cudaFree(d_v_temp));
    gpuErrchk(cudaFree(d_eta_new_temp));
    gpuErrchk(cudaFree(d_iters));

    // gpuErrchk(cudaStreamDestroy(streams[0]));

    return h_iters;
}