#pragma once

#include <iostream>
#include <stdint.h>
#include "gpuassert.cuh"
#include "types.cuh"
#include "pcg_block.cuh"
#include <ctime>

#define tic      double tic_t = clock();
#define toc      std::cout << (clock() - tic_t)/CLOCKS_PER_SEC \
                           << " seconds" << std::endl;

/* TODO: have a interface for accepting h_S in other formats*/
template<typename T>
uint32_t solvePCGBlock(
        T *h_Sdb,
        T *h_Sob,
        T *h_Pinvdb,
        T *h_Pinvob,
        T *h_gamma,
        T *h_lambda,
        unsigned stateSize,
        unsigned knotPoints,
        struct pcg_config<T> *config) {
    if (!config->empty_pinv)
        printf("This api can only be called with no preconditioner\n");

    const uint32_t states_sq = stateSize * stateSize;
    /* Create device memory d_Sdb, d_Sob, d_Pinvdb, d_Pinvob
     * d_gamma, d_lambda, d_r, d_p, d_v_temp
	d_eta_new_temp */
    T *d_Sdb, *d_Sob, *d_gamma, *d_lambda;
    gpuErrchk(cudaMalloc(&d_lambda, stateSize * knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_Sdb, stateSize * knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_Sob, 2 * states_sq * knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, stateSize * knotPoints * sizeof(T)));


    T *d_Pinvdb, *d_Pinvob;
    gpuErrchk(cudaMalloc(&d_Pinvdb, stateSize * knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_Pinvob, 2 * states_sq * knotPoints * sizeof(T)));

    /*   PCG vars   */
    T *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, stateSize * knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, stateSize * knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knotPoints * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knotPoints * sizeof(T)));


    /* Copy S, Pinv, gamma, lambda*/
    gpuErrchk(cudaMemcpy(d_Sdb, h_Sdb, stateSize * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Sob, h_Sob, 2 * states_sq * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinvdb, h_Pinvdb, stateSize * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinvob, h_Pinvob, 2 * states_sq * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda, stateSize * knotPoints * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma, h_gamma, stateSize * knotPoints * sizeof(T), cudaMemcpyHostToDevice));


    uint32_t pcg_iters = solvePCGBlock(stateSize, knotPoints,
                                       d_Sdb,
                                       d_Sob,
                                       d_Pinvdb,
                                       d_Pinvob,
                                       d_gamma,
                                       d_lambda,
                                       d_r,
                                       d_p,
                                       d_v_temp,
                                       d_eta_new_temp,
                                       config);


    /* Copy data back */
    gpuErrchk(cudaMemcpy(h_lambda, d_lambda, stateSize * knotPoints * sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_gamma, d_gamma, stateSize * knotPoints * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(d_lambda);
    cudaFree(d_Sdb);
    cudaFree(d_Sob);
    cudaFree(d_gamma);
    cudaFree(d_Pinvdb);
    cudaFree(d_Pinvob);
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_v_temp);
    cudaFree(d_eta_new_temp);

    return pcg_iters;
}


template<typename T>
uint32_t solvePCGBlock(const uint32_t state_size,
                       const uint32_t knot_points,
                       T *d_Sdb,
                       T *d_Sob,
                       T *d_Pinvdb,
                       T *d_Pinvob,
                       T *d_gamma,
                       T *d_lambda,
                       T *d_r,
                       T *d_p,
                       T *d_v_temp,
                       T *d_eta_new_temp,
                       struct pcg_config<T> *config) {
    uint32_t *d_pcg_iters;
    gpuErrchk(cudaMalloc(&d_pcg_iters, sizeof(uint32_t)));
    bool *d_pcg_exit;
    gpuErrchk(cudaMalloc(&d_pcg_exit, sizeof(bool)));

    void *pcg_kernel = (void *) pcgBlock<T, STATE_SIZE, KNOT_POINTS>;

    // checkPcgOccupancy<T>(pcg_kernel, config->pcg_block, state_size, knot_points);
    void *kernelArgs[] = {
            (void *) &d_Sdb,
            (void *) &d_Sob,
            (void *) &d_Pinvdb,
            (void *) &d_Pinvob,
            (void *) &d_gamma,
            (void *) &d_lambda,
            (void *) &d_r,
            (void *) &d_p,
            (void *) &d_v_temp,
            (void *) &d_eta_new_temp,
            (void *) &d_pcg_iters,
            (void *) &d_pcg_exit,
            (void *) &config->pcg_max_iter,
            (void *) &config->pcg_exit_tol,
            (void *) &config->empty_pinv
    };
    uint32_t h_pcg_iters;

    size_t ppcg_kernel_smem_size = pcgBlockSharedMemSize<T>(state_size, knot_points);

    tic
    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, pcg_constants::DEFAULT_BLOCK, kernelArgs,
                                          ppcg_kernel_smem_size));
    toc
    gpuErrchk(cudaMemcpy(&h_pcg_iters, d_pcg_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));


    gpuErrchk(cudaFree(d_pcg_iters));
    gpuErrchk(cudaFree(d_pcg_exit));

    return h_pcg_iters;
}