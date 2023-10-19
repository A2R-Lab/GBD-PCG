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


/* TODO: have a interface for accepting h_S in other formats*/
template <typename T>
uint32_t solvePCG(
    T *h_S, 
    T *h_gamma, 
    T *h_lambda, 
    unsigned stateSize, 
    unsigned knotPoints, 
    pcg_config *config)
{

    /* Create device memory d_s, d_Pinv, d_gamma, d_lambda, d_r, d_p, d_v_temp
	d_eta_new_temp */
	T *d_S, *d_gamma, *d_lambda;
	gpuErrchk(cudaMalloc(&d_lambda, state_size*knot_points*sizeof(T)));
	gpuErrchk(cudaMalloc(&d_S, 3*states_sq*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_gamma, state_size*knot_points*sizeof(T)));


	T *d_Pinv;
    gpuErrchk(cudaMalloc(&d_Pinv, 3*states_sq*knot_points*sizeof(T)));
    
    /*   PCG vars   */
    T  *d_r, *d_p, *d_v_temp, *d_eta_new_temp;
    gpuErrchk(cudaMalloc(&d_r, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_p, state_size*knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_v_temp, knot_points*sizeof(T)));
    gpuErrchk(cudaMalloc(&d_eta_new_temp, knot_points*sizeof(T)));


	/* pcgconfig*/
	pcg_config config;
	config->empty_pinv = 1;


	/* Copy s, gamma, lambda*/
	gpuErrchk(cudaMemcpy(d_S, h_S, 3 * states_sq * knot_points * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_lambda, h_lambda, state_size * knot_points * sizeof(T), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_gamma, h_gamma, state_size * knot_points * sizeof(T), cudaMemcpyHostToDevice));


	solvePCG(state_size, knot_points,
                  d_S,
                  d_Pinv,
                  d_gamma,
                  d_lambda,
                  d_r,
                  d_p,
                  d_v_temp,
                  d_eta_new_temp,
                  &config)


    /* Copy data back */
	gpuErrchk(cudaMemcpy(h_S, d_S, 3 * states_sq * knot_points * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_lambda, d_lambda, state_size * knot_points * sizeof(T), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_gamma, d_gamma, state_size * knot_points * sizeof(T), cudaMemcpyDeviceToHost));

	return 1;
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
        (void *)&config->pcg_exit_tol,
		(void *)&config->empty_pinv
    };
    uint32_t h_iters;

    size_t ppcg_kernel_smem_size = pcgSharedMemSize<T>(state_size, knot_points);


    gpuErrchk(cudaLaunchCooperativeKernel(pcg_kernel, knot_points, PCG_NUM_THREADS, kernelArgs, ppcg_kernel_smem_size));    
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaMemcpy(&h_iters, d_iters, sizeof(uint32_t), cudaMemcpyDeviceToHost));



    gpuErrchk(cudaFree(d_iters));

    return h_iters;
}