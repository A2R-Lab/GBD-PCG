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
struct device_ptr {
	T *ptr;
	bool copy_into_smem;
	device_ptr() : ptr(nullptr), copy_into_smem(false) {}
};
template <typename T>
struct pcg_problem {
    pcg_config<T> config;
	struct device_ptr<T> d_S;
	struct device_ptr<T> d_Pinv;
	struct device_ptr<T> d_lambda_buffer;
    T *d_x;
	struct device_ptr<T> d_r_tilde;
	struct device_ptr<T> d_r_buffer;
	struct device_ptr<T> d_p_buffer;
	struct device_ptr<T> d_v;
	struct device_ptr<T> d_upsilon;
	struct device_ptr<T> d_eta_new;
	struct device_ptr<T> d_v_temp;
	struct device_ptr<T> d_eta_new_temp;
	struct device_ptr<T> d_gamma;
	
	size_t shared_mem_size;
	uint32_t *d_pcg_iters;
	bool *d_pcg_exit;
};

template <typename T>
size_t pcgSharedMemSize(uint32_t state_size, uint32_t knot_points){
    // return sizeof(T) * max(
    //                     (2*3*state_size*state_size + 
    //                     10 * state_size + 
    //                     2*max(state_size, knot_points)),
    //                     (9 * state_size*state_size));
    return sizeof(T) * (2*3*state_size*state_size + 
                        12 * state_size + 
                        2*max(state_size, knot_points));
}

template <typename T>
__device__
void element_wise_vector_mult(uint32_t, T *, T *, T *);


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

    int numProcs = deviceProp.multiProcessorCount; 
    int numBlocksPerSm;
    gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, block.x*block.y*block.z, smem_size));

    if(knot_points > numProcs*numBlocksPerSm){
        printf("Too many knot points ([%d]). Device supports [%d] active blocks, over [%d] SMs.\n", knot_points, numProcs*numBlocksPerSm, numProcs);
        exit(6);
    }

    return true;
}




template <typename T, uint32_t state_size, uint32_t knot_points>
__global__
void pcg(
         T *d_S, 
         T *d_Pinv, 
         T *d_gamma,  				
         T *d_lambda, 
         T  *d_r, 
         T  *d_p, 
         T *d_v_temp, 
         T *d_eta_new_temp,
         uint32_t *d_iters, 
         bool *d_max_iter_exit,
         uint32_t max_iter, 
         T exit_tol, 
		 int empty_pinv)
{   

    const cgrps::thread_block block = cgrps::this_thread_block();	 
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;

    extern __shared__ T s_temp[];
    
    // If empty pinv, Load Pinv from Schur

	/* TODO: Check pinv before after for correctness*/
	if(empty_pinv){
		for(unsigned ind=blockIdx.x; ind<knot_points; ind+=gridDim.x){
			gato_ss_from_schur<T>(
				state_size, knot_points,
				d_S,
				d_Pinv,
				s_temp,
				ind
			);
		}
		grid.sync();
	}

    //
    // complete Pinv
    //
    for(unsigned ind=blockIdx.x; ind<knot_points; ind+=gridDim.x){
        gato_form_ss_inner<T>(
            state_size, knot_points,
            d_S,
            d_Pinv,
            s_temp,
            ind
        );
    }
    grid.sync();


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

    bool max_iter_exit = true;

    // populate shared memory
    for (unsigned ind = thread_id; ind < 3*states_sq; ind += block_dim){
        if(block_id == 0 && ind < states_sq){ continue; }
        if(block_id == knot_points-1 && ind >= 2*states_sq){ continue; }

        s_S[ind] = d_S[block_id*states_sq*3 + ind];
        s_Pinv[ind] = d_Pinv[block_id*states_sq*3 + ind];
    }
    glass::copy<T>(state_size, &d_gamma[block_x_statesize], s_gamma);


    //
    // PCG
    //

    // r = gamma - S * lambda
    loadbdVec<T, state_size, knot_points-1>(s_lambda, block_id, &d_lambda[block_x_statesize]);
    __syncthreads();
    bdmv<T>(s_r_b, s_S, s_lambda, state_size, knot_points-1,  block_id);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[block_x_statesize + ind] = s_r_b[ind]; 
    }
    
    grid.sync(); //-------------------------------------

    // r_tilde = Pinv * r
    loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
    __syncthreads();
    bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1,  block_id);
    __syncthreads();
    
    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        s_p_b[ind] = s_r_tilde[ind];
        d_p[block_x_statesize + ind] = s_p_b[ind]; 
    }


    // eta = r * r_tilde
    glass::dot<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde);
    if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
    grid.sync(); //-------------------------------------
    glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
    __syncthreads();
    eta = s_eta_new_b[0];
    

    // MAIN PCG LOOP
	if (abs(eta) > exit_tol) {
		for(iter = 0; iter < max_iter; iter++){

			// upsilon = S * p
			loadbdVec<T, state_size, knot_points-1>(s_p, block_id, &d_p[block_x_statesize]);
			__syncthreads();
			bdmv<T>(s_upsilon,  s_S, s_p,state_size, knot_points-1, block_id);
			__syncthreads();

			// alpha = eta / p * upsilon
			glass::dot<T>(s_v_b, state_size, s_p_b, s_upsilon);
			__syncthreads();
			if(thread_id == 0){ d_v_temp[block_id] = s_v_b[0]; }
			grid.sync(); //-------------------------------------
			glass::reduce<T>(s_v_b, knot_points, d_v_temp);
			__syncthreads();

			// HERE
			if(s_v_b[0] == 0)
				break;
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
			loadbdVec<T, state_size, knot_points-1>(s_r, block_id, &d_r[block_x_statesize]);
			__syncthreads();
			bdmv<T>(s_r_tilde, s_Pinv, s_r, state_size, knot_points-1, block_id);
			__syncthreads();

			// eta = r * r_tilde
			glass::dot<T>(s_eta_new_b, state_size, s_r_b, s_r_tilde);
			__syncthreads();
			if(thread_id == 0){ d_eta_new_temp[block_id] = s_eta_new_b[0]; }
			grid.sync(); //-------------------------------------
			glass::reduce<T>(s_eta_new_b, knot_points, d_eta_new_temp);
			__syncthreads();
			eta_new = s_eta_new_b[0];

			if(abs(eta_new) < exit_tol){ iter++; max_iter_exit = false; break; }

			// beta = eta_new / eta
			// eta = eta_new

			// HERE

			if(eta == 0)
				break;
			beta = eta_new / eta;
			eta = eta_new;

			// p = r_tilde + beta*p
			for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
				s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
				d_p[block_x_statesize + ind] = s_p_b[ind];
			}
			grid.sync(); //-------------------------------------
		}
	}

    // save output
    if(block_id == 0 && thread_id == 0){ d_iters[0] = iter; d_max_iter_exit[0] = max_iter_exit; }
    
    __syncthreads();
    glass::copy<T>(state_size, s_lambda_b, &d_lambda[block_x_statesize]);

    grid.sync();
}




template <typename T>
__device__ __forceinline__
T clip(T x, T u, T l){
    if(x < l){
        return l;
    }
    else if(x > u){
        return u;
    }
    else{
        return x;
    }
}




template <typename T, uint32_t state_size, uint32_t knot_points>
__global__
void pcg_dynamic_mem(
        T * d_gamma, 
        T * d_g, 
        T * d_A, 
        T* d_admm_lambda, 
        T*d_z, 
        T *d_weight_mat_d,
        T rho, 
        T sigma, 
        T* d_l, 
        T* d_u, 
        T *d_invE, 
        T *d_invD, 
        T *d_H, 
        T *d_primal_res, 
        T *d_dual_res, 
        T *d_primal_normalizer, 
        T *d_dual_normalizer, 
        T c,
        pcg_problem<T> problem,
        bool compute_residuals
)
{       
    const pcg_problem<T> *prob = &problem;
    const cgrps::grid_group grid = cgrps::this_grid();
    const uint32_t block_id = blockIdx.x;
    const uint32_t block_dim = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_x_statesize = block_id * state_size;
    const uint32_t states_sq = state_size * state_size;
    const uint32_t max_iter = problem.config.pcg_max_iter;
    const T exit_tol = problem.config.pcg_exit_tol;

    extern __shared__ T s_temp[];

    
    // compute gamma

    /* z = d_weight_mat * rho * d_z - lambda */
    /* Atx = A.T * z_diff */
    /* gamma = -g + sigma * x */
    /* gamma = Atz + gamma */

    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < NC; i += blockDim.x * gridDim.x){
        d_z[i] = (d_z[i] * d_weight_mat_d[i] * rho) - d_admm_lambda[i];
    }
    grid.sync();
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < NX; i += blockDim.x * gridDim.x){
        T Atz_i = 0;
        for(int j = 0; j < NC; j++){
            Atz_i += d_A[i*NC + j] * d_z[j];
        }
        d_gamma[i] = Atz_i + (- d_g[i] + sigma * prob->d_x[i]);
    }
    grid.sync();

    
    T *S_b, *Pinv_b, *gamma_b, 
      *lambda_bm1, *r_tilde_b, *upsilon_b,
      *v_b, *eta_new_b, *r_bm1, *p_bm1, 
      *r_b, *p_b, *lambda_b;
    T *s_end = s_temp;
    // S
    if(prob->d_S.copy_into_smem){
        S_b = s_end;
        s_end = S_b +3*states_sq;
        glass::copy<T>(3*states_sq, &(prob->d_S.ptr[block_id*states_sq*3]), S_b);
    }
    else{
        S_b = prob->d_S.ptr + block_id*states_sq*3;
    }

    // v
    if(prob->d_v.copy_into_smem){
        v_b = s_end;
        s_end = v_b + max(knot_points, state_size);
    }
    else{
        v_b = prob->d_v.ptr + block_id * max(knot_points, state_size);
    }

    // eta_new
    if(prob->d_eta_new.copy_into_smem){
        eta_new_b = s_end;
        s_end = eta_new_b + max(knot_points, state_size);
    }
    else{
        eta_new_b = prob->d_eta_new.ptr + block_id * max(knot_points, state_size);
    }

    // p
    if(prob->d_p_buffer.copy_into_smem){
        p_bm1 = s_end;
        s_end = p_bm1 + 3*state_size;
        p_b = p_bm1 + state_size;
    }
    else{
        p_bm1 = prob->d_p_buffer.ptr + block_x_statesize;
        p_b = p_bm1 + state_size;
    }


    // r_tilde
    if(prob->d_r_tilde.copy_into_smem){
        r_tilde_b = s_end;
        s_end = r_tilde_b + state_size;
    }
    else{
        r_tilde_b = prob->d_r_tilde.ptr + block_x_statesize;
    }


    // lambda
    if(prob->d_lambda_buffer.copy_into_smem){
        lambda_bm1 = s_end;
        s_end = lambda_bm1 + 3*state_size;
        glass::copy<T>(3*state_size, &(prob->d_lambda_buffer.ptr[block_x_statesize]), lambda_bm1);
        lambda_b = lambda_bm1 + state_size;
    }
    else{
        lambda_bm1 = prob->d_lambda_buffer.ptr + block_x_statesize;
        lambda_b = lambda_bm1 + state_size;
    }
    
    // r
    if(prob->d_r_buffer.copy_into_smem){
        r_bm1 = s_end;
        s_end = r_bm1 + 3*state_size;
        r_b = r_bm1 + state_size;
    }
    else{
        r_bm1 = prob->d_r_buffer.ptr + block_x_statesize;
        r_b = r_bm1 + state_size;
    }


    // upsilon
    if(prob->d_upsilon.copy_into_smem){
        upsilon_b = s_end;
        s_end = upsilon_b + state_size;
    }
    else{
        upsilon_b = prob->d_upsilon.ptr + block_x_statesize;
    }

    // gamma
    if(prob->d_gamma.copy_into_smem){
        gamma_b = s_end;
        s_end = gamma_b + state_size;
        glass::copy<T>(state_size, &(prob->d_gamma.ptr[block_x_statesize]), gamma_b);
    }
    else{
        gamma_b = prob->d_gamma.ptr + block_x_statesize;
    }

    // Pinv
    if(prob->d_Pinv.copy_into_smem){
        Pinv_b = s_end;
        s_end = Pinv_b + 3*states_sq;
        glass::copy<T>(3*states_sq, &(prob->d_Pinv.ptr[block_id*states_sq*3]), Pinv_b);
    }
    else{
        Pinv_b = prob->d_Pinv.ptr + block_id*states_sq*3;
    }

    uint32_t iter = 0;
    T alpha, beta, eta, eta_new;

    bool max_iter_exit = true;

    //
    // PCG
    //

    // r = gamma - S * lambda
    __syncthreads();
    glass::gemv<T, false>(state_size, 3*state_size, static_cast<T>(1.0), S_b, lambda_bm1, r_b);
    __syncthreads();
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        r_b[ind] = gamma_b[ind] - r_b[ind];
        if(prob->d_r_buffer.copy_into_smem){
            prob->d_r_buffer.ptr[state_size + block_x_statesize + ind] = r_b[ind];
        }
    }

    grid.sync(); //-------------------------------------

    // r_tilde = Pinv * r
    if(prob->d_r_buffer.copy_into_smem){
        glass::copy<T>(3*state_size, &(prob->d_r_buffer.ptr[block_x_statesize]), r_bm1);
    }
    __syncthreads();
    glass::gemv<T, false>(state_size, 3*state_size, static_cast<T>(1.0), Pinv_b, r_bm1, r_tilde_b);
    __syncthreads();
    
    // p = r_tilde
    for (unsigned ind = thread_id; ind < state_size; ind += block_dim){
        p_b[ind] = r_tilde_b[ind];
        if(prob->d_p_buffer.copy_into_smem){
            prob->d_p_buffer.ptr[state_size + block_x_statesize + ind] = p_b[ind];
        }
    }


    // eta = r * r_tilde
    glass::dot<T>(eta_new_b, state_size, r_b, r_tilde_b);
    if(thread_id == 0){ prob->d_eta_new_temp.ptr[block_id] = eta_new_b[0]; }
    grid.sync(); //-------------------------------------
    glass::reduce<T>(eta_new_b, knot_points, prob->d_eta_new_temp.ptr);
    __syncthreads();
    eta = eta_new_b[0];
    
    // MAIN PCG LOOP
	if (abs(eta) > exit_tol) {
		for(iter = 0; iter < max_iter; iter++){

			// upsilon = S * p
            if(prob->d_p_buffer.copy_into_smem){
                glass::copy<T>(3*state_size, &(prob->d_p_buffer.ptr[block_x_statesize]), p_bm1);
            }
			__syncthreads();
            glass::gemv<T, false>(state_size, 3*state_size, static_cast<T>(1.0), S_b, p_bm1, upsilon_b);
			__syncthreads();

			// alpha = eta / p * upsilon
			glass::dot<T>(v_b, state_size, p_b, upsilon_b);
			__syncthreads();
			if(thread_id == 0){ prob->d_v_temp.ptr[block_id] = v_b[0]; }
			grid.sync(); //-------------------------------------
			glass::reduce<T>(v_b, knot_points, prob->d_v_temp.ptr);
			__syncthreads();

			if(v_b[0] == 0){
				break;
            }
			alpha = eta / v_b[0];
			// lambda = lambda + alpha * p
			// r = r - alpha * upsilon
			for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
				lambda_b[ind] += alpha * p_b[ind];
				r_b[ind] -= alpha * upsilon_b[ind];
                if(prob->d_r_buffer.copy_into_smem){
				    prob->d_r_buffer.ptr[state_size + block_x_statesize + ind] = r_b[ind];
                }
			}

			grid.sync(); //-------------------------------------

			// r_tilde = Pinv * r
            if(prob->d_r_buffer.copy_into_smem){
                glass::copy<T>(3*state_size, &(prob->d_r_buffer.ptr[block_x_statesize]), r_bm1);
            }
			__syncthreads();
            glass::gemv<T, false>(state_size, 3*state_size, static_cast<T>(1.0), Pinv_b, r_bm1, r_tilde_b);
			__syncthreads();

			// eta = r * r_tilde
			glass::dot<T>(eta_new_b, state_size, r_b, r_tilde_b);
			__syncthreads();
			if(thread_id == 0){ prob->d_eta_new_temp.ptr[block_id] = eta_new_b[0]; }
			grid.sync(); //-------------------------------------
			glass::reduce<T>(eta_new_b, knot_points, prob->d_eta_new_temp.ptr);
			__syncthreads();
			eta_new = eta_new_b[0];

			if(abs(eta_new) < exit_tol){ 
                iter++; 
                max_iter_exit = false; 
                break; 
            }

			if(eta == 0)
				break;

			beta = eta_new / eta;
			eta = eta_new;

			// p = r_tilde + beta*p
			for(uint32_t ind = thread_id; ind < state_size; ind += block_dim){
				p_b[ind] = r_tilde_b[ind] + beta*p_b[ind];
                if(prob->d_p_buffer.copy_into_smem){
				    prob->d_p_buffer.ptr[state_size + block_x_statesize + ind] = p_b[ind];
                }
			}
			grid.sync(); //-------------------------------------
		}
	}

    // save output
    if(block_id == 0 && thread_id == 0){ prob->d_pcg_iters[0] = iter; prob->d_pcg_exit[0] = max_iter_exit; }
    
    if(prob->d_lambda_buffer.copy_into_smem){
        __syncthreads();
        glass::copy<T>(state_size, lambda_b, &(prob->d_lambda_buffer.ptr[state_size + block_x_statesize]));
    }


    // update z and lambda
    grid.sync();

    T *s_Ax = s_temp;
    T *s_Axz = s_Ax + NC;

    /* Ax = A * x */
    /* z = Ax + 1/rho * lambda */
    /* z = clip(z) */
    /* Axz = Ax - z*/
    /* Axz = Axz * rho element-wise */
    /* lambda = lambda + Axz */

    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < NC; i += blockDim.x * gridDim.x){
        s_Ax[i] = 0;
        for(int j = 0; j < NX; j++){
            s_Ax[i] += d_A[j*NC + i] * prob->d_x[j];
        }

        d_z[i] = clip(s_Ax[i] + d_admm_lambda[i] / (rho * d_weight_mat_d[i]), d_u[i], d_l[i]);
        s_Axz[i] = (s_Ax[i] - d_z[i]) * (rho * d_weight_mat_d[i]);
        d_admm_lambda[i] += s_Axz[i];
    }

    // compute residuals and normalizers
    if(compute_residuals){
        grid.sync();

        if(block_id == 0){
            // compute primal residual
            T *s_invE_Ax_z = s_temp;

            for(int row = threadIdx.x; row < NC; row += blockDim.x){
                s_invE_Ax_z[row] = 0;
                for(int col = 0; col < NX; col++){
                    s_invE_Ax_z[row] += d_A[col*NC + row] * prob->d_x[col];
                }
                s_invE_Ax_z[row] -= d_z[row];
                s_invE_Ax_z[row] *= d_invE[row];
            }
            __syncthreads();

            glass::infnorm<T>(NC, s_invE_Ax_z);
            __syncthreads();

            if(thread_id == 0){ d_primal_res[0] = s_invE_Ax_z[0]; }
        }


        if(block_id == 1 || gridDim.x < 4 && block_id == 0){
            // compute dual residual
            T *s_HX_g_AT_lambda = s_temp;

            for(int row = threadIdx.x; row < NX; row += blockDim.x){
                T Hx_temp = 0;
                T Atl_temp = 0;
                for(int col = 0; col < NX; col++){
                    Hx_temp += d_H[col*NX + row] * prob->d_x[col];
                }
                for(int col = 0; col < NC; col++){
                    Atl_temp += d_A[row*NC + col] * d_admm_lambda[col];
                }
                s_HX_g_AT_lambda[row] = d_invD[row] * (Hx_temp + Atl_temp + d_g[row]) / c;
            }
            __syncthreads();

            glass::infnorm<T>(NX, s_HX_g_AT_lambda);
            __syncthreads();

            if(thread_id == 0){ d_dual_res[0] = s_HX_g_AT_lambda[0]; }
        }

        if(block_id == 2 || gridDim.x < 4 && block_id == 0){
            // compute primal normalizer
            
            T *s_invE_A_x = s_temp;
            T *s_invE_z = s_invE_A_x + NC;

            /* invE_A_x = invE * Ax */
            /* invE_z = invE * z */
            for(int i = threadIdx.x; i < NC; i += blockDim.x){
                T Ax_temp = 0;
                for(int col = 0; col < NX; col++){
                    Ax_temp += d_A[col*NC + i] * prob->d_x[col];
                }
                s_invE_A_x[i] = d_invE[i] * Ax_temp;
                s_invE_z[i] = d_invE[i] * d_z[i];
            }
            __syncthreads();

            // get norm(invE*A*x), norm(invE*z)
            glass::infnorm<T>(NC, s_invE_A_x);
            glass::infnorm<T>(NC, s_invE_z);

            __syncthreads();
            if(threadIdx.x == 0)
            {
                // primal_normalizer = max(norm(invE*A*x), norm(invE*z), 1e-4)
                d_primal_normalizer[0] = max( s_invE_A_x[0], max( s_invE_z[0], static_cast<T>(1e-4) ) );
            }
        }

        if(block_id == 3 || gridDim.x < 4 && block_id == 0){
            // compute dual normalizer

            T *s_invD_H_x = s_temp;
            T *s_invD_g = s_invD_H_x + NX;
            T *s_invD_At_lambda = s_invD_g + NX;

            // inv_H_x = ( invD * Hx ) / c
            // invD_g = ( invD * g ) / c
            // invD_At_lambda = ( invD * Atl ) / c
            for(int i = threadIdx.x; i < NX; i += blockDim.x){
                T Hx_temp = 0;
                T Atl_temp = 0;
                for(int col = 0; col < NX; col++){
                    Hx_temp += d_H[col*NX + i] * prob->d_x[col];
                }
                for(int col = 0; col < NC; col++){
                    Atl_temp += d_A[i*NC + col] * d_admm_lambda[col];
                }
                s_invD_H_x[i] = (d_invD[i] * Hx_temp) / c;
                s_invD_g[i] = (d_invD[i] * d_g[i]) / c;
                s_invD_At_lambda[i] = (d_invD[i] * Atl_temp) / c;
            }
            __syncthreads();
            // get norm(invD*H*x)/c, norm(invD*g)/c, norm(invD*At*lam)/c
            glass::infnorm<T>(NX, s_invD_H_x);
            glass::infnorm<T>(NX, s_invD_g);
            glass::infnorm<T>(NX, s_invD_At_lambda);

            __syncthreads();
            // dual_normalizer = max(norm(invD*H*x)/c, norm(invD*g)/c, norm(invD*At*lam)/c, 1e-4)
            if(threadIdx.x==0)
            {
                d_dual_normalizer[0] = max( max( max ( s_invD_H_x[0], 
                                                    s_invD_g[0] ),
                                                s_invD_At_lambda[0] ),
                                            static_cast<T>(1e-4) );
            }
        }

        grid.sync();
    }
}