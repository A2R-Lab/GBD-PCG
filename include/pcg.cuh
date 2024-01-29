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









template <typename T, uint32_t state_size, uint32_t knot_points>
__global__
void pcg_dynamic_mem(
        T * d_gamma, T * d_g, T * d_A, T* d_admm_lambda, T*d_z, T *d_rho_mat, T sigma, T* d_l, T* d_u,
         pcg_problem<T> problem
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
    if(block_id == 0){

        T *s_zdiff = s_temp;
        T *s_Atz = s_zdiff + NC;
        /* z_diff = d_rho_mat * d_z */
        element_wise_vector_mult<T>(NC, s_zdiff, d_rho_mat, d_z);

        /* z_diff = z_diff - lambda */
        glass::axpby<T>(NC, static_cast<T>(1.0), s_zdiff, -1, d_admm_lambda, s_zdiff);
        __syncthreads();

        /* Atx = A.T * z_diff */
        glass::gemv<T, true>(NX, NC, static_cast<T>(1.0), d_A, s_zdiff, s_Atz);
        __syncthreads();

        /* gamma = -g + sigma * x */
        glass::axpby<T>(NX, -1, d_g, sigma, prob->d_x, d_gamma);
        __syncthreads();

        /* gamma = Atz + gamma */
        glass::axpy<T>(NX, 1, s_Atz, d_gamma);
    }
    grid.sync();


    
    T *S_b, *Pinv_b, *gamma_b, 
      *lambda_bm1, *r_tilde_b, *upsilon_b,
      *v_b, *eta_new_b, *r_bm1, *p_bm1, 
      *r_b, *p_b, *lambda_b;
    T *s_end = s_temp;

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

    // S
    if(prob->d_S.copy_into_smem){
        S_b = s_end;
        s_end = S_b +3*states_sq;
        glass::copy<T>(3*states_sq, &(prob->d_S.ptr[block_id*states_sq*3]), S_b);
    }
    else{
        S_b = prob->d_S.ptr + block_id*states_sq*3;
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


    uint32_t iter;
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

    if(block_id == 0){
        T *s_Ax = s_temp;
        T *s_Axz = s_Ax + NC;

        /* Ax = A * x */
        glass::gemv<T, false>(NC, NX, 1, d_A, prob->d_x, s_Ax);

        /* z = Ax + 1/rho * lambda */
        for(int i=blockIdx.x*blockDim.x + threadIdx.x; i < NC; i += blockDim.x)
        {
            d_z[i] = s_Ax[i] + d_admm_lambda[i] / d_rho_mat[i];
        }

        /* z = clip(z) */
        glass::clip(NC, d_z, d_l, d_u);

        /* Axz = Ax - z*/
        glass::axpby<T>(NC, 1, s_Ax, -1, d_z, s_Axz);

        /* Axz = Axz * rho element-wise */
        element_wise_vector_mult(NC, s_Axz, d_rho_mat, s_Axz);

        /* lambda = lambda + Axz */
        glass::axpy<T>(NC, static_cast<T>(1.0), s_Axz, d_admm_lambda);
    }
}