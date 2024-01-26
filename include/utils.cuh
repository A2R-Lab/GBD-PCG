#pragma once
#include <stdint.h>
#include <cooperative_groups.h>
#include "types.cuh"
#include "glass.cuh"

namespace cgrps = cooperative_groups;

template <typename T, uint32_t block_dim, uint32_t max_block_id>
__device__
void loadbdVec(T *s_var, 
               const uint32_t block_id,
               T *d_var_b)
{
    glass::copy<T>(3*block_dim, &d_var_b[-1 * (int) block_dim], s_var);

    // // Need to load b also now
    // for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x){
    //     s_var[ind + block_dim] = *(d_var_b + ind); 
    // }
    
    // if(block_id == 0){
    //     for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x){
    //         s_var[ind] = static_cast<T>(0);
    //         s_var[ind + 2*block_dim] = *(d_var_b + block_dim + ind); 
    //     }
    // }
    // else if (block_id == max_block_id){
    //     for (unsigned ind = threadIdx.x; ind < block_dim; ind += blockDim.x){
    //         s_var[ind] = *(d_var_b - block_dim + ind);
    //         s_var[ind + 2*block_dim] = static_cast<T>(0);
    //     }
    // }
    // else{
    //     T *dst, *src;
    //     for (unsigned ind = threadIdx.x; ind < 2*block_dim; ind += blockDim.x){
    //         dst = s_var + ind + (ind >= block_dim) * block_dim;
    //         src = d_var_b + ind - (ind < block_dim) * block_dim;
    //         *dst = *src;
    //     }
    // }
}


// 
// block-diagonal matrix-vector product
// 
template <typename T>
__device__ 
void bdmv(T *s_dst, 
          T *s_mat, 
          T *s_vec,
          uint32_t b_dim, 
          uint32_t max_block_id, 
          uint32_t block_id)
{

    // if(block_id == 0){
    //     glass::gemv<T, false>(b_dim, 2*b_dim, static_cast<T>(1.0), &s_mat[b_dim*b_dim], &s_vec[b_dim], s_dst);
    // }
    // else if(block_id == max_block_id){
    //     glass::gemv<T, false>(b_dim, 2*b_dim, static_cast<T>(1.0), s_mat, s_vec, s_dst);
    // }
    // else{
    //     glass::gemv<T, false>(b_dim, 3*b_dim, static_cast<T>(1.0), s_mat, s_vec, s_dst);
    // }
    glass::gemv<T, false>(b_dim, 3*b_dim, static_cast<T>(1.0), s_mat, s_vec, s_dst);
    
    // T val;

    // if(block_id == 0){
    //     for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x){
    //         val = static_cast<T>(0);
    //         for(unsigned c = 0; c < 2*b_dim; c++){
    //             T one =s_mat[b_dim*b_dim + b_dim * c + r]; 
    //             // if(block_id==0){
    //             //    printf("c: %d, b_dim: %d, one: %p\n", c, b_dim, &(s_vec[0]));
    //             // }
    //             T two = s_vec[c + b_dim];
    //             val += one * two; // var and var+1
    //         }
    //         s_dst[r] = val;
    //     }
    // }
    // else if (block_id == max_block_id){
    //     for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x){
    //         val = static_cast<T>(0);
    //         for(unsigned c = 0; c < 2*b_dim; c++){
    //             val += s_mat[b_dim * c + r] * s_vec[c];
    //         }
    //         s_dst[r] = val;
    //     }
    // }
    // else{
    //     for (unsigned r = threadIdx.x; r < b_dim; r += blockDim.x){
    //         val = static_cast<T>(0);
    //         for(unsigned c = 0; c < 3*b_dim; c++){
    //             val += s_mat[b_dim * c + r] * s_vec[c];
    //         }
    //         s_dst[r] = val;
    //     }
    // }
}

template <typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts){
	unsigned ind;
	for(ind=threadIdx.x; ind < size_Ts; ind+=blockDim.x){
		dst[ind] = src[ind];
	}
}

template <typename T>
__device__
void load_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned bcol, unsigned brow, bool transpose=false, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
    
    if(bcol > 2 || brow > m_dim-1){
        printf("doing somehting wrong in load_block_bd\n");
        return;
    }
    

    unsigned block_row_offset, block_col_offset;

    block_row_offset = brow * (3 * b_dim * b_dim);
    block_col_offset = bcol*b_dim*b_dim;

    if(!transpose){

        gato_memcpy<T>(
            dst,
            src+block_row_offset+block_col_offset,
            b_dim*b_dim
        );

    }
    else{

        unsigned ind, transpose_col, transpose_row;

        for(ind=threadIdx.x; ind<b_dim*b_dim; ind+=blockDim.x){
            transpose_col = ind%b_dim * b_dim;
            transpose_row = ind/b_dim;
            dst[transpose_col + transpose_row] = src[block_row_offset + block_col_offset + ind];    
        }
    }
}

template <typename T>
__device__
void store_block_bd(uint32_t b_dim, uint32_t m_dim, T *src, T *dst, unsigned col, unsigned BLOCKNO, int multiplier=1, cooperative_groups::thread_group g = cooperative_groups::this_thread_block()){
    
    unsigned block_row_offset, block_col_offset, ind;


    block_row_offset = BLOCKNO * (3 * b_dim * b_dim);
    block_col_offset = col*b_dim*b_dim;


    if(multiplier==1){

        glass::copy<T>(b_dim*b_dim, src, &dst[block_row_offset+block_col_offset]);

        gato_memcpy<T>(
            dst+block_row_offset+block_col_offset,
            src,
            b_dim*b_dim
        );

    }
    else{
        
        for(ind=g.thread_rank(); ind<b_dim*b_dim; ind+=g.size()){
            dst[block_row_offset + block_col_offset + ind] = src[ind] * multiplier;
        }

    }
}

template <typename T>
__device__
T *get_block_bd_address(uint32_t b_dim, uint32_t m_dim, T *src, unsigned col, unsigned BLOCKNO)
{
    unsigned block_row_offset, block_col_offset;


    block_row_offset = BLOCKNO * (3 * b_dim * b_dim);
    block_col_offset = col*b_dim*b_dim;

    return &src[block_row_offset + block_col_offset];
}


    ///TODO: this could be more better
    template <typename T>
    __device__
    void mat_mat_prod(T *out, T *mat_A, T *mat_B, int A_rows, int A_cols, int B_rows, int B_cols, bool transposeB = false){

        if(!transposeB){

            unsigned ind, thing;
            unsigned maxind = A_rows*B_cols;
            T res;
            int row, col;

            for(ind=threadIdx.x; ind<maxind; ind+=blockDim.x){
                // ind x takes row x/A_cols and col x%b_rows
                res = 0;
                row = ind % A_rows;
                col = ind / A_rows;

                for(thing=0; thing<A_cols; thing++){
                    res += mat_A[thing*A_rows+row] * mat_B[col*B_rows+thing];
                }

                out[col*A_rows+row] = res;

            } 
        }
        else{                       // transpose matrix B


            unsigned ind, thing;
            unsigned maxind = A_rows*B_rows;
            T res;
            int row, col;

            for(ind=threadIdx.x; ind<maxind; ind+=blockDim.x){
                // ind x takes row x/A_cols and col x%b_rows
                res = 0;
                row = ind % A_rows;
                col = ind / A_rows;

                for(thing=0; thing<A_cols; thing++){
                    res += mat_A[thing*A_rows+row] * mat_B[thing*B_rows+col];
                }

                out[col*A_rows+row] = res;

            } 

        }
    }

template <typename T>
__device__
void gato_form_ss_inner(uint32_t state_size, uint32_t knot_points, T *d_S, T *d_Pinv, T *s_temp, unsigned blockrow){

    const uint32_t states_sq = state_size*state_size;
    
    //  STATE OF DEVICE MEM
    //  S:      -Q0_i in spot 00, phik left off-diagonal, thetak main diagonal, phik_T right off-diagonal
    //  Phi:    -Q0 in spot 00, theta_invk main diagonal
    //  gamma:  -Q0_i*q0 spot 0, gammak

    T *s_end = s_temp + states_sq;

    T *d_Phi_right_diag = get_block_bd_address<T>(state_size, knot_points, d_Pinv, 2, blockrow);
    T *d_Phi_left_diag = get_block_bd_address<T>(state_size, knot_points, d_Pinv, 0, blockrow);
    T *d_thetaInv_k = get_block_bd_address<T>(state_size, knot_points, d_Pinv, 1, blockrow);

    // // load thetaInv_k
    // load_block_bd<T>(
    //     state_size, knot_points,
    //     d_Pinv,
    //     s_thetaInv_k,
    //     1,
    //     blockrow
    // );
    // __syncthreads();//----------------------------------------------------------------


    if(blockrow!=0){

        T *d_phi_k = get_block_bd_address<T>(state_size, knot_points, d_S, 0, blockrow);
        T *d_thetaInv_km1 = get_block_bd_address<T>(state_size, knot_points, d_Pinv, 1, blockrow-1);

        // compute left off diag    
        glass::gemm<T, false>(
            state_size,
            state_size,
            state_size,
            static_cast<T>(1.0),
            d_thetaInv_k,
            d_phi_k,
            s_temp
        );
        __syncthreads();//----------------------------------------------------------------
        
        glass::gemm<T, false>(
            state_size,
            state_size,
            state_size,
            static_cast<T>(-1.0),
            s_temp,
            d_thetaInv_km1,
            d_Phi_left_diag
        );
    }
    __syncthreads();//----------------------------------------------------------------


    if(blockrow!=(knot_points - 1)){

        T *d_phi_kp1 = get_block_bd_address<T>(state_size, knot_points, d_S, 0, blockrow+1);
        T *d_thetaInv_kp1 = get_block_bd_address<T>(state_size, knot_points, d_Pinv, 1, blockrow+1);

        glass::gemm<T, true>(
            state_size,
            state_size,
            state_size,
            static_cast<T>(1.0),
            d_thetaInv_k,
            d_phi_kp1,
            s_temp
        );
        __syncthreads();//----------------------------------------------------------------

        glass::gemm<T, false>(
            state_size,
            state_size,
            state_size,
            static_cast<T>(-1.0),
            s_temp,
            d_thetaInv_kp1,
            d_Phi_right_diag
        );
    }
}

template <typename T>
    __device__ __forceinline__
    void loadIdentity(uint32_t DIM, T *A){
        for (unsigned ind = threadIdx.x; ind < DIM*DIM; ind += blockDim.x){
            unsigned r, c;
            r = ind % DIM; 
            c = ind / DIM;
            A[ind] = static_cast<T>(r == c);
        }
    }


/*
 STATE OF S
        //  S:      -Q0_i in spot 00, phik left off-diagonal, thetak main diagonal, phik_T right off-diagonal

 MAKE STATE OF PHI
        //  Phi:    -Q0 in spot 00, theta_invk main diagonal

*/
 template <typename T>
    __device__
    void gato_ss_from_schur(uint32_t state_size, uint32_t knot_points, T *d_S, T *d_Pinv, T *s_temp, unsigned blockrow){

	const uint32_t states_sq = state_size*state_size;
	T *s_diag = s_temp;
	T *s_diag_i = s_temp + states_sq;
	T *s_scratch = s_diag_i + states_sq; 

	/* Load diag S*/
	load_block_bd<T>(
            state_size, knot_points,
            d_S,
            s_diag,
            1,
            blockrow
        );


	__syncthreads();//----------------------------------------------------------------


	/* Invert diag S*/

	loadIdentity<T>( state_size, s_diag_i);

	__syncthreads();//----------------------------------------------------------------

	glass::invertMatrix<T>(
		state_size,
		s_diag,
		s_scratch
	);

	__syncthreads();//----------------------------------------------------------------


	/* Store in Pinv*/
	store_block_bd<T>(
                state_size, knot_points,
                s_diag_i, 
                d_Pinv,
                1,
                blockrow,
                1
        );

	__syncthreads();//----------------------------------------------------------------

    }

