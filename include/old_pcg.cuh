#include <cooperative_groups.h>



namespace cgrps = cooperative_groups;



namespace oldpcgutils{
    /// TODO: this is really stupid but has to be done for now
/// TODO: error check?
template <typename T>
__device__
void gato_memcpy(T *dst, T *src, unsigned size_Ts){
    unsigned ind;
    for(ind=threadIdx.x; ind < size_Ts; ind+=blockDim.x){
        dst[ind] = src[ind];
    }
}


template <typename T, unsigned BLOCK_DIM>
__device__
void loadBlockTriDiagonal_offDiagonal(T *s_var, T *d_var_b, cgrps::thread_block block, cgrps::grid_group grid){

    cgrps::thread_group tile = cgrps::tiled_partition(block, 32);
    int tileId = threadIdx.x / 32;

    if(tileId == 0){
        // Need to load b also now
        for (unsigned ind = tile.thread_rank(); ind < BLOCK_DIM; ind += tile.size()){
            s_var[ind + BLOCK_DIM] = *(d_var_b + ind); 
        }
    }
    else if(tileId==1){
        // if first block just want b and b+1 (and already have b)
        if((blockIdx.x == 0)){
            for (unsigned ind = tile.thread_rank(); ind < BLOCK_DIM; ind += tile.size()){
                s_var[ind + 2*BLOCK_DIM] = *(d_var_b + BLOCK_DIM + ind); // just b+1
            }

        }
        // if last block just want b-1 and b (and already have b)
        else if (GATO_LAST_BLOCK){
            for (unsigned ind = tile.thread_rank(); ind < BLOCK_DIM; ind += tile.size()){
                s_var[ind] = *(d_var_b - BLOCK_DIM + ind); // just b-1
            }

        }
        // else want b-1 and b and b+1 (and already have b)
        else{
            T *dst, *src;
            for (unsigned ind = tile.thread_rank(); ind < 2*BLOCK_DIM; ind += tile.size()){
                dst = s_var + ind + (ind >= BLOCK_DIM) * BLOCK_DIM;
                src = d_var_b + ind - (ind < BLOCK_DIM) * BLOCK_DIM;
                *dst = *src;
            }
        }
    }
}

template <typename T, unsigned BLOCK_DIM>
__device__ 
void matVecMultBlockTriDiagonal(T *s_dst, T *s_mat, T *s_vec, cgrps::thread_block block, cgrps::grid_group grid){
    // First or Last block only 2 mults (var and either var+1 or var-1)
    T val;
    if((blockIdx.x == 0)){
        for (unsigned r = threadIdx.x; r < BLOCK_DIM; r += blockDim.x){
            val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM*BLOCK_DIM + BLOCK_DIM * c + r] * s_vec[c + BLOCK_DIM]; // var and var+1
            }
            s_dst[r] = val;
        }
    }
    else if (GATO_LAST_BLOCK){
        for (unsigned r = threadIdx.x; r < BLOCK_DIM; r += blockDim.x){
            val = static_cast<T>(0);
            for(unsigned c = 0; c < 2*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c]; // var and var-1
            }
            s_dst[r] = val;
        }
    }
    else{
        // else 3 mults
        for (unsigned r = threadIdx.x; r < BLOCK_DIM; r += blockDim.x){
            val = static_cast<T>(0);
            for(unsigned c = 0; c < 3*BLOCK_DIM; c++){
                val += s_mat[BLOCK_DIM * c + r] * s_vec[c];
            }
            s_dst[r] = val;
        }
    }
}

template <typename T, unsigned VEC_SIZE>
__device__
void reducePlus(T *dstTemp, cgrps::thread_block block){
    unsigned size_left = VEC_SIZE;
    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = threadIdx.x; ind < size_left; ind += blockDim.x){
            dstTemp[ind] += dstTemp[ind + size_left];
        }	
        // add the odd size adjust if needed
        if ((threadIdx.x == 0) && odd_flag){dstTemp[0] += dstTemp[2*size_left];}
        // sync and repeat
        block.sync();
    }
    // when we get really small sum up what is left
    if ((threadIdx.x == 0)){
        for(unsigned ind = 1; ind < size_left; ind++){dstTemp[0] += dstTemp[ind];}
    }
}

template <typename T>
__device__
void reducePlus(T *dstTemp, unsigned VEC_SIZE, cgrps::thread_block block){
    unsigned size_left = VEC_SIZE;
    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = threadIdx.x; ind < size_left; ind += blockDim.x){
            dstTemp[ind] += dstTemp[ind + size_left];
        }	
        // add the odd size adjust if needed
        if ((threadIdx.x == 0) && odd_flag){dstTemp[0] += dstTemp[2*size_left];}
        // sync and repeat
        block.sync();
    }
    // when we get really small sum up what is left
    if ((threadIdx.x == 0)){
        for(unsigned ind = 1; ind < size_left; ind++){dstTemp[0] += dstTemp[ind];}
    }
}


template <typename T>
__device__
void reducePlus_copy(T *dst, T *src, unsigned VEC_SIZE, cgrps::thread_block block){
    
    gato_memcpy<T>(dst, src, VEC_SIZE);
    block.sync();

    unsigned size_left = VEC_SIZE;
    // loop until only a few values left
    while (size_left > 3){
        // determine if odd_adjust needed and update size
        bool odd_flag = size_left % 2;
        size_left = (size_left - odd_flag)/2; 
        // reduce in half
        for (unsigned ind = threadIdx.x; ind < size_left; ind += blockDim.x){
            dst[ind] += dst[ind + size_left];
        }	
        // add the odd size adjust if needed
        if ((threadIdx.x == 0) && odd_flag){dst[0] += dst[2*size_left];}
        // sync and repeat
        block.sync();
    }
    // when we get really small sum up what is left
    if ((threadIdx.x == 0)){
        for(unsigned ind = 1; ind < size_left; ind++){dst[0] += dst[ind];}
    }
}

template <typename T, unsigned VEC_SIZE>
__device__
void dotProd(T *dstTemp, T *vec1, T *vec2, cgrps::thread_block block){
    // first compute temp sums across all threads
    for (unsigned ind = threadIdx.x; ind < VEC_SIZE; ind += blockDim.x){
        dstTemp[ind] = vec1[ind] * vec2[ind];
    }
    block.sync();
    // then reduce
    reducePlus<T,VEC_SIZE>(dstTemp,block);
}

template <typename T>
__device__
void dotProd(T *dstTemp, T *vec1, T *vec2,unsigned VEC_SIZE, cgrps::thread_block block){
    // first compute temp sums across all threads
    for (unsigned ind = threadIdx.x; ind < VEC_SIZE; ind += blockDim.x){
        dstTemp[ind] = vec1[ind] * vec2[ind];
    }
    block.sync();
    // then reduce
    reducePlus<T>(dstTemp,VEC_SIZE,block);
}

template <typename T>
__device__
void gato_axpy(T *z, T a, T *x, T *y, unsigned n, cgrps::thread_group g){
    for(unsigned i = g.thread_rank(); i < n; i+=g.size()){
        z[i] = a * x[i] + y[i];
    }
}
}


namespace oldpcg{


template <typename T, bool USE_TRACE = false>
__device__
void parallelPCG_inner(float  *s_S, float  *s_pinv, float  *s_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float *d_v_temp, float *d_eta_new_temp,	// global vectors and scalars
                        float  *s_temp, int *iters,	int maxIters, float exitTol, 		    // shared mem for use in CG step and constants
                        cgrps::thread_block block, cgrps::grid_group grid){                      
    //Initialise shared memory
    float  *s_lambda = s_temp;
    float  *s_r_tilde = s_lambda + 3*14;
    float  *s_upsilon = s_r_tilde + 14;
    float  *s_v_b = s_upsilon + 14;
    float  *s_eta_new_b = s_v_b + max(50, 14);

    float  *s_r = s_eta_new_b + max(50, 14);
    float  *s_p = s_r + 3*14;

    float  *s_r_b = s_r + 14;
    float  *s_p_b = s_p + 14;
    float *s_lambda_b = s_lambda + 14;

    T alpha, beta;
    T eta = static_cast<T>(0);	T eta_new = static_cast<T>(0);

    // Used when writing to device memory
    int bIndStateSize = 14 * GATO_BLOCK_ID;

    // Initililiasation before the main-pcg loop
    // note that in this formulation we always reset lambda to 0 and therefore we can simplify this step
    // Therefore, s_r_b = s_gamma_b

    // We find the s_r, load it into device memory, initialise lambda to 0

    oldpcgutils::loadBlockTriDiagonal_offDiagonal<float ,14>(s_lambda,&d_lambda[bIndStateSize],block,grid);
    block.sync();
    oldpcgutils::matVecMultBlockTriDiagonal<float ,14>(s_r_b,s_S,s_lambda,block,grid);
    block.sync();
    for (unsigned ind = threadIdx.x; ind < 14; ind += blockDim.x){
        s_r_b[ind] = s_gamma[ind] - s_r_b[ind];
        d_r[bIndStateSize + ind] = s_r_b[ind]; 

        //Already initialised before
        // s_lambda_b[ind] = d_lambda[bIndStateSize + ind];      // d_lambda already initialized to zeros or guess if warm start
    }
    // Make eta_new zero
    // if((threadIdx.x == 0) && (blockIdx.x == 0)){
    //     d_eta_new_temp[0] = static_cast<T>(0);
    //     d_v_temp[0] = static_cast<T>(0);
    // }

    if((blockIdx.x == 0) && (threadIdx.x == 0)){
        *iters = maxIters;
    }
    // Need to sync before loading from other blocks
    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    oldpcgutils::loadBlockTriDiagonal_offDiagonal<float ,14>(s_r,&d_r[bIndStateSize],block,grid);
    block.sync();
    oldpcgutils::matVecMultBlockTriDiagonal<float ,14>(s_r_tilde,s_pinv,s_r,block,grid);
    block.sync();

    // We copy p from r_tilde and write to device, since it will be required by other blocks
    for (unsigned ind = threadIdx.x; ind < 14; ind += blockDim.x){
        s_p_b[ind] = s_r_tilde[ind];
        d_p[bIndStateSize + ind] = s_p_b[ind]; 
    }

    oldpcgutils::dotProd<float ,14>(s_eta_new_b,s_r_b,s_r_tilde,block);
    block.sync();

    if((threadIdx.x == 0)){
        // printf("Partial sums of Block %d: %f\n", GATO_BLOCK_ID, s_eta_new_b[0] );
        d_eta_new_temp[GATO_BLOCK_ID] = s_eta_new_b[0];
    }

    grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
    oldpcgutils::reducePlus_copy<T>(s_eta_new_b, d_eta_new_temp, GATO_NUM_BLOCKS, block);
    block.sync();
    eta = s_eta_new_b[0];

    // if((blockIdx.x == 0) && (threadIdx.x == 0)){
    //     printf("Before main loop eta %f > exitTol %f\n",eta, exitTol);
    // }
    
    for(unsigned iter = 0; iter < maxIters; iter++){
        oldpcgutils::loadBlockTriDiagonal_offDiagonal<float ,14>(s_p,&d_p[bIndStateSize],block,grid);
        block.sync();
        oldpcgutils::matVecMultBlockTriDiagonal<float ,14>(s_upsilon,s_S,s_p,block,grid);
        block.sync();
        oldpcgutils::dotProd<float ,14>(s_v_b,s_p_b,s_upsilon,block);
        block.sync();

        if((threadIdx.x == 0)){
            d_v_temp[GATO_BLOCK_ID] = s_v_b[0];
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        oldpcgutils::reducePlus_copy<T>(s_v_b, d_v_temp, GATO_NUM_BLOCKS, block);
        block.sync();

        alpha = eta / s_v_b[0];

        // if(false){
        //     printf("d_pSp[%f] -> alpha[%f]\n",*d_v,alpha);
        // }

        // block.sync();

        // Move this loop into a function
        for(unsigned ind = threadIdx.x; ind < 14; ind += blockDim.x){
            s_lambda_b[ind] += alpha * s_p_b[ind];
            s_r_b[ind] -= alpha * s_upsilon[ind];
            d_r[bIndStateSize + ind] = s_r_b[ind];
            }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER

        oldpcgutils::loadBlockTriDiagonal_offDiagonal<float ,14>(s_r,&d_r[bIndStateSize],block,grid);
        block.sync();
        oldpcgutils::matVecMultBlockTriDiagonal<float ,14>(s_r_tilde,s_pinv,s_r,block,grid);
        block.sync();
        oldpcgutils::dotProd<float ,14>(s_eta_new_b,s_r_b,s_r_tilde,block);
        block.sync();
        if((threadIdx.x == 0)){
            d_eta_new_temp[GATO_BLOCK_ID] = s_eta_new_b[0];
        }
        grid.sync(); //---------------------------------------------------------------------------------------------------GLOBAL BARRIER
        oldpcgutils::reducePlus_copy<T>(s_eta_new_b, d_eta_new_temp, GATO_NUM_BLOCKS, block);
        block.sync();
        eta_new = s_eta_new_b[0];
        
#if DEBUG_MODE
        if(GATO_BLOCK_ID==0&&threadIdx.x==0){
            printf("eta_new[%f]\n",abs(eta_new));
        }
        block.sync();
#endif /* #if DEBUG_MODE */

        if(abs(eta_new) < exitTol){

            if((blockIdx.x == 0) && (threadIdx.x == 0)){
                *iters = iter;
            }

            break;
        }
        
        // else compute d_p for next loop
        else{
            beta = eta_new / eta;
            for(unsigned ind = threadIdx.x; ind < 14; ind += blockDim.x){
                s_p_b[ind] = s_r_tilde[ind] + beta*s_p_b[ind];
                d_p[bIndStateSize + ind] = s_p_b[ind];
            }
            eta = eta_new;
        }

        // if((blockIdx.x == 0) && (threadIdx.x == 0)){
        //     printf("Executing iter %d with eta %f > exitTol %f------------------------------------------------------\n", iter, abs(eta_new), exitTol);
        // }
        // then global sync for next loop
        grid.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
        
    }
    // save final lambda to global
    block.sync(); //-------------------------------------------------------------------------------------------------------BARRIER
    for(unsigned ind = threadIdx.x; ind < 14; ind += blockDim.x){
        d_lambda[bIndStateSize + ind] = s_lambda_b[ind];
    }
    
}

template <typename T, bool USE_TRACE = false>
__global__
void parallelPCG(float  *d_S, float  *d_pinv, float  *d_gamma,  				// block-local constant temporary variable inputs
                        float  *d_lambda, float  *d_r, float  *d_p, float *d_v_temp, float *d_eta_new_temp,	// global vectors and scalars
                        int *iters, int maxIters=100, T exitTol = 1e-6 	// shared mem for use in CG step and constants
                        ){
    
    __shared__ T s_temp[(3*14*14 + 3*14*14 + 10 * 14 + 2*50)];
    float  *s_S = s_temp;
    float  *s_pinv = s_S +3*14*14;
    float  *s_gamma = s_pinv + 3*14*14;
    float  *shared_mem = s_gamma + 14;

    cgrps::thread_block block = cgrps::this_thread_block();	 
    cgrps::grid_group grid = cgrps::this_grid();

    int bIndStateSize = 14 * GATO_BLOCK_ID;
    for (unsigned ind = threadIdx.x; ind < 3 * 14 * 14; ind += blockDim.x){
        if((blockIdx.x == 0) && ind < (14*14)){ continue; }
        if(GATO_LAST_BLOCK && ind >= 2*(14*14)){ continue; }
        s_S[ind] = d_S[bIndStateSize*14*3 + ind];
        s_pinv[ind] = d_pinv[bIndStateSize*14*3 + ind];
    }
    for (unsigned ind = threadIdx.x; ind < 14; ind += blockDim.x){
        s_gamma[ind] = d_gamma[bIndStateSize + ind];
    }
    grid.sync();

    parallelPCG_inner<float>(s_S, s_pinv, s_gamma, d_lambda, d_r, d_p, d_v_temp, d_eta_new_temp, shared_mem, iters, maxIters, exitTol, block, grid);
    grid.sync();
}




}