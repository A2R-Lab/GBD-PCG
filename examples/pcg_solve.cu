#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"



int main(){

    const uint32_t state_size = 2;
    const uint32_t knot_points = 3;

    float h_S[36] = {0,0,0,0,
                     -.999, 0, 0, -.999,
                     .999, .0999, -.98, .999,
                     .999, -.98, .0999, .999,
                     -2.008, .8801, .8801, -3.0584,
                     .999, .0999, -.98, .999,
                     .999, -.98, .0999, .999,
                     -1.019, .8801, .8801, -2.0694,
                     0,0,0,0};

    float h_Pinv[36] = {0,0,0,0,
                        -1.001, -0, -0, -1.001,
                        -.409, -.221, .2031, -.3906,
                        -.409, .2031, -.221, -.3906,
                        -.5699, -.164, -.164, -.3742, 
                        -.6482, -.4527, -.0849, -.2955,
                        -.6482, -.0849, -.4527, -.2955, 
                        -1.5512, -.6597, -.6597, -.7638,
                        0,0,0,0};
    
    float h_gamma[6] = {3.1385, 0, 0, 3.0788, .0031, 3.0788};
    float h_lambda[6] = {0,0,0,0,0,0};

    float *d_S, *d_Pinv, *d_gamma, *d_lambda;
    gpuErrchk(cudaMalloc(&d_S, 36*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_Pinv, 36*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_gamma, 6*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_lambda, 6*sizeof(float)));

    gpuErrchk(cudaMemcpy(d_S, h_S, 36*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Pinv, h_Pinv, 36*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_gamma, h_gamma, 6*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_lambda, h_lambda, 6*sizeof(float), cudaMemcpyHostToDevice));

    pcg_config config;
    uint32_t res = solvePCG<float>(state_size, knot_points, d_S, d_Pinv, d_gamma, d_lambda, &config);

    gpuErrchk(cudaMemcpy(h_lambda, d_lambda, 6*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_S));
    gpuErrchk(cudaFree(d_Pinv));
    gpuErrchk(cudaFree(d_gamma));
    gpuErrchk(cudaFree(d_lambda));

    std::cout << "GPU-PCG returned in " << res << " iters." << std::endl;
    std::cout << "Lambda: " << std::endl;
    for(int i = 0; i < 6; i++){
        std::cout << h_lambda[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

