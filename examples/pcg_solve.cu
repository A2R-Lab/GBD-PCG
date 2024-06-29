#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"




int main(){

    const uint32_t state_size = 2;
    const uint32_t knot_points = 3;

    float h_Pinv[36] ={1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1,
                       1, 0, 0, 1};

    float h_S[36] = {0,0,0,0,
                     2, 0, 0, 3,
                     -1, 0.1, 0.2, -1,
                     -1, 0.2, 0.1, -1,
                     1, 0, 0 ,1,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     1, 0, 0, 1,
                     0,0,0,0};
    
    float h_gamma[6] = {1, 1, 1, 1, 1, 1};
    float h_lambda[6] = {0,0,0,0,0,0};


    struct pcg_config<float> config;
    uint32_t res = solvePCG<float>(h_S,
                                    h_Pinv,
									h_gamma,
									h_lambda,
									state_size,
									knot_points,
									&config);

    std::cout << "GBD-PCG returned in " << res << " iters." << std::endl;
    std::cout << "Lambda: " << std::endl;
    for(int i = 0; i < 6; i++){
        std::cout << h_lambda[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

