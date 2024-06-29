#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"


int main() {

    const uint32_t state_size = 2;
    const uint32_t knot_points = 3;

    // identity preconditioner Pinv
    float h_Pinvob[24] = {1, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 0, 0, 1};
    float h_Pinvdb[6] = {1, 1, 1, 1, 1, 1};

    float h_Sob[24] = {0, 0, 0, 0,
                       -1, 0.1, 0.2, -0.5,
                       -1, 0.2, 0.1, -0.5,
                       -0.8, 0.2, 0.1, -0.9,
                       -0.8, 0.1, 0.2, -0.9,
                       0, 0, 0, 0};
    float h_Sdb[6] = {2, 3, 1, 1, 5, 4};


    float h_gamma[6] = {1, 1, 1, 1, 1, 1};
    float h_lambda[6] = {0, 0, 0, 0, 0, 0};


    struct pcg_config<float> config;
    uint32_t res = solvePCGBlock<float>(h_Sdb,
                                        h_Sob,
                                        h_Pinvdb,
                                        h_Pinvob,
                                        h_gamma,
                                        h_lambda,
                                        state_size,
                                        knot_points,
                                        &config);

    std::cout << "GBD-PCG-Block returned in " << res << " iters." << std::endl;
    std::cout << "Lambda: " << std::endl;
    for (int i = 0; i < 6; i++) {
        std::cout << h_lambda[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

