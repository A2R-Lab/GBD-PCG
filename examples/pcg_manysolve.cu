#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"

int main(){

    const uint32_t state_size = 2;
    const uint32_t knot_points = 3;
	int i, j;

    float h_S[36] = {0,0,0,0,
                     -.999, 0, 0, -.999,
                     .999, .0999, -.98, .999,
                     .999, -.98, .0999, .999,
                     -2.008, .8801, .8801, -3.0584,
                     .999, .0999, -.98, .999,
                     .999, -.98, .0999, .999,
                     -1.019, .8801, .8801, -2.0694,
                     0,0,0,0};

    float h_gamma[6] = {3.1385, 0, 0, 3.0788, .0031, 3.0788};

	float h_S10[10][36];
	float h_gamma10[10][6];
	float h_lambda10[10][6] = {{0}};

    for (i = 0; i < 10; i++) {
        for (j = 0; j < 6; j++) {
            h_gamma10[i][j] = h_gamma[j] * (i + 1);
        }
        for (j = 0; j < 36; j++) {
            h_S10[i][j] = h_S[j] * (i + 1);
        }
    }

    struct pcg_config<float> config[10];
    uint32_t res[10] = {0};
    for (i = 0; i < 10; i++) {
        res[i] = solvePCG<float>(
            h_S10[i],
            h_gamma10[i],
            h_lambda10[i],
            state_size,
            knot_points,
            config + i
        );
        std::cout << "GDB-PCG returned in " << res[i] << " iters\n";
        std::cout << "\tLambda:\t";
        for (int j = 0; j < 6; j++) {
            std::cout << h_lambda10[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << std::endl;

    return 0;
}

