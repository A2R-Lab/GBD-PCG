#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"

int main(){

    const uint32_t state_size = 2;
    const uint32_t knot_points = 3;
	int i;

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
    float h_lambda[6] = {0,0,0,0,0,0};

	float h_S10[360];
	float h_gamma10[60];
	float h_lambda10[60];

	for (i = 0; i < 60; i++) {
		h_gamma10[i] = h_gamma[i%6] * (i / 6 + 1);
		h_lambda10[i] = 0;
	}
	for (i = 0; i < 360; i++) {
		h_S10[i] = h_S[i%36] * (i / 36 + 1);
	}

    struct pcg_config<float> config;
    uint32_t res = solvePCG<float>(h_S10,
									h_gamma10,
									h_lambda10,
									state_size,
									knot_points*10,
									&config);

    std::cout << "GBD-PCG returned in " << res << " iters." << std::endl;
    std::cout << "Lambda: " << std::endl;
    for(int i = 0; i < 60; i++){
        std::cout << h_lambda10[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}

