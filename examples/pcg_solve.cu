#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
#include "read_array.h"


int main() {

    const uint32_t state_size = STATE_SIZE;
    const uint32_t knot_points = KNOT_POINTS;
    const int matrix_size = 3 * knot_points * state_size * state_size;
    const int vector_size = state_size * knot_points;

    float h_Pinv[matrix_size];
    float h_S[matrix_size];
    readArrayFromFile(matrix_size, "data/S.txt", h_S);
    readArrayFromFile(matrix_size, "data/P.txt", h_Pinv);

    float h_gamma[vector_size];
    readArrayFromFile(vector_size, "data/gamma.txt", h_gamma);
    float h_lambda[vector_size];
    for (int i = 0; i < vector_size; i++) {
        h_lambda[i] = 0;
    }
    struct pcg_config<float> config;
    uint32_t res = solvePCG<float>(h_S,
                                   h_Pinv,
                                   h_gamma,
                                   h_lambda,
                                   state_size,
                                   knot_points,
                                   &config);

    std::cout << "GBD-PCG returned in " << res << " iters." << std::endl;
    float norm = 0;
    for (int i = 0; i < vector_size; i++) {
        norm += h_lambda[i] * h_lambda[i];
    }
    std::cout << "Lambda norm: " << sqrt(norm) << std::endl;

    return 0;
}

