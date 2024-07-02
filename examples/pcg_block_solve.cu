#include <iostream>
#include <stdio.h>
#include "gpu_pcg.cuh"
#include "gpuassert.cuh"
#include "read_array.h"


int main(int argc, char *argv[]) {

    const uint32_t state_size = STATE_SIZE;
    const uint32_t knot_points = KNOT_POINTS;
    const int matrix_size = 2 * knot_points * state_size * state_size;
    const int vector_size = state_size * knot_points;

    float h_Pinvob[matrix_size];
    float h_Pinvdb[vector_size];
    float h_Sob[matrix_size];
    float h_Sdb[vector_size];

    readArrayFromFile(matrix_size, "data/Pob.txt", h_Pinvob);
    readArrayFromFile(vector_size, "data/Pdb.txt", h_Pinvdb);
    readArrayFromFile(matrix_size, "data/Sob.txt", h_Sob);
    readArrayFromFile(vector_size, "data/Sdb.txt", h_Sdb);

    float h_gamma[vector_size];
    readArrayFromFile(vector_size, "data/gamma_tilde.txt", h_gamma);
    float h_lambda[vector_size];
    for (int i = 0; i < vector_size; i++) {
        h_lambda[i] = 0;
    }

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
    float norm = 0;
    for (int i = 0; i < vector_size; i++) {
        norm += h_lambda[i] * h_lambda[i];
    }
    std::cout << "Lambda norm: " << sqrt(norm) << std::endl;

    return 0;
}

