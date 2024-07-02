//
// Created by huier on 24-7-2.
//

#ifndef MPCGPU_READMATRIX_H
#define MPCGPU_READMATRIX_H

#endif //MPCGPU_READMATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

void readArrayFromFile(uint32_t size, const char *filename,
                       float *matrix) {
    FILE *myFile;
    myFile = fopen(filename, "r");
    if (myFile == NULL) {
        printf("Error Reading File\n");
        exit(0);
    }

    for (uint32_t i = 0; i < size; i++) {
        int ret = fscanf(myFile, "%f,", &matrix[i]);
    }

    fclose(myFile);
    return;
}
