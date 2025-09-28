/****************************************************************************
 *
 * main_omp.c - 1D feed-forward multi-layer neural network with OpenMP
 *
 * Copyright (C) 2025 by Alessio Pittiglio
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp main_omp.c -o main_omp
 *
 * Run with:
 *
 *      ./main_omp [N] [K]
 *
 * (N = number of neurons in the input layer, K = number of layers).
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef R
#define R 5
#endif

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Fills layer, weights, and bias with random values in [0,1] */
static void fill(float *layer, int n, float *W, int r, float *b) {
    for (int i = 0; i < n; i++) {
        layer[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < r * n; i++) {
        W[i] = (float)rand() / (float)RAND_MAX;
    }
    *b = (float)rand() / (float)RAND_MAX;
}

/* Forward pass through a 1D multi-layer network */
void forward_pass(float *layers[2], int N, int K, float *W, float b) {
    int current_in = 0;
    int current_size = N;

    for (int t = 0; t < K - 1; ++t) {
        int next_size = current_size - (R - 1);
        const float *x = layers[current_in];
        float *y = layers[1 - current_in];

        #pragma omp parallel for schedule(static) default(none) \
            shared(x, y, W, next_size, b)
        for (int i = 0; i < next_size; ++i) {
            const float *p_x = &x[i];
            const float *p_W  = &W[i * R];
            float sum = b;

            #if R == 3
                sum += p_x[0] * p_W[0];
                sum += p_x[1] * p_W[1];
                sum += p_x[2] * p_W[2];
            #elif R == 4
                sum += p_x[0] * p_W[0];
                sum += p_x[1] * p_W[1];
                sum += p_x[2] * p_W[2];
                sum += p_x[3] * p_W[3];
            #elif R == 5
                sum += p_x[0] * p_W[0];
                sum += p_x[1] * p_W[1];
                sum += p_x[2] * p_W[2];
                sum += p_x[3] * p_W[3];
                sum += p_x[4] * p_W[4];
            #else
                #warning "Compiling with genercic loop for R."
                for (int r = 0; r < R; ++r) {
                    sum += x[i + r] * W[i * R + r];
                }
            #endif
            y[i] = sigmoid(sum);
        }

        current_in = 1 - current_in;
        current_size = next_size;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    if (N <= 0 || K <= 0) {
        fprintf(stderr, "N and K must be positive integers.\n");
        return EXIT_FAILURE;
    }

    if (N - (K - 1) * (R - 1) <= 0) {
        fprintf(stderr, "Error: Number of neurons becomes non-positive after %d layers.\n", K);
        fprintf(stderr, "N must be > (K-1)*(R-1).\n");
        return EXIT_FAILURE;
    }

    float *layers[2];
    layers[0] = (float*)malloc(N * sizeof(float));
    layers[1] = (float*)malloc(N * sizeof(float));
    float *W = (float*)malloc(N * R * sizeof(float));
    float b;

    if (!layers[0] || !layers[1] || !W) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(layers[0]);
        free(layers[1]);
        free(W);
        return EXIT_FAILURE;
    }

    fill(layers[0], N, W, R, &b);

    printf("1D NN forward pass (N=%d, K=%d, R=%d)\n", N, K, R);

    double start_time = hpc_gettime();
    forward_pass(layers, N, K, W, b);
    double end_time = hpc_gettime();

    printf("Execution time %f\n", end_time - start_time);

    free(layers[0]);
    free(layers[1]);
    free(W);

    return EXIT_SUCCESS;
}
