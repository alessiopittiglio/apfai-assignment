/****************************************************************************
 *
 * main_cuda.cu - 1D feed-forward multi-layer neural network with CUDA
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
 *      nvcc main_cuda.cu -o main_cuda
 *
 * Run with:
 *
 *      ./main_cuda [N] [K]
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

#define BLKDIM 512
#define RADIUS R - 1

typedef struct {
    float *d_base;
    float *d_layers[2];  /* buffer layers */
    float *d_W;          /* device copy of weights */
} DeviceMemory;

void allocate_device_memory(DeviceMemory *buf, int N) {
    const size_t n_size = N * sizeof(float);
    const size_t w_size = N * R * sizeof(float);
    const size_t total_size = 2 * n_size + w_size;

    cudaError_t err = cudaMalloc((void**)&buf->d_base, total_size);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        buf->d_layers[0] = buf->d_layers[1] = buf->d_W = NULL;
        return;
    }

    buf->d_layers[0] = buf->d_base;
    buf->d_layers[1] = buf->d_base + N;
    buf->d_W = buf->d_base + 2 * N;
}

void free_device_memory(DeviceMemory* buf) {
    if (buf->d_base) {
        cudaFree(buf->d_base);
    }
    buf->d_base = buf->d_layers[0] = buf->d_layers[1] = buf->d_W = NULL;
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Fills layer, weights, and bias with random values in [0,1] */
void fill(float *layer, int n, float *W, int r_dim, float *b) {
    for (int i = 0; i < n; i++) {
        layer[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int r = 0; r < r_dim; r++) {
        for (int i = 0; i < n; i++) {
            W[r * n + i] = (float)rand() / (float)RAND_MAX;
        }
    }
    *b = (float)rand() / (float)RAND_MAX;
}

/* Compute the next layer; this version does not use shared memory. */
__global__ void compute_layer_global(
    const float *x,
    float *y,
    const float *W,
    float b,
    int N,
    int out_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= out_size) return;

    float sum = b;
    for (int r = 0; r < R; ++r) {
        sum += x[index + r] * W[r * N + index];
    }
    y[index] = sigmoid(sum);
}

/* Compute the next layer; this version uses shared memory. */
__global__ void compute_layer_shared(
    const float *x,
    float *y,
    const float *W,
    float b,
    int N,
    int out_size
) {
    __shared__ float tile[BLKDIM + R - 1];

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x;

    if (gindex < N) {
        tile[lindex] = x[gindex];
    } else {
        tile[lindex] = 0.0f;
    }

    int block_end = blockIdx.x * blockDim.x + blockDim.x;
    int halo_idx = block_end + threadIdx.x;
    // right halo
    if (lindex < (R - 1)) {
        if (halo_idx < N)
            tile[blockDim.x + lindex] = x[halo_idx];
        else
            tile[blockDim.x + lindex] = 0.0f;
    }

    __syncthreads();

    if (gindex >= out_size) return;

    float sum = b;
    for (int r = 0; r < R; ++r) {
        sum += tile[lindex + r] * W[r * N + gindex];
    }
    y[gindex] = sigmoid(sum);
}

int check_result(const float *ref, const float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(ref[i] - res[i]) > 1e-5f) {
            printf("Check failed: y[%d] = %f, expected %f\n", i, res[i], ref[i]);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    if (N <= 0 || K <= 0) {
        fprintf(stderr, "Invalid arguments.\n");
        return EXIT_FAILURE;
    }

    if (N - (K - 1) * (R - 1) <= 0) {
        fprintf(stderr, "Error: Number of neurons becomes non-positive after %d layers.\n", K);
        return EXIT_FAILURE;
    }
    const size_t n_size = N * sizeof(float);
    const size_t w_size = N * R * sizeof(float);

    float *h_x = (float*)malloc(n_size);  /* host copy of input layer */
    float *h_y = (float*)malloc(n_size);  /* host copy of output layer */
    float *h_W = (float*)malloc(w_size);  /* host copy of weights */
    float h_b;                            /* host copy of bias */

    int final_size = N - (K - 1) * (R - 1);
    float *output_ref = (float*)malloc(final_size * sizeof(float));

    if (!h_x || !h_y || !h_W) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    fill(h_x, N, h_W, R, &h_b);

    DeviceMemory buf;
    allocate_device_memory(&buf, N);

    /* Copy inputs to device */
    cudaMemcpy(buf.d_layers[0], h_x, n_size, cudaMemcpyHostToDevice);
    cudaMemcpy(buf.d_W, h_W, w_size, cudaMemcpyHostToDevice);

    printf("1D NN forward pass (N=%d, K=%d, R=%d)\n", N, K, R);

    double tstart, tstop, tnoshared, tshared;

    /* -------------------------------------------------------------------
     * Forward pass WITHOUT shared memory
     * ----------------------------------------------------------------- */
    printf("No shared memory:\t");

    tstart = hpc_gettime();
    int current_in_idx = 0;
    int current_size = N;

    for (int t = 0; t < K - 1; ++t) {
        int next_size = current_size - (R - 1);
        int grid_size = (next_size + BLKDIM - 1) / BLKDIM;

        compute_layer_global<<<grid_size, BLKDIM>>>(
            buf.d_layers[current_in_idx],
            buf.d_layers[1 - current_in_idx],
            buf.d_W,
            h_b,
            N,
            next_size
        );
        
        current_in_idx = 1 - current_in_idx;
        current_size = next_size;
    }
    cudaDeviceSynchronize();
    tstop = hpc_gettime();

    tnoshared = tstop - tstart;
    printf("%.6f s\n", tnoshared);

    /* Copy result back to host */
    cudaMemcpy(output_ref,
        buf.d_layers[current_in_idx],
        final_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    /* -------------------------------------------------------------------
     * Forward pass WITH shared memory
     * ----------------------------------------------------------------- */
    printf("Shared memory:\t");

    tstart = hpc_gettime();
    current_in_idx = 0;
    current_size = N;

    for (int t = 0; t < K - 1; ++t) {
        int next_size = current_size - (R - 1);
        int grid_size = (next_size + BLKDIM - 1) / BLKDIM;
        
        compute_layer_shared<<<grid_size, BLKDIM>>>(
            buf.d_layers[current_in_idx],
            buf.d_layers[1 - current_in_idx],
            buf.d_W,
            h_b,
            N,
            next_size
        );
        
        current_in_idx = 1 - current_in_idx;
        current_size = next_size;
    }
    cudaDeviceSynchronize();
    tstop = hpc_gettime();

    tshared = tstop - tstart;
    printf("%.6f s (%.2fx speedup)\n", tshared, tnoshared / tshared);

    /* Copy result back to host and check correctness */
    cudaMemcpy(h_y,
        buf.d_layers[current_in_idx],
        final_size * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    
    check_result(output_ref, h_y, final_size);

    /* Cleanup */
    free(h_x);
    free(h_y);
    free(h_W);
    free(output_ref);
    free_device_memory(&buf);

    return EXIT_SUCCESS;
}
