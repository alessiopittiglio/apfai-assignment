# Architectures and Platforms for AI - Programming Assignment

This project implements the evaluation of a multi-layer Neural Network using OpenMP and CUDA. Each output neuron depends only on R consecutive input neurons. The network consists of K layers. Each output node is computed as:

$$
y_i = f \left(\sum_{r=0}^{R-1} x_{i+r} \times W_{i,r} + b\right)
$$

with $f(x)$ as the sigmoid activation function and a shared bias $b$. Because each output depends only on R inputs, the network is sparsely connected. 

## Project structure

```
├── build
├── docs
│   └── Assignment-NeuralNet.pdf    # Assignment description
├── include
│   └── hpc.h                       # Header file for HPC utilities
├── plot_results.ipynb
├── plots
├── README.md                       # This file
├── report
├── results
├── scripts
│   ├── compare_cpu_gpu.sh          # Compare CPU and GPU performance
│   ├── strong_scaling.sh           # Strong scaling experiments
│   └── weak_scaling.sh             # Weak scaling experiments
├── src_cuda/                       # CUDA implementation
│   └── main_cuda.cu                # Source code
└── src_omp/                        # OpenMP implementation
    └── main_omp.c                  # Source code
```

## Compilation

You need `gcc` with OpenMP support and the `nvcc` compiler from the NVIDIA CUDA Toolkit.

### OpenMP version

To compile the OpenMP program, navigate to the `src_omp` directory and run `make`:

```
cd src_omp
make
```

This will create an executable named `main_omp`. The value of `R` can be changed by editing the `R` variable in the `Makefile`.

### CUDA version

To compile the CUDA program, navigate to the `src_cuda` directory and run `make`:

```
cd src_cuda
make
```

This will create an executable named `main_cuda`. The value of `R` can be changed by editing the `R_VAL` variable in the `Makefile`.

## Execution

### OpenMP program

The program takes two command-line arguments: `N` (number of neurons in the input layer) and `K` (total number of layers).

```
./main_omp <N> <K>

# Example:
./main_omp 1000000 100
```

### CUDA program

The program takes two command-line arguments: `N`, `K`.

```
./main_cuda <N> <K>
```

## OpenMP Scalability Analysis

Regarding the performance analysis of the OpenMP implementation, a scalability study was carried out, considering both strong scaling and weak scaling.

```
cd scripts
./strong_scaling.sh
./weak_scaling.sh
```

These scripts run `main_omp` with various thread counts and problem sizes, saving the execution times to `results/strong_scaling_results.txt` and `results/weak_scaling_results.txt`.

## CUDA Performance Analysis

```
cd scripts
./compare_cpu_gpu.sh
```

This script runs both implementations across different problem sizes and saves the results to `results/compare_cpu_gpu_results.txt`.

To generate the performance plots (as seen in the report), you can run the `plot_results.py`. It reads the data files from the `results/` directory and saves the plots as PDFs in the plots/ directory.
