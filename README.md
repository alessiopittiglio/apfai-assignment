# Architectures and Platforms for AI - Programming Assignment

This project implements the evaluation of a sparsely connected multi-layer Neural Network using OpenMP and CUDA. Each output neuron depends only on R consecutive input neurons.

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

```bash
./main_cuda <N> <K>
