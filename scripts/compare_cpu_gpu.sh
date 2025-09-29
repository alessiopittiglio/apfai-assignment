#!/bin/bash

# This script compares the performance of the OpenMP and CUDA
# implementations across a range of increasing problem sizes. For each
# problem size $N$, it measures the average wall-clock time and
# calculates the computational throughput in GB/s.
#-----------------------------------------------------------------------
# The "Best OpenMP" version is run with a number of threads equal to the
# number of physical cores on the machine.

# Last updated on 2025-09-29
# Alessio Pittiglio

N_SIZES=(250000 500000 1000000 2000000 4000000 8000000) # array of problem sizes (N) to test (adjust as needed)
K=1000                  # number of layers (adjust as needed)
R=5
BLKDIM=512
NREPS=5                 # number of replications for stable measurements

PROG_OMP=./src_omp/main_omp
OMP_BEST_THREADS=6      # optimal number of threads for OpenMP (adjust as needed)

PROG_CUDA=./src_cuda/main_cuda

if [ ! -f "$PROG_OMP" ]; then
    echo
    echo "$PROG_OMP not found"
    echo
    exit 1
fi
if [ ! -f "$PROG_CUDA" ]; then
    echo
    echo "$PROG_CUDA not found"
    echo
    exit 1
fi
echo -e "Problem_Size_(N)\tOMP_Time_(s)\tOMP_Tput_(GB/s)\tOMP_FLOPs_(GFLOPs/s)\t\
CUDA_Global_Time_(s)\tCUDA_Shared_Time_(s)\tCUDA_Global_Tput_(GB/s)\t\
CUDA_Shared_Tput_(GB/s)\tCUDA_Global_FLOPs_(GFLOPs/s)\tCUDA_Shared_FLOPs_(GFLOPs/s)"

for N in "${N_SIZES[@]}"; do

    EFFECTIVE_BYTES=0
    TOTAL_FLOPS=0

    for (( t=1; t<K; t++ )); do
        OUT_SIZE=$(( N - t * (R-1) ))
        if [ $OUT_SIZE -le 0 ]; then
            break
        fi

        BYTES_LAYER=$(( OUT_SIZE * (2*R + 1) * 4 ))
        EFFECTIVE_BYTES=$(( EFFECTIVE_BYTES + BYTES_LAYER ))
        
        FLOPS_LAYER=$(( OUT_SIZE * 2 * R ))
        TOTAL_FLOPS=$(( TOTAL_FLOPS + FLOPS_LAYER ))
    done

    # ----------------- OpenMP -----------------
    TOTAL_TIME_OMP=0
    for rep in $(seq $NREPS); do
        TIME_OMP="$(OMP_NUM_THREADS=$OMP_BEST_THREADS "$PROG_OMP" "$N" "$K" | grep -oE '[0-9]+\.[0-9]+')"
        TOTAL_TIME_OMP=$(echo "$TOTAL_TIME_OMP + $TIME_OMP" | bc)
    done
    AVG_TIME_OMP=$(printf "%.6f" "$(echo "scale=6; $TOTAL_TIME_OMP / $NREPS" | bc)")
    THROUGHPUT_OMP=$(echo "scale=2; $EFFECTIVE_BYTES / 1000000000 / $AVG_TIME_OMP" | bc)
    FLOPS_OMP=$(echo "scale=2; $TOTAL_FLOPS / 1000000000 / $AVG_TIME_OMP" | bc)

    # ----------------- CUDA -----------------
    TOTAL_TIME_CUDA_GLOBAL=0
    TOTAL_TIME_CUDA_SHARED=0
    for rep in $(seq $NREPS); do
          OUTPUT=$("$PROG_CUDA" "$N" "$K")
          TIME_GLOBAL=$(echo "$OUTPUT" | grep "No shared memory" | grep -oE '[0-9]+\.[0-9]+')
          TIME_SHARED=$(echo "$OUTPUT" | grep "Shared memory" | grep -oE '[0-9]+\.[0-9]+' | head -n1)
          TOTAL_TIME_CUDA_GLOBAL=$(echo "$TOTAL_TIME_CUDA_GLOBAL + $TIME_GLOBAL" | bc)
          TOTAL_TIME_CUDA_SHARED=$(echo "$TOTAL_TIME_CUDA_SHARED + $TIME_SHARED" | bc)
    done
    AVG_TIME_CUDA_GLOBAL=$(printf "%.6f" "$(echo "scale=6; $TOTAL_TIME_CUDA_GLOBAL / $NREPS" | bc)")
    AVG_TIME_CUDA_SHARED=$(printf "%.6f" "$(echo "scale=6; $TOTAL_TIME_CUDA_SHARED / $NREPS" | bc)")

    THROUGHPUT_CUDA_GLOBAL=$(echo "scale=2; $EFFECTIVE_BYTES / 1000000000 / $AVG_TIME_CUDA_GLOBAL" | bc)
    THROUGHPUT_CUDA_SHARED=$(echo "scale=2; $EFFECTIVE_BYTES / 1000000000 / $AVG_TIME_CUDA_SHARED" | bc)
    FLOPS_CUDA_GLOBAL=$(echo "scale=2; $TOTAL_FLOPS / 1000000000 / $AVG_TIME_CUDA_GLOBAL" | bc)
    FLOPS_CUDA_SHARED=$(echo "scale=2; $TOTAL_FLOPS / 1000000000 / $AVG_TIME_CUDA_SHARED" | bc)

    echo -e "$N\t$AVG_TIME_OMP\t$THROUGHPUT_OMP\t$FLOPS_OMP\t\
    $AVG_TIME_CUDA_GLOBAL\t$AVG_TIME_CUDA_SHARED\t$THROUGHPUT_CUDA_GLOBAL\t\
    $THROUGHPUT_CUDA_SHARED\t$FLOPS_CUDA_GLOBAL\t$FLOPS_CUDA_SHARED"
done
