#!/bin/bash

# This script compares the performance of the OpenMP and CUDA
# implementations across a range of increasing problem sizes. For each
# problem size $N$, it measures the average wall-clock time and
# calculates the computational throughput in millions of neurons per
# second.
#-----------------------------------------------------------------------
# The "Best OpenMP" version is run with a number of threads equal to the
# number of physical cores on the machine.

# Last updated on 2025-09-30
# Alessio Pittiglio

N_SIZES=(10000 20000 50000 100000 200000 250000 500000 1000000 2000000 5000000)
K=1000                  # number of layers (adjust as needed)
R=5
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
echo -e "Problem_Size_(N)\tOMP_Time_(s)\tOMP_Tput_(Mneurons/s)\t\
CUDA_Global_Time_(s)\tCUDA_Global_Tput_(Mneurons/s)\t\
CUDA_Shared_Time_(s)\tCUDA_Shared_Tput_(Mneurons/s)"

for N in "${N_SIZES[@]}"; do

    TOTAL_NEURONS=0
    for (( t=1; t<K; t++ )); do
        OUT_SIZE=$(( N - t * (R-1) ))
        if [ $OUT_SIZE -le 0 ]; then
            break
        fi
        TOTAL_NEURONS=$(( TOTAL_NEURONS + OUT_SIZE ))
    done

    # ----------------- OpenMP -----------------
    TOTAL_TIME_OMP=0
    for rep in $(seq $NREPS); do
        TIME_OMP="$(OMP_NUM_THREADS=$OMP_BEST_THREADS "$PROG_OMP" "$N" "$K" | grep -oE '[0-9]+\.[0-9]+')"
        TOTAL_TIME_OMP=$(echo "$TOTAL_TIME_OMP + $TIME_OMP" | bc)
    done
    AVG_TIME_OMP=$(printf "%.6f" "$(echo "scale=6; $TOTAL_TIME_OMP / $NREPS" | bc)")
    THROUGHPUT_OMP=$(echo "scale=2; $TOTAL_NEURONS / 1000000 / $AVG_TIME_OMP" | bc)

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

    THROUGHPUT_CUDA_GLOBAL=$(echo "scale=2; $TOTAL_NEURONS / 1000000 / $AVG_TIME_CUDA_GLOBAL" | bc)
    THROUGHPUT_CUDA_SHARED=$(echo "scale=2; $TOTAL_NEURONS / 1000000 / $AVG_TIME_CUDA_SHARED" | bc)

    echo -e "$N\t$AVG_TIME_OMP\t$THROUGHPUT_OMP\t\
    $AVG_TIME_CUDA_GLOBAL\t$THROUGHPUT_CUDA_GLOBAL\t\
    $AVG_TIME_CUDA_SHARED\t$THROUGHPUT_CUDA_SHARED"
done
