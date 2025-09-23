#!/bin/bash

# This script executes the parallel `main_omp` program with an
# increasing number of threads p, from 1 up to the number of logical
# cores. For each run, we use the same input size so that the
# execution times can be used to compute the speedup and the strong
# scaling efficiency. Each run is repeated `NREPS` times; the script
# prints all individual execution times on standard output.
#-----------------------------------------------------------------------
# NOTE: the problem size is determined by two main parameters: `N` 
# (the number of neurons in the input layer) and `K` (the number of
# layers). You may want to change these values according to the
# performance of the text machine. Ideally, the problem size should be
# large enough (at least a few seconds, ideally 10s or more) so that
# the OpenMP and scheduling overheads are minimized.

# Last updated on 2025-09-23
# Alessio Pittiglio

PROG=./src_omp/main_omp # name of the executable
N=1000000               # number of neurons in input layer (adjust as needed)
K=1500                  # number of layers (adjust as needed)
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
NREPS=5                 # number of replications.

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5"

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $N $K | grep "Execution time" | sed 's/Execution time //' )"
        echo -n -e "${EXEC_TIME}\t"
    done
    echo ""
done
