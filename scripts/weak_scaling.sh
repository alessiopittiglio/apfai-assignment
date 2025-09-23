#!/bin/bash

# This script executes the parallel `main_omp` program with an
# increasing number of threads $p$, from 1 up to the number of logical
# cores. For each value of $p$, the program sets the input size so
# that the amount of work executed by each OpenMP thread is kept
# constant. For the `main_omp` program, this means that the input
# size (i.e., the number of neurons in the input layer) is $N0 \times 
# p$, where $N0$ is a user-defined constant and represents the input 
# size when $p=1$.
#
#-----------------------------------------------------------------------
# NOTE: the computation of the problem size above is valid ONLY FOR
# THIS PROGRAM; therefore, the input size for computing the weak
# scaling efficiency should be defined by considering the actual
# computational cost of the algorithm under test. The `main_omp`
# program requires serial time $O(n)$, but other algorithms may have
# a different cost and therefore may require a different formula for
# the input size.

# Last updated on 2025-09-23
# Alessio Pittiglio

PROG=./src_omp/main_omp # name of the executable
N0=1000000              # base problem size; you may want to change this
K=1000                  # number of layers
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
NREPS=5                 # number of replications

if [ ! -f "$PROG" ]; then
    echo
    echo "$PROG not found"
    echo
    exit 1
fi

echo -e "p\tt1\tt2\tt3\tt4\tt5"

for p in `seq $CORES`; do
    echo -n -e "$p\t"
    N=$(( N0 * p ))
    for rep in `seq $NREPS`; do
        EXEC_TIME="$( OMP_NUM_THREADS=$p "$PROG" $N $K | grep "Execution time" | sed 's/Execution time //' )"
        echo -n -e "${EXEC_TIME}\t"
    done
    echo ""
done
