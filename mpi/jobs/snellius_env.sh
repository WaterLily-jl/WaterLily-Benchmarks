#!/bin/bash
# Software environment of the MPI benchmarks on Snellius (rome partition: 2 AMD EPYC 7H12, 128 cores per node).
# Source it on the login node before `julia mpi/setup.jl --backend=SIMD`, and the jobs source it again: MPI.jl
# is bound to the libmpi loaded here, so it has to be the same in both. Check the module with `module spider OpenMPI`.
module load 2024
module load OpenMPI/5.0.3-GCC-13.3.0
export PATH="$HOME/.juliaup/bin:$PATH"
export WATERLILY_DIR="${WATERLILY_DIR:-$HOME/WaterLily.jl}"  # WaterLily.jl clone, branch mpi-igg
export UCX_WARN_UNUSED_ENV_VARS=n  # Snellius sets UCX variables that OpenMPI 5 does not use
