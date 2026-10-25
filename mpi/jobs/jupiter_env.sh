#!/bin/bash
# Software environment of the MPI benchmarks on the JUPITER Booster (4 GH200 per node, one MPI rank per GPU).
# Source it on the login node before `julia mpi/setup.jl --cuda-local`, and the jobs source it again: MPI.jl is
# bound to the libmpi loaded here, so it has to be the same in both.
#
# Module names follow the JSC documentation (https://apps.fz-juelich.de/jsc/hps/jupiter/), where ParaStationMPI
# and OpenMPI are both CUDA-aware. Check them before the first run, for instance with
#     module avail; module spider ParaStationMPI; ml show MPI-settings
# and pass your budget to sbatch (-A <budget>).

module purge                    # keeps the (sticky) Stages module; --force would drop it too
module load GCC ParaStationMPI
module load MPI-settings/CUDA   # CUDA-aware settings of the MPI module
module load CUDA
module load Julia               # WaterLily mpi-igg needs Julia >= 1.11; juliaup works too (put it first in PATH)

# The Julia depot needs a few GB and compute nodes have no internet: keep it in the project file system
# $PROJECT is set by `jutil env activate -p <project>`; without it the depot and the clone default to $HOME
WL_BASE="${PROJECT:+$PROJECT/$USER}"; WL_BASE="${WL_BASE:-$HOME}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$WL_BASE/julia_depot}"
export WATERLILY_DIR="${WATERLILY_DIR:-$WL_BASE/WaterLily.jl}"  # WaterLily.jl clone, branch mpi-igg
# ImplicitGlobalGrid sends the halos of GPU arrays straight from device memory (1), or stages them on the host (0)
export IGG_CUDAAWARE_MPI="${IGG_CUDAAWARE_MPI:-1}"
