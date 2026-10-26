#!/bin/bash
# Software environment of the MPI benchmarks on the JUPITER Booster (4 GH200 per node, one MPI rank per GPU).
# Source it on the login node before `julia mpi/setup.jl --cuda-local`, and the jobs source it again: MPI.jl is
# bound to the libmpi loaded here, so it has to be the same in both.
#
# Julia comes from juliaup (its installer puts it in PATH), not from the Julia module. The MPI modules show once a
# compiler is loaded (`ml GCC; ml avail`); ParaStationMPI and OpenMPI are both CUDA-aware on JUPITER (ParaStationMPI
# loads MPI-settings/CUDA, which sets PSP_CUDA=1), and WATERLILY_MPI picks one (ParaStationMPI by default; run
# setup.jl again after changing it). Pass your budget to sbatch (-A <budget>).

module purge                    # keeps the (sticky) Stages module; --force would drop it too
module load GCC "${WATERLILY_MPI:-ParaStationMPI}" CUDA
command -v julia >/dev/null || echo "jupiter_env.sh: no julia in PATH, install juliaup" >&2
export UCX_CUDA_COPY_DMABUF=no  # UCX 1.19 has a GPU-memory registration issue on JUPITER
export JULIA_CUDA_MEMORY_POOL=none  # CUDA-aware MPI cannot pass memory of CUDA.jl's pool (MPI.jl known issues)
# Julia raises SIGSEGV on purpose to stop its threads for the GC: neither pscom nor UCX may take it as an error.
# MPI.jl sets UCX_ERROR_SIGNALS too, but only after the MPI library has loaded UCX, which has then read its settings.
export PSP_SIGSEGV=0
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE
export UCX_WARN_UNUSED_ENV_VARS=n  # MPI.jl sets UCX_MEMTYPE_CACHE, which UCX 1.19 no longer has: one warning per rank
# srun does not take --cpus-per-task from the allocation: pass it on, or the Julia threads of a rank share one core
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"; fi

# The Julia depot needs a few GB and compute nodes have no internet: keep it in the project file system
# $PROJECT is set by `jutil env activate -p <project>`; without it the depot and the clone default to $HOME
WL_BASE="${PROJECT:+$PROJECT/$USER}"; WL_BASE="${WL_BASE:-$HOME}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$WL_BASE/julia_depot}"
export WATERLILY_DIR="${WATERLILY_DIR:-$WL_BASE/WaterLily.jl}"  # WaterLily.jl clone, branch mpi-igg
# ImplicitGlobalGrid sends the halos of GPU arrays straight from device memory (1), or stages them on the host (0)
export IGG_CUDAAWARE_MPI="${IGG_CUDAAWARE_MPI:-1}"
