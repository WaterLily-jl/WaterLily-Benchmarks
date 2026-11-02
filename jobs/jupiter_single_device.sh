#!/bin/bash
#SBATCH --job-name=wl-single
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=slurm-single-%j.out

# One device of a JUPITER Booster node (a GH200: one Grace CPU of 72 cores and one Hopper GPU) with benchmark.sh and
# WaterLily v1.11 (KernelAbstractions 0.10): one Grace core (CPUx01, the SIMD backend), the 72 cores of that Grace
# (CPUx72, KernelAbstractions on PoCL threads) and the GPU (CUDA.jl 6.5), for the TGV, the sphere and the moving
# cylinder at four sizes each, 100 steps of the transient (the largest sizes have no developed-flow checkpoints).
# Submit from the repository root:
#
#   sbatch -A <budget> jobs/jupiter_single_device.sh
#   sbatch -A <budget> jobs/jupiter_single_device.sh -t "72" -b CuArray   # extra benchmark.sh args replace the defaults
#
# Prepare on a login node (compute nodes have no internet), once: a clone of WaterLily at v1.11.0 next to this one,
# and the two environments benchmark.sh runs in (this directory for the CPU, gpu/CUDA for the GPU), resolved,
# downloaded and precompiled, so that the job runs offline:
#
#   cd .. && git clone https://github.com/WaterLily-jl/WaterLily.jl WaterLily.jl-master && cd WaterLily-Benchmarks
#   export WATERLILY_DIR=$PWD/../WaterLily.jl-master && source mpi/jobs/jupiter_env.sh
#   git -C $WATERLILY_DIR checkout v1.11.0
#   julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["WATERLILY_DIR"]); Pkg.update(); Pkg.precompile()'
#   mkdir -p gpu/CUDA && cp Project.toml gpu/CUDA/ && julia --project=gpu/CUDA -e \
#       'using Pkg; Pkg.develop(path=ENV["WATERLILY_DIR"]); Pkg.add("CUDA"); Pkg.update(); Pkg.precompile()' \
#       && cp Project.toml gpu/CUDA/Project.base.toml
#   julia --project=gpu/CUDA -e 'using CUDA; CUDA.versioninfo()'   # the runtime CUDA.jl picked (the job prints it too)
#
# The cores: the node is the job's (--exclusive), and taskset keeps the runs on the 72 cores of the first Grace (0-71,
# NUMA domain 0, next to GPU 0, the one CUDA.jl takes); POCL_AFFINITY=1 pins each PoCL thread to a core of its own.
# CUDA.jl's memory pool stays on (jupiter_env.sh turns it off for CUDA-aware MPI). Time: about 2 h of the SIMD core
# for the largest sizes, the rest minutes.

cd "$SLURM_SUBMIT_DIR"
WL_BASE="${WL_BASE:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd)}"
export WATERLILY_DIR="${WATERLILY_DIR:-$WL_BASE/WaterLily.jl-master}"
source mpi/jobs/jupiter_env.sh       # modules, the depot next to this clone
unset JULIA_CUDA_MEMORY_POOL         # one GPU, no MPI: CUDA.jl's default pool, as anyone running WaterLily has it
export JULIA_PKG_OFFLINE=true        # the environments were prepared on the login node
export POCL_AFFINITY=1               # pin PoCL's threads (KA 0.10)
lscpu | grep -E "^NUMA node[0-9]+ CPU\(s\)"; taskset -cp $$; nvidia-smi -L
echo "WaterLily $(git -C "$WATERLILY_DIR" describe --tags) in $WATERLILY_DIR, $(julia --version)"
julia --project=gpu/CUDA -e 'using CUDA; CUDA.versioninfo()'

DEFAULTS=(-b "Array CuArray" -t "1 72" -w "v1.11.0" -u false -dev "" -f
          -c "tgv sphere cylinder" -p "6,7,8,9 3,4,5,6 4,5,6,7" -s 100 -ft Float32 -dd "data/jupiter_single_device")
taskset -c 0-71 ./benchmark.sh "${DEFAULTS[@]}" "$@"
julia --project=. compare.jl --data_dir=data/jupiter_single_device --patterns="tgv sphere cylinder" --speedup_base=CPUx01 --sort=1 --markdown
