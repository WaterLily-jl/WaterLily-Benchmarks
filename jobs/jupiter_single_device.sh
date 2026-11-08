#!/bin/bash
#SBATCH --job-name=wl-single
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-single-%j.out

# One device of a JUPITER Booster node (a GH200: one Grace CPU of 72 cores and one Hopper GPU) with benchmark.sh and
# WaterLily v1.11 (KernelAbstractions 0.10): the 72 cores of a Grace (CPUx72, KernelAbstractions on PoCL threads) and
# the GPU (CUDA.jl 6.5), for the TGV, the sphere and the moving cylinder at four sizes each, from developed-flow
# checkpoints, 3 repetitions of 5 runs of 25 steps (benchmark.sh's defaults). One Grace core (CPUx01, SIMD) is
# jobs/jupiter_single_device_simd.sh, whose largest sizes take hours. The checkpoints are read from $CKPT (by default
# $SCRATCH_e_ext_2025e01_127/bernat/benchmark_checkpoints), where jobs/jupiter_develop.sh writes the sizes that
# checkpoints/ lacks (tgv 9, sphere 6, cylinder 6 and 7). Submit from the repository root:
#
#   sbatch -A <budget> --dependency=afterok:<develop job> jobs/jupiter_single_device.sh
#   sbatch -A <budget> jobs/jupiter_single_device.sh -b CuArray   # extra benchmark.sh args replace the defaults
#
# Prepare on a login node (compute nodes have no internet), once: a clone of WaterLily at v1.11.0 next to this one,
# and the two environments benchmark.sh runs in (this directory for the CPU, gpu/CUDA for the GPU), resolved,
# downloaded and precompiled, so that the job runs offline:
#
#   cd .. && git clone https://github.com/WaterLily-jl/WaterLily.jl WaterLily.jl-master && cd WaterLily-Benchmarks
#   module purge; export WATERLILY_DIR=$PWD/../WaterLily.jl-master
#   export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$PWD/../julia_depot}   # as mpi/jobs/jupiter_env.sh
#   export JULIAUP_CHANNEL=1.13 && juliaup add 1.13; julia --version   # Julia 1.13, as the job (an error if installed)
#   git -C $WATERLILY_DIR checkout v1.11.0
#   julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["WATERLILY_DIR"]); Pkg.update(); Pkg.precompile()'
#   mkdir -p gpu/CUDA && cp Project.toml gpu/CUDA/ && julia --project=gpu/CUDA -e \
#       'using Pkg; Pkg.develop(path=ENV["WATERLILY_DIR"]); Pkg.add("CUDA"); Pkg.update(); Pkg.precompile()' \
#       && cp Project.toml gpu/CUDA/Project.base.toml
#   julia --project=gpu/CUDA -e 'using CUDA; CUDA.versioninfo()'   # the runtime CUDA.jl picked (the job prints it too)
#
# The cores: the node is the job's (--exclusive), and taskset keeps the runs on the 72 cores of the first Grace (0-71,
# NUMA domain 0, next to GPU 0, the one CUDA.jl takes); POCL_AFFINITY=1 pins each PoCL thread to a core of its own.
# CUDA.jl's memory pool stays on (mpi/jobs/jupiter_env.sh turns it off for CUDA-aware MPI). benchmark.sh and the
# SIMD job write the same LocalPreferences.toml (the WaterLily backend of the next run), so the two must not overlap:
# submit the SIMD job with --dependency=afterany on this one. Time: benchmark.sh reads the checkpoint of a size before
# its warm-up and before each of its 5 runs, so 3 repetitions read about 180 GB per backend (cylinder 7 alone 3.7 GB a
# time), an hour or more per backend from scratch, plus about an hour of 72-thread runs of the largest sizes: several
# hours, so the limit is JUPITER's 12 h (only the time used is charged).
#
# Modules: GCC and git only, not mpi/jobs/jupiter_env.sh. There is no MPI here, and with its CUDA module on
# LD_LIBRARY_PATH the cuBLAS of CUDA.jl's own runtime binds to the module's older libcublasLt.so.13 and does not load
# (undefined symbol cublasLtZZZMatmulAlgoGetHeuristicForStream); the git module's libcurl, older than Julia's, makes
# Pkg's downloads log `curl_easy_setopt: 48`, so the preparation above loads no module at all.
#
# Julia 1.13 (JULIAUP_CHANNEL selects it for every `julia` here and in benchmark.sh, whatever juliaup's default): the
# environments are resolved with it on the login node, and on Julia 1.11 every PoCL launch still allocates.

cd "$SLURM_SUBMIT_DIR"
WL_BASE="${WL_BASE:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd)}"
export WATERLILY_DIR="$WL_BASE/WaterLily.jl-master"  # never the submitting shell's (jupiter_env.sh points it at the
# mpi-igg clone), since benchmark.sh checks out v1.11.0 in it
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$WL_BASE/julia_depot}"  # as mpi/jobs/jupiter_env.sh
module purge; module load GCC git    # git: the compute nodes have none (benchmark.sh checks out v1.11.0)
unset JULIA_CUDA_MEMORY_POOL         # one GPU, no MPI: CUDA.jl's default pool, as anyone running WaterLily has it
export JULIA_PKG_OFFLINE=true        # the environments were prepared on the login node
export POCL_AFFINITY=1               # pin PoCL's threads (KA 0.10)
export JULIAUP_CHANNEL="${JULIAUP_CHANNEL:-1.13}"
[[ $(julia --version) == "julia version 1.13."* ]] || { echo "Julia 1.13 needed, got $(julia --version)" >&2; exit 1; }
CKPT="${CKPT:-${SCRATCH_e_ext_2025e01_127:?}/bernat/benchmark_checkpoints}"
for f in tgv_{6,7,8,9} sphere_{3,4,5,6} cylinder_{4,5,6,7}; do  # benchmark.sh stops the sweep at a missing one
    [ -f "$CKPT/${f}_Float32.jld2" ] || { echo "no checkpoint $CKPT/${f}_Float32.jld2 (jobs/jupiter_develop.sh)" >&2; exit 1; }
done
lscpu | grep -E "^NUMA node[0-9]+ CPU\(s\)" | grep -v "CPU(s):\s*$"; nvidia-smi -L
echo "WaterLily $(git -C "$WATERLILY_DIR" describe --tags) in $WATERLILY_DIR, $(julia --version)"
julia --project=gpu/CUDA -e 'using CUDA; CUDA.versioninfo()'

DEFAULTS=(-b "Array CuArray" -t "72" -w "v1.11.0" -u false -dev "$CKPT" -r 3 -f -c "tgv sphere cylinder"
          -p "6,7,8,9 3,4,5,6 4,5,6,7" -ft Float32 -dd "data/jupiter_single_device_developed")
taskset -c 0-71 ./benchmark.sh "${DEFAULTS[@]}" "$@"
julia --project=. compare.jl --data_dir=data/jupiter_single_device_developed --patterns="tgv sphere cylinder" --sort=1 --markdown
