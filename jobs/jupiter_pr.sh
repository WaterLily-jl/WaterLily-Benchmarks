#!/bin/bash
#SBATCH --job-name=wl-pr
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=slurm-pr-%j.out

# A WaterLily pull request against main on one device of a JUPITER Booster node (a GH200), with benchmark.sh and its
# default cases (tgv 6,7 and jelly-biotsavart 5,6, 25 steps, Float32, from the developed-flow checkpoints of
# checkpoints/): one Grace core (CPUx01, SIMD), 4 and 8 PoCL threads (KernelAbstractions) and the GPU (CUDA.jl), 3
# repetitions of the whole sweep, the order of the two versions reversed on the second. The defaults are main at
# 9d7c8ca and WaterLily-jl/WaterLily.jl#408 (bf/dual-always-inline) at 63f776b. Submit from the repository root:
#
#   sbatch jobs/jupiter_pr.sh
#   WL_VERSIONS="<main commit> <PR commit>" DATA=data/jupiter_pr<n> sbatch jobs/jupiter_pr.sh   # another PR
#   sbatch jobs/jupiter_pr.sh -c "tgv sphere cylinder"   # extra benchmark.sh arguments replace the defaults
#
# Commits, not branch names: the result files carry the short hash of the checkout, and a hash cannot move.
#
# Prepare on a login node (compute nodes have no internet), from the repository root: both commits in the WaterLily
# clone benchmark.sh checks out, and the two environments it runs in (this directory for the CPU, gpu/CUDA for the
# GPU) resolved, downloaded and precompiled for main, whose dependencies the PR shares. This replaces the environments
# resolved for another WaterLily (v1.11.0 lacks AcceleratedKernels, so main does not load in its CPU environment):
#
#   module purge; export WATERLILY_DIR=$PWD/../WaterLily.jl-master
#   export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$PWD/../julia_depot}   # as mpi/jobs/jupiter_env.sh
#   export JULIAUP_CHANNEL=1.13; julia --version                      # Julia 1.13, as the job
#   git -C $WATERLILY_DIR fetch origin && git -C $WATERLILY_DIR checkout --detach 9d7c8ca
#   julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["WATERLILY_DIR"]); Pkg.update(); Pkg.precompile()'
#   mkdir -p gpu/CUDA && cp Project.toml gpu/CUDA/ && julia --project=gpu/CUDA -e \
#       'using Pkg; Pkg.develop(path=ENV["WATERLILY_DIR"]); Pkg.add("CUDA"); Pkg.update(); Pkg.precompile()' \
#       && cp Project.toml gpu/CUDA/Project.base.toml
#
# The job stops before the first run if a commit is missing from the clone or WaterLily does not load from it in
# either environment, and prints the packages it runs with. benchmark.sh checks the versions out in WaterLily.jl-master
# and writes LocalPreferences.toml (the WaterLily backend of the next run), so no other benchmark.sh job may run at
# the same time; the clone is left at the last version it ran. Cores, modules and Julia as jobs/jupiter_single_device.sh:
# the first Grace (cores 0-71, next to GPU 0), each PoCL thread pinned to a core, GCC and git only, Julia 1.13.
# Time: 24 processes (4 backends, 2 versions, 3 repetitions), each a few minutes of compilation and runs.

cd "$SLURM_SUBMIT_DIR"
WL_BASE="${WL_BASE:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd)}"
export WATERLILY_DIR="$WL_BASE/WaterLily.jl-master"  # the clone the environments are resolved on
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$WL_BASE/julia_depot}"  # as mpi/jobs/jupiter_env.sh
module purge; module load GCC git    # git: the compute nodes have none (benchmark.sh checks the versions out)
unset JULIA_CUDA_MEMORY_POOL         # one GPU, no MPI: CUDA.jl's default pool
export JULIA_PKG_OFFLINE=true POCL_AFFINITY=1 JULIAUP_CHANNEL="${JULIAUP_CHANNEL:-1.13}"
[[ $(julia --version) == "julia version 1.13."* ]] || { echo "Julia 1.13 needed, got $(julia --version)" >&2; exit 1; }
WL_VERSIONS="${WL_VERSIONS:-9d7c8ca 63f776b}"  # main, then the PR
DATA="${DATA:-data/jupiter_pr408}"
for v in $WL_VERSIONS; do
    git -C "$WATERLILY_DIR" cat-file -e "$v^{commit}" 2>/dev/null ||
        { echo "no commit $v in $WATERLILY_DIR: git fetch origin there, on a login node" >&2; exit 1; }
done
git -C "$WATERLILY_DIR" checkout -q --detach ${WL_VERSIONS%% *} || exit 1
SHOW='using Pkg, WaterLily, BiotSavartBCs; println(pathof(WaterLily)); Pkg.status()'
julia --project=. -e "$SHOW" && julia --project=gpu/CUDA -e "using CUDA; $SHOW; CUDA.versioninfo()" ||
    { echo "WaterLily ${WL_VERSIONS%% *} does not load: prepare the environments on a login node (see the header)" >&2; exit 1; }
lscpu | grep -E "^NUMA node[0-9]+ CPU\(s\)" | grep -v "CPU(s):\s*$"; nvidia-smi -L
echo "WaterLily $WL_VERSIONS in $WATERLILY_DIR, $(julia --version), results in $DATA"

taskset -c 0-71 ./benchmark.sh -b "Array CuArray" -t "1 4 8" -w "$WL_VERSIONS" -u false -r 3 -f -dd "$DATA" "$@" || exit 1
julia --project=. compare.jl --data_dir="$DATA" --sort=1 --markdown
