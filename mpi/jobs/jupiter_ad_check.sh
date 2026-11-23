#!/bin/bash
#SBATCH --job-name=wl-ad-check
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=00:40:00
#SBATCH --output=wl-ad-check-%j.out

# Correctness of forward-mode AD through MPI on GPUs: the TGV with one partial (d/dν, mpi/scaling.jl --partials=1)
# from its initial condition over a short, laminar horizon, the same global grid in every run (strong mode). The
# functional is the kinetic energy (and the enstrophy) after a fixed number of time steps, whose sizes depend on ν.
# Two comparisons, not to be mixed:
#   1. AD on 1 (with MPI), 4 and 8 GPUs against AD on one GPU without MPI. With more than one rank this also holds
#      the effect of the decomposition on the flow itself (the convection stencil is one-sided at the rank faces),
#      which the value columns show next to the derivatives. Device-to-device halos must be identical to staged ones,
#      and the values of the Dual run equal those of a Float32 run within round-off.
#   2. AD against central finite differences on the same 8 GPUs, in Float64 (ν(1±EPS); in Float32 the round-off of
#      the difference hides the derivative): this tests the AD itself.
# Submit from the repository root (it fits in the develbooster reservation):
#
#   sbatch -A <budget> --reservation=develbooster mpi/jobs/jupiter_ad_check.sh [LOG2P=7] [STEPS=100] [CHECK=10] [EPS=1e-4]

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
P="${1:-7}"; STEPS="${2:-100}"; CHECK="${3:-10}"; EPS="${4:-1e-4}"; ENV_DIR="${WATERLILY_ENV:-mpi}"
DIR="data/mpi/jupiter_ad_check_${SLURM_JOB_ID}"; mkdir -p "$DIR"
RE_PLUS=$(awk -v e="$EPS" 'BEGIN { printf "%.17g", 1600/(1 + e) }'); RE_MINUS=$(awk -v e="$EPS" 'BEGIN { printf "%.17g", 1600/(1 - e) }')
echo "TGV log2p=$P, $STEPS steps, diagnostics every $CHECK steps, finite differences at Re=$RE_PLUS and $RE_MINUS -> $DIR"
srun -N 1 -n 1 julia --project="$ENV_DIR" -e 'using Pkg; Pkg.precompile(); using CUDA, MPI, WaterLily' || exit 1

run () { # name, srun options..., then -- and scaling.jl options
    local name=$1; shift; local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    echo; echo "=== $name: srun ${opts[*]} ... $*"
    { echo "# run=$name job=$SLURM_JOB_ID"
      srun "${opts[@]}" julia --project="$ENV_DIR" -t 4 mpi/scaling.jl --case=tgv --log2p="$P" \
          --steps="$STEPS" --warmup=0 --check="$CHECK" --out="$DIR/$name.csv" "$@"; } > "$DIR/$name.txt" 2>&1
    grep -E "Info: (SERIAL|MPI np)|^WaterLily|^MPI:|Warn|ERROR|[Ee]rror|srun:|signal|[Kk]illed" "$DIR/$name.txt" || echo "[ FAILED: $name ]"
}
GPU="--backend=CuArray"
run ad_gpu_serial     -N 1 -n 1 -- $GPU --partials=1 --serial
run ad_gpu_np1        -N 1 -n 1 -- $GPU --partials=1
run ad_gpu_np4        -N 1 -n 4 -- $GPU --partials=1
WATERLILY_GPU_AWARE_MPI=0 run ad_gpu_np8        -N 2 -n 8 -- $GPU --partials=1
WATERLILY_GPU_AWARE_MPI=1 run ad_gpu_np8_device -N 2 -n 8 -- $GPU --partials=1
run f32_gpu_np8       -N 2 -n 8 -- $GPU
run ad64_gpu_np8      -N 2 -n 8 -- $GPU --T=Float64 --partials=1
run fd64_plus_gpu_np8  -N 2 -n 8 -- $GPU --T=Float64 --Re="$RE_PLUS"
run fd64_minus_gpu_np8 -N 2 -n 8 -- $GPU --T=Float64 --Re="$RE_MINUS"
run ad_cpu_np8        -N 2 -n 8 -- --backend=Array --partials=1

echo; echo "=== 1. AD on GPU ranks against AD on one GPU without MPI (relative differences; dKE_dnu is the derivative)"
julia mpi/check.jl "$DIR"/ad_gpu_serial.csv "$DIR"/ad_gpu_np1.csv "$DIR"/ad_gpu_np4.csv "$DIR"/ad_gpu_np8.csv
echo; echo "=== 8 GPUs: device-to-device halos, the Float32 run (values only) and 8 CPU ranks against the staged AD run"
julia mpi/check.jl "$DIR"/ad_gpu_np8.csv "$DIR"/ad_gpu_np8_device.csv "$DIR"/f32_gpu_np8.csv "$DIR"/ad_cpu_np8.csv
echo; echo "=== 2. AD against central finite differences on 8 GPUs, Float64"
julia mpi/check.jl --fd "$DIR"/ad64_gpu_np8.csv "$DIR"/fd64_plus_gpu_np8.csv "$DIR"/fd64_minus_gpu_np8.csv
