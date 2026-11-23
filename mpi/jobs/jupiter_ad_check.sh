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
# Three comparisons, not to be mixed:
#   1. AD on 8 GPUs against AD on one GPU without MPI, in Float64 with a converged pressure solve (--tol=TOL). What
#      is left is the effect of the decomposition on the flow (the convection stencil is one-sided at the rank
#      faces), which the value columns show next to the derivatives.
#   2. AD against central finite differences on the same 8 GPUs, Float64, converged pressure solve, ν(1±ε) for each
#      ε of EPS: this tests the AD itself. In Float32 the round-off of the difference hides the derivative, and
#      without a converged solve the solver's stopping test and adaptive relaxation make the result jump with ν.
#      A large ε crosses kinks of the scheme (the time step is a maximum over the cells, the QUICK limiter a median,
#      the fluxes at the rank faces switch with the sign of the velocity), so the difference falls as ε does.
#   3. The settings of the timed runs (Float32, WaterLily's pressure tolerance): AD on 1 (with MPI), 4 and 8 GPUs
#      against one GPU without MPI. MPI on one GPU must be identical, device-to-device halos identical to staged
#      ones, and the values of the Dual run equal to a Float32 run within round-off. The derivatives agree less
#      than the values here: the solver stops on the residual of the values, not of the partials, so the derivative
#      holds the solver's tolerance at the percent level, with or without MPI (compare with 1).
# The runs write to data/mpi/jupiter_adcheck_<job>/ (their diagnostics as .csv, the tables below as summary.txt).
# Submit from the repository root (it fits in the develbooster reservation):
#
#   sbatch -A <budget> --reservation=develbooster mpi/jobs/jupiter_ad_check.sh [LOG2P=7] [STEPS=100] [CHECK=10]
#
# Variables: EPS ["1e-5 1e-6 1e-7"], TOL [1e-9], RUN_MINUTES [10] (the time limit of each run), WATERLILY_ENV [mpi].

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
P="${1:-7}"; STEPS="${2:-100}"; CHECK="${3:-10}"; EPS="${EPS:-1e-5 1e-6 1e-7}"; TOL="${TOL:-1e-9}"; ENV_DIR="${WATERLILY_ENV:-mpi}"
DIR="data/mpi/jupiter_adcheck_${SLURM_JOB_ID}"; mkdir -p "$DIR"  # not jupiter_ad_*: these runs are no timings
re () { awk -v e="$1" -v s="$2" 'BEGIN { printf "%.17g", 1600/(1 + s*e) }'; }  # the Reynolds number of ν(1 + s ε)
echo "TGV log2p=$P, $STEPS steps, diagnostics every $CHECK steps, finite differences at ε = $EPS, converged solves at tol=$TOL -> $DIR"
srun -N 1 -n 1 julia --project="$ENV_DIR" -e 'using Pkg; Pkg.precompile(); using CUDA, MPI, WaterLily' || exit 1

run () { # name, srun options..., then -- and scaling.jl options
    local name=$1; shift; local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    echo; echo "=== $name: srun ${opts[*]} ... $*"
    { echo "# run=$name job=$SLURM_JOB_ID"
      srun --time="${RUN_MINUTES:-10}" "${opts[@]}" julia --project="$ENV_DIR" -t 4 mpi/scaling.jl --case=tgv --log2p="$P" \
          --steps="$STEPS" --warmup=0 --check="$CHECK" --out="$DIR/$name.csv" "$@"; } > "$DIR/$name.txt" 2>&1
    grep -E "Info: (SERIAL|MPI np)|^WaterLily|^MPI:|Warn|ERROR|[Ee]rror|srun:|signal|[Kk]illed" "$DIR/$name.txt" || echo "[ FAILED: $name ]"
}
GPU="--backend=CuArray"; F64="--T=Float64 --tol=$TOL"
run ad_gpu_serial     -N 1 -n 1 -- $GPU --partials=1 --serial
run ad_gpu_np1        -N 1 -n 1 -- $GPU --partials=1
run ad_gpu_np4        -N 1 -n 4 -- $GPU --partials=1
WATERLILY_GPU_AWARE_MPI=0 run ad_gpu_np8        -N 2 -n 8 -- $GPU --partials=1
run ad64_gpu_serial   -N 1 -n 1 -- $GPU $F64 --partials=1 --serial
run ad64_gpu_np8      -N 2 -n 8 -- $GPU $F64 --partials=1
for e in $EPS; do
    run "fd64_plus_${e}_gpu_np8"  -N 2 -n 8 -- $GPU $F64 --Re="$(re "$e" 1)"
    run "fd64_minus_${e}_gpu_np8" -N 2 -n 8 -- $GPU $F64 --Re="$(re "$e" -1)"
done
run f32_gpu_np8       -N 2 -n 8 -- $GPU
run ad_cpu_np8        -N 2 -n 8 -- --backend=Array --partials=1
WATERLILY_GPU_AWARE_MPI=1 run ad_gpu_np8_device -N 2 -n 8 -- $GPU --partials=1  # last: no Dual halo has gone device to device here

{ echo; echo "=== 1. Float64, converged pressure solve: AD on 8 GPUs against AD on one GPU without MPI (dKE_dnu is the derivative)"
  julia mpi/check.jl "$DIR"/ad64_gpu_serial.csv "$DIR"/ad64_gpu_np8.csv
  for e in $EPS; do
      echo; echo "=== 2. Float64, converged pressure solve: AD against central finite differences on 8 GPUs, ε = $e"
      julia mpi/check.jl --fd "$DIR"/ad64_gpu_np8.csv "$DIR/fd64_plus_${e}_gpu_np8.csv" "$DIR/fd64_minus_${e}_gpu_np8.csv"
  done
  echo; echo "=== 3. Float32, WaterLily's pressure tolerance (the timed runs): AD on GPU ranks against AD on one GPU without MPI"
  julia mpi/check.jl "$DIR"/ad_gpu_serial.csv "$DIR"/ad_gpu_np1.csv "$DIR"/ad_gpu_np4.csv "$DIR"/ad_gpu_np8.csv
  echo; echo "=== 8 GPUs: device-to-device halos, the Float32 run (values only) and 8 CPU ranks against the staged AD run"
  julia mpi/check.jl "$DIR"/ad_gpu_np8.csv "$DIR"/ad_gpu_np8_device.csv "$DIR"/f32_gpu_np8.csv "$DIR"/ad_cpu_np8.csv
} 2>&1 | tee "$DIR/summary.txt"
