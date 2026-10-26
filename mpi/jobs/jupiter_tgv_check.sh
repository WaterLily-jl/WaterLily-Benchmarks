#!/bin/bash
#SBATCH --job-name=wl-tgv-check
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=wl-tgv-check-%j.out

# Correctness of the multi-GPU TGV: the same run on one GPU without MPI, then on 1, 4 and 8 GPUs (two nodes), with
# CUDA-aware and host-staged halos, and on 8 CPU ranks. Every run saves its global diagnostics (kinetic energy,
# enstrophy, max|u|, max|p|, Δt, Poisson iterations) and mpi/check.jl compares them with the one-GPU run.
# Submit from the repository root:
#
#   sbatch -A <budget> mpi/jobs/jupiter_tgv_check.sh [LOG2P=7] [STEPS=200] [CHECK=10]
#
# Expected (see docs/src/mpi.md): MPI on 1 GPU and the host-staged 8-GPU run are identical to the serial and
# CUDA-aware runs that they repeat. With more ranks, the convection term takes its one-sided boundary form at the
# rank faces (the halo is one cell wide), so the diagnostics differ slightly from the serial run (~1e-4 to 1e-3);
# the 8-GPU and 8-CPU runs use the same decomposition and differ by FP32 round-off.

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
P="${1:-7}"; STEPS="${2:-200}"; CHECK="${3:-10}"
DIR="data/mpi/jupiter_tgv_check_${SLURM_JOB_ID}"; mkdir -p "$DIR"
echo "WaterLily: $(git -C "$WATERLILY_DIR" log -1 --format='%h %s')   WaterLily-Benchmarks: $(git log -1 --format=%h)"
echo "TGV log2p=$P, $STEPS steps, diagnostics every $CHECK steps -> $DIR"

run () { # name, srun options..., then -- and scaling.jl options
    local name=$1; shift; local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    echo; echo "=== $name: srun ${opts[*]} ... $*"
    srun "${opts[@]}" julia --project=mpi -t 4 mpi/scaling.jl --case=tgv --log2p="$P" \
        --steps="$STEPS" --warmup=0 --check="$CHECK" --out="$DIR/$name.csv" "$@" 2>&1 | tee "$DIR/$name.log" \
        | grep -E "Info|WaterLily|MPI:|rank|Warn|ERROR" || echo "[ FAILED: $name ]"
}

run gpu_serial            -N 1 -n 1 -- --backend=CuArray --serial
run gpu_np1               -N 1 -n 1 -- --backend=CuArray
run gpu_np4               -N 1 -n 4 -- --backend=CuArray
run gpu_np8               -N 2 -n 8 -- --backend=CuArray
IGG_CUDAAWARE_MPI=0 run gpu_np8_hoststaged -N 2 -n 8 -- --backend=CuArray
run cpu_np8               -N 2 -n 8 -- --backend=Array

echo; echo "=== Diagnostics against the one-GPU run (largest relative difference over the run)"
julia mpi/check.jl "$DIR/gpu_serial.csv" "$DIR"/gpu_np1.csv "$DIR"/gpu_np4.csv "$DIR"/gpu_np8.csv "$DIR"/gpu_np8_hoststaged.csv "$DIR"/cpu_np8.csv
echo; echo "=== 8 GPUs, host-staged against CUDA-aware halos"
julia mpi/check.jl "$DIR/gpu_np8.csv" "$DIR/gpu_np8_hoststaged.csv"
echo; echo "=== 8 GPUs against 8 CPU ranks (same decomposition)"
julia mpi/check.jl "$DIR/gpu_np8.csv" "$DIR/cpu_np8.csv"
