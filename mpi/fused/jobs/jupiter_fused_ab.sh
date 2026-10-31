#!/bin/bash
#SBATCH --job-name=wl-fused-ab
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=01:00:00
#SBATCH --output=wl-fused-ab-%j.out

# WaterLily's own fused halos (WaterLily branch mpi-igg-fused, without ImplicitGlobalGrid) in each setting of
# VARIANTS (a name, then the environment of its runs; default: GPU halos staged through host memory, device to
# device with CUDA-aware MPI, device to device for the halos of at least 1 MiB, staged without the overlap):
#   1. the TGV 512^3, timed, on 4 and 8 GPUs;
#   2. the TGV 256^3 on 8 GPUs with --check, compared with mpi/check.jl: every run must be identical, also to REF
#      (default overlap_ab/check_np8_m1048576.csv, the ImplicitGlobalGrid fork's fused halos, when it exists).
# Each run writes to $OUT (default mpi/fused/data/ab/, not tracked), one file per run. The runs use the environment
# mpi/fused, built once on the login node from a clone of its own, so that the clone of mpi/ keeps its branch:
#
#   source mpi/jobs/jupiter_env.sh
#   git clone -b mpi-igg-fused https://github.com/WaterLily-jl/WaterLily.jl $WATERLILY_DIR-fused  # next to the clone of mpi/
#   WATERLILY_DIR=$WATERLILY_DIR-fused julia mpi/setup.jl --env=mpi/fused --cuda-local
#
# Then submit from the repository root:
#
#   sbatch -A <budget> --reservation=develbooster mpi/fused/jobs/jupiter_fused_ab.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
OUT="${OUT:-mpi/fused/data/ab}"; mkdir -p "$OUT"
REF="${REF:-overlap_ab/check_np8_m1048576.csv}"
VARIANTS="${VARIANTS:-staged:WATERLILY_GPU_AWARE_MPI=0 device:WATERLILY_GPU_AWARE_MPI=1 device1MiB:WATERLILY_GPU_AWARE_MPI=1048576 nooverlap:WATERLILY_GPU_AWARE_MPI=0,WATERLILY_OVERLAP_MIN=1000000000000}"
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi/fused -e 'using Pkg; Pkg.precompile(); import WaterLily as W
    isdefined(W, :_device_selectors) || error("WaterLily in mpi/fused is not on branch mpi-igg-fused")' || exit 1

tgv() { local o=$1 e=$2; shift 2; env ${e//,/ } srun -N $1 -n $2 julia --project=mpi/fused -t 4 mpi/scaling.jl --case=tgv --backend=CuArray "${@:3}" > "$OUT/$o.txt" 2>&1 || echo "[ FAILED: $o ]"; }

checks=()
for v in $VARIANTS; do
    name=${v%%:*} e=${v#*:}
    echo "== $name: ${e//,/ }"
    tgv tgv_np4_$name $e 1 4 --log2p=9 --steps=100 --warmup=20
    tgv tgv_np8_$name $e 2 8 --log2p=9 --steps=100 --warmup=20
    tgv check_np8_$name $e 2 8 --log2p=8 --steps=50 --warmup=0 --check=10 --out="$OUT/check_np8_$name.csv"
    grep -h "Info: MPI np" "$OUT/tgv_np4_$name.txt" "$OUT/tgv_np8_$name.txt"
    [ -f "$OUT/check_np8_$name.csv" ] && checks+=("$OUT/check_np8_$name.csv")
done
[ ${#checks[@]} -gt 1 ] && julia mpi/check.jl "${checks[@]}"
[ ${#checks[@]} -gt 0 ] && [ -f "$REF" ] && julia mpi/check.jl "$REF" "${checks[@]}"
