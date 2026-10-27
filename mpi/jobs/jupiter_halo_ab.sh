#!/bin/bash
#SBATCH --job-name=wl-halo-ab
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=01:15:00
#SBATCH --output=wl-halo-ab-%j.out

# ImplicitGlobalGrid's update_halo! against its fused updates (IGG_FUSED=1: one round of messages per dimension,
# 2: one round for all; fork branch halo-fused), with the timers on (IGG_TRACE=1):
#   1. mpi/halo_trace.jl at the halo sizes of a 512^3 TGV on 4 and 8 GPUs (34^3 and 258^3 blocks, scalar and velocity);
#   2. the TGV 512^3 itself, timed, on 4 and 8 GPUs;
#   3. the TGV 256^3 on 8 GPUs with --check, compared with mpi/check.jl: the fused runs must be identical.
# Then the same with CUDA-aware MPI (device buffers) and IGG_FUSED=2 on one node, where NVLink moves large faces
# about twice as fast as host buffers. Each run writes to $OUT (default trace_ab/), one file per run.
# Submit from the repository root, with WaterLily on 23775aa (its Poisson iterations call update_halo!) and the
# fork on halo-fused developed into mpi/:
#
#   sbatch -A <budget> --reservation=develbooster mpi/jobs/jupiter_halo_ab.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
OUT="${OUT:-trace_ab}"; mkdir -p "$OUT"
export IGG_TRACE=1
# precompile on one rank, so that the first runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile(); import ImplicitGlobalGrid as G
    isdefined(G, :FUSED) || error("ImplicitGlobalGrid in mpi/ is not the fork, branch halo-fused")' || exit 1

trace() { local o=$1; shift; srun -N $1 -n $2 julia --project=mpi mpi/halo_trace.jl "${@:3}" > "$OUT/$o.txt" 2>&1 || echo "[ FAILED: $o ]"; }
tgv() { local o=$1; shift; srun -N $1 -n $2 julia --project=mpi -t 4 mpi/scaling.jl --case=tgv --backend=CuArray "${@:3}" > "$OUT/$o.txt" 2>&1 || echo "[ FAILED: $o ]"; }

for f in 0 1 2; do
    export IGG_FUSED=$f
    trace n4_64_f$f 1 4 64 1 CuArray;     trace n4_512_f$f 1 4 512 1 CuArray
    trace n8_64_f$f 2 8 64 1 CuArray;     trace n8_512_f$f 2 8 512 1 CuArray;  trace n8_512_u_f$f 2 8 512 3 CuArray
    tgv tgv_np4_f$f 1 4 --log2p=9 --steps=100 --warmup=20
    tgv tgv_np8_f$f 2 8 --log2p=9 --steps=100 --warmup=20
    tgv check_np8_f$f 2 8 --log2p=8 --steps=50 --warmup=0 --check=10 --out="$OUT/check_np8_f$f.csv"
    grep -h "Info: MPI np" "$OUT/tgv_np4_f$f.txt" "$OUT/tgv_np8_f$f.txt"
done
julia mpi/check.jl "$OUT/check_np8_f0.csv" "$OUT/check_np8_f1.csv" "$OUT/check_np8_f2.csv"

export IGG_CUDAAWARE_MPI=1 IGG_FUSED=2
trace n4_512_aware_f2 1 4 512 1 CuArray
tgv tgv_np4_aware_f2 1 4 --log2p=9 --steps=100 --warmup=20
grep -h "Info: MPI np" "$OUT/tgv_np4_aware_f2.txt"
