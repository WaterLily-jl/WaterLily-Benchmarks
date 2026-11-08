#!/bin/bash
#SBATCH --job-name=wl-overlap-ab
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=00:45:00
#SBATCH --output=wl-overlap-ab-%j.out

# The Poisson halos of WaterLily overlapped with their stencil loops (WaterLily branch mpi-igg-overlap: the loop runs
# on the cells that read no ghost while the messages travel), with the fused halo updates of the ImplicitGlobalGrid
# fork (branch halo-fused, IGG_FUSED=2) and the timers on (IGG_TRACE=1). For each WATERLILY_OVERLAP_MIN in MINS (the
# fewest cells a loop needs to be split; default "1048576 0 1000000000000": the default, every multigrid level, none):
#   1. the TGV 512^3, timed, on 4 and 8 GPUs;
#   2. the TGV 256^3 on 8 GPUs with --check, compared with mpi/check.jl: every run must be identical.
# Each run writes to $OUT (default overlap_ab/), one file per run. Submit from the repository root, with WaterLily on
# mpi-igg-overlap and the fork on halo-fused developed into mpi/:
#
#   sbatch -A <budget> --reservation=develbooster mpi/jobs/jupiter_overlap_ab.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
OUT="${OUT:-overlap_ab}"; mkdir -p "$OUT"
MINS="${MINS:-1048576 0 1000000000000}"
export IGG_TRACE=1 IGG_FUSED=2
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile(); import ImplicitGlobalGrid as G, WaterLily as W
    isdefined(G, :update_halo_start!) || error("ImplicitGlobalGrid in mpi/ is not the fork, branch halo-fused")
    isdefined(W, :shell) || error("WaterLily in mpi/ is not on branch mpi-igg-overlap")' || exit 1

tgv() { local o=$1; shift; srun -N $1 -n $2 julia --project=mpi -t 4 mpi/scaling.jl --case=tgv --backend=CuArray "${@:3}" > "$OUT/$o.txt" 2>&1 || echo "[ FAILED: $o ]"; }

for m in $MINS; do
    export WATERLILY_OVERLAP_MIN=$m
    tgv tgv_np4_m$m 1 4 --log2p=9 --steps=100 --warmup=20
    tgv tgv_np8_m$m 2 8 --log2p=9 --steps=100 --warmup=20
    tgv check_np8_m$m 2 8 --log2p=8 --steps=50 --warmup=0 --check=10 --out="$OUT/check_np8_m$m.csv"
    grep -h "Info: MPI np" "$OUT/tgv_np4_m$m.txt" "$OUT/tgv_np8_m$m.txt"
done
set -- $MINS
[ -f "$OUT/check_np8_m$1.csv" ] && julia mpi/check.jl "$OUT/check_np8_m$1.csv" $(shift; for m in "$@"; do echo "$OUT/check_np8_m$m.csv"; done)
