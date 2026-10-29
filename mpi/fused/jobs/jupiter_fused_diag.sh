#!/bin/bash
#SBATCH --job-name=wl-fused-diag
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=00:45:00
#SBATCH --output=wl-fused-diag-%j.out

# WaterLily's own fused halos (branch mpi-igg-fused, mpi/fused) against the ImplicitGlobalGrid fork's (IGG_FUSED=2,
# mpi/), all in one allocation: TGV 512^3, timed, on 8 GPUs of 2 nodes and on the 4 GPUs of one node:
#   1. the fork (WaterLily mpi-igg-overlap in mpi/), with its timers (IGG_TRACE=1), on 8 GPUs;
#   2. mpi-igg-fused, halos staged through host memory, on 8 and 4 GPUs;
#   3. mpi-igg-fused without the overlap (WATERLILY_OVERLAP_MIN beyond any loop), and device to device
#      (WATERLILY_GPU_AWARE_MPI=1), on 8 GPUs;
#   4. mpi-igg-fused with the time of each phase of each halo (mpi/fused/halo_diag.jl, DIAG_LAPS=1), on 8 GPUs;
#   5. the fork again, without its timers, on 8 and 4 GPUs.
# The first run of this job (2574387, data in mpi/fused/data/diag/, before WaterLily f38c69a) found the branch at 47.0 ms on 8 GPUs against the
# fork's 33.7 and 32.5, the same with the fork's host buffers (DIAG_BUFFERS=register, 47.2), and its host waiting
# for the pack and unpack kernels: they searched the halo regions per element, 3 ms for a 258^3 halo of 26 regions.
# Each run writes to $OUT (default mpi/fused/data/diag_<job id>/, not tracked). Submit from the repository root, with mpi/ and
# mpi/fused built as for mpi/jobs/jupiter_overlap_ab.sh and mpi/fused/jobs/jupiter_fused_ab.sh:
#
#   sbatch -A <budget> --reservation=develbooster mpi/fused/jobs/jupiter_fused_diag.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
OUT="${OUT:-mpi/fused/data/diag_$SLURM_JOB_ID}"; mkdir -p "$OUT"
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile(); import ImplicitGlobalGrid as G, WaterLily as W
    isdefined(G, :update_halo_start!) || error("ImplicitGlobalGrid in mpi/ is not the fork, branch halo-fused")
    isdefined(W, :shell) || error("WaterLily in mpi/ is not on branch mpi-igg-overlap")' || exit 1
srun -N 1 -n 1 julia --project=mpi/fused -e 'using Pkg; Pkg.precompile(); import WaterLily as W
    isdefined(W, :_device_selectors) || error("WaterLily in mpi/fused is not on branch mpi-igg-fused")' || exit 1

ARGS_TGV="--case=tgv --backend=CuArray --log2p=9 --steps=100 --warmup=20"
# run NAME NODES ENVIRONMENT VARIABLES SCRIPT: one timed run on the 4 GPUs of each node, the comma-separated VARIABLES set
run() { env ${4//,/ } srun -N $2 -n $((4*$2)) julia --project=$3 -t 4 $5 $ARGS_TGV > "$OUT/$1.txt" 2>&1 || echo "[ FAILED: $1 ]"; }
run fork_trace       2 mpi       IGG_FUSED=2,IGG_TRACE=1                                     mpi/scaling.jl
run fused            2 mpi/fused WATERLILY_GPU_AWARE_MPI=0                                   mpi/scaling.jl
run fused_nooverlap  2 mpi/fused WATERLILY_GPU_AWARE_MPI=0,WATERLILY_OVERLAP_MIN=1000000000000 mpi/scaling.jl
run fused_device     2 mpi/fused WATERLILY_GPU_AWARE_MPI=1                                   mpi/scaling.jl
run fused_laps       2 mpi/fused WATERLILY_GPU_AWARE_MPI=0,DIAG_LAPS=1                       mpi/fused/halo_diag.jl
run fork             2 mpi       IGG_FUSED=2,IGG_TRACE=0                                     mpi/scaling.jl
run fork_np4         1 mpi       IGG_FUSED=2,IGG_TRACE=0                                     mpi/scaling.jl
run fused_np4        1 mpi/fused WATERLILY_GPU_AWARE_MPI=0                                   mpi/scaling.jl
for o in fork_trace fused fused_nooverlap fused_device fused_laps fork fork_np4 fused_np4; do
    printf "%-16s %s\n" $o "$(grep -h "Info: MPI np" "$OUT/$o.txt" | sed 's/.*Info: MPI //; s/  case.*per_step=/ /; s/ levels.*//')"
done
