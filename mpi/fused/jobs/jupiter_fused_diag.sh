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

# Why WaterLily's own fused halos (branch mpi-igg-fused, mpi/fused) lose the inter-node gain of the ImplicitGlobalGrid
# fork's (IGG_FUSED=2, mpi/): TGV 512^3 on 8 GPUs of 2 nodes, all in one allocation, timed:
#   1. the fork (WaterLily mpi-igg-overlap in mpi/), with its timers (IGG_TRACE=1);
#   2. mpi-igg-fused as it is: host buffers from cuMemHostAlloc;
#   3. mpi-igg-fused with the fork's host buffers (a Julia array registered with cuMemHostRegister, mpi/fused/halo_diag.jl);
#   4. and 5. as 2 and 3, with the time of each phase of each halo (DIAG_LAPS=1);
#   6. the fork again, without its timers.
# Each run writes to $OUT (default mpi/fused/data/diag/, not tracked). Submit from the repository root, with mpi/ and
# mpi/fused built as for mpi/jobs/jupiter_overlap_ab.sh and mpi/fused/jobs/jupiter_fused_ab.sh:
#
#   sbatch -A <budget> --reservation=develbooster mpi/fused/jobs/jupiter_fused_diag.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
OUT="${OUT:-mpi/fused/data/diag}"; mkdir -p "$OUT"
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile(); import ImplicitGlobalGrid as G, WaterLily as W
    isdefined(G, :update_halo_start!) || error("ImplicitGlobalGrid in mpi/ is not the fork, branch halo-fused")
    isdefined(W, :shell) || error("WaterLily in mpi/ is not on branch mpi-igg-overlap")' || exit 1
srun -N 1 -n 1 julia --project=mpi/fused -e 'using Pkg; Pkg.precompile(); import WaterLily as W
    isdefined(W, :_device_selectors) || error("WaterLily in mpi/fused is not on branch mpi-igg-fused")' || exit 1

ARGS_TGV="--case=tgv --backend=CuArray --log2p=9 --steps=100 --warmup=20"
# run NAME ENVIRONMENT VARIABLES SCRIPT: one timed run, with the comma-separated VARIABLES set
run() { env ${3//,/ } srun -N 2 -n 8 julia --project=$2 -t 4 $4 $ARGS_TGV > "$OUT/$1.txt" 2>&1 || echo "[ FAILED: $1 ]"; }
run fork_trace    mpi       IGG_FUSED=2,IGG_TRACE=1                                     mpi/scaling.jl
run alloc         mpi/fused WATERLILY_GPU_AWARE_MPI=0                                   mpi/scaling.jl
run register      mpi/fused WATERLILY_GPU_AWARE_MPI=0,DIAG_BUFFERS=register             mpi/fused/halo_diag.jl
run alloc_laps    mpi/fused WATERLILY_GPU_AWARE_MPI=0,DIAG_LAPS=1                       mpi/fused/halo_diag.jl
run register_laps mpi/fused WATERLILY_GPU_AWARE_MPI=0,DIAG_BUFFERS=register,DIAG_LAPS=1 mpi/fused/halo_diag.jl
run fork          mpi       IGG_FUSED=2,IGG_TRACE=0                                     mpi/scaling.jl
for o in fork_trace alloc register alloc_laps register_laps fork; do
    printf "%-14s %s\n" $o "$(grep -h "Info: MPI np" "$OUT/$o.txt" | sed 's/.*per_step=//; s/ .*//')"
done
