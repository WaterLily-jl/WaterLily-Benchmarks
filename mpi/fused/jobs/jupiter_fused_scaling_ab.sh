#!/bin/bash
#SBATCH --job-name=wl-fused-scaling-ab
#SBATCH --partition=booster
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=01:00:00
#SBATCH --output=wl-fused-scaling-ab-%j.out

# The points of the scaling curves on 32 and 64 nodes (128 and 256 GPUs) of every halo variant, all in one
# allocation: at that size two allocations can differ by ~15 % (the IGG fork's weak 256 GPUs took 45.45 ms in one,
# 39.62 in another), more than some of the differences between the variants. The variants (VARIANTS, all by default):
#   stock  ImplicitGlobalGrid's update_halo! (mpi/, IGG_FUSED=0: the fork does not fuse)
#   fork   the ImplicitGlobalGrid fork's fused halos with the overlap (mpi/, IGG_FUSED=2)
#   fused  WaterLily's own halos, staged through host memory, with the overlap (mpi/fused, branch mpi-igg-fused)
# The points (TGV, 100 timed steps after 20): weak 256^3 cells per GPU on 128, 256 and again 128 GPUs (the repeat
# shows the spread within the allocation), strong 512^3 on 128 GPUs, strong 1024^3 on 128 and 256 GPUs, each point
# with every variant in turn. The curves on fewer nodes are those of mpi/jobs/submit.sh (data/mpi/jupiter_*_fused/,
# and the stock ones with WATERLILY_TAG=stock). Each run writes to $OUT (default mpi/fused/data/scaling_ab_<job id>/,
# not tracked); the job ends with a table of the times. Submit from the repository root, with mpi/ (WaterLily
# mpi-igg-overlap, the fork in branch halo-fused) and mpi/fused built as for mpi/fused/jobs/jupiter_fused_diag.sh:
#
#   sbatch -A <budget> mpi/fused/jobs/jupiter_fused_scaling_ab.sh
#   VARIANTS="stock fused" sbatch -A <budget> mpi/fused/jobs/jupiter_fused_scaling_ab.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
VARIANTS="${VARIANTS:-stock fork fused}"
OUT="${OUT:-mpi/fused/data/scaling_ab_$SLURM_JOB_ID}"; mkdir -p "$OUT"
echo "variants: $VARIANTS"
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile(); import ImplicitGlobalGrid as G, WaterLily as W
    isdefined(G, :update_halo_start!) || error("ImplicitGlobalGrid in mpi/ is not the fork, branch halo-fused")
    isdefined(W, :shell) || error("WaterLily in mpi/ is not on branch mpi-igg-overlap")' || exit 1
srun -N 1 -n 1 julia --project=mpi/fused -e 'using Pkg; Pkg.precompile(); import WaterLily as W
    isdefined(W, :_device_selectors) || error("WaterLily in mpi/fused is not on branch mpi-igg-fused")' || exit 1

# run VARIANT NAME NP ARGS...: one timed run of mpi/scaling.jl on NP GPUs, 4 per node
run() {
    local v=$1 name=$2 np=$3; shift 3
    local proj=mpi vars="IGG_FUSED=0"
    case $v in fork) vars="IGG_FUSED=2" ;; fused) proj=mpi/fused vars="WATERLILY_GPU_AWARE_MPI=0" ;; esac
    env $vars srun -N $(( (np + 3) / 4 )) -n $np julia --project=$proj -t 4 mpi/scaling.jl --case=tgv --backend=CuArray \
        --steps=100 --warmup=20 "$@" > "$OUT/${name}_$v.txt" 2>&1 || echo "[ FAILED: ${name}_$v ]"
}
point() { local name=$1; shift; for v in $VARIANTS; do run $v $name "$@"; done; }
point weak_np128    128 --log2p=8 --weak
point weak_np256    256 --log2p=8 --weak
point weak_np128b   128 --log2p=8 --weak
point strong9_np128 128 --log2p=9
point strong10_np128 128 --log2p=10
point strong10_np256 256 --log2p=10

printf "%-16s" ms/step; for v in $VARIANTS; do printf "%10s" $v; done; echo
for name in weak_np128 weak_np256 weak_np128b strong9_np128 strong10_np128 strong10_np256; do
    printf "%-16s" $name
    for v in $VARIANTS; do
        t=$(grep -ho 'per_step=[0-9.]*' "$OUT/${name}_$v.txt" 2>/dev/null | cut -d= -f2)
        printf "%10s" "${t:--}"
    done
    echo
done
