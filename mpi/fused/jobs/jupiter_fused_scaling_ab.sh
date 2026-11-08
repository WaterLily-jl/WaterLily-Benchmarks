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
# 39.62 in another), more than some of the differences between the variants. The variants (VARIANTS, by default
# stock fork fused):
#   stock   ImplicitGlobalGrid's update_halo! (mpi/, IGG_FUSED=0: the fork does not fuse)
#   fork    the ImplicitGlobalGrid fork's fused halos with the overlap (mpi/, IGG_FUSED=2)
#   fused*  WaterLily's own halos, staged through host memory, with the overlap (branch mpi-igg-fused): the
#           environment mpi/<variant>, so fused is mpi/fused and fused-f38c69a mpi/fused-f38c69a (as built for
#           mpi/fused/jobs/jupiter_fused_mg_ab.sh)
# and any of them with variables set, as <variant>:<VARIABLE>=<value>[,...] (fused:WATERLILY_OVERLAP_MIN=1099511627776:
# fused without the overlap; the files have _ for the : , and =).
# The points (POINTS; TGV, 100 timed steps after 20) are named weak_np<GPUs> (256^3 cells per GPU) or
# strong<log2 N>_np<GPUs> (N^3 cells), with any suffix (weak_np128b repeats weak_np128); by default weak 128, 256 and
# again 128 GPUs (the repeat shows the spread within the allocation), strong 512^3 on 128 GPUs, strong 1024^3 on 128
# and 256 GPUs. Each of ROUNDS rounds (1 by default) runs every point with every variant in turn. The curves on fewer
# nodes are those of mpi/jobs/submit.sh (data/mpi/jupiter_*_fused/, and the stock ones with WATERLILY_TAG=stock).
# Each run writes to $OUT (default mpi/fused/data/scaling_ab_<job id>/, not tracked), as <point>_<variant>.txt with
# one round and <point>_<round>_<variant>.txt with more; the job ends with a table of the times (and their mean over
# the rounds). The points on 128 GPUs alone need 32 nodes (sbatch --nodes=32). Submit from the repository root, with
# mpi/ (WaterLily mpi-igg-overlap, the fork in branch halo-fused) and mpi/fused built as for
# mpi/fused/jobs/jupiter_fused_diag.sh:
#
#   sbatch -A <budget> mpi/fused/jobs/jupiter_fused_scaling_ab.sh
#   VARIANTS="stock fused" sbatch -A <budget> mpi/fused/jobs/jupiter_fused_scaling_ab.sh
#   POINTS="strong9_np128 strong10_np256" ROUNDS=3 VARIANTS="fork fused-f38c69a fused" \
#       sbatch -A <budget> mpi/fused/jobs/jupiter_fused_scaling_ab.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
VARIANTS="${VARIANTS:-stock fork fused}"
POINTS="${POINTS:-weak_np128 weak_np256 weak_np128b strong9_np128 strong10_np128 strong10_np256}"
ROUNDS="${ROUNDS:-1}"
OUT="${OUT:-mpi/fused/data/scaling_ab_$SLURM_JOB_ID}"; mkdir -p "$OUT"
# point_args POINT: the number of GPUs and the arguments of mpi/scaling.jl for the point
point_args() {
    local np=${1#*_np} log2p
    np=${np%%[!0-9]*}
    case $1 in
        weak_np*) log2p=8 ;;
        strong*_np*) log2p=${1#strong}; log2p=${log2p%%_*} ;;
    esac
    [[ $np =~ ^[0-9]+$ && $log2p =~ ^[0-9]+$ ]] || { echo "point $1: not weak_np<GPUs> or strong<log2 N>_np<GPUs>" >&2; return 1; }
    echo "$np --log2p=$log2p$([[ $1 == weak* ]] && echo " --weak")"
}
for p in $POINTS; do point_args $p > /dev/null || exit 1; done
for v in $VARIANTS; do
    b=${v%%:*}
    [[ $b == stock || $b == fork || ($b == fused* && -d mpi/$b) ]] || { echo "variant $v: not stock, fork or mpi/fused*"; exit 1; }
done
echo "variants: $VARIANTS; points: $POINTS; rounds: $ROUNDS"
ENVS=" $(for v in $VARIANTS; do echo ${v%%:*}; done | sort -u | tr '\n' ' ')"  # the variants without their variables
file() { echo ${1//[:=,]/_}; }  # file VARIANT: the variant in the names of the files
# precompile on one rank, so that the runs do not all compile at once; print the WaterLily commit of each environment
if [[ $ENVS == *" stock "* || $ENVS == *" fork "* ]]; then
    srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile(); import ImplicitGlobalGrid as G, WaterLily as W
        isdefined(G, :update_halo_start!) || error("ImplicitGlobalGrid in mpi/ is not the fork, branch halo-fused")
        isdefined(W, :shell) || error("WaterLily in mpi/ is not on branch mpi-igg-overlap")
        println("mpi/: WaterLily ", readchomp(`git -C $(pkgdir(W)) log -1 --format="%h %s"`))' || exit 1
fi
for v in $ENVS; do
    [[ $v == fused* ]] || continue
    srun -N 1 -n 1 julia --project=mpi/$v -e 'using Pkg; Pkg.precompile(); import WaterLily as W
        isdefined(W, :_device_selectors) || error("WaterLily in $(dirname(Base.active_project())) is not on branch mpi-igg-fused")
        println("$(relpath(dirname(Base.active_project()))): WaterLily ", readchomp(`git -C $(pkgdir(W)) log -1 --format="%h %s"`))' || exit 1
done

# run VARIANT FILE NP ARGS...: one timed run of mpi/scaling.jl on NP GPUs, 4 per node
run() {
    local full=$1 v=${1%%:*} file=$2 np=$3; shift 3
    local proj=mpi vars="IGG_FUSED=0"
    case $v in fork) vars="IGG_FUSED=2" ;; fused*) proj=mpi/$v vars="WATERLILY_GPU_AWARE_MPI=0" ;; esac
    [[ $full == *:* ]] && vars="$vars ${full#*:}" && vars=${vars//,/ }
    env $vars srun -N $(( (np + 3) / 4 )) -n $np julia --project=$proj -t 4 mpi/scaling.jl --case=tgv --backend=CuArray \
        --steps=100 --warmup=20 "$@" > "$OUT/$file.txt" 2>&1 || echo "[ FAILED: $file ]"
}
# name POINT ROUND: the name of the runs of the point in the round
name() { [ "$ROUNDS" -eq 1 ] && echo $1 || echo $1_$2; }
for k in $(seq $ROUNDS); do
    for p in $POINTS; do
        for v in $VARIANTS; do run $v $(name $p $k)_$(file $v) $(point_args $p); done
    done
done

i=0; for v in $VARIANTS; do i=$((i + 1)); [ ${#v} -le 14 ] || echo "variant $i: $v"; done
printf "%-20s" ms/step; i=0
for v in $VARIANTS; do i=$((i + 1)); [ ${#v} -le 14 ] && printf "%15s" $v || printf "%15s" "variant $i"; done; echo
for p in $POINTS; do
    for k in $(seq $ROUNDS) mean; do
        [[ $k == mean && $ROUNDS -eq 1 ]] && continue
        printf "%-20s" "$([ $k = mean ] && echo "$p mean" || name $p $k)"
        for v in $VARIANTS; do
            if [ $k = mean ]; then
                t=$(for j in $(seq $ROUNDS); do grep -ho 'per_step=[0-9.]*' "$OUT/$(name $p $j)_$(file $v).txt" 2>/dev/null; done |
                    awk -F= '{s += $2; n++} END {if (n) printf "%.2f (%d)", s/n, n}')
            else
                t=$(grep -ho 'per_step=[0-9.]*' "$OUT/$(name $p $k)_$(file $v).txt" 2>/dev/null | cut -d= -f2)
            fi
            printf "%15s" "${t:--}"
        done
        echo
    done
done
