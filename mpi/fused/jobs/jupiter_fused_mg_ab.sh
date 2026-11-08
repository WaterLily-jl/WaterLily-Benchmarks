#!/bin/bash
#SBATCH --job-name=wl-fused-mg-ab
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=00:45:00
#SBATCH --output=wl-fused-mg-ab-%j.out

# The pressure solver of mpi-igg-fused with fewer halos and collectives (NEW, WaterLily d843529: no residual
# update, so no halo, after the smoothing of the coarse levels; no halo of x before the first residual of a solve,
# whose ghosts the previous solve left in place; the count of solved cells kept between updates, and L1 and Linf in one
# collective) against f38c69a (BASE), all in one allocation:
#   1. TGV 512^3, timed, on 8 GPUs of 2 nodes and on the 4 GPUs of one node, each version twice in turn;
#   2. the time of each phase of each halo (mpi/fused/halo_diag.jl, DIAG_LAPS=1) on 8 GPUs: 64 pressure halos per
#      step with f38c69a (job 2578550), 48 expected;
#   3. TGV 256^3 on 8 GPUs with --check, compared with mpi/check.jl: the two must be identical.
# BASE (default mpi/fused-f38c69a) and NEW (default mpi/fused) are environments like mpi/fused with WaterLily at
# f38c69a and at d843529. Jobs that start later compile the clone of mpi/fused at whatever commit it then holds, so
# while other jobs of mpi/fused wait in the queue, keep that clone where it is and build both, once, on a login node
# after `source mpi/jobs/jupiter_env.sh` (for c in f38c69a d843529; from the remote, which has every pushed commit, as
# a clone of the clone has only its local branches):
#
#   git clone $(git -C $WATERLILY_DIR-fused remote get-url origin) $WATERLILY_DIR-fused-$c
#   git -C $WATERLILY_DIR-fused-$c checkout $c
#   mkdir -p mpi/fused-$c && cp mpi/fused/{Project,Manifest,LocalPreferences}.toml mpi/fused-$c/
#   julia --project=mpi/fused-$c -e "using Pkg; Pkg.develop(path=\"$WATERLILY_DIR-fused-$c\")"
#
# Each run writes to $OUT (default mpi/fused/data/mg_ab_<job id>/, not tracked); the job ends with a table of the
# times and the halos per step. Submit from the repository root:
#
#   NEW=mpi/fused-d843529 sbatch -A <budget> mpi/fused/jobs/jupiter_fused_mg_ab.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
BASE="${BASE:-mpi/fused-f38c69a}"; NEW="${NEW:-mpi/fused}"
OUT="${OUT:-mpi/fused/data/mg_ab_$SLURM_JOB_ID}"; mkdir -p "$OUT"
# precompile on one rank, so that the runs do not all compile at once; print the commit of each environment (the
# last line of its output)
commit() {
    srun -N 1 -n 1 julia --project=$1 -e 'using Pkg; Pkg.precompile(); import WaterLily as W
        isdefined(W, :_device_selectors) || error("WaterLily in $(dirname(Base.active_project())) is not on branch mpi-igg-fused")
        println(readchomp(`git -C $(pkgdir(W)) log -1 --format="%h %s"`))' | tail -n 1
}
new=$(commit $NEW) || exit 1
base=$(commit $BASE) || exit 1
echo "new:  $NEW, WaterLily $new"; echo "base: $BASE, WaterLily $base"
[ "${new%% *}" != "${base%% *}" ] || { echo "$NEW and $BASE have the same WaterLily commit"; exit 1; }

TGV="--case=tgv --backend=CuArray --log2p=9 --steps=100 --warmup=20"
# run NAME NODES ENVIRONMENT VARIABLES SCRIPT ARGS...: one run on the 4 GPUs of each node, the comma-separated
# VARIABLES set (staged halos, as by default)
run() {
    local name=$1 nodes=$2 proj=$3 vars=$4; shift 4
    env WATERLILY_GPU_AWARE_MPI=0 ${vars//,/ } srun -N $nodes -n $((4*nodes)) julia --project=$proj -t 4 "$@" \
        > "$OUT/$name.txt" 2>&1 || echo "[ FAILED: $name ]"
}
for k in 1 2; do
    for v in base new; do
        proj=$NEW; [ $v = base ] && proj=$BASE
        run np8_${v}_$k 2 $proj "" mpi/scaling.jl $TGV
        run np4_${v}_$k 1 $proj "" mpi/scaling.jl $TGV
    done
done
for v in base new; do
    proj=$NEW; [ $v = base ] && proj=$BASE
    run laps_$v 2 $proj DIAG_LAPS=1 mpi/fused/halo_diag.jl $TGV
    run check_$v 2 $proj "" mpi/scaling.jl --case=tgv --backend=CuArray --log2p=8 --steps=50 --warmup=0 --check=10 \
        --out="$OUT/check_$v.csv"
done

[ -f "$OUT/check_base.csv" ] && julia mpi/check.jl "$OUT/check_base.csv" "$OUT/check_new.csv"
echo "ms per step"
for name in np8_base_1 np8_new_1 np8_base_2 np8_new_2 np4_base_1 np4_new_1 np4_base_2 np4_new_2 laps_base laps_new; do
    printf "%-12s %s\n" $name "$(grep -ho 'per_step=[0-9.]*' "$OUT/$name.txt" 2>/dev/null | cut -d= -f2)"
done
echo "halos per step on 8 GPUs (velocity and pressure), and their ms per step"
for v in base new; do
    awk -v v=$v '/^array/ {t=1; next} t && /^total/ {printf "%-12s %5.0f halos, %s ms\n", v, c, $2; t=0} t {c += $3}' \
        "$OUT/laps_$v.txt"
done
