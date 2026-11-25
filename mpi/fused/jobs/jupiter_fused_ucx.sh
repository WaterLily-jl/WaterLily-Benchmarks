#!/bin/bash
#SBATCH --job-name=wl-fused-ucx
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=01:00:00
#SBATCH --output=wl-fused-ucx-%j.out

# Device-to-device halos of mpi-igg-fused with UCX's rendezvous threshold lowered. Nsight Systems (job 2594773) put
# the cost of device to device in UCX's choice of protocol, not in CUDA copies: for device memory UCX sends eager
# (copied through its own buffers) up to 134399 B inside a node and 656795 B between nodes, and only above that zero
# copy (cuda_ipc, GPUDirect RDMA); for host memory it switches at 13537 and 25203 B. So between nodes a 256 KiB
# device message took 55.3 us one way against 17.9 from host memory, and the fine pressure halo of TGV 512^3 on 8 GPUs
# (524288 B messages) and every coarser one were slower device to device. UCX_RNDV_THRESH sets the size from which
# UCX uses the rendezvous protocols for every memory type; the job runs, for the UCX default and each of THRESHOLDS
# (by default 16384, near the default for host memory, and 65536, below the 131072 B messages of the 130^3 level):
#   1. the ping-pong of mpi/cuda_mpi_test.jl between two GPUs of one node and of two nodes, device and host buffers
#      of the sizes of the halo messages of TGV 512^3 on 8 GPUs (one message per neighbour, both faces of a periodic
#      pair, Float32: 32 B at 4^3 to 524288 B at 258^3, 1572864 B the velocity), UCX_PROTO_INFO=y (the protocol UCX
#      picks for each memory type, message size and transport, summarized at the end);
#   2. TGV 512^3 on 8 GPUs of 2 nodes (mpi/scaling.jl of PROJ, default mpi/fused), timed: staged (the default),
#      device to device (WATERLILY_GPU_AWARE_MPI=1), and device to device from halos of 1 MiB (the fine pressure and
#      the velocity) and of 256 KiB (also the 130^3 level), the others staged; then the staged default again;
#   3. the same TGV with the time of each halo by array size (mpi/fused/halo_diag.jl, DIAG_LAPS=1), staged and
#      device to device: which halos win which way, so the WATERLILY_GPU_AWARE_MPI threshold to pick.
# UCX_RNDV_THRESH also moves the switch for host memory, hence the staged runs at each threshold; the first thing to
# read is whether the protocol tables at the end moved the switch at all. The times compare within the job (PROJ is
# what mpi/fused holds when the job starts, whose commit the job prints first). Each run writes to
# $OUT (default mpi/fused/data/ucx_<job id>/, not tracked); the job ends with the tables. Submit from the repository
# root, with mpi/ and mpi/fused built as for mpi/fused/jobs/jupiter_fused_diag.sh:
#
#   sbatch -A <budget> mpi/fused/jobs/jupiter_fused_ucx.sh
#   THRESHOLDS="8192 32768" sbatch -A <budget> mpi/fused/jobs/jupiter_fused_ucx.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
PROJ="${PROJ:-mpi/fused}"; THRESHOLDS="${THRESHOLDS:-16384 65536}"
OUT="${OUT:-mpi/fused/data/ucx_$SLURM_JOB_ID}"; mkdir -p "$OUT"
echo "UCX: $(command -v ucx_info >/dev/null && ucx_info -v | head -n 2 | tr '\n' ' ' || echo 'no ucx_info')"
command -v ucx_info >/dev/null && ucx_info -c | grep -iE 'RNDV|ZCOPY|PROTO_ENABLE|_TLS=|MEMTYPE'
echo "UCX and ParaStation variables set:"; env | grep -E '^(UCX|PSP)_' | sort
# precompile on one rank, so that the runs do not all compile at once; print the WaterLily commit of PROJ
srun -N 1 -n 1 julia --project=$PROJ -e 'using Pkg; Pkg.precompile(); import WaterLily as W
    isdefined(W, :_device_selectors) || error("WaterLily in $(dirname(Base.active_project())) is not on branch mpi-igg-fused")
    println("$(relpath(dirname(Base.active_project()))): WaterLily ", readchomp(`git -C $(pkgdir(W)) log -1 --format="%h %s"`))' || exit 1
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile()' || exit 1

# run NAME NODES RANKS THRESHOLD VARIABLES COMMAND...: one run on RANKS ranks of NODES nodes, UCX_RNDV_THRESH set to
# THRESHOLD (unless it is default) and the comma-separated VARIABLES too
run() {
    local name=$1 nodes=$2 ranks=$3 t=$4 vars=$5; shift 5
    [ "$t" = default ] || vars="UCX_RNDV_THRESH=$t,$vars"
    env ${vars//,/ } srun -N $nodes -n $ranks "$@" > "$OUT/$name.txt" 2>&1 || echo "[ FAILED: $name ]"
}
export PINGPONG_BYTES="8 512 2048 8192 32768 131072 524288 1572864 8388608"
TGV="--case=tgv --backend=CuArray --log2p=9 --steps=100 --warmup=20"
declare -A AWARE=([staged]=0 [device]=1 [from1M]=1048576 [from256K]=262144)
SETTINGS="default $THRESHOLDS"
for t in $SETTINGS; do
    run pingpong_node_$t 1 2 $t UCX_PROTO_INFO=y julia --project=mpi mpi/cuda_mpi_test.jl
    run pingpong_nodes_$t 2 2 $t UCX_PROTO_INFO=y julia --project=mpi mpi/cuda_mpi_test.jl
done
for t in $SETTINGS; do
    for h in staged device from1M from256K; do
        run tgv_${h}_$t 2 8 $t WATERLILY_GPU_AWARE_MPI=${AWARE[$h]} julia --project=$PROJ -t 4 mpi/scaling.jl $TGV
    done
done
run tgv_staged_default_again 2 8 default WATERLILY_GPU_AWARE_MPI=0 julia --project=$PROJ -t 4 mpi/scaling.jl $TGV
for t in $SETTINGS; do
    for h in staged device; do
        run laps_${h}_$t 2 8 $t WATERLILY_GPU_AWARE_MPI=${AWARE[$h]},DIAG_LAPS=1 julia --project=$PROJ -t 4 \
            mpi/fused/halo_diag.jl $TGV
    done
done

echo; echo "UCX protocols: the size from which each memory type goes rendezvous"
for t in $SETTINGS; do
    for p in node nodes; do
        awk -v t=$t '/\| ucp_context_[0-9]+ (intra|inter)-node cfg#[0-9]+ +\| tagged message by ucp_tag_send\* from/ {
                match($0, /(intra|inter)-node/); place = substr($0, RSTART, RLENGTH)
                match($0, /from [^|]*/); mem = substr($0, RSTART + 5, RLENGTH - 5); sub(/ +$/, "", mem); on = 1; next }
            on && /rendezvous/ { split($0, f, "|"); r = f[2]; gsub(/ /, "", r); print t ": " place ", " mem ": " r; on = 0 }' \
            "$OUT/pingpong_${p}_$t.txt" | sort -u
    done
done
for p in node nodes; do
    echo; echo "ping-pong, one way [us], $([ $p = node ] && echo "two GPUs of one node" || echo "two nodes"): device host"
    printf "%10s" bytes; for t in $SETTINGS; do printf "%22s" "$t"; done; echo
    for b in $PINGPONG_BYTES; do
        printf "%10s" $b
        for t in $SETTINGS; do
            printf "%22s" "$(awk -v b=$b '$1 == b && NF == 5 {printf "%9.2f %9.2f", $2, $4}' "$OUT/pingpong_${p}_$t.txt")"
        done
        echo
    done
done
echo; echo "TGV 512^3 on 8 GPUs, ms per step"
printf "%-12s" ""; for h in staged device from1M from256K; do printf "%10s" $h; done; echo
for t in $SETTINGS; do
    printf "%-12s" $t
    for h in staged device from1M from256K; do
        printf "%10s" "$(grep -ho 'per_step=[0-9.]*' "$OUT/tgv_${h}_$t.txt" 2>/dev/null | cut -d= -f2)"
    done
    echo
done
printf "%-12s%10s\n" "again" "$(grep -ho 'per_step=[0-9.]*' "$OUT/tgv_staged_default_again.txt" 2>/dev/null | cut -d= -f2)"
echo; echo "halos of TGV 512^3 on 8 GPUs, us per call (median over the ranks), staged and device to device"
files=(); for t in $SETTINGS; do for h in staged device; do files+=("$OUT/laps_${h}_$t.txt"); done; done
printf "%-22s %6s" array calls; for t in $SETTINGS; do printf "%20s" "$t"; done; echo
printf "%-29s" ""; for t in $SETTINGS; do printf "%10s%10s" staged device; done; echo
awk -v n=${#files[@]} 'FNR == 1 {f++; t = 0} /^array/ {t = 1; next} t && /^total/ {tot[f] = $2; t = 0; next}
    t { k = $1 " " $2; if (!(k in c)) {keys[++m] = k; c[k] = $3}; u[k, f] = $11 }
    END {
        for (i = 1; i <= m; i++) { k = keys[i]; printf "%-22s %6s", k, c[k]
            for (j = 1; j <= n; j++) printf "%10s", ((k, j) in u ? u[k, j] : "-"); print "" }
        printf "%-22s %6s", "total ms/step", ""; for (j = 1; j <= n; j++) printf "%10s", (j in tot ? tot[j] : "-"); print ""
    }' "${files[@]}"
