#!/bin/bash
#SBATCH --job-name=wl-scaling
#SBATCH --partition=rome
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00

# One point (or a few) of a strong- or weak-scaling curve on Snellius CPUs, one thread per rank (SIMD backend:
# julia mpi/setup.jl --backend=SIMD). WATERLILY_RPN ranks per node (default 64) are spread evenly over the
# CCXs of each node: the per-node slowdown comes from ranks sharing a CCX, not from memory bandwidth.
#
#   sbatch -N <nodes> mpi/jobs/snellius_scaling.sh MODE CASE LOG2P [STEPS=100] [WARMUP=20] [NPS=RPN×nodes]
#
# MODE is strong (LOG2P sets the global size) or weak (LOG2P sets the size per rank); 1 in NPS runs without MPI.
# mpi/jobs/submit.sh submits a whole curve, one job per node count. --ear=on is not set: it crashes MPI.jl at init.

cd "$SLURM_SUBMIT_DIR"
source mpi/jobs/snellius_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
MODE="${1:?usage: snellius_scaling.sh MODE CASE LOG2P [STEPS] [WARMUP] [NPS]}"; CASE="${2:?missing CASE}"; P="${3:?missing LOG2P}"
RPN="${WATERLILY_RPN:-64}"
STEPS="${4:-100}"; WARMUP="${5:-20}"; NPS="${6:-$((RPN*SLURM_JOB_NUM_NODES))}"
[ "$MODE" = weak ] && WEAK="--weak" || WEAK=""
echo "WaterLily: $(git -C "$WATERLILY_DIR" log -1 --format='%h %s')"
echo "$MODE scaling, $CASE log2p=$P, $STEPS steps after $WARMUP, ranks: $NPS ($RPN per node)"

bind () { # ranks per node -> srun binding: an even stride over the 128 cores (equal ranks per CCX)
    local tpn=$1
    [ $(( 128 % tpn )) -ne 0 ] && { echo "--cpu-bind=cores"; return; }
    echo "--cpu-bind=map_cpu:$(seq -s, 0 $(( 128 / tpn )) 127) -c $(( 128 / tpn ))"
}
for np in $NPS; do
    args=(--case="$CASE" --log2p="$P" $WEAK --backend=Array --steps="$STEPS" --warmup="$WARMUP")
    if [ "$np" -eq 1 ]; then
        julia --project="${WATERLILY_ENV:-mpi}" -t 1 mpi/scaling.jl "${args[@]}" --serial 2>&1 | grep -E "Info|ERROR" || echo "[ FAILED: $CASE serial ]"
        continue
    fi
    nodes=$(( (np + RPN - 1) / RPN ))
    if [ "$nodes" -gt "$SLURM_JOB_NUM_NODES" ]; then echo "[ SKIPPED np=$np: needs $nodes nodes ]"; continue; fi
    tpn=$(( (np + nodes - 1) / nodes ))
    srun --mpi=pmix $(bind "$tpn") -N "$nodes" -n "$np" --ntasks-per-node="$tpn" julia --project="${WATERLILY_ENV:-mpi}" -t 1 \
        mpi/scaling.jl "${args[@]}" 2>&1 | grep -E "Info|MPI:|ERROR" || echo "[ FAILED: $CASE np=$np ]"
done
