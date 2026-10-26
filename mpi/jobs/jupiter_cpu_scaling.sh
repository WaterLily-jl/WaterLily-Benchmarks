#!/bin/bash
#SBATCH --job-name=wl-scaling-cpu
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

# One point (or a few) of a strong- or weak-scaling curve on the CPUs of JUPITER Booster nodes (4 Grace CPUs of 72
# cores each), one thread per rank as on Snellius: the SIMD backend, in its own environment, built once with
#
#   julia mpi/setup.jl --env=mpi/cpu --backend=SIMD --cuda-local
#
# WATERLILY_RPN ranks per node (default 256, a power of 2 that splits the 2^p grids, which 288 does not), the same
# number on each Grace CPU, spread evenly over its cores.
#
#   sbatch -A <budget> -N <nodes> mpi/jobs/jupiter_cpu_scaling.sh MODE CASE LOG2P [STEPS=100] [WARMUP=20] [NPS=RPN×nodes]
#
# MODE is strong (LOG2P sets the global size) or weak (LOG2P sets the size per rank); 1 in NPS runs without MPI.
# mpi/jobs/submit.sh submits a whole curve, one job per node count (bash mpi/jobs/submit.sh jupiter_cpu ...).

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
MODE="${1:?usage: jupiter_cpu_scaling.sh MODE CASE LOG2P [STEPS] [WARMUP] [NPS]}"; CASE="${2:?missing CASE}"; P="${3:?missing LOG2P}"
RPN="${WATERLILY_RPN:-256}"
STEPS="${4:-100}"; WARMUP="${5:-20}"; NPS="${6:-$((RPN*SLURM_JOB_NUM_NODES))}"
[ "$MODE" = weak ] && WEAK="--weak" || WEAK=""
echo "$MODE scaling, $CASE log2p=$P, $STEPS steps after $WARMUP, ranks: $NPS ($RPN per node)"
lscpu | grep -E "^NUMA node[0-9]+ CPU\(s\): +[0-9]"  # bind takes the cores of the Grace CPUs to be 0-71, 72-143, 144-215, 216-287

bind () { # ranks per node -> one core per rank: the same number of ranks on each Grace CPU, in rank order, spread evenly over its 72 cores
    local tpn=$1 q=$(( ($1 + 3) / 4 )) r cores=()
    for (( r = 0; r < tpn; r++ )); do cores+=($(( r / q * 72 + r % q * 72 / q ))); done
    local IFS=,; echo "--cpu-bind=map_cpu:${cores[*]}"
}
for np in $NPS; do
    nodes=$(( (np + RPN - 1) / RPN ))
    if [ "$nodes" -gt "$SLURM_JOB_NUM_NODES" ]; then echo "[ SKIPPED np=$np: needs $nodes nodes ]"; continue; fi
    tpn=$(( (np + nodes - 1) / nodes ))
    serial=""; [ "$np" -eq 1 ] && serial="--serial"
    srun $(bind "$tpn") -N "$nodes" -n "$np" --ntasks-per-node="$tpn" julia --project=mpi/cpu -t 1 mpi/scaling.jl \
        --case="$CASE" --log2p="$P" $WEAK --backend=Array --steps="$STEPS" --warmup="$WARMUP" $serial 2>&1 \
        | grep -E "Info|^WaterLily|MPI:|Warn|ERROR" || echo "[ FAILED: $CASE np=$np ]"
done
