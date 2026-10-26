#!/bin/bash
#SBATCH --job-name=wl-scaling
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00

# One point (or a few) of a strong- or weak-scaling curve on JUPITER GPUs: one rank per GPU, 4 per node. Each
# srun is a new Julia process; the steps of a job run one after the other on the same exclusive nodes.
#
#   sbatch -A <budget> -N <nodes> mpi/jobs/jupiter_scaling.sh MODE CASE LOG2P [STEPS=100] [WARMUP=20] [NPS=4×nodes]
#
# MODE is strong (LOG2P sets the global size) or weak (LOG2P sets the size per GPU). NPS lists the numbers of
# GPUs to run, at most 4 per node; 1 runs on one GPU without MPI, the reference of the efficiency.
# mpi/jobs/submit.sh submits a whole curve, one job per node count.

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
MODE="${1:?usage: jupiter_scaling.sh MODE CASE LOG2P [STEPS] [WARMUP] [NPS]}"; CASE="${2:?missing CASE}"; P="${3:?missing LOG2P}"
STEPS="${4:-100}"; WARMUP="${5:-20}"; NPS="${6:-$((4*SLURM_JOB_NUM_NODES))}"
[ "$MODE" = weak ] && WEAK="--weak" || WEAK=""
echo "WaterLily: $(git -C "$WATERLILY_DIR" log -1 --format='%h %s')   IGG_CUDAAWARE_MPI=$IGG_CUDAAWARE_MPI"
echo "$MODE scaling, $CASE log2p=$P, $STEPS steps after $WARMUP, GPUs: $NPS"

for np in $NPS; do
    nodes=$(( (np + 3) / 4 ))
    if [ "$nodes" -gt "$SLURM_JOB_NUM_NODES" ]; then echo "[ SKIPPED np=$np: needs $nodes nodes ]"; continue; fi
    serial=""; [ "$np" -eq 1 ] && serial="--serial"
    srun -N "$nodes" -n "$np" julia --project=mpi -t 4 mpi/scaling.jl --case="$CASE" --log2p="$P" $WEAK \
        --backend=CuArray --steps="$STEPS" --warmup="$WARMUP" $serial 2>&1 \
        | grep -E "Info|MPI:|Warn|ERROR" || echo "[ FAILED: $CASE np=$np ]"
done
