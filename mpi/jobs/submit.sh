#!/bin/bash
# Submit a strong- or weak-scaling curve, one job per node count, from the repository root:
#
#   bash mpi/jobs/submit.sh CLUSTER MODE CASE LOG2P "NODES" [STEPS=100] [WARMUP=20] [-- extra sbatch options]
#
#   bash mpi/jobs/submit.sh jupiter strong tgv 9 "1 2 4 8 16" -- -A <budget>
#   bash mpi/jobs/submit.sh jupiter weak   tgv 8 "1 2 4 8 16 32" -- -A <budget>
#   bash mpi/jobs/submit.sh snellius strong tgv 8 "1 2 4 8"
#
# CLUSTER selects mpi/jobs/<CLUSTER>_scaling.sh. The first job also runs the points inside one node (on JUPITER
# 1, 2 and 4 GPUs, the first without MPI). The outputs go to data/mpi/<CLUSTER>_<MODE>_<CASE>_<LOG2P>/, named
# <MODE>_*.out, where mpi/plot_strong.jl and mpi/plot_weak.jl find them. With WATERLILY_BACKEND=Array (the CPUs of
# the JUPITER nodes, see jupiter_scaling.sh) the directory is data/mpi/<CLUSTER>_cpu_<MODE>_<CASE>_<LOG2P>/.
set -euo pipefail
CLUSTER="${1:?usage: submit.sh CLUSTER MODE CASE LOG2P \"NODES\" [STEPS] [WARMUP] [-- sbatch options]}"
MODE="$2"; CASE="$3"; P="$4"; NODES="$5"; shift 5
STEPS=100; WARMUP=20
[ $# -gt 0 ] && [ "$1" != "--" ] && { STEPS="$1"; shift; }
[ $# -gt 0 ] && [ "$1" != "--" ] && { WARMUP="$1"; shift; }
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
case "$MODE" in strong|weak) ;; *) echo "MODE must be strong or weak" >&2; exit 1 ;; esac
JOB="mpi/jobs/${CLUSTER}_scaling.sh"; [ -f "$JOB" ] || { echo "no $JOB" >&2; exit 1; }
RPN="${WATERLILY_RPN:-$([ "$CLUSTER" = jupiter ] && echo 4 || echo 64)}"  # ranks per node
TAG="$CLUSTER"; [ "${WATERLILY_BACKEND:-}" = Array ] && TAG="${CLUSTER}_cpu"
DIR="data/mpi/${TAG}_${MODE}_${CASE}_${P}"; mkdir -p "$DIR"

first=true
for n in $NODES; do
    nps=$(( n * RPN ))
    if $first; then  # the points inside the first node: 1, 2, 4, ... ranks
        nps=""; np=1; while [ "$np" -lt $(( n * RPN )) ]; do nps="$nps $np"; np=$(( np * 2 )); done
        nps="${nps# } $(( n * RPN ))"; first=false
    fi
    sbatch --nodes="$n" --job-name="wl-${TAG}-${MODE}-${CASE}-${P}-N${n}" --output="$DIR/${MODE}_${CASE}_${P}_N${n}_%j.out" \
        "$@" "$JOB" "$MODE" "$CASE" "$P" "$STEPS" "$WARMUP" "$nps"
done
echo "Outputs in $DIR"
