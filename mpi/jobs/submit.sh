#!/bin/bash
# Submit a strong- or weak-scaling curve, one job per node count, from the repository root:
#
#   bash mpi/jobs/submit.sh CLUSTER MODE CASE LOG2P "NODES" [STEPS=100] [WARMUP=20] [-- extra sbatch options]
#
#   bash mpi/jobs/submit.sh jupiter strong tgv 9 "1 2 4 8 16" -- -A <budget>
#   bash mpi/jobs/submit.sh jupiter weak   tgv 8 "1 2 4 8 16 32" -- -A <budget>
#   bash mpi/jobs/submit.sh jupiter_cpu strong tgv 8 "1 2 4 8 16" -- -A <budget>
#   bash mpi/jobs/submit.sh snellius strong tgv 8 "1 2 4 8"
#
# CLUSTER selects mpi/jobs/<CLUSTER>_scaling.sh. The job on one node also runs the points inside it (on JUPITER
# 1, 2 and 4 GPUs, the first without MPI); the others run one point each, so a single point can be submitted again. The outputs go to data/mpi/<CLUSTER>_<MODE>_<CASE>_<LOG2P>/, named
# <MODE>_*.out, where mpi/plot_strong.jl and mpi/plot_weak.jl find them. WATERLILY_RPN sets the ranks per node: on
# JUPITER 4 (one per GPU), on its CPUs (jupiter_cpu) 256, on Snellius 64. WATERLILY_ENV sets the Julia environment of
# the runs (by default mpi, mpi/cpu for jupiter_cpu), and the outputs then go to <...>_<LOG2P>_<its last directory>/:
#
#   WATERLILY_ENV=mpi/fused bash mpi/jobs/submit.sh jupiter weak tgv 8 "1 2 4 8 16 32" -- -A <budget>  # ..._8_fused/
set -euo pipefail
CLUSTER="${1:?usage: submit.sh CLUSTER MODE CASE LOG2P \"NODES\" [STEPS] [WARMUP] [-- sbatch options]}"
MODE="$2"; CASE="$3"; P="$4"; NODES="$5"; shift 5
STEPS=100; WARMUP=20
[ $# -gt 0 ] && [ "$1" != "--" ] && { STEPS="$1"; shift; }
[ $# -gt 0 ] && [ "$1" != "--" ] && { WARMUP="$1"; shift; }
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
case "$MODE" in strong|weak) ;; *) echo "MODE must be strong or weak" >&2; exit 1 ;; esac
JOB="mpi/jobs/${CLUSTER}_scaling.sh"; [ -f "$JOB" ] || { echo "no $JOB" >&2; exit 1; }
case "$CLUSTER" in jupiter) RPN=4 ;; jupiter_cpu) RPN=256 ;; *) RPN=64 ;; esac
RPN="${WATERLILY_RPN:-$RPN}"  # ranks per node
DIR="data/mpi/${CLUSTER}_${MODE}_${CASE}_${P}${WATERLILY_ENV:+_$(basename "$WATERLILY_ENV")}"; mkdir -p "$DIR"

for n in $NODES; do
    nps=$(( n * RPN ))
    if [ "$n" -eq 1 ]; then  # the points inside the node: 1, 2, 4, ... ranks
        nps=""; np=1; while [ "$np" -lt "$RPN" ]; do nps="$nps $np"; np=$(( np * 2 )); done
        nps="${nps# } $RPN"
    fi
    sbatch --nodes="$n" --job-name="wl-${CLUSTER}-${MODE}-${CASE}-${P}-N${n}" --output="$DIR/${MODE}_${CASE}_${P}_N${n}_%j.out" \
        "$@" "$JOB" "$MODE" "$CASE" "$P" "$STEPS" "$WARMUP" "$nps"
done
echo "Outputs in $DIR"
