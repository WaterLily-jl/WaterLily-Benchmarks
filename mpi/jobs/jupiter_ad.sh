#!/bin/bash
#SBATCH --job-name=wl-ad
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=01:30:00
#SBATCH --output=wl-ad-%j.out

# The scaling tests of mpi/jobs/jupiter_scaling.sh with forward-mode AD numbers (ForwardDiff.Dual, mpi/scaling.jl
# --partials) next to Float32: TGV, one rank per GPU. At every point the runs follow one another in this allocation:
# Float32, Dual with each number of partials of AD_PARTIALS, Float32 again (the two Float32 runs give the scatter of
# the ratio), then Float32 and the first of AD_PARTIALS with the halo timers (--laps). Submit from the repository root:
#
#   sbatch -A <budget> -N <nodes> mpi/jobs/jupiter_ad.sh ["MODE LOG2P NP [NP ...]" ...]
#
# Each argument is a set of points: MODE strong (LOG2P sets the global size) or weak (the size per GPU), and the
# numbers of GPUs (1 runs without MPI). Without arguments, the points of the node count, which make the weak curve at
# 256^3 per GPU and the strong curve at 512^3 when submitted on 1, 2, 4, 8 and 16 nodes (1, 4, 8, 16, 32, 64 GPUs):
#   1 node    "weak 7 1" "weak 8 1 4" "strong 9 1 4"   one GPU at 128^3, 256^3 and 512^3, 4 GPUs
#   N nodes   "weak 8 4N" "strong 9 4N"
# Variables (defaults in brackets):
#   AD_PARTIALS       [2 1 3]          the partials of the Dual runs, in this order
#   AD_LAPS           [1]              0: no runs with halo timers
#   AD_TOL            [2e-6]           also time Float32 and the first of AD_PARTIALS with this pressure tolerance at
#                                      the weak points of 256^3 per GPU and more: the solver stops on the residual of
#                                      the values, not of the partials, and the derivative needs more iterations
#                                      than WaterLily's 2e-3 gives (empty: skip)
#   AD_MORE_PARTIALS  [4 8, none beyond 2 nodes]  more partials, timed once each at the weak points
#   AD_DEVICE         [1, 0 beyond 2 nodes]       also time Float32 and the first of AD_PARTIALS with the halos passed
#                                      device to device (WATERLILY_GPU_AWARE_MPI=1; the others are staged through host memory)
#   AD_ARGS           []               more options of scaling.jl for every run
#   RUN_MINUTES       [10]             the time limit of each run (srun --time), so that one that hangs does not
#                                      take the rest of the job with it
#   STEPS [100], WARMUP [20], WATERLILY_ENV [mpi]
# The runs with the tolerance, more partials and device to device come last, so that a time limit cuts them first.
# Every run writes its output to data/mpi/jupiter_ad_<job>/<point>_<run>.txt; the job prints their table and writes
# runs.csv there (mpi/collect.jl, which also merges several jobs: julia mpi/collect.jl --out=ad.csv data/mpi/jupiter_ad_*).
# Up to 4 nodes fit in the develbooster reservation (--reservation=develbooster).

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
N="$SLURM_JOB_NUM_NODES"; NP=$((4*N))
if [ $# -gt 0 ]; then POINTS=("$@")
elif [ "$N" -eq 1 ]; then POINTS=("weak 7 1" "weak 8 1 4" "strong 9 1 4")
else POINTS=("weak 8 $NP" "strong 9 $NP"); fi
few=$([ "$N" -le 2 ] && echo 1 || echo 0)
PARTIALS="${AD_PARTIALS:-2 1 3}"; FIRST="${PARTIALS%% *}"; LAPS="${AD_LAPS:-1}"; DEVICE="${AD_DEVICE:-$few}"
MORE="${AD_MORE_PARTIALS-$([ "$few" -eq 1 ] && echo "4 8")}"; TOL="${AD_TOL-2e-6}"
STEPS="${STEPS:-100}"; WARMUP="${WARMUP:-20}"; ENV_DIR="${WATERLILY_ENV:-mpi}"
DIR="data/mpi/jupiter_ad_${SLURM_JOB_ID}"; mkdir -p "$DIR"
echo "points: $(printf '"%s" ' "${POINTS[@]}"), partials: $PARTIALS, laps: $LAPS, tolerance: ${TOL:-none}, more partials: ${MORE:-none}, device to device: $DEVICE"
echo "$STEPS steps after $WARMUP${AD_ARGS:+, $AD_ARGS}, environment $ENV_DIR -> $DIR"
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project="$ENV_DIR" -e 'using Pkg; Pkg.precompile(); using CUDA, MPI, WaterLily
    isdefined(WaterLily, :HaloBackend) || error("WaterLily in this environment has no halo exchange of its own (mpi-igg since 57443f8)")' || exit 1
srun -N 1 -n 4 bash -c 'echo "  rank $SLURM_PROCID: cores$(taskset -cp $$ | cut -d: -f2)"' | sort -k2n

run () { # NAME NP PARTIALS GPU_AWARE [scaling.jl options]: one run, its whole output in $DIR/NAME.txt
    local name=$1 np=$2 partials=$3 aware=$4; shift 4
    local nodes=$(( (np + 3) / 4 )) serial=""; [ "$np" -eq 1 ] && serial="--serial"
    if [ "$nodes" -gt "$N" ]; then echo "[ SKIPPED $name: needs $nodes nodes ]"; return; fi
    { echo "# run=$name job=$SLURM_JOB_ID nodes=$nodes"
      WATERLILY_GPU_AWARE_MPI="$aware" srun --time="${RUN_MINUTES:-10}" -N "$nodes" -n "$np" julia --project="$ENV_DIR" -t 4 mpi/scaling.jl --case=tgv \
          --backend=CuArray --steps="$STEPS" --warmup="$WARMUP" --partials="$partials" $serial "$@" ${AD_ARGS:-}; } > "$DIR/$name.txt" 2>&1
    printf "%-30s %s\n" "$name" "$(grep -h "Info: \(SERIAL\|MPI np\)" "$DIR/$name.txt" | sed 's/.*Info: //; s/  case.*per_step=/ /; s/ levels=[0-9]*//; s/cost.*partials/partials/' || true)"
    grep -q "per_step=" "$DIR/$name.txt" || { echo "[ FAILED: $name ]"; grep -hE "ERROR|[Ee]rror|srun:|signal|[Kk]illed" "$DIR/$name.txt" | head -5; }
}
each () { # runs FUNCTION MODE LOG2P NP for every point
    local f=$1 point mode p nps np
    for point in "${POINTS[@]}"; do
        read -r mode p nps <<< "$point"
        for np in $nps; do "$f" "$mode" "$p" "$np"; done
    done
}
opts () { [ "$1" = weak ] && echo "--weak --log2p=$2" || echo "--log2p=$2"; }

timed () { # Float32, every Dual, Float32 again, then the halo timers
    local id="$1$2_np$3" n
    run "${id}_f32_a" "$3" 0 0 $(opts "$1" "$2")
    for n in $PARTIALS; do run "${id}_dual${n}" "$3" "$n" 0 $(opts "$1" "$2"); done
    run "${id}_f32_b" "$3" 0 0 $(opts "$1" "$2")
    if [ "$LAPS" = 1 ] && [ "$3" -gt 1 ]; then
        run "${id}_f32_laps" "$3" 0 0 $(opts "$1" "$2") --laps
        run "${id}_dual${FIRST}_laps" "$3" "$FIRST" 0 $(opts "$1" "$2") --laps
    fi
}
more () { # the pressure tolerance and more partials at the weak points, and device-to-device halos
    local id="$1$2_np$3" n
    if [ "$1" = weak ] && [ "$2" -ge 8 ] && [ -n "$TOL" ]; then
        run "${id}_f32_tol" "$3" 0 0 $(opts "$1" "$2") --tol="$TOL"
        run "${id}_dual${FIRST}_tol" "$3" "$FIRST" 0 $(opts "$1" "$2") --tol="$TOL"
    fi
    if [ "$1" = weak ]; then for n in $MORE; do run "${id}_dual${n}" "$3" "$n" 0 $(opts "$1" "$2"); done; fi
    if [ "$DEVICE" = 1 ] && [ "$3" -gt 1 ]; then
        run "${id}_f32_device" "$3" 0 1 $(opts "$1" "$2")
        run "${id}_dual${FIRST}_device" "$3" "$FIRST" 1 $(opts "$1" "$2")
    fi
}
each timed
each more

echo; julia mpi/collect.jl --table --out="$DIR/runs.csv" "$DIR"
