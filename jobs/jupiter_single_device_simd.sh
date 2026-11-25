#!/bin/bash
#SBATCH --job-name=wl-single-simd
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-single-simd-%j.out

# The single-core part of jobs/jupiter_single_device.sh: one Grace core (CPUx01, the SIMD backend) with benchmark.sh
# and WaterLily v1.11, the same cases, sizes, checkpoints and repetitions. At 130 to 350 ns per cell and step, the
# largest sizes (134 to 226 Mcells) take 20 to 80 s per step and 525 steps each (3 repetitions of 50 warm-up steps and
# 5 runs of 25): cylinder 7 alone could take the 12 h limit. So the sweep runs as five benchmark.sh at once, each on a
# core of its own (one per Grace and its memory, two on the second): tgv and cylinder 4 to 6, sphere, and the three
# repetitions of cylinder 7 apart (compare.jl merges files with the same tags as repetitions, wherever they are).
# Each writes its own directory of data/jupiter_single_device_developed, since benchmark.jl rewrites the
# environments.toml of its directory after every case. Submit from the repository root, after the GPU and PoCL job
# (the two write the same LocalPreferences.toml):
#
#   sbatch -A <budget> --dependency=afterany:<single-device job> jobs/jupiter_single_device_simd.sh
#
# Environment as jobs/jupiter_single_device.sh. No -w here: five checkouts at once would collide, so the job checks
# that WaterLily.jl-master is at v1.11.0 instead (results carry its commit either way).

cd "$SLURM_SUBMIT_DIR"
WL_BASE="${WL_BASE:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd)}"
export WATERLILY_DIR="$WL_BASE/WaterLily.jl-master"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$WL_BASE/julia_depot}"  # as mpi/jobs/jupiter_env.sh
module purge; module load GCC git    # git: the compute nodes have none
export JULIA_PKG_OFFLINE=true POCL_AFFINITY=1 JULIAUP_CHANNEL="${JULIAUP_CHANNEL:-1.13}"
[[ $(julia --version) == "julia version 1.13."* ]] || { echo "Julia 1.13 needed, got $(julia --version)" >&2; exit 1; }
[ "$(git -C "$WATERLILY_DIR" describe --tags --exact-match 2>/dev/null)" = v1.11.0 ] ||
    { echo "$WATERLILY_DIR is not at v1.11.0" >&2; exit 1; }
CKPT="${CKPT:-${SCRATCH_e_ext_2025e01_127:?}/bernat/benchmark_checkpoints}"
for f in tgv_{6,7,8,9} sphere_{3,4,5,6} cylinder_{4,5,6,7}; do
    [ -f "$CKPT/${f}_Float32.jld2" ] || { echo "no checkpoint $CKPT/${f}_Float32.jld2 (jobs/jupiter_develop.sh)" >&2; exit 1; }
done
echo "WaterLily v1.11.0 in $WATERLILY_DIR, $(julia --version), checkpoints in $CKPT"

DD=data/jupiter_single_device_developed
# core | repetitions | cases (-c) | sizes (-p) | directory and log name
RUNS=("0|3|tgv cylinder|6,7,8,9 4,5,6|tgv_cylinder456" "72|3|sphere|3,4,5,6|sphere"
      "108|1|cylinder|7|cylinder7_a" "144|1|cylinder|7|cylinder7_b" "216|1|cylinder|7|cylinder7_c")
pids=()
for r in "${RUNS[@]}"; do
    IFS='|' read -r core reps cases sizes tag <<< "$r"
    echo "core $core: $cases $sizes, -r $reps, log slurm-single-simd-$SLURM_JOB_ID-$tag.log"
    taskset -c $core ./benchmark.sh -b Array -t 1 -u false -dev "$CKPT" -r $reps -f -c "$cases" -p "$sizes" -ft Float32 \
        -dd "$DD/simd_$tag" > "slurm-single-simd-$SLURM_JOB_ID-$tag.log" 2>&1 &
    pids+=($!)
    sleep 30  # benchmark.sh writes LocalPreferences.toml (SIMD) just before each julia: not at the same moment
done
fail=0
for k in "${!pids[@]}"; do wait ${pids[$k]} || { fail=1; echo "FAILED: ${RUNS[$k]}"; }; done
julia --project=. compare.jl --data_dir=$DD --patterns="tgv sphere cylinder" --speedup_base=CPUx01 --sort=1 --markdown
echo "== $SECONDS s, $([ $fail = 0 ] && echo ok || echo FAILED)"
exit $fail
