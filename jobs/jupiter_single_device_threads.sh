#!/bin/bash
#SBATCH --job-name=wl-single-threads
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-single-threads-%j.out

# The thread counts between jobs/jupiter_single_device_simd.sh (CPUx01) and jobs/jupiter_single_device.sh (CPUx72):
# KernelAbstractions on 2, 4, 8, 16 and 32 PoCL threads of one Grace, with benchmark.sh and WaterLily v1.11, the same
# cases, sizes, checkpoints and repetitions (3 of 5 runs of 25 steps). Each Grace of the node runs a slot of its own,
# a list of runs one after the other, each on the first threads+1 cores of that Grace (taskset; PoCL keeps its
# threads within them, POCL_AFFINITY=1 pins each to one) and in its own directory of
# data/jupiter_single_device_developed, since benchmark.jl rewrites the environments.toml of its directory after
# every case. compare.jl merges files with the same tags as repetitions, wherever they are. Two parts, one node each:
#
#   sbatch -A <budget> --dependency=afterany:<simd job> jobs/jupiter_single_device_threads.sh 1   # 2 threads
#   sbatch -A <budget> --dependency=afterany:<simd job> jobs/jupiter_single_device_threads.sh 2   # 4 to 32 threads
#
# After the SIMD job, which writes the same LocalPreferences.toml with another backend (these runs all want
# KernelAbstractions, so the two parts may overlap). Time: at the per-thread cost of 72 PoCL threads (about 750 ns per
# cell and step, 175 steps per size and repetition), the longest slot of each part takes 9.4 and 8.9 h, cylinder 7
# alone 4.5 h per repetition on 2 threads (so its three repetitions run apart); fewer threads per Grace can only be
# cheaper per thread. Environment as jobs/jupiter_single_device.sh. No -w here: runs at once would collide in the
# checkout, so the job checks that WaterLily.jl-master is at v1.11.0 instead (results carry its commit either way).

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

T="tgv:6,7,8,9" S="sphere:3,4,5,6" C="cylinder:4,5,6,7"
# One slot per Grace, its runs one after the other: threads:repetitions:case:sizes[:tag]
case "${1:?part 1 or 2}" in
    1) SLOTS=("2:3:$T" "2:3:$S" "2:3:cylinder:4,5,6 2:1:cylinder:7:a" "2:1:cylinder:7:b 2:1:cylinder:7:c") ;;
    2) SLOTS=("4:3:$T 4:3:$S" "4:3:$C" "8:3:$T 8:3:$S 8:3:$C" "16:3:$T 16:3:$S 16:3:$C 32:3:$T 32:3:$S 32:3:$C") ;;
    *) echo "part 1 or 2, not $1" >&2; exit 1 ;;
esac

DD=data/jupiter_single_device_developed
slot () {  # slot <grace> <runs...>
    local g=$1 ok=0; shift
    for run in "$@"; do
        IFS=':' read -r t reps c sizes tag <<< "$run"
        local name="t${t}_${c}${tag:+_$tag}" cores="$((72*g))-$((72*g+t))"
        echo "Grace $g, cores $cores: $c $sizes on $t threads, -r $reps, log slurm-single-threads-$SLURM_JOB_ID-$name.log"
        taskset -c $cores ./benchmark.sh -b Array -t $t -u false -dev "$CKPT" -r $reps -f -c "$c" -p "$sizes" \
            -ft Float32 -dd "$DD/pocl_$name" > "slurm-single-threads-$SLURM_JOB_ID-$name.log" 2>&1 ||
            { echo "FAILED: $run (Grace $g)"; ok=1; }
    done
    return $ok
}
pids=()
for g in "${!SLOTS[@]}"; do
    slot $g ${SLOTS[$g]} &
    pids+=($!)
    sleep 30  # benchmark.sh writes LocalPreferences.toml just before each julia: not at the same moment
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
julia --project=. compare.jl --data_dir=$DD --patterns="tgv sphere cylinder" --speedup_base=CPUx01 --sort=1 --markdown
echo "== $SECONDS s, $([ $fail = 0 ] && echo ok || echo FAILED)"
exit $fail
