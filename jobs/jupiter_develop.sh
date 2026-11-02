#!/bin/bash
#SBATCH --job-name=wl-develop
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm-develop-%j.out

# Developed-flow checkpoints (develop.jl) on the 4 GPUs of a JUPITER Booster node, one case size per GPU, all at once:
# by default the sizes of jobs/jupiter_single_device.sh that checkpoints/ lacks (tgv 9, sphere 6, cylinder 6 and 7).
# They go to $CKPT (by default $SCRATCH_e_ext_2025e01_127/bernat/benchmark_checkpoints, field data on scratch), next
# to links to the checkpoints of checkpoints/, and the single-device jobs read them from there (-dev $CKPT). Never
# copy them into checkpoints/ (git-LFS) without deciding to publish them. Submit from the repository root:
#
#   sbatch -A <budget> jobs/jupiter_develop.sh
#   sbatch -A <budget> jobs/jupiter_develop.sh "sphere:6 cylinder:7"   # other case:size pairs, at most 4
#
# Environment: that of jobs/jupiter_single_device.sh (gpu/CUDA, Julia 1.13, prepared on the login node), run offline.
# Time: cylinder 7 (1152×768×256 cells, the cylinder measured every step) to tU/L=100 at L=128, tens of thousands of
# steps at about half a second each, is the longest: hours; the others take less than one.

cd "$SLURM_SUBMIT_DIR"
WL_BASE="${WL_BASE:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd)}"
export WATERLILY_DIR="$WL_BASE/WaterLily.jl-master"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$WL_BASE/julia_depot}"  # as mpi/jobs/jupiter_env.sh
module purge; module load GCC git    # not the CUDA module: see jobs/jupiter_single_device.sh
unset JULIA_CUDA_MEMORY_POOL
export JULIA_PKG_OFFLINE=true JULIAUP_CHANNEL="${JULIAUP_CHANNEL:-1.13}"
[[ $(julia --version) == "julia version 1.13."* ]] || { echo "Julia 1.13 needed, got $(julia --version)" >&2; exit 1; }
echo "WaterLily $(git -C "$WATERLILY_DIR" describe --tags) in $WATERLILY_DIR, $(julia --version)"; nvidia-smi -L

CKPT="${CKPT:-${SCRATCH_e_ext_2025e01_127:?}/bernat/benchmark_checkpoints}"; mkdir -p "$CKPT"
for f in "$PWD"/checkpoints/*.jld2; do [ -e "$CKPT/$(basename "$f")" ] || ln -s "$f" "$CKPT/"; done
printf '[WaterLily]\nbackend = "KernelAbstractions"\n' > gpu/CUDA/LocalPreferences.toml

SIZES=(${1:-tgv:9 sphere:6 cylinder:6 cylinder:7})
(( ${#SIZES[@]} <= 4 )) || { echo "at most 4 case:size pairs (one per GPU)" >&2; exit 1; }
pids=()
for k in "${!SIZES[@]}"; do
    case=${SIZES[$k]%%:*}; p=${SIZES[$k]##*:}
    log="$CKPT/develop_${case}_${p}_$SLURM_JOB_ID.log"
    echo "GPU $k, cores $((72*k))-$((72*k+71)): $case p=$p, log $log"
    CUDA_VISIBLE_DEVICES=$k taskset -c $((72*k))-$((72*k+71)) julia --project=gpu/CUDA --startup-file=no develop.jl \
        --backend=CuArray --cases="[\"$case\"]" --log2p="[($p,)]" --data_dir="$CKPT/" > "$log" 2>&1 &
    pids+=($!)
    sleep 20  # one process precompiles, the others then find the cache
done
fail=0
for k in "${!pids[@]}"; do
    wait ${pids[$k]} || { fail=1; echo "FAILED: ${SIZES[$k]}"; }
    tail -n 3 "$CKPT/develop_${SIZES[$k]%%:*}_${SIZES[$k]##*:}_$SLURM_JOB_ID.log"
done
ls -la "$CKPT"
echo "== $SECONDS s, $([ $fail = 0 ] && echo ok || echo FAILED)"
exit $fail
