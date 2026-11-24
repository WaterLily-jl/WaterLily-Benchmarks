#!/bin/bash
#SBATCH --job-name=wl-ad-cost
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=01:30:00
#SBATCH --output=wl-ad-cost-%j.out

# Why the cost of ForwardDiff.Dual is not linear in the number of partials N (docs/src/mpi.md, "The cost of a
# partial"), on one node, no MPI. mpi/ad_cost.jl times, for each N, a step and each piece of it for WaterLily as it
# is (plain), with every GPU kernel compiled with always_inline (inline) and with @inline fast-math methods for Dual
# numbers defined by the script (full), the three in turn at every repetition, next to a Float32 reference.
# The four GPUs of the node each run their own list, at the same time:
#   0  256^3, N = 0 to 5, then the timed runs of the scaling tests (mpi/scaling.jl, 256^3) for 0, 4, 8 and 12
#      partials as they are and with --fastdual (the same methods, mpi/fastdual.jl)
#   1  256^3, N = 6 to 9
#   2  256^3, N = 10 and 12, then 128^3 up to N = 16
#   3  Float64 at 128^3 (the calls appear at the same N as in Float32, with twice the bytes per element), then the
#      registers and stack of every kernel at 64^3
# The runs write to data/mpi/jupiter_adcost_<job>/ (summary.txt: ms per step per N and variant). Submit from the
# repository root:
#
#   sbatch --reservation=develbooster mpi/jobs/jupiter_ad_cost.sh
#
# Variables: REPS [10], WATERLILY_ENV [mpi], and RUN0 to RUN3 to replace a GPU's list: runs separated by ";", each
# a name, a script and its options.

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
set -uo pipefail
export ENV_DIR="${WATERLILY_ENV:-mpi}" REPS="${REPS:-10}"
export DIR="data/mpi/jupiter_adcost_${SLURM_JOB_ID}"; mkdir -p "$DIR"
C="mpi/ad_cost.jl --reps=$REPS"; T="mpi/scaling.jl --case=tgv --backend=CuArray --log2p=8 --steps=100 --warmup=20 --serial"
export RUN0="${RUN0:-f32_256_a $C --n=256 --Ns=0,1,2,3,4,5;tgv_p0 $T;tgv_p4 $T --partials=4;tgv_p4_fastdual $T --partials=4 --fastdual;tgv_p8 $T --partials=8;tgv_p8_fastdual $T --partials=8 --fastdual;tgv_p12 $T --partials=12;tgv_p12_fastdual $T --partials=12 --fastdual}"
export RUN1="${RUN1:-f32_256_b $C --n=256 --Ns=0,6,7,8,9}"
export RUN2="${RUN2:-f32_256_c $C --n=256 --Ns=0,10,12;f32_128 $C --n=128 --Ns=0,4,5,6,8,9,12,16}"
export RUN3="${RUN3:-f64_128 $C --n=128 --T=Float64 --Ns=0,4,5,6,7,8,9;kernels_64 mpi/ad_cost.jl --n=64 --Ns=0,4,5,6,8,9 --reps=1 --kernels}"
echo "mpi/ad_cost.jl on the 4 GPUs of $(hostname), $REPS repetitions -> $DIR"
srun -N 1 -n 1 julia --project="$ENV_DIR" -e 'using Pkg; Pkg.precompile(); using CUDA, WaterLily
    println("Julia $VERSION, WaterLily $(pkgversion(WaterLily)), CUDA.jl $(pkgversion(CUDA)), ForwardDiff $(pkgversion(WaterLily.ForwardDiff)), ", CUDA.name(CUDA.device()))' || exit 1
srun -N 1 -n 4 bash -c 'v=RUN$SLURM_PROCID; IFS=";" read -ra runs <<< "${!v}"
    for r in "${runs[@]}"; do set -- $r; name=$1; shift
        julia --project="$ENV_DIR" -t 4 "$@" > "$DIR/$name.txt" 2>&1 || echo "[ FAILED: $name ]"
    done'
{ for f in "$DIR"/*.txt; do
      grep -q "^ROW" "$f" || continue
      case "$f" in */kernels*) continue ;; esac  # the kernel listing: its timings carry the hook
      echo; echo "=== $(basename "$f" .txt): $(head -1 "$f")"
      grep -E "ERROR|Error|[Kk]illed" "$f"
      # ms per step (the minimum of the repetitions) per N and variant, and its ratio to this run's N = 0
      awk '$1 == "ROW" && $4 == "step" { t[$2, $3] = $5; N[$2] }
           END { printf "%4s %10s %10s %10s %10s | %s\n", "N", "plain", "inline", "full", "ref", "t(N)/t(0): plain inline full"
                 for (i = 0; i <= 64; i++) if (i in N) printf "%4d %10.2f %10.2f %10.2f %10.2f | %6.2f %6.2f %6.2f\n", i, t[i, "plain"],
                     t[i, "inline"], t[i, "full"], t[i, "ref"], t[i, "plain"]/t[0, "plain"], t[i, "inline"]/t[0, "plain"], t[i, "full"]/t[0, "plain"] }' "$f"
  done
  echo; echo "=== mpi/scaling.jl, TGV 256^3 on one GPU, as it is and with --fastdual"
  for f in "$DIR"/tgv_*.txt; do
      printf "%-18s %s\n" "$(basename "$f" .txt)" "$(grep -h "Info: SERIAL" "$f" | sed 's/.*per_step=/per_step=/; s/  levels.*T=/  T=/' || true)"
      grep -q "per_step=" "$f" || { echo "[ FAILED: $f ]"; grep -hE "ERROR|[Ee]rror|[Kk]illed" "$f" | head -3; }
  done; } 2>&1 | tee "$DIR/summary.txt"
