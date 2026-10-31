#!/bin/bash
#SBATCH --job-name=wl-fused-nsys
#SBATCH --partition=booster
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4  # with --gpus-per-task, or Slurm rejects the GRES it adds by default
#SBATCH --time=00:45:00
#SBATCH --output=wl-fused-nsys-%j.out

# Why the halos of mpi-igg-fused are slower device to device (WATERLILY_GPU_AWARE_MPI=1) than staged through host
# memory, the small ones most (job 2583225, TGV 512^3 on 8 GPUs of 2 nodes: 37.0 against 32.1 ms per step, 33.5 with
# halos of at least 1 MiB device to device), traced with Nsight Systems:
#   1. the ping-pong of mpi/cuda_mpi_test.jl, device and host buffers of 8 B to 64 MB, between two GPUs of one node
#      and of two nodes, with every CUDA API call (--cuda-trace-all-apis, pointer queries included);
#   2. TGV 512^3 on 8 GPUs, staged and device to device, 5 timed steps, each halo and its phases marked
#      (mpi/fused/halo_diag.jl, DIAG_NVTX=1), on ranks 0 and 4 (one per node): with the default CUDA API trace for the
#      times, then with every call for the counts;
#   3. the ping-pong between two nodes again, with UCX's API calls (--trace=ucx) too.
# UCX_PROTO_INFO=y makes UCX print the protocol it picks for each memory type, message size and transport.
# The explanation to test: for every message on a device buffer the MPI library queries the pointer and then, between
# nodes, copies the buffer through host memory itself (no GPUDirect RDMA, UCX_CUDA_COPY_DMABUF=no), inside a node
# over NVLink (cuda_ipc), synchronizing a stream for each copy; staged, it sends the host memory the pack kernel wrote
# and copies nothing. Refuted if the device run shows neither more CUDA calls nor memory copies than the staged one in
# post, wait_recv and wait_send: then the cost is the MPI library's own (a rendezvous handshake, say).
# The time per step that scaling.jl prints is not one under nsys (it includes the start of the capture): the times are
# those of the NVTX ranges. Each run writes to $OUT (default mpi/fused/data/nsys_<job id>/, not tracked), the reports
# and their SQLite exports as <run>_<rank>.nsys-rep and .sqlite; read them with
#
#   python3 -I mpi/fused/nsys_halos.py --steps=5 $OUT/tgv_device_0.sqlite $OUT/tgv_device_4.sqlite
#   nsys stats --report nvtx_sum,cuda_api_sum,cuda_gpu_mem_size_sum $OUT/pingpong_nodes_0.sqlite
#
# Submit from the repository root, with mpi/ and mpi/fused built as for mpi/fused/jobs/jupiter_fused_diag.sh:
#
#   sbatch -A <budget> mpi/fused/jobs/jupiter_fused_nsys.sh

cd "$SLURM_SUBMIT_DIR"
ulimit -c 0  # no core dumps: one crashed rank can write tens of GB
source mpi/jobs/jupiter_env.sh  # before set -u: module functions may read unset variables
module load Nsight-Systems
set -uo pipefail
command -v nsys >/dev/null || { echo "no nsys after module load Nsight-Systems"; exit 1; }
nsys --version
export JULIA_CUDA_NSYS="$(command -v nsys)"  # CUDA.@profile starts the capture through it
OUT="${OUT:-mpi/fused/data/nsys_$SLURM_JOB_ID}"; mkdir -p "$OUT"
# precompile on one rank, so that the runs do not all compile at once
srun -N 1 -n 1 julia --project=mpi/fused -e 'using Pkg; Pkg.precompile(); import WaterLily as W
    isdefined(W, :_device_selectors) || error("WaterLily in mpi/fused is not on branch mpi-igg-fused")' || exit 1
srun -N 1 -n 1 julia --project=mpi -e 'using Pkg; Pkg.precompile()' || exit 1

NSYS="--trace=cuda,nvtx --sample=none --cpuctxsw=none --export=sqlite --force-overwrite=true"
ALL=""  # every CUDA API call, if this nsys has the option (grep reads all of the help: with -q, nsys may get SIGPIPE)
nsys profile --help 2>&1 | grep -- --cuda-trace-all-apis >/dev/null && ALL="--cuda-trace-all-apis=true"
CAPTURE="--capture-range=cudaProfilerApi --capture-range-end=stop"  # the timed steps (scaling.jl --profile)
echo "nsys profile $NSYS, all APIs: ${ALL:-not available}"
export UCX_PROTO_INFO=y DIAG_NVTX=1

# trace NAME NODES RANKS TRACED VARIABLES OPTIONS COMMAND...: one run on RANKS ranks of NODES nodes, the ranks TRACED
# (separated by spaces) under `nsys profile OPTIONS`, with the comma-separated VARIABLES set
trace() {
    local name=$1 nodes=$2 ranks=$3 traced=$4 vars=$5 opts=$6; shift 6
    env ${vars//,/ } NSYS_RANKS="$traced" NSYS_OUT="$OUT/$name" NSYS_ARGS="$opts" \
        srun -N "$nodes" -n "$ranks" mpi/fused/jobs/nsys_ranks.sh "$@" > "$OUT/$name.txt" 2>&1 || echo "[ FAILED: $name ]"
}
TGV="mpi/fused/halo_diag.jl --case=tgv --backend=CuArray --log2p=9 --steps=5 --warmup=20 --profile"
trace pingpong_node      1 2 "0 1" "" "$NSYS $ALL" julia --project=mpi mpi/cuda_mpi_test.jl
trace pingpong_nodes     2 2 "0 1" "" "$NSYS $ALL" julia --project=mpi mpi/cuda_mpi_test.jl
trace tgv_staged         2 8 "0 4" WATERLILY_GPU_AWARE_MPI=0 "$NSYS $CAPTURE" julia --project=mpi/fused -t 4 $TGV
trace tgv_device         2 8 "0 4" WATERLILY_GPU_AWARE_MPI=1 "$NSYS $CAPTURE" julia --project=mpi/fused -t 4 $TGV
trace tgv_staged_all     2 8 "0 4" WATERLILY_GPU_AWARE_MPI=0 "$NSYS $ALL $CAPTURE" julia --project=mpi/fused -t 4 $TGV
trace tgv_device_all     2 8 "0 4" WATERLILY_GPU_AWARE_MPI=1 "$NSYS $ALL $CAPTURE" julia --project=mpi/fused -t 4 $TGV
trace pingpong_nodes_ucx 2 2 "0 1" "" "${NSYS/--trace=cuda,nvtx/--trace=cuda,nvtx,ucx} $ALL" julia --project=mpi mpi/cuda_mpi_test.jl
ls -la "$OUT"
