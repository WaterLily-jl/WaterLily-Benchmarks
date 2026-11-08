#!/bin/bash
# Run a command under Nsight Systems on some MPI ranks and plainly on the others, so that profiling a few ranks
# does not slow all of them down. Every rank runs this script (srun or mpiexec):
#   NSYS_RANKS  the ranks to profile, separated by spaces [0]
#   NSYS_OUT    the path of the reports, to which _<rank> is appended
#   NSYS_ARGS   the options of `nsys profile`
# The other ranks run the command without the --profile of mpi/scaling.jl, which would run CUDA.jl's own profiler.
#
#   NSYS_RANKS="0 4" NSYS_OUT=out/tgv NSYS_ARGS="--trace=cuda,nvtx" srun -n 8 mpi/fused/jobs/nsys_ranks.sh julia ...
rank="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}}"
if [[ " ${NSYS_RANKS:-0} " == *" $rank "* ]]; then
    exec nsys profile $NSYS_ARGS --output="${NSYS_OUT:?set NSYS_OUT}_$rank" "$@"
else
    args=(); for a in "$@"; do [[ $a == --profile ]] || args+=("$a"); done
    exec "${args[@]}"
fi
