# Where ImplicitGlobalGrid's update_halo! of WaterLily-sized arrays spends its time, phase by phase, next to the bare
# MPI exchange of the same faces. Needs the timers of the ImplicitGlobalGrid fork (branch halo-trace) and IGG_TRACE=1:
#
#   IGG_TRACE=1 srun -n 8 julia --project=mpi mpi/halo_trace.jl [N=128] [FIELDS=1] [BACKEND=CuArray] [BARRIER=1]
#
# The ranks split a periodic N^3 box as WaterLily does (one ghost cell per side) and exchange the halos of FIELDS
# arrays (1: a scalar such as the pressure, 3: the velocity) along the split dimensions, 1000 times. BARRIER=1 puts an
# MPI.Barrier before each call, so that the instants the timers record (since the call entry) compare between ranks:
# a rank waits in `wait_recv` for its neighbours' `t_first_isend`, and `t_recv_done` minus the latest `t_first_isend`
# is what the transfer itself took. With CUDA arrays, CUDA.jl's implicit stream synchronization is off, as in
# WaterLily since 12c339c (OWNERSYNC=1 turns it back on). IGG_CUDAAWARE_MPI=1 passes device buffers to MPI.
#
# The bare exchange then times the MPI calls alone, in update_halo!'s order (receives from both neighbours of a
# dimension, sends to both, wait), with buffers of the face sizes (all fields in one message): per dimension and for
# all split dimensions one after another, from plain host arrays, page-locked host arrays (as update_halo! stages GPU
# halos) and device arrays.
using MPI, CUDA, ImplicitGlobalGrid, Printf
MPI.Init()
const comm = MPI.COMM_WORLD
const me, np = MPI.Comm_rank(comm), MPI.Comm_size(comm)
N = parse(Int, get(ARGS, 1, "128")); nfields = parse(Int, get(ARGS, 2, "1")); backend = get(ARGS, 3, "CuArray")
barrier = get(ARGS, 4, "1") != "0"; ownersync = get(ENV, "OWNERSYNC", "0") != "0"
gpu = backend == "CuArray"
isdefined(ImplicitGlobalGrid, :halo_trace) || error("this ImplicitGlobalGrid has no timers: use the fork, branch halo-trace")
mpidims = Tuple(Int.(MPI.Dims_create(np, [0, 0, 0])))
nxyz = N .÷ mpidims .+ 2
init_global_grid(nxyz...; dimx=mpidims[1], dimy=mpidims[2], dimz=mpidims[3], periodx=1, periody=1, periodz=1,
                 overlaps=(2, 2, 2), halowidths=(1, 1, 1), init_MPI=false, select_device=false, quiet=true)
splitdims = Tuple(findall(>(1), collect(mpidims)))
sync() = gpu && CUDA.synchronize()
A = [gpu ? CUDA.rand(Float32, nxyz...) : rand(Float32, nxyz...) for _ in 1:nfields]; sync()
gpu && foreach(a -> CUDA.enable_synchronization!(a, ownersync), A)
calls(n) = for _ in 1:n
    barrier && MPI.Barrier(comm)
    update_halo!(A...; dims=splitdims)
end

calls(50); sync(); MPI.Barrier(comm)
halo_trace_reset!()
ncalls = 1000
t = @elapsed (calls(ncalls); sync(); MPI.Barrier(comm))
if me == 0
    host = gethostname()
    @printf("update_halo! of %d %s field(s) of %s, split %s, %d ranks (rank 0 on %s), IGG_CUDAAWARE_MPI=%s, OWNERSYNC=%d, BARRIER=%d: %.1f us per call (wall time, barriers included)\n",
            nfields, backend, join(nxyz, "x"), join(mpidims, "x"), np, host, get(ENV, "IGG_CUDAAWARE_MPI", ""), ownersync, barrier, 1e6t / ncalls)
end
halo_trace(byrank=np <= 16)

# The bare MPI exchange, in update_halo!'s order: receive from the right and the left neighbour, send to the left and
# the right one, wait for the receives, then for the sends (tag 0, as update_halo!; pairs match by order)
function exchange!(s, r, d)
    lo, hi = ImplicitGlobalGrid.neighbors(d)
    rq = (MPI.Irecv!(r[2], comm; source=hi, tag=0), MPI.Irecv!(r[1], comm; source=lo, tag=0))
    sq = (MPI.Isend(s[1], comm; dest=lo, tag=0), MPI.Isend(s[2], comm; dest=hi, tag=0))
    MPI.Wait(rq[1]); MPI.Wait(rq[2]); MPI.Wait(sq[1]); MPI.Wait(sq[2])
end
function per_call(f, n)  # the slowest rank's mean time per call, barriers excluded (as in the timers)
    for _ in 1:20; f(); end
    MPI.Barrier(comm); t = 0.0
    for _ in 1:n
        barrier && MPI.Barrier(comm)
        t0 = time_ns(); f(); t += (time_ns() - t0) * 1e-3
    end
    MPI.Barrier(comm); MPI.Allreduce(t / n, MPI.MAX, comm)
end
facelen(d) = nfields * prod(nxyz[k] for k in 1:3 if k != d)
kinds = gpu ? ("host", "pinned", "device") : ("host",)
function buffers(kind, d)
    b = [zeros(Float32, facelen(d)) for _ in 1:4]
    kind == "pinned" && foreach(CUDA.pin, b)
    kind == "device" && (b = [CUDA.zeros(Float32, facelen(d)) for _ in 1:4]; sync())
    (b[1:2], b[3:4])
end
me == 0 && @printf("\nbare MPI exchange (%s barrier before each call; the slowest rank's mean), us per call: %s\n", barrier ? "with a" : "without",
                   join(("$k" for k in kinds), ", "))
for d in splitdims
    ts = Float64[]
    for k in kinds
        s, r = buffers(k, d)
        push!(ts, per_call(() -> exchange!(s, r, d), 1000))
    end
    me == 0 && @printf("  dim %d, %d bytes per message: %s\n", d, 4facelen(d), join((@sprintf("%.1f", x) for x in ts), ", "))
end
bufs = Dict(k => Dict(d => buffers(k, d) for d in splitdims) for k in kinds)
tseq = [per_call(() -> foreach(d -> exchange!(bufs[k][d]..., d), splitdims), 1000) for k in kinds]
me == 0 && @printf("  all split dims one after another: %s\n", join((@sprintf("%.1f", x) for x in tseq), ", "))
finalize_global_grid(finalize_MPI=false)
MPI.Finalize()
