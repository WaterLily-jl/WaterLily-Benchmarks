# Where ImplicitGlobalGrid's update_halo! of WaterLily-sized device arrays spends its time, without WaterLily.
#
#   srun -n 2 julia --project=mpi mpi/halo_profile.jl [N=128] [FIELDS=1] [OWNERSYNC=1] [CONCURRENT=0]
#
# The ranks split a periodic N^3 box as WaterLily does (one ghost cell per side) and exchange the halos of FIELDS
# device arrays (1: a scalar such as the pressure, 3: the velocity) along the split dimensions. OWNERSYNC=0 turns
# off CUDA.jl's implicit synchronization when a kernel uses an array last used on another stream
# (CUDA.enable_synchronization!): ImplicitGlobalGrid packs and unpacks each side on its own stream, so CUDA.jl waits
# for the previous side every time, although ImplicitGlobalGrid synchronizes its streams itself. CONCURRENT=1 exchanges
# each field with the routine of WaterLily's `face_comm!` (the Poisson iterations), which posts the messages of all
# the split dimensions at once instead of one dimension after the other (face ghosts only). Rank 0 prints the
# time per update_halo!, then
#   1. CUDA.@profile of 100 calls: the CUDA API calls (launches, syncs, copies) and the kernels behind them,
#   2. Julia's sampling profiler over about 2 s of calls: where the host spends that time, by line of
#      ImplicitGlobalGrid, MPI.jl and CUDA.jl, and in the MPI, UCX and CUDA libraries (main thread only).
# IGG_CUDAAWARE_MPI=0 stages the halos through host memory instead of passing device buffers to MPI.
using MPI, CUDA, ImplicitGlobalGrid, Printf, Profile, WaterLily
MPI.Init()
const comm = MPI.COMM_WORLD
const me, np = MPI.Comm_rank(comm), MPI.Comm_size(comm)
N = parse(Int, get(ARGS, 1, "128")); nfields = parse(Int, get(ARGS, 2, "1")); ownersync = get(ARGS, 3, "1") != "0"
concurrent = get(ARGS, 4, "0") != "0"
mpidims = Tuple(Int.(MPI.Dims_create(np, [0, 0, 0])))
nxyz = N .÷ mpidims .+ 2
init_global_grid(nxyz...; dimx=mpidims[1], dimy=mpidims[2], dimz=mpidims[3], periodx=1, periody=1, periodz=1,
                 overlaps=(2, 2, 2), halowidths=(1, 1, 1), init_MPI=false, select_device=false, quiet=true)
splitdims = Tuple(findall(>(1), collect(mpidims)))
A = [CUDA.rand(Float32, nxyz...) for _ in 1:nfields]; CUDA.synchronize()
foreach(a -> CUDA.enable_synchronization!(a, ownersync), A)
const face_halo! = Base.get_extension(WaterLily, :WaterLilyMPIExt)._face_halo!
calls(n) = for _ in 1:n
    concurrent ? foreach(a -> face_halo!(ImplicitGlobalGrid.wrap_field(a), splitdims), A) : update_halo!(A...; dims=splitdims)
end

calls(50); CUDA.synchronize(); MPI.Barrier(comm)
t = @elapsed (calls(1000); CUDA.synchronize(); MPI.Barrier(comm))
me == 0 && @printf("update_halo! of %d field(s) of %s, split %s, IGG_CUDAAWARE_MPI=%s, OWNERSYNC=%d, CONCURRENT=%d: %.1f us per call\n",
                   nfields, join(nxyz, "x"), join(mpidims, "x"), get(ENV, "IGG_CUDAAWARE_MPI", ""), ownersync, concurrent, 1e3t)

prof = CUDA.@profile calls(100)
if me == 0
    println("\nCUDA.@profile of 100 calls:")
    show(IOContext(stdout, :displaysize => (10_000, 200)), MIME("text/plain"), prof)
end

# The same number of calls on every rank, or the halo exchanges would not pair up
ncalls = MPI.Allreduce(clamp(round(Int, 2 / (t / 1000)), 100, 20_000), MPI.MIN, comm)
Profile.init(n=10^7, delay=0.001)
MPI.Barrier(comm)
Profile.@profile calls(ncalls)
if me == 0
    libs = r"ImplicitGlobalGrid|@MPI|@CUDA|@GPUCompiler|gc\.c|libmpi|libpscom|libuc[pst]|libcuda"
    for (by, n, keep) in ((:count, 50, l -> occursin(libs, l)), (:overhead, 25, l -> true))
        io = IOBuffer()
        Profile.print(IOContext(io, :displaysize => (10_000, 200)); format=:flat, sortedby=by, C=true,
                      threads=Threads.threadid())
        lines = split(String(take!(io)), '\n')
        total = filter(startswith("Total snapshots"), lines)
        body = filter(l -> keep(l) && !(l in total), lines[3:end])
        println("\n$ncalls calls, main thread, frames by $(by == :count ? "total" : "self") samples. ", join(total))
        foreach(println, lines[1:2]); foreach(println, body[max(1, end-n+1):end])
    end
end
finalize_global_grid(finalize_MPI=false)
MPI.Finalize()
