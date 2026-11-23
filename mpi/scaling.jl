# Strong and weak scaling, and correctness checks, of WaterLily's MPI extension (WaterLily.jl branch mpi-igg) on CPUs
# and GPUs. Since 57443f8 the branch exchanges the halos itself, or with ImplicitGlobalGrid's update_halo! under
# WATERLILY_HALO=igg; up to 12c339c ImplicitGlobalGrid carried them. --project=mpi/fused runs branch mpi-igg-fused, the
# same exchange before it was merged (for the records of mpi/fused/jobs). Every rank runs this script; it detects an
# MPI launch (mpiexec/srun) and runs serial otherwise.
#
#   julia --project=mpi mpi/scaling.jl --case=tgv --log2p=7                           # serial CPU
#   mpiexec -n 4 julia --project=mpi mpi/scaling.jl --case=tgv --log2p=7              # 4 ranks, CPU
#   srun -n 8 julia --project=mpi mpi/scaling.jl --case=tgv --log2p=8 --backend=CuArray # 8 ranks, one GPU each
#
# Options (defaults in brackets):
#   --case=tgv|sphere   [tgv] the cases of src/cases.jl: `tgv_periodic` (triply periodic) and `sphere` (exitBC)
#   --log2p=p           [6]   size as in src/cases.jl: tgv is a (2^p)^3 box, sphere a 16D×6D×6D box with D=2^p
#   --dims=a,b,c              size in cells, instead of --log2p
#   --weak                    weak scaling: the size is per rank and the global grid grows with the number of ranks
#   --backend=Array|CuArray [Array]
#   --T=Float32|Float64 [Float32]
#   --partials=N        [0]   ForwardDiff.Dual numbers of N partials on top of --T (tgv only): partial 1 is d/dν,
#                             partial 2 d/dU (the amplitude of the initial velocity), the others repeat d/dν
#   --Re=x              [1600 tgv, 3700 sphere]  Reynolds number (ν(1±ε) for finite differences: Re/(1±ε))
#   --tol=x --itmx=n          tolerance and largest number of iterations of the pressure solver (WaterLily's by
#                             default: 2e-3, 32). A finite difference needs a converged solve (--tol=1e-9 in Float64):
#                             the solver's stopping test and adaptive relaxation make the result jump with ν otherwise
#   --steps=n [100] --warmup=n [20]  timed and untimed steps
#   --check=k           [0]   print and save the global diagnostics every k steps (0: timing only); with --partials
#                             also their derivatives (dKE_dnu, ...), which mpi/check.jl compares
#   --out=file.csv            where rank 0 writes the diagnostics of --check
#   --serial                  run serial even under an MPI launcher (a reference run inside a Slurm job)
#   --laps                    time the phases of every halo of WaterLily's own exchange in the timed steps (a copy of
#                             its routine with timers): rank 0 prints them by array and their sum per step. The
#                             backend is synchronized around the pack, the overlapped loop and the unpack, so that
#                             each phase holds its own GPU time; LAPS_SYNC=0 leaves the routine's one synchronization,
#                             and the pack then also waits for the kernels launched before the halo
#   --profile                 profile the timed steps with CUDA.@profile (CuArray): rank 0 prints the summary of the
#                             API calls and kernels; under `nsys profile --capture-range=cudaProfilerApi` it marks them
#   --cpuprofile              sample the timed steps with Julia's profiler: rank 0 prints where the main thread spends
#                             its time (self samples, C frames such as MPI and UCX included)
#
# With IGG_TRACE=1 and the ImplicitGlobalGrid fork's timers (branch halo-trace), rank 0 also prints the time per step
# that the timed steps spent in update_halo!, by array size, phase and dimension (ImplicitGlobalGrid.halo_trace), when
# ImplicitGlobalGrid carries the halos (WATERLILY_HALO=igg, or WaterLily up to 12c339c).
#
# Rank 0 prints one line per run, which mpi/plot_strong.jl, mpi/plot_weak.jl and mpi/collect.jl read, and the memory
# of its block: the bytes per cell of the arrays of the simulation and of the halo plans and buffers, the memory in use
# on its GPU (the CUDA context and other processes included) and the peak resident memory of its process:
#   [ Info: MPI np=8  case=tgv  grid=256x256x256  local=128x128x128  100 steps  per_step=12.3ms  ...
#   [ Info: MEMORY  cells=2097152  arrays=143.1B/cell  halos=1.2B/cell  device=310.2MB  host_rss=2100.0MB
#   [ Info: HALO  calls=52.0  wait=..  pack=..  post=..  work=..  wait_recv=..  unpack=..  wait_send=..  halo=8.1ms  (--laps)
using StaticArrays, Printf, LibGit2

arg(key, default) = (i = findlast(a -> startswith(a, "--$key="), ARGS); isnothing(i) ? default : split(ARGS[i], "="; limit=2)[2])
flag(key) = any(==("--$key"), ARGS) || arg(key, "0") in ("1", "true")

const case    = arg("case", "tgv")
const log2p   = parse(Int, arg("log2p", "6"))
const weak    = flag("weak")
const bname   = arg("backend", "Array")
const V       = Dict("Float32" => Float32, "Float64" => Float64)[arg("T", "Float32")]  # the eltype, or its value type
const Npart   = parse(Int, arg("partials", "0"))
const Nsteps  = parse(Int, arg("steps", "100"))
const Nwarmup = parse(Int, arg("warmup", "20"))
const check   = parse(Int, arg("check", "0"))
const outfile = arg("out", "")
const profile = flag("profile")
const cpuprofile = flag("cpuprofile")
const laps    = flag("laps")
const poisson_kwargs = (; (Symbol(k) => parse(F, arg(k, "")) for (k, F) in ("tol" => Float64, "itmx" => Int) if !isempty(arg(k, "")))...)
case in ("tgv", "sphere") || error("unknown --case=$case: use tgv or sphere")
(Npart > 0 && case != "tgv") && error("--partials needs --case=tgv")
const Re = let r = arg("Re", case == "tgv" ? "1600" : "3700"); something(tryparse(Int, r), parse(Float64, r)); end
bname in ("Array", "CuArray") || error("unknown --backend=$bname: use Array or CuArray")
(profile && bname != "CuArray") && error("--profile needs --backend=CuArray")

# Size of the case: global for strong scaling, per rank for weak scaling
const dims = let d = arg("dims", "")
    !isempty(d) ? Tuple(parse.(Int, split(d, ","))) :
    case == "tgv" ? ntuple(_ -> 2^log2p, 3) : (16, 6, 6) .* 2^log2p
end

# srun --mpi=pmix sets PMIX_RANK/SLURM_PROCID, mpiexec sets PMI_RANK (MPICH) or OMPI_COMM_WORLD_RANK (OpenMPI)
const use_mpi = !flag("serial") && any(k -> haskey(ENV, k),
    ("PMI_RANK", "PMI_SIZE", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALRANKID", "PMIX_RANK", "SLURM_PROCID"))

# The GPU package is loaded before init_waterlily_mpi, so that each node-local rank selects its own GPU
bname == "CuArray" && (using CUDA; CUDA.allowscalar(false))
const mem = bname == "CuArray" ? CuArray : Array
using WaterLily
bname == "CuArray" && WaterLily.backend != "KernelAbstractions" &&
    error("GPU runs need the KernelAbstractions backend, this environment has \"$(WaterLily.backend)\": julia --project=mpi mpi/setup.jl --backend=KernelAbstractions")
# Dual numbers from WaterLily's own ForwardDiff, so that the environment needs none
const FD = WaterLily.ForwardDiff
struct Seed end  # their tag
const T = Npart == 0 ? V : FD.Dual{typeof(FD.Tag(Seed(), V)),V,Npart}
const params = ("nu", "U")[1:min(Npart, 2)]  # the parameters of partials 1 and 2
seeded(x, on) = Npart == 0 ? V(x) : T(V(x), FD.Partials(ntuple(k -> V(on(k)), Npart)))  # x with partial k = on(k)
perdir = case == "tgv" ? (1, 2, 3) : ()
if use_mpi
    using MPI
    # ImplicitGlobalGrid: the halo backend of WATERLILY_HALO=igg, or the halos of WaterLily up to 12c339c, whose extension
    # needs it (no HaloBackend there); the environment of branch mpi-igg-fused has none
    !isnothing(Base.identify_package("ImplicitGlobalGrid")) &&
        (!isdefined(WaterLily, :HaloBackend) || get(ENV, "WATERLILY_HALO", "native") == "igg") && @eval using ImplicitGlobalGrid
    MPI.Initialized() || MPI.Init()
    const nranks = MPI.Comm_size(MPI.COMM_WORLD)
    # weak scaling: a balanced topology, which the shape-aware decomposition of init_waterlily_mpi reproduces for cubic blocks
    const gdims = weak ? dims .* Tuple(Int.(MPI.Dims_create(nranks, zeros(Int, length(dims))))) : dims
    const local_dims, me, comm = init_waterlily_mpi(gdims; perdir)
    weak && local_dims != dims && error("weak-scaling topology mismatch: local=$local_dims, requested=$dims")
else
    const nranks, me, comm = 1, 0, nothing
    const gdims = local_dims = dims
end
const igg_grid = use_mpi && @isdefined(ImplicitGlobalGrid) && ImplicitGlobalGrid.grid_is_initialized()  # WaterLily up to 12c339c
const halo = !use_mpi ? "none" : isdefined(WaterLily, :HaloBackend) ?
    (nameof(typeof(WaterLily.par_mode[].halo)) == :IGGHalo ? "igg" : "native") : igg_grid ? "igg" : "native"
const igg = halo == "igg"  # the halos are ImplicitGlobalGrid's

# ── Halo timers (--laps): WaterLily's exchange with the time of each phase ─────────────────────────────
const phases = ("wait", "pack", "post", "work", "wait_recv", "unpack", "wait_send")
const lap_sync = get(ENV, "LAPS_SYNC", "1") != "0"
const lap_log = Dict{Any,Vector{NTuple{7,Float64}}}()  # (size, eltype) of the array => the phases of each call, in us
if laps
    halo == "native" && isdefined(Base.get_extension(WaterLily, :WaterLilyMPIExt), :_exchange!) ||
        error("--laps times WaterLily's own halo exchange (mpi-igg since 57443f8): it needs MPI ranks and not WATERLILY_HALO=igg")
    @eval Base.get_extension(WaterLily, :WaterLilyMPIExt) function _exchange!(work, a, h)
        t0 = time_ns(); Main.lap_sync && _sync(a)  # the kernels launched before the halo: not its cost
        t1 = time_ns(); WaterLily.pack!(h.s, a, h.S); _sync(a)
        t2 = time_ns()
        for (i, p) in enumerate(h.peers)
            MPI.Irecv!(view(h.rh, h.ranges[i]), _comm(), h.rreq[i]; source=p, tag=0)
        end
        for (i, p) in enumerate(h.peers)
            MPI.Isend(view(h.sh, h.ranges[i]), _comm(), h.sreq[i]; dest=p, tag=0)
        end
        t3 = time_ns(); work(); Main.lap_sync && _sync(a)  # the loop that overlaps the messages
        t4 = time_ns(); MPI.Waitall(h.rreq)
        t5 = time_ns(); WaterLily.unpack!(a, h.r, h.R); Main.lap_sync && _sync(a)
        t6 = time_ns(); MPI.Waitall(h.sreq)
        t = (t0, t1, t2, t3, t4, t5, t6, time_ns())
        push!(get!(() -> NTuple{7,Float64}[], Main.lap_log, (size(a), eltype(a))), ntuple(k -> (t[k+1] - t[k])/1e3, 7))
        return nothing
    end
end
# Rank 0 prints, per array, the calls per timed step and the median over the ranks of the mean time of each phase
function print_laps()
    isempty(lap_log) && error("--laps recorded no halo: is a dimension split across the ranks?")
    name(k) = "$(join(k[1], "x")) $(k[2] <: FD.Dual ? "Dual$(FD.npartials(k[2]))" : k[2])"
    ks = sort!(collect(keys(lap_log)), by=k -> (-prod(k[1]), name(k)))  # the same arrays on every rank
    row(v) = [length(v)/Nsteps; [sum(x -> x[j], v)/length(v) for j in 1:7]]
    all_ = MPI.Gather(reduce(vcat, row.(getindex.(Ref(lap_log), ks))), comm; root=0)
    me == 0 || return
    all_ = reshape(all_, 8, length(ks), :); med(x) = sort(vec(x))[(length(x) + 1) ÷ 2]
    println("Halos per timed step: calls, then the median over the ranks of the mean time of each phase per call (us); " *
            "ms/step without `wait` and `work`")
    @printf("%-24s %6s%s %9s\n", "array", "calls", join(lpad.(phases, 10)), "ms/step")
    total = zeros(8); cost = [j ∉ (1, 4) for j in 1:7]  # all phases but `wait` and `work`
    for (i, k) in enumerate(ks)
        c, p = med(all_[1, i, :]), [med(all_[j+1, i, :]) for j in 1:7]
        total .+= [c; c .* p ./ 1e3]
        @printf("%-24s %6.1f%s %9.3f\n", name(k), c, join(lpad.(round.(p, digits=1), 10)), c*sum(p[cost])/1e3)
    end
    @info "HALO  calls=$(round(total[1], digits=1))  " * join(("$ph=$(@sprintf("%.3f", total[j+1]))" for (j, ph) in enumerate(phases)), "  ") *
          "  halo=$(@sprintf("%.3f", sum(total[2:end][cost])))ms  sync=$(Int(lap_sync))"
end

# ── Cases (as in src/cases.jl, sized by the global grid) ───────────────────────────────────────────────
function tgv(dims, gdims; Re=1600)
    # strong: one period over the global box (tgv_periodic); weak: one period per rank block, so the flow and
    # the work of every rank stay the same as the number of ranks grows
    L = weak ? dims[1] : gdims[1]
    κ = V(2π/L); U = seeded(1, ==(2)); ν = seeded(1/(κ*Re), !=(2))
    function u0(i, xyz)
        x, y, z = @. xyz*κ
        i == 1 && return -U*sin(x)*cos(y)*cos(z)
        i == 2 && return  U*cos(x)*sin(y)*cos(z)
        return 0*U
    end
    Simulation(dims, (0, 0, 0), 1/κ; U, u0, ν, T, mem, perdir=(1, 2, 3))
end
function sphere(dims, gdims; Re=3700, U=1)
    D = gdims[1] ÷ 16; ν = U*D/Re
    center = SA{T}[1.5D, 3D, 3D]; radius = T(D/2)
    body = AutoBody((x, t) -> √sum(abs2, x .- center) - radius)
    Simulation(dims, (U, 0, 0), D; U, ν, body, T, mem, exitBC=true)
end
sim = case == "tgv" ? tgv(local_dims, gdims; Re) : sphere(local_dims, gdims; Re)

# ── Global diagnostics, reduced over all ranks ─────────────────────────────────────────────────────────
wide(x) = Float64(x)
wide(x::FD.Dual{Tg,<:Any,N}) where {Tg,N} = convert(FD.Dual{Tg,Float64,N}, x)
gsum(a)  = WaterLily.global_allreduce(wide(sum(a)))
gmax(a)  = WaterLily.global_max(maximum(abs, a))
function diagnostics(sim)
    σ, u, p = sim.flow.σ, sim.flow.u, sim.flow.p   # σ is scratch between time steps
    R = inside(σ); n = prod(gdims)
    @inside σ[I] = WaterLily.ke(I, u)
    KE = gsum(view(σ, R.indices...))/n
    @inside σ[I] = WaterLily.ω_mag(I, u)^2
    enstrophy = gsum(view(σ, R.indices...))/2n
    (t = WaterLily.time(sim.flow), dt = sim.flow.Δt[end], KE, enstrophy,
     umax = gmax(view(u, R.indices..., :)), pmax = gmax(view(p, R.indices...)), iters = Int(sim.pois.n[end]))
end
# A record as plain numbers: the values, ν, then the derivatives of the reduced diagnostics for each seeded parameter
function record(step)
    d = diagnostics(sim)
    ders = [Symbol(:d, k, :_d, p) => FD.partials(d[k], i) for k in (:KE, :enstrophy, :umax, :pmax) for (i, p) in enumerate(params)]
    (; step, map(FD.value, d)..., nu = Float64(FD.value(sim.flow.ν)), ders...)
end
rows = NamedTuple[]
record!(step) = (check > 0 && step % check == 0) && push!(rows, record(step))

# ── Memory of this rank ────────────────────────────────────────────────────────────────────────────────
# The arrays of the simulation, each once (the top Poisson level's L is the flow's μ₀)
arrays(x::AbstractArray{<:Number}) = Any[x]
arrays(x::Union{Tuple,AbstractArray}) = isempty(x) ? Any[] : mapreduce(arrays, vcat, x)
arrays(x::Union{Simulation,WaterLily.AbstractFlow,WaterLily.AbstractPoisson}) = unique(objectid,
    mapreduce(f -> arrays(getfield(x, f)), vcat, fieldnames(typeof(x))))
arrays(x) = Any[]
# The index lists and buffers of WaterLily's halo plans (the staged buffers are host memory mapped into the device)
function halo_bytes()
    X = use_mpi ? Base.get_extension(WaterLily, :WaterLilyMPIExt) : nothing
    (isnothing(X) || !isdefined(X, :_halos)) && return 0
    sum(h -> hasproperty(h, :S) ? sizeof(h.S) + sizeof(h.R) + sizeof(h.s) + sizeof(h.r) : 0, values(X._halos); init=0)
end
# The memory in use on this rank's GPU: all of it, the CUDA context and other processes included
device_bytes() = bname == "CuArray" ? (GC.gc(); CUDA.reclaim(); CUDA.total_memory() - CUDA.free_memory()) : 0

# ── Header ─────────────────────────────────────────────────────────────────────────────────────────────
# The commit of a clone, through Julia's libgit2, which needs no git command on the compute nodes (JUPITER's have one
# only from the git module, which mpi/jobs/jupiter_env.sh loads)
commit(dir) = try
    LibGit2.with(LibGit2.GitRepo(dir)) do repo
        c = LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))
        "$(string(LibGit2.GitHash(c))[1:7]) $(first(split(LibGit2.message(c), '\n')))"
    end
catch
    "unknown"
end
devices = bname == "CuArray" ? "$(CUDA.name(CUDA.device())) [uuid $(string(CUDA.uuid(CUDA.device()))[1:8])]" : "CPU"
devices = use_mpi ? MPI.gather(devices, comm) : [devices]
if me == 0
    println("WaterLily $(pkgversion(WaterLily)) ($(commit(pkgdir(WaterLily)))), WaterLily-Benchmarks $(first(commit(dirname(@__DIR__)), 7)), " *
            "Julia $VERSION, backend $(WaterLily.backend), $bname, $V, partials=$Npart")
    isempty(poisson_kwargs) || println("pressure solver: $poisson_kwargs")
    if use_mpi
        println("MPI: $(first(split(MPI.Get_library_version(), '\n'))), has_cuda=$(MPI.has_cuda()), halo=$halo, " *
                (igg ? "IGG_CUDAAWARE_MPI=$(get(ENV, "IGG_CUDAAWARE_MPI", "unset"))" :
                       "WATERLILY_GPU_AWARE_MPI=$(get(ENV, "WATERLILY_GPU_AWARE_MPI", "unset"))"))
        bname == "CuArray" && length(unique(devices)) < nranks && @warn "$(length(unique(devices))) distinct GPUs for $nranks ranks"
    end
    foreach(((r, d),) -> println("  rank $(r-1): $d"), enumerate(devices))
    flush(stdout)
end

# ── Run ────────────────────────────────────────────────────────────────────────────────────────────────
sync() = bname == "CuArray" && CUDA.synchronize()
barrier() = (sync(); use_mpi && MPI.Barrier(comm))
step!() = isempty(poisson_kwargs) ? sim_step!(sim; remeasure=false) : sim_step!(sim; remeasure=false, poisson_kwargs)
for s in 1:Nwarmup; step!(); record!(s); end
GC.gc(); barrier()
const halo_timers = igg && get(ENV, "IGG_TRACE", "0") != "0" && isdefined(ImplicitGlobalGrid, :halo_trace)
halo_timers && ImplicitGlobalGrid.halo_trace_reset!()
empty!(lap_log)
timed_steps() = (for s in Nwarmup+1:Nwarmup+Nsteps; step!(); record!(s); end; barrier())
prof = nothing  # the CUDA.@profile summary (nothing under an external profiler)
cpuprofile && (using Profile; Profile.init(n=10^7, delay=0.0005))
t = @elapsed if profile
    # under Nsight Systems: start and stop its capture (nsys profile --capture-range=cudaProfilerApi), which CUDA.jl
    # would not detect, since nsys collects nothing before the capture starts
    prof = @eval CUDA.@profile external=$(haskey(ENV, "NSYS_PROFILING_SESSION_ID")) timed_steps()
elseif cpuprofile
    @eval Profile.@profile timed_steps()
else
    timed_steps()
end

if me == 0
    ms = t/Nsteps*1e3
    iters = sim.pois.n[end-Nsteps+1:end]
    prefix = use_mpi ? "MPI np=$nranks" : "SERIAL"
    @info "$prefix  case=$case$(weak ? "  mode=weak" : "")  grid=$(join(gdims, "x"))  local=$(join(local_dims, "x"))  " *
          "$Nsteps steps  per_step=$(round(ms, digits=2))ms  levels=$(length(sim.pois.levels))  " *
          "mean_iters=$(round(sum(iters)/length(iters), digits=1))  cost=$(@sprintf("%.3f", ms*1e6/prod(gdims)))ns/DOF/dt  backend=$bname  " *
          "T=$V  partials=$Npart"
    cells = prod(local_dims); mb(b) = @sprintf("%.1f", b/2^20); per(b) = @sprintf("%.2f", b/cells)
    @info "MEMORY  cells=$cells  arrays=$(per(sum(sizeof, arrays(sim))))B/cell  halos=$(per(halo_bytes()))B/cell  " *
          "device=$(mb(device_bytes()))MB  host_rss=$(mb(Sys.maxrss()))MB"
    isnothing(prof) || show(IOContext(stdout, :displaysize => (10_000, 200)), MIME("text/plain"), prof)
    # main thread: frames of the packages, MPI/UCX and the GC by total samples, then all frames by self samples
    pkgs = r"WaterLily|ImplicitGlobalGrid|@MPI|@CUDA|@GPUArrays|@KernelAbstractions|gc\.c|libmpi|libpscom|libuc[pst]|libcuda"
    for (by, n, keep) in (cpuprofile ? ((:count, 50, l -> occursin(pkgs, l)), (:overhead, 25, l -> true)) : ())
        io = IOBuffer()
        Profile.print(IOContext(io, :displaysize => (10_000, 200)); format=:flat, sortedby=by, C=true,
                      threads=Threads.threadid())
        lines = split(String(take!(io)), '\n')
        total = filter(startswith("Total snapshots"), lines)
        body = filter(l -> keep(l) && !(l in total), lines[3:end])
        println("\nMain thread, frames by $(by == :count ? "total" : "self") samples. ", join(total))
        foreach(println, lines[1:2]); foreach(println, body[max(1, end-n+1):end])
    end
    if check > 0
        cols = keys(first(rows))
        println(join(cols, "  "))
        foreach(r -> println(join((v isa AbstractFloat ? @sprintf("%.10e", v) : string(v) for v in r), "  ")), rows)
        if !isempty(outfile)
            mkpath(dirname(abspath(outfile)))
            open(outfile, "w") do io
                println(io, join(cols, ","))
                foreach(r -> println(io, join((v isa AbstractFloat ? @sprintf("%.16e", v) : string(v) for v in r), ",")), rows)
            end
            println("diagnostics written to $outfile")
        end
    end
end
halo_timers && ImplicitGlobalGrid.halo_trace(per=Nsteps, unit="step")
laps && print_laps()
igg_grid && finalize_global_grid()
