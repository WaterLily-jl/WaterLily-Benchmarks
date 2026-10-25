# Strong and weak scaling, and correctness checks, of WaterLily's MPI extension (WaterLily.jl branch mpi-igg)
# on CPUs and GPUs. Every rank runs this script; it detects an MPI launch (mpiexec/srun) and runs serial otherwise.
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
#   --steps=n [100] --warmup=n [20]  timed and untimed steps
#   --check=k           [0]   print and save the global diagnostics every k steps (0: timing only)
#   --out=file.csv            where rank 0 writes the diagnostics of --check
#   --serial                  run serial even under an MPI launcher (a reference run inside a Slurm job)
#
# Rank 0 prints one line per run, which mpi/plot_strong.jl and mpi/plot_weak.jl read:
#   [ Info: MPI np=8  case=tgv  grid=256x256x256  local=128x128x128  100 steps  per_step=12.3ms  ...
using StaticArrays, Printf

arg(key, default) = (i = findlast(a -> startswith(a, "--$key="), ARGS); isnothing(i) ? default : split(ARGS[i], "="; limit=2)[2])
flag(key) = any(==("--$key"), ARGS) || arg(key, "0") in ("1", "true")

const case    = arg("case", "tgv")
const log2p   = parse(Int, arg("log2p", "6"))
const weak    = flag("weak")
const bname   = arg("backend", "Array")
const T       = Dict("Float32" => Float32, "Float64" => Float64)[arg("T", "Float32")]
const Nsteps  = parse(Int, arg("steps", "100"))
const Nwarmup = parse(Int, arg("warmup", "20"))
const check   = parse(Int, arg("check", "0"))
const outfile = arg("out", "")
case in ("tgv", "sphere") || error("unknown --case=$case: use tgv or sphere")
bname in ("Array", "CuArray") || error("unknown --backend=$bname: use Array or CuArray")

# Size of the case: global for strong scaling, per rank for weak scaling
const dims = let d = arg("dims", "")
    !isempty(d) ? Tuple(parse.(Int, split(d, ","))) :
    case == "tgv" ? ntuple(_ -> 2^log2p, 3) : (16, 6, 6) .* 2^log2p
end

# srun --mpi=pmix sets PMIX_RANK/SLURM_PROCID, mpiexec sets PMI_RANK (MPICH) or OMPI_COMM_WORLD_RANK (OpenMPI)
const use_mpi = !flag("serial") && any(k -> haskey(ENV, k),
    ("PMI_RANK", "PMI_SIZE", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALRANKID", "PMIX_RANK", "SLURM_PROCID"))

# The GPU package is loaded before init_waterlily_mpi, so that ImplicitGlobalGrid selects one GPU per node-local rank
bname == "CuArray" && (using CUDA; CUDA.allowscalar(false))
const mem = bname == "CuArray" ? CuArray : Array
using WaterLily
bname == "CuArray" && WaterLily.backend != "KernelAbstractions" &&
    error("GPU runs need the KernelAbstractions backend, this environment has \"$(WaterLily.backend)\": julia --project=mpi mpi/setup.jl --backend=KernelAbstractions")
perdir = case == "tgv" ? (1, 2, 3) : ()
if use_mpi
    using ImplicitGlobalGrid, MPI
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

# ── Cases (as in src/cases.jl, sized by the global grid) ───────────────────────────────────────────────
function tgv(dims, gdims; Re=1600)
    # strong: one period over the global box (tgv_periodic); weak: one period per rank block, so the flow and
    # the work of every rank stay the same as the number of ranks grows
    L = weak ? dims[1] : gdims[1]
    U = T(1); κ = T(2π/L); ν = T(1/(κ*Re))
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
sim = case == "tgv" ? tgv(local_dims, gdims) : sphere(local_dims, gdims)

# ── Global diagnostics, reduced over all ranks ─────────────────────────────────────────────────────────
gsum(a)  = WaterLily.global_allreduce(Float64(sum(a)))
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
rows = NamedTuple[]
record!(step) = (check > 0 && step % check == 0) && push!(rows, (; step, diagnostics(sim)...))

# ── Header ─────────────────────────────────────────────────────────────────────────────────────────────
wl_commit = try readchomp(`git -C $(pkgdir(WaterLily)) log -1 --format="%h %s"`) catch; "unknown" end
devices = bname == "CuArray" ? "$(CUDA.name(CUDA.device())) [uuid $(string(CUDA.uuid(CUDA.device()))[1:8])]" : "CPU"
devices = use_mpi ? MPI.gather(devices, comm) : [devices]
if me == 0
    println("WaterLily $(pkgversion(WaterLily)) ($wl_commit), Julia $VERSION, backend $(WaterLily.backend), $bname, $T")
    if use_mpi
        println("MPI: $(first(split(MPI.Get_library_version(), '\n'))), has_cuda=$(MPI.has_cuda()), IGG_CUDAAWARE_MPI=$(get(ENV, "IGG_CUDAAWARE_MPI", "unset"))")
        bname == "CuArray" && length(unique(devices)) < nranks && @warn "$(length(unique(devices))) distinct GPUs for $nranks ranks"
    end
    foreach(((r, d),) -> println("  rank $(r-1): $d"), enumerate(devices))
    flush(stdout)
end

# ── Run ────────────────────────────────────────────────────────────────────────────────────────────────
sync() = bname == "CuArray" && CUDA.synchronize()
barrier() = (sync(); use_mpi && MPI.Barrier(comm))
for s in 1:Nwarmup; sim_step!(sim; remeasure=false); record!(s); end
GC.gc(); barrier()
t = @elapsed begin
    for s in Nwarmup+1:Nwarmup+Nsteps; sim_step!(sim; remeasure=false); record!(s); end
    barrier()
end

if me == 0
    ms = t/Nsteps*1e3
    iters = sim.pois.n[end-Nsteps+1:end]
    prefix = use_mpi ? "MPI np=$nranks" : "SERIAL"
    @info "$prefix  case=$case$(weak ? "  mode=weak" : "")  grid=$(join(gdims, "x"))  local=$(join(local_dims, "x"))  " *
          "$Nsteps steps  per_step=$(round(ms, digits=2))ms  levels=$(length(sim.pois.levels))  " *
          "mean_iters=$(round(sum(iters)/length(iters), digits=1))  cost=$(@sprintf("%.3f", ms*1e6/prod(gdims)))ns/DOF/dt  backend=$bname"
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
use_mpi && finalize_global_grid()
