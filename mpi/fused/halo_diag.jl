# mpi/scaling.jl with the halos of WaterLily branch mpi-igg-fused changed for a diagnostic (the arguments are those of
# scaling.jl; GPU runs only):
#   DIAG_BUFFERS=alloc     [default] the branch's host buffers: cuMemHostAlloc (CUDA.jl's HostMemory)
#   DIAG_BUFFERS=register  a Julia array registered with cuMemHostRegister, as the ImplicitGlobalGrid fork stages them
#   DIAG_LAPS=1            time the phases of every halo (a copy of the branch's _halo! with timers); rank 0 prints
#                          them per timed step, by array size
#   DIAG_NVTX=1            mark every halo and its phases with NVTX ranges (domain "halo", the halo named by the size
#                          and eltype of its array) for Nsight Systems (mpi/fused/jobs/jupiter_fused_nsys.sh)
#
#   srun -n 8 julia --project=mpi/fused -t 4 mpi/fused/halo_diag.jl --case=tgv --backend=CuArray --log2p=9
using CUDA, MPI, WaterLily, Printf
const X = Base.get_extension(WaterLily, :WaterLilyMPIExt)
const buffers = get(ENV, "DIAG_BUFFERS", "alloc")
buffers in ("alloc", "register") || error("DIAG_BUFFERS must be alloc or register, got $buffers")

if buffers == "register"
    const registered = Any[]  # the host arrays and their registrations, kept alive to the end
    function WaterLily._host_buffers(::CuArray, T, n)
        h = zeros(T, max(n, 1))
        r = CUDA.register(CUDA.HostMemory, pointer(h), sizeof(h), CUDA.MEMHOSTREGISTER_DEVICEMAP)
        push!(registered, (h, r))
        (unsafe_wrap(CuArray, convert(CuPtr{T}, r), n), h)
    end
end

const phases = ("pack", "sync", "post", "work", "wait_recv", "unpack", "wait_send")
const laps = Dict{Any,Vector{NTuple{7,Float64}}}()  # (size, eltype) => the phases of each call, in us
get(ENV, "DIAG_LAPS", "0") == get(ENV, "DIAG_NVTX", "0") == "1" && error("DIAG_LAPS and DIAG_NVTX each redefine _halo!: set one")
if get(ENV, "DIAG_LAPS", "0") == "1"
    @eval X function _halo!(work, a, h)
        t0 = time_ns()
        WaterLily.pack!(h.s, a, h.S); t1 = time_ns()
        _sync(a); t2 = time_ns()
        for (i, p) in enumerate(h.peers)
            MPI.Irecv!(view(h.rh, h.ranges[i]), _comm(), h.rreq[i]; source=p, tag=0)
        end
        for (i, p) in enumerate(h.peers)
            MPI.Isend(view(h.sh, h.ranges[i]), _comm(), h.sreq[i]; dest=p, tag=0)
        end
        t3 = time_ns()
        work(); t4 = time_ns()
        MPI.Waitall(h.rreq); t5 = time_ns()
        WaterLily.unpack!(a, h.r, h.R); t6 = time_ns()
        MPI.Waitall(h.sreq); t7 = time_ns()
        t = (t0, t1, t2, t3, t4, t5, t6, t7)
        push!(get!(() -> NTuple{7,Float64}[], Main.laps, (size(a), eltype(a))), ntuple(k -> (t[k+1] - t[k])/1e3, 7))
        return nothing
    end
end
if get(ENV, "DIAG_NVTX", "0") == "1"
    # NVTX.jl comes with CUDA.jl, not as a dependency of the environment; its ranges are no-ops outside a profiler
    const NVTX = Base.loaded_modules[Base.PkgId(Base.UUID("5da4648a-3479-48b8-97b9-01cb529c0a1f"), "NVTX")]
    const halo_domain = NVTX.Domain("halo")
    @eval X function _halo!(work, a, h)
        Main.NVTX.@range "$(join(size(a), "x")) $(eltype(a))" domain=Main.halo_domain begin
            Main.NVTX.@range "pack" domain=Main.halo_domain WaterLily.pack!(h.s, a, h.S)
            Main.NVTX.@range "sync" domain=Main.halo_domain _sync(a)
            Main.NVTX.@range "post" domain=Main.halo_domain begin
                for (i, p) in enumerate(h.peers)
                    MPI.Irecv!(view(h.rh, h.ranges[i]), _comm(), h.rreq[i]; source=p, tag=0)
                end
                for (i, p) in enumerate(h.peers)
                    MPI.Isend(view(h.sh, h.ranges[i]), _comm(), h.sreq[i]; dest=p, tag=0)
                end
            end
            Main.NVTX.@range "work" domain=Main.halo_domain work()
            Main.NVTX.@range "wait_recv" domain=Main.halo_domain MPI.Waitall(h.rreq)
            Main.NVTX.@range "unpack" domain=Main.halo_domain WaterLily.unpack!(a, h.r, h.R)
            Main.NVTX.@range "wait_send" domain=Main.halo_domain MPI.Waitall(h.sreq)
        end
        return nothing
    end
end

include(joinpath(@__DIR__, "..", "scaling.jl"))

me == 0 && println("DIAG_BUFFERS=$buffers")

# The calls of the timed steps: the last of each array's calls, as many as the timed steps make of all steps
function lap_row(v)
    n = max(1, round(Int, length(v)*Nsteps/(Nwarmup + Nsteps))); w = v[end-n+1:end]
    [n/Nsteps; [sum(x -> x[j], w)/n for j in 1:7]]
end
function print_laps()
    ks = sort!(collect(keys(laps)), by=k -> (-prod(k[1]), string(k)))  # the same arrays on every rank
    all_ = MPI.Gather(reduce(vcat, lap_row.(getindex.(Ref(laps), ks))), MPI.COMM_WORLD; root=0)
    me == 0 || return
    all_ = reshape(all_, 8, length(ks), :)
    med(x) = sort(vec(x))[(length(x) + 1) ÷ 2]
    println("\nHalos per timed step: calls, then the median over ranks of the mean time of each phase per call (us)")
    @printf("%-22s %6s%s %9s %9s\n", "array", "calls", join(lpad.(phases, 10)), "us/call", "ms/step")
    total = 0.0
    for (i, k) in enumerate(ks)
        c, p = med(all_[1, i, :]), [med(all_[j+1, i, :]) for j in 1:7]
        total += c*sum(p)/1e3
        @printf("%-22s %6.1f%s %9.1f %9.2f\n", "$(join(k[1], "x")) $(k[2])", c, join(lpad.(round.(p, digits=1), 10)), sum(p), c*sum(p)/1e3)
    end
    @printf("%-22s %6s%s %9s %9.2f\n", "total", "", " "^70, "", total)
end
isempty(laps) || print_laps()
