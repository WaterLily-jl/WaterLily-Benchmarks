# Where the cost of ForwardDiff.Dual goes as the number of partials N grows (docs/src/mpi.md, "The cost of a partial").
# One process on one device, no MPI; for each N a TGV per variant (the case of mpi/scaling.jl --partials=N), and every
# repetition times all variants one after the other, so that a load on the machine hits them alike:
#   plain   WaterLily as it is
#   inline  the same simulation, GPU kernels compiled with CUDABackend(always_inline=true)
#   full    Duals of another tag, for which this script defines @inline methods of Base.FastMath's operations
#           (forwarded to ForwardDiff's own) and @inline methods of WaterLily's quick and median
#   ref     a Float32 simulation, the same at every N: its times tell whether the machine was busy
#
#   julia --project=mpi -t 4 mpi/ad_cost.jl --n=128 --Ns=0,4,5,6,8 [--variants=plain,inline,full] [--T=Float32]
#         [--backend=CuArray] [--reps=10] [--pieces=step,conv_diff,...] [--kernels]
#
# Output: `ROW N variant piece min median` (ms, backend synchronized), `INV N variant function: calls left` (what
# Julia's optimizer did not inline, before any backend), and with --kernels `KERNEL N variant piece regs= local=
# shared= name` (registers and stack bytes per thread; its hook costs time: do not use the timings of such a run).
using Printf
med(x) = (y = sort(x); m = length(y); isodd(m) ? y[(m+1)÷2] : (y[m÷2] + y[m÷2+1])/2)
arg(k, d) = (i = findfirst(startswith("--$k="), ARGS); isnothing(i) ? d : split(ARGS[i], "="; limit=2)[2])
flag(k) = "--$k" in ARGS
const n = parse(Int, arg("n", "64")); const Ns = parse.(Int, split(arg("Ns", "0,4,8"), ","))
const V = Dict("Float32" => Float32, "Float64" => Float64)[arg("T", "Float32")]
const bname = arg("backend", "CuArray"); const reps = parse(Int, arg("reps", "10"))
const gpu = bname == "CuArray"
const variants = filter(v -> gpu || v != "inline", String.(split(arg("variants", "plain,inline,full"), ",")))
const only_pieces = String.(split(arg("pieces", ""), ","; keepempty=false))
gpu && (using CUDA; CUDA.allowscalar(false))
using WaterLily
import WaterLily: KernelAbstractions as KA
const FD = WaterLily.ForwardDiff
const mem = gpu ? CuArray : Array
const INL = Ref(false)
gpu && @eval KA.get_backend(::CUDA.CuArray) = CUDA.CUDABackend(; always_inline=INL[])
S() = gpu && CUDA.synchronize()

struct Seed end
struct FullSeed end
const FullD = FD.Dual{<:FD.Tag{FullSeed}}
# `full`: @fastmath turns a*b into Base.FastMath.mul_fast(a, b), and for a Dual that is Base's generic method, which
# is not @inline (and promotes a Float32 factor to a Dual first). These methods hand the operation to ForwardDiff's.
for (ff, f) in ((:add_fast, :+), (:sub_fast, :-), (:mul_fast, :*), (:div_fast, :/), (:max_fast, :max), (:min_fast, :min),
                (:eq_fast, :(==)), (:ne_fast, :!=), (:lt_fast, :<), (:le_fast, :<=), (:pow_fast, :^))
    @eval begin
        @inline Base.FastMath.$ff(x::D, y::D) where {D<:FullD} = $f(x, y)
        @inline Base.FastMath.$ff(x::FullD, y::Real) = $f(x, y)
        @inline Base.FastMath.$ff(x::Real, y::FullD) = $f(x, y)
    end
end
for (ff, f) in ((:sub_fast, :-), (:abs_fast, :abs), (:abs2_fast, :abs2), (:sqrt_fast, :sqrt), (:inv_fast, :inv), (:sign_fast, :sign))
    @eval @inline Base.FastMath.$ff(x::FullD) = $f(x)
end
@inline Base.FastMath.pow_fast(x::FullD, ::Val{p}) where {p} = x^p
@eval WaterLily begin
    @inline @fastmath quick(u::D, c::D, d::D) where {D<:Main.FullD} = median((5c+2d-u)/6, c, median(10c-9u, c, d))
    @inline @fastmath median(a::D, b::D, c::D) where {D<:Main.FullD} = max(min(a, b), min(max(a, b), c))
    # loops of mom_project!, CFL and the solver as functions of their own
    probe_div!(a, b) = @inside b.z[I] = div(I, a.u)
    probe_corr!(a, b) = for i ∈ 1:ndims(a.p); @loop a.u[I,i] -= b.L[I,i]*∂(i,I,b.x) over I ∈ inside(b.x); end
    probe_flux!(a) = @inside a.σ[I] = flux_out(I, a.u)
    probe_mult!(l) = @inside l.z[I] = mult(I, l.L, l.D, l.x)
    probe_scale!(l) = @inside l.ϵ[I] = l.r[I]*l.iD[I]
    probe_add!(l) = @inside l.ϵ[I] = l.r[I]+l.x[I]
end

dual(N, seed) = N == 0 ? V : FD.Dual{typeof(FD.Tag(seed, V)),V,N}
function tgv(N, seed=Seed())
    T = dual(N, seed)
    seeded(x, on) = N == 0 ? V(x) : T(V(x), FD.Partials(ntuple(k -> V(on(k)), N)))
    κ = V(2π/n); U = seeded(1, ==(2)); ν = seeded(1/(κ*1600), ==(1))
    A = ntuple(j -> seeded(0, ==(j + 2)), max(N - 2, 0))
    function u0(i, xyz)
        x, y, z = @. xyz*κ
        i == 3 && return 0*U
        u = i == 1 ? -U*sin(x)*cos(y)*cos(z) : U*cos(x)*sin(y)*cos(z)
        for j in eachindex(A)
            m = j + 1; a = @inbounds A[j]
            u += i == 1 ? -a*sin(m*x)*cos(m*y)*cos(m*z) : a*cos(m*x)*sin(m*y)*cos(m*z)
        end
        return u
    end
    Simulation((n, n, n), (0, 0, 0), 1/κ; U, u0, ν, T, mem, perdir=(1, 2, 3))
end

function pieces_of(sim)
    a, b = sim.flow, sim.pois; l = b.levels[1]; c = b.levels[2]
    mask = WaterLily.coarsen_mask(l.x, c.x)
    t1 = sum(a.Δt); t0 = t1 - a.Δt[end]
    ps = [
        "step"       => () -> sim_step!(sim; remeasure=false),
        "copy_u"     => () -> (a.u⁰ .= a.u),
        "fill_f"     => () -> fill!(a.f, 0),
        "conv_diff"  => () -> WaterLily.conv_diff!(a.f, a.u⁰, a.σ, a.λ; ν=a.ν, perdir=a.perdir),
        "BDIM"       => () -> WaterLily.BDIM!(a),
        "BC"         => () -> WaterLily.BC!(a.u, a.uBC, a.exitBC, a.perdir, t1),
        "div"        => () -> WaterLily.probe_div!(a, l),
        "corr"       => () -> WaterLily.probe_corr!(a, l),
        "flux_out"   => () -> WaterLily.probe_flux!(a),
        "max_sigma"  => () -> maximum(a.σ),
        "CFL"        => () -> WaterLily.CFL(a),
        "scale_x"    => () -> (l.x .*= 1),
        "p_scale"    => () -> WaterLily.probe_scale!(l),
        "p_add"      => () -> WaterLily.probe_add!(l),
        "mult"       => () -> WaterLily.probe_mult!(l),
        "residual"   => () -> WaterLily.residual!(l),
        "sum_r"      => () -> WaterLily.global_sum(l.r),
        "L1Linf"     => () -> WaterLily.L₁L∞(l),
        "Jacobi"     => () -> WaterLily.Jacobi!(l),
        "smooth"     => () -> WaterLily.smooth!(l),
        "restrict"   => () -> WaterLily.restrict!(c.r, l.r, mask),
        "prolongate" => () -> WaterLily.prolongate!(l.ϵ, c.x, mask),
        "pin"        => () -> WaterLily.pin_pressure!(l),
        "Vcycle"     => () -> WaterLily.Vcycle!(b),
        "solver"     => () -> WaterLily.solver!(b),
        "predict"    => () -> WaterLily.mom_predict!(a, t0, t1),
        "project"    => () -> WaterLily.mom_project!(a, b, 1, t1),
        "correct"    => () -> WaterLily.mom_correct!(a, t1),
    ]
    isempty(only_pieces) ? ps : filter(p -> p[1] in only_pieces, ps)
end

# --kernels: every kernel compiled, recorded by a method of cufunction defined here
const KERNELS = Dict{Any,Any}()
if gpu && flag("kernels")
    @eval CUDA.CUDACore function cufunction(f::F, tt::Type{TT}; kwargs...) where {F,TT<:Tuple}
        k = invoke(cufunction, Tuple{Any,Any}, f, tt; kwargs...)
        Main.KERNELS[(F, tt, Main.INL[])] = k
        k
    end
end
function kernel_info(k)
    name = replace(string(typeof(k).parameters[1]), r"^typeof\(|\)$" => "")
    m = CUDA.memory(k)
    (; name, regs=CUDA.registers(k), loc=getproperty(m, :local), shared=m.shared)
end
# calls that Julia's optimizer leaves as calls in the convection stencil and in a product (before any backend)
function invokes(N, v)
    N == 0 && return
    T = dual(N, v == "full" ? FullSeed() : Seed()); u = zeros(T, 4, 4, 4, 3)
    left(f, tt) = (ci = first(only(code_typed(f, tt))); [string(s.args[1] isa Core.MethodInstance ? s.args[1].def.name : s.args[1].def.def.name) for s in ci.code if Meta.isexpr(s, :invoke)])
    tally(x) = isempty(x) ? "none" : join(("$(count(==(y), x))× $y" for y in unique(x)), ", ")
    @printf("INV %2d %-6s ϕu: %s | quick: %s | a*b: %s | a/b: %s | x*a: %s\n", N, v,
            tally(left(WaterLily.ϕu, (Int, CartesianIndex{4}, typeof(u), T, typeof(WaterLily.quick)))),
            tally(left(WaterLily.quick, (T, T, T))), tally(left(Base.FastMath.mul_fast, (T, T))),
            tally(left(Base.FastMath.div_fast, (T, T))), tally(left(Base.FastMath.mul_fast, (V, T))))
end

mode!(v) = (INL[] = v == "inline")
simkey(v) = v == "full" ? "full" : "dual"
function build(N)
    sims = Dict{String,Any}()
    for v in variants
        haskey(sims, simkey(v)) || (sims[simkey(v)] = tgv(N, v == "full" ? FullSeed() : Seed()))
    end
    sims
end
const REF = Ref{Any}(nothing)

function probe(N)
    vs = N == 0 ? ["plain"] : variants
    foreach(v -> v == "inline" || invokes(N, v), vs)
    # first pass: compile every piece of every variant, and attribute each kernel to the first piece that compiles it
    sims = build(N)
    for v in vs
        mode!(v); seen = copy(KERNELS)
        for (name, f) in pieces_of(sims[simkey(v)])
            f(); S()
            for (k, kern) in KERNELS
                haskey(seen, k) && continue
                i = kernel_info(kern)
                @printf("KERNEL %2d %-6s %-10s regs=%3d local=%5d shared=%d %s\n", N, v, name, i.regs, i.loc, i.shared, first(i.name, 60))
            end
            seen = copy(KERNELS)
        end
    end
    # fresh simulations for the timings (the first pass ran the pieces on the initial state)
    empty!(sims); GC.gc(); gpu && CUDA.reclaim()
    sims = build(N)
    for v in vs; simkey(v) == v || v == "plain" || continue; mode!("plain"); for _ in 1:10; sim_step!(sims[simkey(v)]; remeasure=false); end; end; S()
    isnothing(REF[]) && (REF[] = tgv(0); mode!("plain"); for _ in 1:10; sim_step!(REF[]; remeasure=false); end; S())
    ps = Dict(v => pieces_of(sims[simkey(v)]) for v in vs); refps = Dict(pieces_of(REF[]))
    state = Dict(k => (copy(s.flow.u), copy(s.flow.p)) for (k, s) in sims)
    time1(f) = (t = time_ns(); f(); S(); (time_ns() - t)/1e6)
    for (i, (name, _)) in enumerate(ps[first(vs)])
        ts = Dict(v => Float64[] for v in vs); tr = Float64[]
        for v in vs; mode!(v); ps[v][i][2](); S(); end  # once untimed
        for _ in 1:reps
            for v in vs; mode!(v); push!(ts[v], time1(ps[v][i][2])); end
            name in ("step", "conv_diff") && (mode!("plain"); push!(tr, time1(refps[name])))
        end
        for v in vs; @printf("ROW %2d %-6s %-10s %10.4f %10.4f\n", N, v, name, minimum(ts[v]), med(ts[v])); end
        isempty(tr) || @printf("ROW %2d %-6s %-10s %10.4f %10.4f\n", N, "ref", name, minimum(tr), med(tr))
        flush(stdout)
        name == "step" || for (k, s) in sims; s.flow.u .= state[k][1]; s.flow.p .= state[k][2]; end; S()
    end
    others = try  # other Julia processes computing on this GPU
        gpu ? filter(l -> !isempty(l) && !startswith(l, "$(getpid()),") && occursin("julia", l),
                     readlines(`nvidia-smi --query-compute-apps=pid,name --format=csv,noheader`)) : String[]
    catch
        String[]
    end
    @printf("INFO %2d sizeof=%d cells=%d others=%d device_MB=%.0f\n", N, sizeof(dual(N, Seed())), n^3, length(others),
            gpu ? (CUDA.total_memory() - CUDA.free_memory())/2^20 : 0)
    empty!(sims); empty!(state); ps = nothing; GC.gc(); gpu && CUDA.reclaim()
end

println("# ad_cost n=$n T=$V backend=$bname ($(WaterLily.backend), $(Threads.nthreads()) threads) variants=$(join(variants, ",")) reps=$reps " *
        "Julia $VERSION WaterLily $(pkgversion(WaterLily)) ForwardDiff $(pkgversion(FD))" * (gpu ? " CUDA.jl $(pkgversion(CUDA)) $(CUDA.name(CUDA.device()))" : ""))
foreach(probe, Ns)
