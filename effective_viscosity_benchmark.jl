# Benchmark: variable effective-viscosity access patterns in `conv_diff!`.
#
# WaterLily's `conv_diff!` takes ν as a scalar OR a callable ν(I) (the
# on-the-fly variable-viscosity hook). A downstream closure can either
#   (a) wrap a PRECOMPUTED per-cell array  (ν = I -> νₑ[I]), refreshed
#       once per step, or
#   (b) compute ν ON THE FLY from a primitive field, with no stored νₑ
#       array and no refresh — e.g. VoF: ν = μ(α)/ρ(α).
#
# This isolates the cost of each in the diffusive flux. The point is the
# CPU-vs-GPU discrepancy:
#   * CPU: ν(I) is evaluated ~2·D times per cell per conv_diff! (once as
#     `I`, once as `I-δ`, per direction). For a divide-heavy blend like
#     μ/ρ that recompute is NOT latency-hidden, so (b) costs more than
#     (a) + the cheap refresh pass.
#   * GPU: bandwidth-bound with massive latency hiding, so the extra
#     arithmetic in (b) is expected to be ~free — and (b) saves a full
#     grid-sized array plus the refresh pass.
#
# Run:
#   julia --project effective_viscosity_benchmark.jl            # CPU
#   julia --project effective_viscosity_benchmark.jl CuArray    # NVIDIA
#   julia --project effective_viscosity_benchmark.jl ROCArray   # AMD
#
# Requires a WaterLily with the callable-ν `conv_diff!` (PR #291).

using WaterLily, BenchmarkTools, StaticArrays, Printf

backend = Array
if !isempty(ARGS)
    ARGS[1] == "CuArray"  && (using CUDA;   global backend = CuArray)
    ARGS[1] == "ROCArray" && (using AMDGPU; global backend = ROCArray)
end
bstr = backend === Array ? "CPUx" * string(Threads.nthreads()) : string(nameof(backend))

# device synchronize so GPU timings are real (no-op on CPU)
sync!() = nothing
if backend !== Array
    if string(nameof(backend)) == "CuArray"
        sync!() = CUDA.synchronize()
    else
        sync!() = AMDGPU.synchronize()
    end
end

# water/air μ/ρ blend (a representative divide-heavy effective viscosity)
const ρw, ρa, μw, μa = 1f3, 1f0, 1f-3, 1.8f-5
@inline blendν(a) = (a * μw + (1 - a) * μa) / (a * ρw + (1 - a) * ρa)

function setup(p, D, backend)
    N = ntuple(_ -> 2^p, D); Ng = N .+ 2
    uc = zeros(Float32, Ng..., D)
    for I in CartesianIndices(uc)        # smooth, divergence-ish velocity
        i = I.I[end]; x = I.I[1:D]
        uc[I] = Float32(sinpi(x[1] / 2^p) + (D > 1 ? cospi(x[2] / 2^p) : 0) + 0.1i)
    end
    αc = zeros(Float32, Ng...)
    for I in CartesianIndices(αc)        # slanted water/air interface
        αc[I] = (I.I[2] < 2^p / 2 + 0.2 * I.I[1]) ? 1f0 : 0f0
    end
    u = uc |> backend
    α = αc |> backend
    νarr = (blendν.(αc)) |> backend
    f = zero(u); Φ = zeros(Float32, Ng...) |> backend
    return (; u, f, Φ, α, νarr, N)
end

cd!(s, ν) = WaterLily.conv_diff!(s.f, s.u, s.Φ, WaterLily.quick; ν)
refresh!(s) = (s.νarr .= blendν.(s.α))   # the per-step cost the array path pays

println("effective-viscosity conv_diff! benchmark — backend=$bstr")
@printf "%-5s %12s %12s %12s | %14s %14s\n" "log2p" "scalar(µs)" "array(µs)" "onfly(µs)" "onfly/array" "perstep o/a"
for p in (5, 6)
    s = setup(p, 3, backend)
    νs = blendν(0.5f0)
    νa = let v = s.νarr; I -> @inbounds v[I]; end          # (a) closure over precomputed array
    νf = let a = s.α;    I -> @inbounds blendν(a[I]); end   # (b) on the fly, no array
    # correctness: (a) and (b) are the same physics
    cd!(s, νa); fa = Array(copy(s.f)); cd!(s, νf); ff = Array(copy(s.f))
    @assert maximum(abs.(fa .- ff)) ≤ 1f-6 * maximum(abs.(fa)) "array vs on-the-fly mismatch"
    ts = @belapsed (cd!($s, $νs); sync!())
    ta = @belapsed (cd!($s, $νa); sync!())
    tf = @belapsed (cd!($s, $νf); sync!())
    tr = @belapsed (refresh!($s); sync!())
    # per-step amortized: array path = 2·conv_diff + 1 refresh; on-the-fly = 2·conv_diff
    perstep = (2tf) / (2ta + tr)
    @printf "%-5d %12.2f %12.2f %12.2f | %14.3f %14.3f\n" p ts*1e6 ta*1e6 tf*1e6 (tf/ta) perstep
end
