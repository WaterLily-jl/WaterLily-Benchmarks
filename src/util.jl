using BenchmarkTools
using Printf
using KernelAbstractions
using JLD2  # WaterLily's save!/load! extension (checkpoints)

iarg(arg) = occursin.(arg, ARGS) |> findfirst
iarg(arg, args) = occursin.(arg, args) |> findfirst
arg_value(arg) = split(ARGS[iarg(arg)], "=")[end]
arg_value(arg, args) = split(args[iarg(arg, args)], "=")[end]
metaparse(x) = eval(Meta.parse(x))
getf(str) = eval(Symbol(replace(str, "-" => "_"))) # case "tgv-periodic" is the function `tgv_periodic`
parsestringlist(x) = filter(!isempty, occursin(',',x) ? split(x,',') : split(x,' '))  .|> x -> filter(x -> !isspace(x), x)

# Developed-flow checkpoints (see develop.jl): one JLD2 file per case, size and float type
checkpoint_name(case, p, ft) = "$(case)_$(p)_$(ft).jld2"
remeasure_case(case) = case in ("cylinder", "cylinder-biotsavart", "jelly-biotsavart")  # moving bodies

function parse_cla(args; cases=["tgv"], log2p=[(6,7)], max_steps=[100], ftype=[Float32], backend=Array, data_dir="data/")
    cases = !isnothing(iarg("cases", args)) ? arg_value("cases", args) |> metaparse : cases
    log2p = !isnothing(iarg("log2p", args)) ? arg_value("log2p", args) |> metaparse : log2p
    max_steps = !isnothing(iarg("max_steps", args)) ? arg_value("max_steps", args) |> metaparse : max_steps
    ftype = !isnothing(iarg("ftype", args)) ? arg_value("ftype", args) |> metaparse : ftype
    backend = !isnothing(iarg("backend", args)) ? arg_value("backend", args) |> x -> eval(Symbol(x)) : backend
    data_dir = !isnothing(iarg("data_dir", args)) ? arg_value("data_dir", args) : data_dir
    return cases, log2p, max_steps, ftype, backend, data_dir
end

macro add_benchmark(args...)
    ex, b, suite, label = args
    return quote
        $suite[$label] = @benchmarkable begin
            $ex
            KernelAbstractions.synchronize($b)
        end
    end |> esc
end

# Warm up until step count and a wall-clock budget are met and force backend sync
function warmup!(sim, ft, remeasure, KA_backend; min_steps=50, seconds=2.0)
    steps = 0; t0 = time()
    while steps < min_steps || (time() - t0) < seconds
        sim_step!(sim, typemax(ft); max_steps=10, verbose=false, remeasure=remeasure)
        KernelAbstractions.synchronize(KA_backend)
        steps += 10
    end
end

# Reset a sim to its checkpoint (in-place load!) and remeasure the body; no-op for the transient (empty dir)
reset_sim!(sim, fname, dir) = !isempty(dir) && (load!(sim.flow; fname, dir); measure!(sim))

# Allocations per step averaged over `s` steps: a single step is not representative when the
# number of V-cycles varies from step to step (e.g. jelly-biotsavart). Deterministic, so one pass is enough.
function block_alloc_count(sim, ft, s, remeasure)
    g0 = Base.gc_num()
    for _ in 1:s
        sim_step!(sim, typemax(ft); max_steps=1, remeasure=remeasure)
    end
    Base.gc_alloc_count(Base.GC_Diff(Base.gc_num(), g0)) ÷ s
end

function add_to_suite!(suite, sim_function; case="", p=(3,4,5), s=100, ft=Float32, backend=Array, bstr="CPU", remeasure=false, developed="", resets=nothing, allocs=nothing)
    suite[bstr] = BenchmarkGroup([bstr])
    for n in p
        # A missing checkpoint is an error, raised before the warm-up
        ckpt = isempty(developed) ? "" : joinpath(developed, checkpoint_name(case, n, ft))
        !isempty(ckpt) && !isfile(ckpt) && error("No developed-flow checkpoint at '$ckpt'. Generate it first " *
            "with `julia --project=. develop.jl` (case=$case, log2p=$n), or pass --developed=\"\" to time the startup transient.")
        sim = sim_function(n, backend; T=ft)
        KA_backend = KernelAbstractions.get_backend(sim.flow.p)
        fname = checkpoint_name(case, n, ft)
        reset_sim!(sim, fname, developed)        # warm up from the developed flow (if any)
        warmup!(sim, ft, remeasure, KA_backend)  # JIT and settle the device clocks
        isnothing(allocs) || (allocs[repr(n)] = block_alloc_count(sim, ft, s, remeasure))
        isnothing(resets) || push!(resets, (sim=sim, fname=fname, dir=developed))
        suite[bstr][repr(n)] = BenchmarkGroup([repr(n)])
        # one sample = one sim_step! (+ sync), so `run(..., samples=s)` times `s` consecutive steps
        @add_benchmark sim_step!($sim, $typemax($ft); max_steps=1, verbose=false, remeasure=$remeasure) $KA_backend suite[bstr][repr(n)] "sim_step!"
    end
end

waterlily_dir = get(ENV, "WATERLILY_DIR", "")
biotsavart_dir = get(ENV, "BIOTSAVART_DIR", "")  # names the BiotSavartBCs refs of -bs runs in compare.jl
short_hash(dir) = read(`git -C $dir rev-parse --short HEAD`, String) |> x -> strip(x, '\n')
git_hash = short_hash(waterlily_dir)
# Name of the git ref at `hash` in the repository `dir` (or `hash` if none). Refs can share a commit, so the pick
# is fixed: local branches, then tags, then remote branches, never `HEAD`, and master/main win a tie.
function find_git_ref(hash; dir=waterlily_dir)
    refs = [split(r, ' ') for r in split(read(`git -C $dir show-ref -d`, String), '\n') if !isempty(r)]
    for prefix in ("refs/heads/", "refs/tags/", "refs/remotes/")
        names = [split(ref[length(prefix)+1:end], '^')[1] for (h, ref) in refs
                 if startswith(h, hash) && startswith(ref, prefix) && !endswith(ref, "/HEAD")]
        prefix == "refs/remotes/" && (names = [split(n, '/'; limit=2)[end] for n in names]) # drop the remote name
        isempty(names) && continue
        return names[something(findfirst(in(("master", "main")), names), 1)]
    end
    return hash
end
is_git_hash(hash) = find_git_ref(hash) == hash
# A -bs run is tagged `<WaterLily hash>+bs<BiotSavartBCs hash>` (benchmark.jl), any other run `<WaterLily hash>`
split_hash(tag) = (h = split(tag, "+bs"; limit=2); (String(h[1]), length(h) == 2 ? String(h[2]) : nothing))
find_bs_ref(hash) = ispath(joinpath(biotsavart_dir, ".git")) ? find_git_ref(hash; dir=biotsavart_dir) : hash
# WaterLily column of a run: the WaterLily ref, and for a -bs run the BiotSavartBCs ref, e.g. "master (bs main)"
function run_ref(tag)
    wl, bs = split_hash(tag)
    return isnothing(bs) ? String(find_git_ref(wl)) : "$(find_git_ref(wl)) (bs $(find_bs_ref(bs)))"
end
# Hashes of a run and the names of their refs, which the --speedup_base values of compare.jl match
function run_hashes(tag)
    wl, bs = split_hash(tag)
    hashes = [wl, find_git_ref(wl)]
    isnothing(bs) || push!(hashes, bs, find_bs_ref(bs))
    return String.(hashes)
end
hostname = gethostname()

# A GPU --backend loads its package (CUDA, AMDGPU) from the environment benchmark.sh makes for it, gpu/CUDA or gpu/AMDGPU
backend_str = Dict(Array => "CPUx"*@sprintf("%.2d", Threads.nthreads()))
backend_arg = isnothing(iarg("--backend=", ARGS)) ? "Array" : arg_value("--backend=", ARGS)
gpu_pkg = Dict("CuArray" => "CUDA", "ROCArray" => "AMDGPU")
haskey(gpu_pkg, backend_arg) && isnothing(Base.find_package(gpu_pkg[backend_arg])) && error("--backend=$(backend_arg) needs " *
    "$(gpu_pkg[backend_arg]): run with --project=$(joinpath(dirname(@__DIR__), "gpu", gpu_pkg[backend_arg])) (made by benchmark.sh)")
backend_arg == "CuArray" && (using CUDA: CuArray, allowscalar; backend_str[CuArray] = "GPU-NVIDIA"; allowscalar(false))
backend_arg == "ROCArray" && (using AMDGPU: ROCArray, allowscalar; backend_str[ROCArray] = "GPU-AMD"; allowscalar(false))

# Plotting packages (Plots, Makie, ...) live in their own environment, plotting/, so that benchmarking (and CI) never
# installs them. Scripts that plot stack it on the load path with this, before `using` them.
function use_plotting_env()
    env = joinpath(dirname(@__DIR__), "plotting")
    any(f -> startswith(f, "Manifest") && endswith(f, ".toml"), readdir(env)) || error(
        "The plotting environment $(env) is not instantiated. Run once: julia --project=$(env) -e 'using Pkg; Pkg.instantiate()'")
    env in LOAD_PATH || push!(LOAD_PATH, env)
end

# Find files utils
using Glob
function rdir(dir, patterns)
    results = String[]
    patterns = [Glob.FilenameMatch("*" * p * "*") for p in patterns]
    for (root, _, files) in walkdir(dir)
        fpaths = joinpath.(root, files)
        length(fpaths) == 0 && continue
        for p in patterns
            push!(results,fpaths[occursin.(Ref(p),fpaths)]...)
        end
    end
    return unique(results)
end

# Benchmark and sizes
all_cases = String["tgv", "tgv-periodic", "sphere", "sphere-biotsavart", "cylinder", "cylinder-biotsavart", "jelly-biotsavart", "donut"]
tests_dets = Dict(
    "tgv" => Dict("size" => (1, 1, 1), "title" => "TGV"),
    "tgv-periodic" => Dict("size" => (1, 1, 1), "title" => "Periodic TGV"),
    "sphere" => Dict("size" => (16, 6, 6), "title" => "Sphere"),
    "sphere-biotsavart" => Dict("size" => (16, 6, 6), "title" => "Sphere, Biot-Savart BCs"),
    "cylinder" => Dict("size" => (9, 6, 2), "title" => "Moving cylinder"),
    "cylinder-biotsavart" => Dict("size" => (9, 6, 2), "title" => "Moving cylinder, Biot-Savart BCs"),
    "donut" => Dict("size" => (2, 1, 1), "title" => "Donut"),
    "jelly-biotsavart" => Dict("size" => (1, 1, 4), "title" => "Jelly, Biot-Savart BCs"),
)

# Time [tU/L] to a developed flow (develop.jl): 10 jelly periods, half the ~20 TU tgv run, 100 for bluff bodies
develop_time = Dict(
    "tgv" => 10.0, "tgv-periodic" => 10.0, "sphere" => 100.0, "sphere-biotsavart" => 100.0, "cylinder" => 100.0,
    "cylinder-biotsavart" => 100.0, "donut" => 100.0, "jelly-biotsavart" => 10*Float64(π),
)

# Sizes of the developed-flow checkpoints, the same as the DEF_LOG2P defaults of benchmark.sh
checkpoint_log2p = Dict(
    "tgv" => (6,7), "tgv-periodic" => (6,7), "sphere" => (3,4), "sphere-biotsavart" => (3,4), "cylinder" => (4,5),
    "cylinder-biotsavart" => (4,5), "donut" => (5,6), "jelly-biotsavart" => (5,6),
)

# CLA of develop.jl and visualize.jl: all cases at their checkpoint sizes in Float32, unless ARGS say otherwise
function parse_checkpoint_cla(args)
    cases = parse_cla(args; cases=all_cases)[1]
    return parse_cla(args; cases, log2p=[checkpoint_log2p[c] for c in cases], ftype=fill(Float32, length(cases)),
                     backend=Array, data_dir="checkpoints/")
end
