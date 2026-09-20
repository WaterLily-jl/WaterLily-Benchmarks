include("cases.jl")
include("util.jl")

const N_RUNS = 5  # runs per benchmark, each timing `max_steps` individual steps

# N_RUNS runs of `s` per-step times. Every run is reset to the developed flow, so all runs time the
# same states. The runs of each case size are appended in order into one Trial, which compare.jl
# reshapes to (s, N_RUNS).
function collect_runs!(group, s, resets, allocs)
    runs = map(1:N_RUNS) do _
        for r in resets; reset_sim!(r.sim, r.fname, r.dir); end   # every case size back to its developed flow
        run(group, samples=s, evals=1, seconds=1e6, gcsample=false, verbose=false)
    end
    merged = runs[1]
    for nk in keys(merged)
        base = merged[nk]["sim_step!"]
        for r in 2:N_RUNS
            append!(base.times, runs[r][nk]["sim_step!"].times)
            append!(base.gctimes, runs[r][nk]["sim_step!"].gctimes)
        end
        # block-averaged allocations instead of the single-step count
        haskey(allocs, nk) && (base.allocs = allocs[nk])
    end
    return merged
end

# Generate benchmarks
function run_benchmarks(cases, log2p, max_steps, ftype, backend, bstr; data_dir="./", developed="")
    for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
        println("Benchmarking: $(case)  ($(N_RUNS) runs × $(s) steps)")
        suite = BenchmarkGroup()
        results = BenchmarkGroup([case, "sim_step!", p, s, ft, bstr, git_hash, string(VERSION)])
        resets = []                  # per-size (sim, fname, dir) to reset before each run
        allocs = Dict{String,Int}()  # per-size block-averaged allocations per step
        add_to_suite!(suite, getf(case); case=case, p=p, s=s, ft=ft, backend=backend, bstr=bstr,
            remeasure = remeasure_case(case), developed=developed, resets=resets, allocs=allocs
        ) # create benchmark
        GC.gc()
        results[bstr] = collect_runs!(suite[bstr], s, resets, allocs) # run!
        fname = "$(case)_$(p...)_$(s)_$(ft)_$(bstr)_$(git_hash)_$VERSION.json"
        BenchmarkTools.save(joinpath(data_dir,fname), results)
    end
end

cases, log2p, max_steps, ftype, backend, data_dir = parse_cla(ARGS;
    cases=["tgv", "jelly"], log2p=[(6,7), (5,6)], max_steps=[25, 25], ftype=[Float32, Float32], backend=Array, data_dir="data/"
)
# `--developed=<dir>`: time from the checkpoints in <dir> (see develop.jl); "" times the transient.
# Exact flag match, since "developed" can also appear inside another value (e.g. a data_dir).
_devi = findfirst(a -> startswith(a, "--developed="), ARGS)
developed = isnothing(_devi) ? "checkpoints" : split(ARGS[_devi], "="; limit=2)[2]

# Generate benchmark data
data_dir = joinpath(data_dir, hostname * "_" * git_hash)
mkpath(data_dir)
run_benchmarks(cases, log2p, max_steps, ftype, backend, backend_str[backend]; data_dir, developed)