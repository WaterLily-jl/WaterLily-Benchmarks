include("cases.jl")
include("util.jl")

const N_RUNS = 5  # number of independent runs; each times `max_steps` individual sim_step!s

# Collect per-step timings as N_RUNS runs of `s` steps each. The benchmarkable is a single
# sim_step! (+ backend sync), so each run's Trial.times holds `s` per-step times. Each `reset!`
# restores its leaf to the developed flow before every run, so all runs time the SAME states
# (the between-run scatter is pure measurement noise). Merge the runs leaf-wise, run-ordered
# (run r occupies samples (r-1)*s+1 : r*s), into one Trial so the existing save/compare pipeline
# still applies; compare.jl reshapes to (s, N_RUNS) and reports reference = min over runs of the
# per-run median, noise = std over runs of that median.
function collect_runs!(group, s, resets, allocs)
    runs = map(1:N_RUNS) do _
        for r in resets; reset_sim!(r.sim, r.fname, r.dir); end   # restore every leaf to its developed flow
        run(group, samples=s, evals=1, seconds=1e6, gcsample=false, verbose=false)
    end
    merged = runs[1]
    for nk in keys(merged)
        base = merged[nk]["sim_step!"]
        for r in 2:N_RUNS
            append!(base.times, runs[r][nk]["sim_step!"].times)
            append!(base.gctimes, runs[r][nk]["sim_step!"].gctimes)
        end
        # replace BenchmarkTools' single-step alloc count with the block average (representative)
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
        resets = []                  # per-size (sim, fname, dir) to restore before each run
        allocs = Dict{String,Int}()  # per-size block-averaged per-step allocation counts
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
# `--developed=<dir>` (default "checkpoints"): time sim_step! from the pre-developed flows in <dir>
# (see develop.jl). A missing checkpoint is a hard error; pass --developed="" to time the transient.
# Match the flag exactly — "developed" can appear inside another value (e.g. a data_dir ".../-developed").
_devi = findfirst(a -> startswith(a, "--developed="), ARGS)
developed = isnothing(_devi) ? "checkpoints" : split(ARGS[_devi], "="; limit=2)[2]

# Generate benchmark data
data_dir = joinpath(data_dir, hostname * "_" * git_hash)
mkpath(data_dir)
run_benchmarks(cases, log2p, max_steps, ftype, backend, backend_str[backend]; data_dir, developed)