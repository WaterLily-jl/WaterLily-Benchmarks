include("cases.jl")
include("util.jl")

# Generate benchmarks
function run_benchmarks(cases, log2p, max_steps, ftype, backend, bstr; data_dir="./", developed="")
    for (case, p, s, ft) in zip(cases, log2p, max_steps, ftype)
        println("Benchmarking: $(case)")
        suite = BenchmarkGroup()
        results = BenchmarkGroup([case, "sim_step!", p, s, ft, bstr, git_hash, string(VERSION)])
        add_to_suite!(suite, getf(case); case=case, p=p, s=s, ft=ft, backend=backend, bstr=bstr,
            remeasure = remeasure_case(case), developed=developed
        ) # create benchmark
        GC.gc()
        results[bstr] = run(suite[bstr], samples=5, evals=1, seconds=1e6, gcsample=true, verbose=true) # run!
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