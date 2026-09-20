include("cases.jl")
include("util.jl")

# Advance each case to a developed flow (`develop_time` in util.jl) and save a JLD2 checkpoint, so that
# benchmarks skip the startup transient. `save!` stores host arrays, so a checkpoint loads on any backend.
# Checkpoints live in `checkpoints/` (git-LFS).
function develop_checkpoints(cases, log2p, ftype, backend, bstr; dir="checkpoints/")
    mkpath(dir)
    for (case, ps, ft) in zip(cases, log2p, ftype)
        for n in ps
            tdev = develop_time[case]
            println("Developing $(case) (p=$n, $ft) to tU/L=$(tdev) on $(bstr) ...")
            sim = getf(case)(n, backend; T=ft)
            sim_step!(sim, ft(tdev); remeasure=remeasure_case(case), verbose=false)
            fname = checkpoint_name(case, n, ft)
            save!(fname, sim.flow; dir)
            println("  → $(joinpath(dir, fname))  (reached tU/L=$(round(sim_time(sim), digits=3)))")
            flush(stdout)
        end
    end
end

# Default sizes as DEF_LOG2P in benchmark.sh (plus donut at 5,6), in `all_cases` order
cases, log2p, max_steps, ftype, backend, data_dir = parse_cla(ARGS;
    cases=all_cases, log2p=[(6,7), (3,4), (4,5), (5,6), (5,6)],
    max_steps=fill(25, length(all_cases)), ftype=fill(Float32, length(all_cases)),
    backend=Array, data_dir="checkpoints/"
)
develop_checkpoints(cases, log2p, ftype, backend, backend_str[backend]; dir=data_dir)
