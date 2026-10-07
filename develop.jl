include("src/cases.jl")
include("src/util.jl")

# Advance each case to a developed flow (`develop_time` in src/util.jl) and save a JLD2 checkpoint, so that
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

cases, log2p, max_steps, ftype, backend, data_dir = parse_checkpoint_cla(ARGS)
develop_checkpoints(cases, log2p, ftype, backend, backend_str[backend]; dir=data_dir)
