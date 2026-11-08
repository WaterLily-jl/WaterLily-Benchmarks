# Compare the diagnostics that mpi/scaling.jl --check --out=<file>.csv writes against a reference run:
#
#   julia mpi/check.jl reference.csv run1.csv [run2.csv ...]
#
# For each run, prints the largest relative difference of every diagnostic over the steps it shares with the
# reference (`iters`: the number of steps with a different iteration count), or "identical" when the run
# matches the reference bit for bit. Plain Julia, no packages.
using Printf

function read_csv(file)
    lines = readlines(file)
    cols = Symbol.(split(lines[1], ","))
    rows = [parse.(Float64, split(l, ",")) for l in lines[2:end] if !isempty(l)]
    Dict(c => [r[k] for r in rows] for (k, c) in enumerate(cols)), cols
end

length(ARGS) ≥ 2 || error("usage: julia mpi/check.jl reference.csv run.csv [run.csv ...]")
ref, cols = read_csv(ARGS[1])
qty = filter(c -> c ∉ (:step, :iters), cols)
println("reference: $(ARGS[1]) ($(length(ref[:step])) records, steps $(Int(ref[:step][1]))-$(Int(ref[:step][end])))")
@printf("%-40s", "run")
foreach(c -> @printf("%12s", c), qty); @printf("%8s\n", "iters")
for file in ARGS[2:end]
    run, _ = read_csv(file)
    common = intersect(ref[:step], run[:step])
    i, j = indexin(common, ref[:step]), indexin(common, run[:step])
    @printf("%-40s", basename(file))
    if all(c -> ref[c][i] == run[c][j], cols)
        println("identical over $(length(common)) records")
        continue
    end
    for c in qty
        a, b = ref[c][i], run[c][j]
        @printf("%12.2e", maximum(abs.(a .- b) ./ max.(abs.(a), eps())))
    end
    @printf("%8d\n", count(ref[:iters][i] .!= run[:iters][j]))
end
