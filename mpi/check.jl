# Compare the diagnostics that mpi/scaling.jl --check --out=<file>.csv writes against a reference run:
#
#   julia mpi/check.jl reference.csv run1.csv [run2.csv ...]
#   julia mpi/check.jl --fd ad.csv plus.csv minus.csv
#
# For each run, prints the relative difference of every diagnostic over the records it shares with the reference: the
# largest (`max`) and that of the last record (`last`; `iters`: the number of records with a different iteration
# count), or "identical" when the run matches the reference bit for bit. Runs with --partials also hold derivatives
# (dKE_dnu, ...), compared like the other columns. A run is compared in the columns it shares with the reference, so
# the values of a run with partials can be checked against a run without.
#
# --fd checks the derivatives of a run with partials against central finite differences of two runs without, at
# ν(1±ε) (scaling.jl --Re=Re/(1±ε), in Float64): (q₊ - q₋)/(ν₊ - ν₋) for every d<q>_dnu, at the records the three
# share. The runs must make the same Poisson iterations, or the two are not one function of ν. Plain Julia, no packages.
using Printf

function read_csv(file)
    lines = readlines(file)
    cols = Symbol.(split(lines[1], ","))
    rows = [parse.(Float64, split(l, ",")) for l in lines[2:end] if !isempty(l)]
    Dict(c => [r[k] for r in rows] for (k, c) in enumerate(cols)), cols
end
rel(a, b) = abs.(a .- b) ./ max.(abs.(a), eps())

function compare(files)
    ref, cols = read_csv(files[1])
    println("reference: $(files[1]) ($(length(ref[:step])) records, steps $(Int(ref[:step][1]))-$(Int(ref[:step][end])))")
    qty = filter(c -> c ∉ (:step, :iters, :nu), cols)
    @printf("%-34s%5s", "run", ""); foreach(c -> @printf("%15s", c), qty); @printf("%7s\n", "iters")
    for file in files[2:end]
        isfile(file) || (@printf("%-34s missing\n", basename(file)); continue)  # a run that failed
        run, rcols = read_csv(file)
        common = intersect(ref[:step], run[:step])
        i, j = indexin(common, ref[:step]), indexin(common, run[:step])
        shared = filter(in(rcols), cols)
        @printf("%-34s", basename(file))
        if all(c -> ref[c][i] == run[c][j], shared)
            println(" identical over $(length(common)) records" * (shared == cols == rcols ? "" : " in the $(length(shared)) shared columns"))
            continue
        end
        for (k, pick) in enumerate((maximum, last))
            k == 2 && @printf("%-34s", "")
            @printf("%5s", k == 1 ? "max" : "last")
            foreach(c -> c in rcols ? @printf("%15.2e", pick(rel(ref[c][i], run[c][j]))) : @printf("%15s", "-"), qty)
            k == 1 ? @printf("%7d\n", count(ref[:iters][i] .!= run[:iters][j])) : println()
        end
    end
end

function finite_differences(ad_file, plus_file, minus_file)
    (ad, cols), (p, _), (m, _) = read_csv(ad_file), read_csv(plus_file), read_csv(minus_file)
    common = intersect(ad[:step], p[:step], m[:step])
    i, j, k = (indexin(common, r[:step]) for r in (ad, p, m))
    h = p[:nu][j] .- m[:nu][k]
    println("AD: $ad_file\nFD: ($plus_file - $minus_file)/(ν₊ - ν₋), ν = $(ad[:nu][1]), ν₊ - ν₋ = $(h[1]) " *
            "($(@sprintf("%.1e", h[1]/ad[:nu][1])) ν), $(length(common)) records, steps $(Int(common[1]))-$(Int(common[end]))")
    maximum(rel(ad[:nu][i], (p[:nu][j] .+ m[:nu][k]) ./ 2)) < 1e-6 || println("WARNING: the AD run is not at the mean ν of the two")
    n = count((ad[:iters][i] .!= p[:iters][j]) .| (ad[:iters][i] .!= m[:iters][k]))
    n == 0 || println("WARNING: the Poisson iterations differ at $n records: the differences below are not of one function of ν")
    @printf("%-18s%18s%18s%12s%12s\n", "derivative", "AD (last)", "FD (last)", "rel (last)", "rel (max)")
    for c in filter(c -> endswith(string(c), "_dnu"), cols)
        q = Symbol(string(c)[2:end-4]); haskey(p, q) || continue
        fd = (p[q][j] .- m[q][k]) ./ h; e = rel(fd, ad[c][i])
        @printf("%-18s%18.9e%18.9e%12.2e%12.2e\n", c, ad[c][i][end], fd[end], e[end], maximum(e))
    end
end

if !isempty(ARGS) && ARGS[1] == "--fd"
    length(ARGS) == 4 || error("usage: julia mpi/check.jl --fd ad.csv plus.csv minus.csv")
    finite_differences(ARGS[2:4]...)
else
    length(ARGS) ≥ 2 || error("usage: julia mpi/check.jl reference.csv run.csv [run.csv ...]")
    compare(ARGS)
end
