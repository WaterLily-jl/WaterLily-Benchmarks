# Run with
# julia --project compare.jl --dir="data" --plot="plots" --patterns=["tgv","sphere","cylinder"] --sort=1
# julia --project compare.jl  --plot="plots" --sort=1 $(find data/ \( -name "tgv*json" -o -name "sphere*json" -o -name "cylinder*json" \) -printf "%T@ %Tc %p\n" | sort -n | awk '{print $7}')

using BenchmarkTools, PrettyTables
include("util.jl")

# Parse CLA and load benchmarks
sort_idx = !isnothing(iarg("sort", ARGS)) ? arg_value("sort", ARGS) |> metaparse : 0
plot_dir = !isnothing(iarg("plot_dir", ARGS)) ? arg_value("plot_dir", ARGS) : nothing
data_dir = !isnothing(iarg("data_dir", ARGS)) ? arg_value("data_dir", ARGS) : "data"
patterns = !isnothing(iarg("patterns", ARGS)) ? arg_value("patterns", ARGS) |> parsepatterns |> metaparse : String["tgv", "jelly"]
benchmarks_list = nothing
if isnothing(iarg("data_dir", ARGS)) && any(split(x, '.')[end] == "json" for x in ARGS)  # passed json files directly
    benchmarks_list = [f for f in ARGS if !any(occursin.(["--sort","--data_dir","--plot_dir"], f))]
elseif !any(split(x, '.')[end] == "json" for x in ARGS) # no json files passed, we rely on --data_dir
    if ispath(data_dir)
        benchmarks_list = rdir(data_dir, patterns)
    else
        @error "--data_dir=$(data_dir) is not a real path."
    end
else
    @error "Cannot pass both --data_dir=$(data_dir) and json files."
end
println("Processing the following benchmarks:")
for f in benchmarks_list
    println("    ", f)
end
benchmarks_all = [BenchmarkTools.load(f)[1] for f in benchmarks_list]

# Separate benchmarks by test case
all_cases = String["tgv", "sphere", "cylinder", "jelly"]
cases_ordered = all_cases[filter(x -> !isnothing(x),[findfirst(x->x==1, contains.(p, all_cases)) for p in patterns])]
length(cases_ordered) == 0 && (cases_ordered = all_cases)
cases_str = [b.tags[1] for b in benchmarks_all] |> unique
benchmarks_all_dict = Dict(Pair{String, Vector{BenchmarkGroup}}(k, []) for k in cases_str)
for b in benchmarks_all
    push!(benchmarks_all_dict[b.tags[1]], b)
end

# Table and plots
!isa(plot_dir, Nothing) &&  mkpath(plot_dir)
for (i, case) in enumerate(cases_ordered)
    benchmarks = benchmarks_all_dict[case]
    # Get backends string vector and assert same case sizes for the different backends
    backends_str = [String.(k)[1] for k in keys.(benchmarks)]
    log2p_str = [String.(keys(benchmarks[i][backend_str])) for (i, backend_str) in enumerate(backends_str)]
    length(unique(log2p_str)) != 1 && @error "Case sizes missmatch."
    log2p_str = sort(log2p_str[1])
    f_test = benchmarks[1].tags[2]
    # Get data for PrettyTables
    header = ["Backend", "WaterLily", "Julia", "Precision", "Allocations", "GC [%]", "Time [s]", "Cost [ns/DOF/dt]", "Speed-up"]
    data, base_speedup = Matrix{Any}(undef, length(benchmarks), length(header)), 1.0
    # plotting_dir := Dict[("WaterLily version", "Julia version", "precision")][backend, log2p, {3}] # times, cost, speedups
    plotting_dict = Dict{NTuple, Array{Float64}}()

    printstyled("Benchmark environment: $case $f_test (max_steps=$(benchmarks[1].tags[4]))\n", bold=true)
    for (k, n) in enumerate(log2p_str)
        printstyled("▶ log2p = $n\n", bold=true)
        for (i, benchmark) in enumerate(benchmarks)
            datap = benchmark[backends_str[i]][n][f_test]
            speedup = i == 1 ? 1.0 : benchmarks[1][backends_str[1]][n][f_test].times[1] / datap.times[1]
            N = prod(tests_dets[case]["size"]) .* 2 .^ (3 .* eval(Meta.parse.(n)))
            cost = datap.times[1] / N / benchmarks[1].tags[4]
            waterlily_ref = String(find_git_ref(benchmark.tags[end-1]))
            data[i, :] .= [backends_str[i], waterlily_ref, benchmark.tags[end], benchmark.tags[end-3],
                datap.allocs, (datap.gctimes[1] / datap.times[1]) * 100.0, datap.times[1] / 1e9, cost, speedup]
            versions_key = (waterlily_ref, benchmark.tags[end], benchmark.tags[end-3])
            backend_idx = findall(x -> x == backends_str[i], unique(backends_str))[1]
            !(versions_key in keys(plotting_dict)) &&
                (plotting_dict[versions_key] = zeros(length(log2p_str), length(unique(backends_str)), 3))
            plotting_dict[versions_key][k, backend_idx, :] = data[i, end-2:end]
        end
        sorted_cond, sorted_idx = 0 < sort_idx <= length(header), nothing
        if sorted_cond
            sorted_idx = sortperm(data[:, sort_idx])
            baseline_idx = findfirst(x->x==1, sorted_idx)
            data .= data[sorted_idx, :]
        end
        hl_base = Highlighter(f=(data, i, j) -> sorted_cond ? i == findfirst(x->x==1, sorted_idx) : i==1,
            crayon=Crayon(foreground=:blue))
        hl_fast = Highlighter(f=(data, i, j) -> i == argmin(data[:, end-1]), crayon=Crayon(foreground=(32,125,56)))
        pretty_table(data; header=header, header_alignment=:c, highlighters=(hl_base, hl_fast), formatters=ft_printf("%.2f", [6,7,8,9]))
    end

    # Plotting each configuration of WaterLily version, Julia version and precision in benchamarks
    if !isa(plot_dir, Nothing)
        # Get cases size
        N = prod(tests_dets[case]["size"]) .* 2 .^ (3 .* eval(Meta.parse.(log2p_str)))
        N_str = (N./1e6) .|> x -> @sprintf("%.2f", x)
        unique_backends_str = unique(backends_str)

        for (k, data_plot) in plotting_dict
            versions_key = join(k, '_')
            # Cost plot
            p_cost = plot()
            for (i, bstr) in enumerate(unique_backends_str)
                scatter!(p_cost, N./1e6, data_plot[:, i, 2], label=unique_backends_str[i], ms=10, ma=1)
            end
            scatter!(p_cost, yaxis=:log10, xaxis=:log10, yminorgrid=true, xminorgrid=true,
                ylims=(1, 1000), xlims=(0.1, 600),
                xlabel="DOF [M]", lw=0, framestyle=:box, grid=:xy, size=(600, 600),
                left_margin=Plots.Measures.Length(:mm, 5), right_margin=Plots.Measures.Length(:mm, 5),
                ylabel="Cost [ns/DOF/dt]", title=tests_dets[case]["title"], legend=:bottomleft
            )
            fancylogscale!(p_cost)
            savefig(p_cost, joinpath(string(@__DIR__), plot_dir, "$(case)_cost_$(versions_key).pdf"))

            # Speedup plot
            groups = repeat(N_str, inner=length(unique_backends_str)) |> CategoricalArray
            levels!(groups, N_str)
            ctg = repeat(unique_backends_str, outer=length(log2p_str)) |> CategoricalArray
            levels!(ctg, unique_backends_str)
            p = annotated_groupedbar(groups, transpose(data_plot[:, :, 1]), ctg;
                series_annotations=vec(transpose(data_plot[:, :, 3])) .|> x -> @sprintf("%d", x) .|> latexstring, bar_width=0.92,
                Dict(:xlabel=>"DOF [M]", :title=>tests_dets[case]["title"],
                    :ylims=>(1e-1, 1e5), :lw=>0, :framestyle=>:box, :yaxis=>:log10, :grid=>true,
                    :color=>reshape(palette([:cyan, :green], length(unique_backends_str))[1:length(unique_backends_str)], (1, length(unique_backends_str))),
                    :size=>(600, 600)
                )...
            )
            plot!(p, ylabel="Time [s]", legend=:topleft, left_margin=Plots.Measures.Length(:mm, 0))
            savefig(p, joinpath(string(@__DIR__), plot_dir, "$(case)_benchmark_$(versions_key).pdf"))
        end
    end
end