# Run with
# julia --project compare.jl --data_dir="data/benchmark" --plot_dir="plots" --patterns=\["tgv","sphere","cylinder"\] --sort=1
# julia --project compare.jl --plot_dir="plots" --sort=1 $(find data/ \( -name "tgv*json" -o -name "sphere*json" -o -name "cylinder*json" \) -printf "%T@ %Tc %p\n" | sort -n | awk '{print $7}')
# julia --project compare.jl --data_dir="data/benchmark" --plot_dir="plots" --patterns=\["tgv","sphere","cylinder"\] --speedup_base="CPUx01" --sort=9

using BenchmarkTools, PrettyTables
include("util.jl")

# Parse CLA and load benchmarks
speedup_base = !isnothing(iarg("speedup_base", ARGS)) ? arg_value("speedup_base", ARGS) : nothing
backend_color = !isnothing(iarg("backend_color", ARGS)) ? arg_value("backend_color", ARGS) |> Symbol : :lightrainbow
sort_idx = !isnothing(iarg("sort", ARGS)) ? arg_value("sort", ARGS) |> metaparse : 0
plot_dir = !isnothing(iarg("plot_dir", ARGS)) ? arg_value("plot_dir", ARGS) : "plots/benchmark"
data_dir = !isnothing(iarg("data_dir", ARGS)) ? arg_value("data_dir", ARGS) : "data/benchmark"
patterns = !isnothing(iarg("patterns", ARGS)) ? arg_value("patterns", ARGS) |> parsepatterns |> metaparse : [""]
patterns = patterns[end] == "" ? patterns[1:end-1] : patterns
benchmarks_list = nothing
!isnothing(plot_dir) &&  mkpath(plot_dir)
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
cases_str = [b.tags[1] for b in benchmarks_all] |> unique
benchmarks_all_dict = Dict(Pair{String, Vector{BenchmarkGroup}}(k, []) for k in cases_str)
for b in benchmarks_all
    push!(benchmarks_all_dict[b.tags[1]], b)
end

# Separate benchmarks by test case and order them
cases_ordered = all_cases[filter(x -> !isnothing(x),[findfirst(x->x==1, contains.(p, all_cases)) for p in patterns])]
if length(cases_ordered) == 0 # No case order specified, follow order above
    cases_ordered = all_cases[any.(contains.(benchmarks_list, c) for c in all_cases)]
end

# Table and plots
for (i, case) in enumerate(cases_ordered)
    benchmarks = benchmarks_all_dict[case]
    # Get backends string vector and assert same case sizes for the different backends
    backends_str = [String.(k)[1] for k in keys.(benchmarks)]
    speedup_base_idx = findfirst(x->x==speedup_base, backends_str)
    log2p_str = [String.(keys(benchmarks[i][backend_str])) for (i, backend_str) in enumerate(backends_str)]
    length(unique(log2p_str)) != 1 && @error "Case sizes missmatch."
    log2p_str = sort(log2p_str[1])
    f_test = benchmarks[1].tags[2]
    # Get data for PrettyTables
    header = ["Backend", "WaterLily", "Julia", "Precision", "Allocations", "GC [%]", "Time [s]", "Cost [ns/DOF/dt]", "Speed-up"]
    data, base_speedup = Matrix{Any}(undef, length(benchmarks), length(header)), 1.0
    plotting_data = zeros(length(log2p_str), length(unique(backends_str)), 3) # times, cost, speedups

    printstyled("Benchmark environment: $case $f_test (max_steps=$(benchmarks[1].tags[4]))\n", bold=true)
    for (k, n) in enumerate(log2p_str)
        printstyled("▶ log2p = $n\n", bold=true)
        for (i, benchmark) in enumerate(benchmarks)
            datap = benchmark[backends_str[i]][n][f_test]
            if !isnothing(speedup_base)
                speedup = benchmarks[speedup_base_idx][speedup_base][n][f_test].times[1] / datap.times[1]
            else
                speedup = i == 1 ? 1.0 : benchmarks[1][backends_str[1]][n][f_test].times[1] / datap.times[1]
            end
            N = prod(tests_dets[case]["size"]) .* 2 .^ (3 .* eval(Meta.parse.(n)))
            cost = datap.times[1] / N / benchmarks[1].tags[4]
            waterlily_ref = String(find_git_ref(benchmark.tags[end-1]))
            data[i, :] .= [backends_str[i], waterlily_ref, benchmark.tags[end], benchmark.tags[end-3],
                datap.allocs, (datap.gctimes[1] / datap.times[1]) * 100.0, datap.times[1] / 1e9, cost, speedup]
            versions_key = (waterlily_ref, benchmark.tags[end], benchmark.tags[end-3])
            backend_idx = findall(x -> x == backends_str[i], unique(backends_str))[1]
            plotting_data[k, backend_idx, :] .= data[i, end-2:end]
        end
        sorted_cond, sorted_idx = 0 < sort_idx <= length(header), nothing
        if sorted_cond
            sorted_idx = sortperm(data[:, sort_idx])
            baseline_idx = findfirst(x->x==1, sorted_idx)
            data .= data[sorted_idx, :]
        end
        if !isnothing(speedup_base)
            hl_base = Highlighter(f=(data, i, j) -> sorted_cond ? i == findfirst(x->x==speedup_base_idx, sorted_idx) : i==speedup_base_idx,
                crayon=Crayon(foreground=:blue))
        else
            hl_base = Highlighter(f=(data, i, j) -> sorted_cond ? i == findfirst(x->x==1, sorted_idx) : i==1,
                crayon=Crayon(foreground=:blue))
        end
        hl_fast = Highlighter(f=(data, i, j) -> i == argmin(data[:, end-1]), crayon=Crayon(foreground=(32,125,56)))
        pretty_table(data; header=header, header_alignment=:c, highlighters=(hl_base, hl_fast), formatters=ft_printf("%.2f", [6,7,8,9]))
    end

    # Plotting each configuration of WaterLily version, Julia version and precision in benchamarks
    if !isnothing(plot_dir)
        # Get cases size
        N = prod(tests_dets[case]["size"]) .* 2 .^ (3 .* eval(Meta.parse.(log2p_str)))
        N_str = (N./1e6) .|> x -> @sprintf("%.2f", x)
        # Sort plotting data
        unique_backends_str = unique(backends_str)
        backends_sort_idxs = sortperm(unique_backends_str, by=length)
        sort!(unique_backends_str, by=length)

        plotting_keys = Dict{Int, NTuple}(k => Tuple(data[k, 2:4]) for k in axes(data, 1))
        versions_key = unique.(collect(zip(values(plotting_keys)...))) |> x->reduce(vcat, x) |> x->join(x, '_')
        data_plot = plotting_data[:, backends_sort_idxs, :]

        cg = cgrad(backend_color, length(unique_backends_str), categorical=true)
        colors = [c for c in cg.colors]

        p_cost = plot()
        for (i, bstr) in enumerate(unique_backends_str)
            scatter!(p_cost, N./1e6, data_plot[:, i, 2], label=unique_backends_str[i], ms=10, ma=1, color=colors[i])
        end
        scatter!(p_cost, yaxis=:log10, xaxis=:log10, yminorgrid=true, xminorgrid=true,
            ylims=(1, 1000), xlims=(0.1, 600),
            xlabel="DOF [M]", lw=0, framestyle=:box, grid=:xy, size=(600, 600),
            left_margin=Plots.Measures.Length(:mm, 0), # right_margin=Plots.Measures.Length(:mm, 0),
            legend=false #, title=tests_dets[case]["title"]
        )
        i == 1 && scatter!(p_cost, ylabel="Cost [ns/DOF/dt]")
        fancylogscale!(p_cost)
        fig_path = joinpath(string(@__DIR__), plot_dir, "$(case)_cost_$(versions_key).pdf")
        savefig(p_cost, fig_path)
        println("Figure stored in $(fig_path)")

        # Speedup plot
        groups = repeat(N_str, inner=length(unique_backends_str)) |> CategoricalArray
        levels!(groups, N_str)
        ctg = repeat(unique_backends_str, outer=length(log2p_str)) |> CategoricalArray
        levels!(ctg, unique_backends_str)
        p = annotated_groupedbar(groups, transpose(data_plot[:, :, 1]), ctg;
            series_annotations=vec(transpose(data_plot[:, :, 3])) .|> x -> @sprintf("%d", x) .|> latexstring, bar_width=0.92,
            Dict(:xlabel=>"DOF [M]", # :title=>tests_dets[case]["title"],
                :ylims=>(1e-1, 1e5), :lw=>0, :framestyle=>:box, :yaxis=>:log10, :grid=>true,
                :color=>reshape(colors, (1, length(unique_backends_str))),
                :size=>(600, 600)
            )...
        )
        plot!(p, left_margin=Plots.Measures.Length(:mm, 0), legend=false)
        i == 1 && plot!(p, legend=:topleft, background_color_legend = RGBA{Float64}(1, 1, 1, 0.7), ylabel="Time [s]")

        fig_path = joinpath(string(@__DIR__), plot_dir, "$(case)_benchmark_$(versions_key).pdf")
        savefig(p, fig_path)
        println("Figure stored in $(fig_path)")
    end
end