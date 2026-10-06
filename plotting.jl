# Plotting helpers (Plots) for compare.jl. The plotting packages live in their own environment, plotting/, stacked on
# top of the benchmark one by `use_plotting_env()` (util.jl) so that benchmarking never installs them.
using Plots, StatsPlots, LaTeXStrings, CategoricalArrays, ColorSchemes

fontsize = 20
speedup_fontsize = 16
Plots.default(
    fontfamily = "Computer Modern",
    linewidth = 1,
    framestyle = :box,
    grid = false,
    left_margin = Plots.Measures.Length(:mm, 24),
    right_margin = Plots.Measures.Length(:mm, 0),
    bottom_margin = Plots.Measures.Length(:mm, 5),
    top_margin = Plots.Measures.Length(:mm, 5),
    legendfontsize = fontsize,
    tickfontsize = fontsize,
    labelfontsize = fontsize,
)

# Fancy logarithmic scale ticks for plotting
# https://github.com/JuliaPlots/Plots.jl/issues/3318
"""
    get_tickslogscale(lims; skiplog=false)
Return a tuple (ticks, ticklabels) for the axis limit `lims`
where multiples of 10 are major ticks with label and minor ticks have no label
skiplog argument should be set to true if `lims` is already in log scale.
"""
function get_tickslogscale(lims::Tuple{T, T}; skiplog::Bool=false) where {T<:AbstractFloat}
    mags = if skiplog
        # if the limits are already in log scale
        floor.(lims)
    else
        floor.(log10.(lims))
    end
    rlims = if skiplog; 10 .^(lims) else lims end

    total_tickvalues = []
    total_ticknames = []

    rgs = range(mags..., step=1)
    for (i, m) in enumerate(rgs)
        if m >= 0
            tickvalues = range(Int(10^m), Int(10^(m+1)); step=Int(10^m))
            ticknames  = vcat([string(round(Int, 10^(m)))],
                              ["" for i in 2:9],
                              [string(round(Int, 10^(m+1)))])
        else
            tickvalues = range(10^m, 10^(m+1); step=10^m)
            ticknames  = vcat([string(10^(m))], ["" for i in 2:9], [string(10^(m+1))])
        end

        if i==1
            # lower bound
            indexlb = findlast(x->x<rlims[1], tickvalues)
            if isnothing(indexlb); indexlb=1 end
        else
            indexlb = 1
        end
        if i==length(rgs)
            # higher bound
            indexhb = findfirst(x->x>rlims[2], tickvalues)
            if isnothing(indexhb); indexhb=10 end
        else
            # do not take the last index if not the last magnitude
            indexhb = 9
        end

        total_tickvalues = vcat(total_tickvalues, tickvalues[indexlb:indexhb])
        total_ticknames = vcat(total_ticknames, ticknames[indexlb:indexhb])
    end
    return (total_tickvalues[1:end-1], total_ticknames[1:end-1])
end

"""
    fancylogscale!(p; forcex=false, forcey=false)
Transform the ticks to log scale for the axis with scale=:log10.
forcex and forcey can be set to true to force the transformation
if the variable is already expressed in log10 units.
"""
function fancylogscale!(p::Plots.Subplot; forcex::Bool=false, forcey::Bool=false)
    kwargs = Dict()
    for (ax, force, lims) in zip((:x, :y), (forcex, forcey), (xlims, ylims))
        axis = Symbol("$(ax)axis")
        ticks = Symbol("$(ax)ticks")

        if force || p.attr[axis][:scale] == :log10
            # Get limits of the plot and convert to Float
            ls = float.(lims(p))
            ts = if force
                (vals, labs) = get_tickslogscale(ls; skiplog=true)
                (log10.(vals), labs)
            else
                get_tickslogscale(ls)
            end
            kwargs[ticks] = ts
        end
    end

    if length(kwargs) > 0
        plot!(p; kwargs...)
    end
    p
end
fancylogscale!(p::Plots.Plot; kwargs...) = (fancylogscale!(p.subplots[1]; kwargs...); return p)
fancylogscale!(; kwargs...) = fancylogscale!(plot!(); kwargs...)

function Base.unique(ctg::CategoricalArray)
    l = levels(ctg)
    newctg = CategoricalArray(l)
    levels!(newctg, l)
end

function annotated_groupedbar(xx, yy, group; series_annotations="", bar_width=1.0, plot_kwargs...)
    gp = groupedbar(xx, yy, group=group, series_annotations="", bar_width=bar_width; plot_kwargs...)
    m = length(unique(group))       # number of items per group
    n = length(unique(xx))          # number of groups
    xt = (1:n) .- 0.5               # plot x-coordinate of groups' centers
    dx = bar_width/m                # each group occupies bar_width units along x
    # dy = diff([extrema(yy)...])[1]
    x2 = [xt[i] + (j - m/2 - 0.4)*dx for j in 1:m, i in 1:n][:]
    k = 1
    for i in 1:n, j in 1:m
        y0 = gp[1][2j][:y][i]*1.4# + 0.04*dy
        if isfinite(y0)
            Plots.annotate!(x2[(i-1)*m + j]*1.02, y0, Plots.text(series_annotations[k], :center, :black, speedup_fontsize))  # qualified: PrettyTables also exports annotate!
            k += 1
        end
    end
    gp
end
