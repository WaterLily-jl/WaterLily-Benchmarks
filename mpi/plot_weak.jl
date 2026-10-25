# Weak-scaling plot from bench result files (`mode=weak` lines emitted by
# mpi/scaling.jl --weak).
#
#   julia --project=plotting mpi/plot_weak.jl <results.txt | dir ...> [--out=weak_scaling.pdf] [--refnp=1]
#
# Each ARG is either a combined results FILE (labeled by filename, legacy) or a DIRECTORY (globs
# its weak_*.out per-point files, groups them into ONE figure labeled by the dir name, and saves
# the PDF INTO that dir — e.g. `plot_weak.jl data/WL-rpn64-ccx --inset=64` → a plot in each).
#
# --refnp picks the efficiency reference: 1 (default) = serial core
# (intra-node view, includes node bandwidth sharing); set it to the
# ranks-per-node of the runs (e.g. 32) for the inter-node view — E is then
# normalized by the 1-full-node time and only np ≥ refnp is plotted (node
# contention cancels; what remains is network/halo cost).
#
# One curve per (case, local block); serial plotted at np=1. The y axis is the weak-scaling
# efficiency E(np) = t(1)/t(np) — ideal stays at 1, real curves decay below it.
# Log₂ x, linear y. LaTeX theme.
# First use: julia --project=plotting -e 'using Pkg; Pkg.instantiate()'

using CairoMakie
using LaTeXStrings

struct Point
    np::Int          # 1 for SERIAL
    ms::Float64
end

# Expand each CLI path: a DIRECTORY → its weak_*.out files (per-point campaign), labeled by the
# dir basename and saved back into that dir; a FILE → itself, labeled by the filename (legacy).
# Returns (pairs, outdir): pairs = [(file,label)...]; outdir = label→dir ("" ⇒ honor --out/cwd).
function resolve_inputs(paths, prefix)
    pairs  = Tuple{String,String}[]
    outdir = Dict{String,String}()
    for p in paths
        if isdir(p)
            ap  = normpath(abspath(p))
            lbl = basename(ap)
            fs  = sort(filter(f -> startswith(basename(f), prefix) && endswith(f, ".out"),
                              readdir(ap; join = true)))
            isempty(fs) && @warn "no $(prefix)*.out files in $p"
            append!(pairs, ((f, lbl) for f in fs))
            outdir[lbl] = ap
        else
            lbl = replace(splitext(basename(p))[1], Regex("^(results_|$prefix)") => "")
            push!(pairs, (p, lbl))
            get!(outdir, lbl, "")
        end
    end
    return pairs, outdir
end

function parse_results(pairs)
    rx = r"(SERIAL|MPI np=(\d+))\s+case=(\w+)\s+mode=weak\s+grid=[\dx]+\s+local=([\dx]+).*?per_step=([\d.]+)ms"
    curves = Dict{Tuple{String,String,String},Vector{Point}}()  # (case, local, label) → points
    for (f, label) in pairs
        for line in eachline(f)
            m = match(rx, line)
            isnothing(m) && continue
            np = isnothing(m[2]) ? 1 : parse(Int, m[2])
            push!(get!(curves, (m[3], m[4], label), Point[]),
                  Point(np, parse(Float64, m[5])))
        end
    end
    for pts in values(curves)
        sort!(pts, by = p -> p.np)
        unique!(p -> p.np, pts)
    end
    return curves
end

function curve_label(case, loc)
    dims = parse.(Int, split(loc, 'x'))
    gstr = allequal(dims) ? "$(dims[1])^3" : join(dims, raw"\times")
    name = case == "tgv" ? "TGV" : uppercasefirst(case)
    latexstring("\\mathrm{$name}\\ $gstr\\ /\\,\\mathrm{rank}")
end

# Plot the curves of ONE machine (one input file).
# inset=N adds a small inter-node panel (bottom-right): E = t_N/t_np for np ≥ N, i.e. the pure
# network/transport efficiency once a node is full at its operating density — separates the
# one-time intra-node contention ramp (main curve) from the MPI scaling (inset).
function plot_weak(curves; out="weak_scaling.pdf", refnp=1, inset=nothing)
    set_theme!(merge(theme_latexfonts(),
                     Theme(figure_padding = (20, 20, 20, 20),   # top 9 matches plot_strong, whose
                                                               # 10^6 tick label overhangs to T≈9
                           fontsize = 24, linewidth = 2, size = (600, 450))))

    nps_all = sort(unique(p.np for pts in values(curves) for p in pts if p.np >= refnp))
    xt      = filter(ispow2, nps_all)   # ticks at powers of 2 only; intermediate np (192, 320,
                                        # ...) are plotted but unlabeled to avoid tick crowding

    fig = Figure()
    ax = Axis(fig[1, 1];
        xscale = log2,
        xticks = (xt, string.(xt)),
        limits = ((refnp, maximum(nps_all)), (0, 1.1)),
        yticks = 0:0.2:1,
        yminorticksvisible = true, yminorticks = IntervalsBetween(2),
        yminorgridvisible = true,
        xlabel = L"n_p",
        ylabel = refnp == 1 ? L"E = t_1 / t_{n_p}" :
                 latexstring("E = t_{$refnp} / t_{n_p}"))

    # legend order: TGV (small→big), then Sphere (small→big) — numeric size, not string sort
    keys_sorted = sort(collect(keys(curves));
                       by = k -> (k[1] == "tgv" ? 0 : 1, prod(parse.(Int, split(k[2], 'x')))))
    # 2-curve figures: lightrainbow's endpoints are both reddish (magenta/red) — use two
    # well-separated samples (teal/red, matching hues 3 and 6 of the 6-curve strong palette)
    colors = length(keys_sorted) == 2 ?
        Makie.resample_cmap(:lightrainbow, 6)[[3, 6]] :
        Makie.resample_cmap(:lightrainbow, max(length(keys_sorted), 2))
    for (i, key) in enumerate(keys_sorted)
        pts = filter(p -> p.np >= refnp, curves[key])
        iref = findfirst(p -> p.np == refnp, pts)
        if isnothing(iref)
            @warn "curve $(key) has no np=$refnp reference point — skipped"
            continue
        end
        np = [p.np for p in pts]; eff = pts[iref].ms ./ [p.ms for p in pts]
        c = colors[mod1(i, length(colors))]
        scatterlines!(ax, np, eff; color = c, marker = :circle, markersize = 12,
                      label = curve_label(key[1], key[2]))
    end
    # ideal weak scaling: E = 1
    hlines!(ax, [1.0]; color = (:gray, 0.6), linestyle = :dash, linewidth = 1.5)
    axislegend(ax; position = :lb, labelsize = 18,
               backgroundcolor = (:white, 0.7), framevisible = true,
               framecolor = :black, framewidth = 0.75)

    if !isnothing(inset)
        nps_in = sort(unique(p.np for pts in values(curves) for p in pts if p.np >= inset))
        # bbox positioning = a TRUE overlay, fully outside the layout: an inset placed in
        # fig[1,1] (even with tellwidth/tellheight=false + Inside alignmode) still couples its
        # protrusions into the row and pushed the main axis down ~26 px. Coordinates are figure
        # units (600×450), origin bottom-left; box ≈ the old Relative(0.44/0.36)@(0.94,0.14).
        ax2 = Axis(fig; bbox = BBox(341, 561, 108, 240),
            backgroundcolor = :white,
            # LINEAR x: with half-node (+64) sampling, equal node increments get equal spacing
            xticks = (nps_in, string.(nps_in)),
            xticklabelsize = 16, yticklabelsize = 16, ytickformat = "{:.2f}",
            title = latexstring("\\text{Internode:}\\ E = t_{$inset} / t_{n_p}"),
            titlesize = 18, titlegap = 3)
        effs_all = Float64[]
        for (i, key) in enumerate(keys_sorted)
            pts = filter(p -> p.np >= inset, curves[key])
            iref = findfirst(p -> p.np == inset, pts)
            isnothing(iref) && continue
            np = [p.np for p in pts]; eff = pts[iref].ms ./ [p.ms for p in pts]
            append!(effs_all, eff)
            scatterlines!(ax2, np, eff; color = colors[mod1(i, length(colors))],
                          marker = :circle, markersize = 8, linewidth = 1.5)
        end
        hlines!(ax2, [1.0]; color = (:gray, 0.6), linestyle = :dash, linewidth = 1.2)
        xlims!(ax2, inset, maximum(nps_in))
        ylims!(ax2, 0.8, 1.02)
        translate!(ax2.blockscene, 0, 0, 150)   # draw the inset above the main axis
    end

    save(out, fig)
    @info "saved $out"
    return fig
end

function main(args)
    out   = "weak_scaling.pdf"
    refnp = 1
    inset = nothing
    files = String[]
    for a in args
        if startswith(a, "--out=")
            out = String(split(a, "=", limit=2)[2])
        elseif startswith(a, "--refnp=")
            refnp = parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, "--inset=")
            inset = parse(Int, split(a, "=", limit=2)[2])
        else
            push!(files, a)
        end
    end
    out = splitext(out)[1] * ".pdf"   # figures are always PDF
    isempty(files) && error("usage: julia --project=plotting mpi/plot_weak.jl <results.txt | dir ...> [--out=plot.pdf]")
    pairs, outdir = resolve_inputs(files, "weak_")
    curves = parse_results(pairs)
    isempty(curves) && error("no `mode=weak` per_step lines found in $(join(files, ", "))")
    labels = sort(unique(k[3] for k in keys(curves)))
    base, ext = splitext(out)
    for label in labels
        sub = Dict(k => v for (k, v) in curves if k[3] == label)
        for (k, pts) in sort(collect(sub); by = first)
            @info "curve $(k[1]) local=$(k[2]) [$label]: $(length(pts)) points (np=$(join([p.np for p in pts], ",")))"
        end
        dir = get(outdir, label, "")
        out_l = !isempty(dir)      ? joinpath(dir, basename(out)) :   # dir arg → save in the data dir
                length(labels) == 1 ? out :
                                      "$(base)_$(label)$(ext)"
        plot_weak(sub; out = out_l, refnp, inset)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
