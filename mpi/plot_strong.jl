# Strong-scaling plot from bench result files (the `[ Info: ...]` lines that
# mpi/scaling.jl prints to the job outputs).
#
#   julia --project=plotting mpi/plot_strong.jl <results.txt | dir ...> [--out=strong_scaling.pdf]
#
# Each ARG is either a combined results FILE or a DIRECTORY:
#   • FILE      → one figure, labeled by the filename (legacy combined-results workflow).
#   • DIRECTORY → globs its strong_*.out files (the per-point campaign, one job per (case,np)),
#                 groups them into ONE figure labeled by the dir name, and saves the PDF INTO
#                 that dir (e.g. `plot_strong.jl data/WL-rpn32-ccx data/WL-rpn64-ccx` → a plot in each).
# One curve per (case, grid); serial plotted at np=1, dashed ideal-scaling guide anchored at each
# curve's first point. Log-log, powers-of-2 ticks, LaTeX theme. Several file-labels → the label is
# inserted into --out (strong.pdf → strong_<label>.pdf); dir args always save as <dir>/<out base>.
# First use: julia --project=plotting -e 'using Pkg; Pkg.instantiate()'

using CairoMakie
using LaTeXStrings
using Printf

# ── Parse ─────────────────────────────────────────────────────────────────────

struct Point
    np::Int          # 1 for SERIAL
    ms::Float64
end

# Expand each CLI path: a DIRECTORY → its strong_*.out files (per-point campaign), labeled by the
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
    rx = r"(SERIAL|MPI np=(\d+))\s+case=(\w+)\s+grid=([\dx]+).*?per_step=([\d.]+)ms"
    curves = Dict{Tuple{String,String,String},Vector{Point}}()  # (case, grid, label) → points
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
        unique!(p -> p.np, pts)   # keep first occurrence if a point was re-run
    end
    return curves
end

# ── Labels ────────────────────────────────────────────────────────────────────

function curve_label(case, grid)
    dims = parse.(Int, split(grid, 'x'))
    gstr = allequal(dims) ? "$(dims[1])^3" : join(dims, raw"\times")
    name = case == "tgv" ? "TGV" : uppercasefirst(case)
    latexstring("\\mathrm{$name}\\ $gstr")
end

# ── Plot ──────────────────────────────────────────────────────────────────────

# Whole-decade ticks for log axes (10¹, 10², …)
struct IntegerTicks end
Makie.get_tickvalues(::IntegerTicks, vmin, vmax) = ceil(Int, vmin):floor(Int, vmax)

# Plot the curves of ONE machine (one input file).
function plot_strong(curves; out="strong_scaling.pdf")
    set_theme!(merge(theme_latexfonts(),
                     Theme(figure_padding = 20, fontsize = 24,
                           linewidth = 2, size = (600, 450))))

    nps_all = sort(unique(p.np for pts in values(curves) for p in pts))
    xt      = length(nps_all) <= 9 ? nps_all :   # past 9 points the labels overlap: every other power of 2
              4 .^ (0:fld(ndigits(maximum(nps_all), base=2) - 1, 2))

    fig = Figure()
    ax = Axis(fig[1, 1];
        xscale = log2, yscale = log10,
        limits = ((1, maximum(nps_all)), nothing),
        xticks = (xt, string.(xt)),
        yticks = LogTicks(IntegerTicks()),
        xminorticksvisible = false, yminorticksvisible = true,
        yminorticks = IntervalsBetween(9), yminorgridvisible = true,
        xlabel = L"n_p", ylabel = L"t \;/\; \Delta t\ \mathrm{[ms]}")

    # legend order: TGV (small→big), then Sphere (small→big) — numeric size, not string sort
    keys_sorted = sort(collect(keys(curves));
                       by = k -> (k[3], k[1] == "tgv" ? 0 : 1, prod(parse.(Int, split(k[2], 'x')))))
    colors = Makie.resample_cmap(:lightrainbow, max(length(keys_sorted), 2))
    for (i, key) in enumerate(keys_sorted)
        pts = curves[key]
        length(pts) < 1 && continue
        np = [p.np for p in pts]; ms = [p.ms for p in pts]
        c = colors[mod1(i, length(colors))]
        # ideal scaling anchored at the curve's first (lowest-np) point
        np_id = [np[1], maximum(nps_all)]
        lines!(ax, np_id, ms[1] .* np[1] ./ np_id;
               color = (:gray, 0.6), linestyle = :dash, linewidth = 1.5)
        scatterlines!(ax, np, ms; color = c, marker = :circle, markersize = 12,
                      label = curve_label(key[1], key[2]))
    end
    axislegend(ax; position = :lb, labelsize = 18,
               backgroundcolor = (:white, 0.7), framevisible = true,
               framecolor = :black, framewidth = 0.75)

    save(out, fig)
    @info "saved $out"
    return fig
end

# ── CLI ───────────────────────────────────────────────────────────────────────

function main(args)
    out   = "strong_scaling.pdf"
    files = String[]
    for a in args
        startswith(a, "--out=") ? (out = String(split(a, "=", limit=2)[2])) : push!(files, a)
    end
    out = splitext(out)[1] * ".pdf"   # figures are always PDF
    isempty(files) && error("usage: julia --project=plotting mpi/plot_strong.jl <results.txt | dir ...> [--out=plot.pdf]")
    pairs, outdir = resolve_inputs(files, "strong_")
    curves = parse_results(pairs)
    isempty(curves) && error("no `per_step` lines found in $(join(files, ", "))")
    labels = sort(unique(k[3] for k in keys(curves)))
    base, ext = splitext(out)
    for label in labels
        sub = Dict(k => v for (k, v) in curves if k[3] == label)
        for (k, pts) in sort(collect(sub); by = first)
            @info "curve $(k[1]) $(k[2]) [$label]: $(length(pts)) points (np=$(join([p.np for p in pts], ",")))"
        end
        dir = get(outdir, label, "")
        out_l = !isempty(dir)      ? joinpath(dir, basename(out)) :   # dir arg → save in the data dir
                length(labels) == 1 ? out :
                                      "$(base)_$(label)$(ext)"
        plot_strong(sub; out = out_l)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
