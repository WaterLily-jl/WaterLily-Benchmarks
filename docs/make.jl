# Build the documentation site (DocumenterVitepress). Locally, from the repository root:
#     julia --project=docs -e 'using Pkg; Pkg.instantiate()'
#     julia --project=docs docs/make.jl
# and serve docs/build/1, e.g. with LiveServer.serve(dir="docs/build/1"). CI deploys it to gh-pages (docs.yml).
using Documenter, DocumenterVitepress

# The vorticity images of the developed-flow checkpoints (visualize.jl), the largest size of each case, for cases.md
viz = joinpath(@__DIR__, "..", "checkpoints", "viz")
assets = mkpath(joinpath(@__DIR__, "src", "assets", "cases"))  # gitignored
case(f) = split(f, "_p")[1]                                    # <case>_p<log2p>.png
log2p(f) = parse(Int, split(splitext(f)[1], "_p")[end])
for c in unique(case.(readdir(viz)))
    cp(joinpath(viz, argmax(log2p, filter(f -> case(f) == c, readdir(viz)))), joinpath(assets, "$c.png"); force=true)
end

makedocs(;
    sitename = "WaterLily-Benchmarks",
    authors = "WaterLily-jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/WaterLily-jl/WaterLily-Benchmarks",
        devbranch = "main",
        devurl = "dev",
        write_inventory = false,  # objects.inv needs a package version
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Guide" => [
            "Running benchmarks" => "benchmarks.md",
            "Comparing results" => "compare.md",
            "Cases and checkpoints" => "cases.md",
            "BiotSavartBCs changes" => "biotsavart.md",
            "Environments" => "environments.md",
            "Profiling" => "profiling.md",
            "Clusters and data" => "clusters.md",
        ],
        "Pull-request benchmarks" => "pull-requests.md",
        "Methodology" => "methodology.md",
    ],
    checkdocs = :none,  # not a package: no docstrings to check
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/WaterLily-jl/WaterLily-Benchmarks",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
