# Environments

The scripts use separate Julia environments, so that each run installs and loads only what it needs.

| Environment | Used by | Contents |
|---|---|---|
| `Project.toml` (root) | CPU runs, `compare.jl`, `develop.jl` | the benchmark packages, WaterLily and BiotSavartBCs |
| `gpu/CUDA`, `gpu/AMDGPU` | `CuArray`, `ROCArray` runs | a copy of the root environment plus CUDA or AMDGPU |
| `plotting/` | `compare.jl --plot_dir`, `visualize.jl`, profiling plots | Plots, StatsPlots, CairoMakie, GLMakie |
| `docs/` | `docs/make.jl` | Documenter and DocumenterVitepress |

## Benchmark environment

`WaterLily` is developed from `$WATERLILY_DIR` (`Pkg.develop`) by `benchmark.sh`, on the first run of a new clone and with `-u true`, so the code that runs is whatever is checked out there. `BiotSavartBCs` follows its `main` branch through the `[sources]` entry of `Project.toml`; `-bs` points it at a local clone for a sweep, see [BiotSavartBCs changes](biotsavart.md).

`benchmark.sh` writes `LocalPreferences.toml` in the environment of each run, choosing WaterLily's backend: `SIMD` for `Array` with 1 thread and `KernelAbstractions` otherwise. It is regenerated for every run, so editing it by hand has no effect on a sweep.

## GPU environments

CUDA and AMDGPU are not in the root `Project.toml`, so CPU runs, `compare.jl` and CI never install them. A `CuArray` or `ROCArray` run uses `gpu/CUDA` or `gpu/AMDGPU` instead: a copy of `Project.toml` plus the GPU package, resolved on its own, so that neither the CPU environment nor the other GPU vendor holds its versions back. A CPU row and a GPU row of one table can therefore run different versions of the packages they share; rows of one backend always share an environment.

`benchmark.sh` makes the GPU environment on the first run of that backend, and makes it again when `Project.toml` changes (it keeps a copy, `gpu/<pkg>/Project.base.toml`, to tell) or with `-u true`. The directories are not tracked by git. To run another script on a GPU, use that environment and pass the backend, e.g.

```sh
julia --project=gpu/CUDA develop.jl --backend=CuArray
```

## Plotting environment

The plotting packages are heavy, so they live in `plotting/`, which is stacked on top of the benchmark environment (`use_plotting_env()` in `src/util.jl`) by the scripts that plot. Instantiate it once:

```sh
julia --project=plotting -e 'using Pkg; Pkg.instantiate()'
```

## Documentation

The documentation is built from `docs/` with [DocumenterVitepress](https://github.com/LuxDL/DocumenterVitepress.jl), which brings its own Node.js:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

and served from `docs/build/1`, e.g. with `LiveServer.serve(dir="docs/build/1")`. A pull request from a branch of this repository that changes `docs/` gets a preview at `previews/PR<number>`, linked from its checks.
