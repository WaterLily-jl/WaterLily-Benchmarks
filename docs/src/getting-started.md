# Getting started

## Requirements

- [Julia](https://julialang.org/downloads/), installed with [juliaup](https://github.com/JuliaLang/juliaup) if you want to compare Julia versions (`-v`).
- A bash shell. On Ubuntu, `sh` is `dash`, which fails on `benchmark.sh` with `Bad substitution`: run the scripts as `./benchmark.sh` or `bash benchmark.sh`. On Windows, use Git Bash.
- `git` and [git-lfs](https://git-lfs.com/), for the developed-flow checkpoints.
- A clone of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl). Point `WATERLILY_DIR` to it, or pass `-wd <path>`. With `-w`, `benchmark.sh` checks out each requested ref in that clone, so keep its working tree clean.
- For the GPU backends, an NVIDIA GPU with its driver (`CuArray`) or an AMD GPU with ROCm (`ROCArray`).

## Install

```sh
git clone https://github.com/WaterLily-jl/WaterLily-Benchmarks
cd WaterLily-Benchmarks
export WATERLILY_DIR=/path/to/WaterLily.jl
```

The checkpoints take about 1 GB. To download only the ones you use, clone with `GIT_LFS_SKIP_SMUDGE=1 git clone ...`: the checkpoints are then small git-LFS pointers, and `benchmark.sh` fetches the ones a sweep needs.

## First benchmark

```sh
./benchmark.sh
julia --project compare.jl
```

This runs the default sweep on the current state of `$WATERLILY_DIR` and the current `julia`: the `tgv` (at `log2p = 6,7`) and `jelly-biotsavart` (`5,6`) cases, 25 steps, `Float32`, on the CPU with 4 threads and on an NVIDIA GPU. Without a GPU, add `-b Array`. The first run installs the benchmark environment and develops WaterLily from `$WATERLILY_DIR` into it, and the first GPU run makes the GPU environment (see [Environments](environments.md)). Add `-u true` to update them when the Julia version or the WaterLily dependencies change (see [Updating the environment](@ref)).

`benchmark.sh` writes one JSON file per case and backend into `data/benchmark/<hostname>_<WaterLily hash>/`, and `compare.jl` prints one table per case and size. [Methodology](methodology.md) explains the columns.

On a laptop, or any machine whose clock speed changes with temperature or power, add `-r 3` to repeat the sweep in separate processes: a single process cannot measure that scatter (see [Noise and repetitions](@ref)). On Linux, `benchmark.sh` also checks the machine first and stops if a CPU is capped below its maximum frequency (see [Machine check](@ref)).

## Comparing WaterLily versions

```sh
./benchmark.sh -w "master my-branch" -b Array -t "1 4" -r 3
julia --project compare.jl --speedup_base=master
```

`-w` takes branches, tags or commit hashes. The `Δ ± σ` and `Signif` columns compare each row with the `master` row of the same backend. See [Running benchmarks](benchmarks.md) for all the options of `benchmark.sh` and [Comparing results](compare.md) for `compare.jl`.

## Plotting

`compare.jl --plot_dir=plots` also saves the cost and speedup plots as PDF. The plotting packages are in their own environment, so that benchmarking never installs them. Instantiate it once:

```sh
julia --project=plotting -e 'using Pkg; Pkg.instantiate()'
```
