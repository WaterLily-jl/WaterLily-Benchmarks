# WaterLily-Benchmarks

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/)

Performance tests for [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl). `benchmark.sh` times `sim_step!` across WaterLily versions, Julia versions, CPU and GPU backends, thread counts, cases and sizes, each combination in a fresh Julia process and starting from a developed flow. `compare.jl` turns the results into tables and plots: the time and cost per step, the speedup, and the difference between versions with its noise and significance.

## Quick start

With a clone of WaterLily.jl in `$WATERLILY_DIR`, compare a branch with `master` on the CPU:

```sh
git clone https://github.com/WaterLily-jl/WaterLily-Benchmarks && cd WaterLily-Benchmarks
./benchmark.sh -w "master my-branch" -b Array -t "1 4" -r 3
julia --project compare.jl --speedup_base=master
```

`-b Array -t "1 4"` runs on the CPU with 1 and 4 threads. To also run on a GPU, add its backend to `-b`: `-b "Array CuArray"` for NVIDIA, `-b "Array ROCArray"` for AMD. On Ubuntu, run the script as `./benchmark.sh`, not `sh benchmark.sh` (`sh` is `dash`); on Windows, use Git Bash. A table of `compare.jl`, here for the `jelly-biotsavart` case comparing WaterLily v1.8.0 with `master` (see the [methodology](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/methodology) for the columns):

```
▶ log2p = 5
┌────────────┬───────────┬────────┬─────────┬───────┬────────┬────────┬─────────────┬─────────┬───────┬──────────────┬─────────┬──────┐
│  Backend   │ WaterLily │ Julia  │   FP    │ Alloc │  Mean  │ Median │    Cost     │ Speedup │ Noise │    Δ ± σ     │ Signif  │ Reps │
│            │           │        │         │  [k]  │  [ms]  │  [ms]  │ [ns/DOF/dt] │         │  [%]  │     [%]      │ [|Δ|/σ] │      │
├────────────┼───────────┼────────┼─────────┼───────┼────────┼────────┼─────────────┼─────────┼───────┼──────────────┼─────────┼──────┤
│     CPUx01 │    master │ 1.11.5 │ Float32 │   0.7 │ 157.99 │ 165.25 │     1205.41 │    1.00 │   0.4 │            - │       - │    3 │
│     CPUx01 │    v1.8.0 │ 1.11.5 │ Float32 │   0.8 │ 161.27 │ 168.63 │     1230.37 │    0.98 │   0.5 │  +2.1 ±  0.7 │    3.1  │    3 │
│     CPUx04 │    master │ 1.11.5 │ Float32 │  33.2 │  56.61 │  57.74 │      431.90 │    2.79 │   1.6 │            - │       - │    3 │
│     CPUx04 │    v1.8.0 │ 1.11.5 │ Float32 │  62.3 │  65.31 │  66.21 │      498.29 │    2.42 │   1.5 │ +15.4 ±  2.2 │    7.0  │    3 │
│ GPU-NVIDIA │    master │ 1.11.5 │ Float32 │  36.0 │   7.67 │   7.80 │       58.55 │   20.59 │   5.6 │            - │       - │    3 │
│ GPU-NVIDIA │    v1.8.0 │ 1.11.5 │ Float32 │  42.6 │   8.29 │   8.37 │       63.26 │   19.05 │   4.5 │  +8.1 ±  7.2 │    1.1  │    3 │
└────────────┴───────────┴────────┴─────────┴───────┴────────┴────────┴─────────────┴─────────┴───────┴──────────────┴─────────┴──────┘
```

## Documentation

The [documentation](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/) covers:

- [Getting started](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/getting-started): requirements, installation and a first benchmark.
- [Running benchmarks](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/benchmarks): the options of `benchmark.sh`, repetitions, the machine check and the output files.
- [Comparing results](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/compare): `compare.jl`, the speedup baseline, sorting and plots.
- [Cases and checkpoints](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/cases): the test cases, the developed-flow checkpoints and how to add a case.
- [Pull-request benchmarks](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/pull-requests): `/benchmark` on WaterLily.jl pull requests and the GitHub Action behind it.
- [Methodology](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/methodology): what is timed and what each column means.
- [BiotSavartBCs changes](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/biotsavart), [environments](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/environments), [profiling](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/profiling) and [clusters and data](https://waterlily-jl.github.io/WaterLily-Benchmarks/dev/clusters).

The data of published studies are in [WaterLily-Benchmarks-data](https://github.com/WaterLily-jl/WaterLily-Benchmarks-data).
