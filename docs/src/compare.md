# Comparing results

[`compare.jl`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/compare.jl) reads the JSON files of `benchmark.sh` and prints one table per case and size, with one row per backend, WaterLily ref, Julia version and float type, where the float type of an AD run (`benchmark.sh -ad`) includes its partials, e.g. `Float32+ad4`. [Methodology](methodology.md) explains how the columns are measured.

```sh
julia --project compare.jl --data_dir=data/benchmark --patterns="tgv jelly-biotsavart" --speedup_base=CPUx01 --sort=Backend
```

## Choosing the files

Either search a directory, or pass the files, not both:

- `--data_dir=<dir>` (default `data/benchmark`) with `--patterns="<p1> <p2>"` (space- or comma-separated, default: every case) reads the JSON files under `<dir>` whose path matches `*<pattern>*`. A pattern is a substring, so `tgv` also matches `tgv-periodic`, and `--patterns="tgv*CPU"` keeps the CPU runs of `tgv`.
- JSON files as arguments, e.g. `julia --project compare.jl data/benchmark/*/tgv_*.json`.

Files with identical tags are repetitions of one benchmark (`benchmark.sh -r`) and are merged into one row.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--speedup_base=<values>` | the first row | the row that `Speedup` divides by and that sets the reference rows of `Δ ± σ`, see below |
| `--sort=<column>` | backend, partials, WaterLily ref | sort the rows by a column, given by (the start of) its name, case-insensitive, e.g. `Backend`, `Mean`, `Speedup`, `Signif`; a column index (1 to 14, counting the hidden `GC`) also works |
| `--markdown` | off | print GitHub markdown tables, as posted on [pull requests](pull-requests.md) |
| `--gc` | off | show the `GC [%]` column |
| `--plot_dir=<dir>` | none | also save plots, see [Plots](@ref) |
| `--backend_color=<scheme>` | `lightrainbow` | [color scheme](https://docs.juliaplots.org/dev/generated/colorschemes/) of the plots |

In the text tables the speedup baseline row is blue and the fastest row of each backend green; in markdown they are bold and italic.

## Speedup baseline

`--speedup_base` takes comma-separated values that a single row must all match: a backend label (`CPUx04`, `GPU-NVIDIA`), a WaterLily ref or hash (`main`, `v1.8.0`, `50cb584`), a Julia version (`1.11.5`) or a float type (`Float32`, or `Float32+ad4` for an AD run), and for a [`-bs` run](biotsavart.md) a BiotSavartBCs ref or hash. The first matching row is the baseline, e.g. `--speedup_base="CPUx01,main"`. Matching is case-sensitive (`CPUx04`, not `cpux04`). If nothing matches, the error lists the rows that are available.

- `Speedup` of every row is the baseline time divided by the row time, so it also compares backends.
- `Δ ± σ` of a row compares it with its *reference row*: the row of the same backend with the WaterLily ref, Julia version and float type of the baseline. With `--speedup_base=main`, each backend is compared with its own `main` row.

## AD runs

The runs of `benchmark.sh -ad` are rows of the same table, listed after the plain rows of their backend with `FP = <ftype>+ad<N>`:

- The plain rows come first, so the default baseline and `--speedup_base=main` are plain rows. `Speedup` of an AD row is then below 1, and its inverse is the cost of the partials: `0.25` means that a step with those partials takes 4 plain steps.
- The reference row of an AD row has the same number of partials: `Δ ± σ` compares the WaterLily refs at equal partials, e.g. `my-branch` with 4 partials against `main` with 4 partials. A row whose number of partials was not run for the baseline's ref prints `-`.
- The fastest row is marked for each backend and number of partials.
- The plots only show the rows with the partials of the baseline.
- A baseline can also be an AD row, e.g. `--speedup_base="main,Float32+ad4"`: `Speedup` is then relative to that row, and each `Δ ± σ` still compares equal partials.

`tgv` at `log2p = 7` on a laptop GPU, `main` against a branch that changes how the `Dual` kernels are compiled ([WaterLily.jl#408](https://github.com/WaterLily-jl/WaterLily.jl/pull/408)), from `./benchmark.sh -w "main dual-always-inline" -b CuArray -c tgv -ad "0 4 8" -r 3` and `--speedup_base=main`:

```
▶ log2p = 7
┌────────────┬────────────────────┬────────┬─────────────┬───────┬────────┬────────┬─────────────┬─────────┬───────┬──────────────┬─────────┬──────┐
│  Backend   │     WaterLily      │ Julia  │     FP      │ Alloc │  Mean  │ Median │    Cost     │ Speedup │ Noise │    Δ ± σ     │ Signif  │ Reps │
│            │                    │        │             │  [k]  │  [ms]  │  [ms]  │ [ns/DOF/dt] │         │  [%]  │     [%]      │ [|Δ|/σ] │      │
├────────────┼────────────────────┼────────┼─────────────┼───────┼────────┼────────┼─────────────┼─────────┼───────┼──────────────┼─────────┼──────┤
│ GPU-NVIDIA │ dual-always-inline │ 1.13.2 │     Float32 │   5.7 │  18.93 │  19.04 │        9.03 │    0.99 │  13.2 │  +1.2 ± 18.6 │    0.1  │    2 │
│ GPU-NVIDIA │               main │ 1.13.2 │     Float32 │   5.7 │  18.71 │  18.94 │        8.92 │    1.00 │  13.0 │            - │       - │    2 │
│ GPU-NVIDIA │ dual-always-inline │ 1.13.2 │ Float32+ad4 │  10.7 │  95.20 │  94.75 │       45.39 │    0.20 │   3.8 │  -7.4 ±  6.6 │    1.1  │    2 │
│ GPU-NVIDIA │               main │ 1.13.2 │ Float32+ad4 │   7.2 │ 102.76 │ 101.42 │       49.00 │    0.18 │   5.4 │            - │       - │    2 │
│ GPU-NVIDIA │ dual-always-inline │ 1.13.2 │ Float32+ad8 │  11.2 │ 193.09 │ 192.90 │       92.07 │    0.10 │   1.9 │ -33.5 ±  1.9 │   17.7  │    2 │
│ GPU-NVIDIA │               main │ 1.13.2 │ Float32+ad8 │   7.5 │ 290.29 │ 290.98 │      138.42 │    0.06 │   0.2 │            - │       - │    2 │
└────────────┴────────────────────┴────────┴─────────────┴───────┴────────┴────────┴─────────────┴─────────┴───────┴──────────────┴─────────┴──────┘
```

A step with 8 partials takes 15.5 plain steps on `main` (`Speedup` 0.06) and a third less on the branch, which changes nothing without partials. The table is of the second and third repetitions: in the first one the machine was still warming up, see [Forward-mode AD](@ref).

WaterLily refs are named from the clone in `$WATERLILY_DIR`: a local branch at that commit, else a tag, else a remote branch, else the hash. Point it to an up-to-date clone when reading results from another machine.

## Plots

`--plot_dir=<dir>` saves two PDF files per case into `<dir>` (relative to the repository root): the cost per cell and time step, `<case>_cost_<versions>.pdf`, and the speedup, `<case>_benchmark_<versions>.pdf`. The plotting packages live in a separate environment that has to be instantiated once, see [Environments](environments.md).
