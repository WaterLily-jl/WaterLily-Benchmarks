# Comparing results

[`compare.jl`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/compare.jl) reads the JSON files of `benchmark.sh` and prints one table per case and size, with one row per backend, WaterLily ref, Julia version and float type. [Methodology](methodology.md) explains how the columns are measured.

```sh
julia --project compare.jl --data_dir=data/benchmark --patterns="tgv jelly-biotsavart" --speedup_base=CPUx01 --sort=Backend
```

## Choosing the files

Either search a directory, or pass the files, not both:

- `--data_dir=<dir>` (default `data/benchmark`) with `--patterns="<p1> <p2>"` (space- or comma-separated, default: every case) reads the files under `<dir>` whose path matches `*<pattern>*`. A pattern is a substring, so `tgv` also matches `tgv-periodic`, and `--patterns="tgv*CPU"` keeps the CPU runs of `tgv`.
- JSON files as arguments, e.g. `julia --project compare.jl data/benchmark/*/tgv_*.json`.

Files with identical tags are repetitions of one benchmark (`benchmark.sh -r`) and are merged into one row.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--speedup_base=<values>` | the first row | the row that `Speedup` divides by and that sets the reference rows of `Δ ± σ`, see below |
| `--sort=<column>` | file order | sort the rows by a column, given by (the start of) its name, case-insensitive, e.g. `Backend`, `Mean`, `Speedup`, `Signif`; a column index (1 to 14, counting the hidden `GC`) also works |
| `--markdown` | off | print GitHub markdown tables, as posted on [pull requests](pull-requests.md) |
| `--gc` | off | show the `GC [%]` column |
| `--plot_dir=<dir>` | none | also save plots, see [Plots](@ref) |
| `--backend_color=<scheme>` | `lightrainbow` | [color scheme](https://docs.juliaplots.org/dev/generated/colorschemes/) of the plots |

In the text tables the speedup baseline row is blue and the fastest row of each backend green; in markdown they are bold and italic.

## Speedup baseline

`--speedup_base` takes comma-separated values that a single row must all match: a backend label (`CPUx04`, `GPU-NVIDIA`), a WaterLily ref or hash (`master`, `v1.8.0`, `50cb584`), a Julia version (`1.11.5`) or a float type (`Float32`), and for a [`-bs` run](biotsavart.md) a BiotSavartBCs ref or hash. The first matching row is the baseline, e.g. `--speedup_base="CPUx01,master"`. Matching is case-sensitive (`CPUx04`, not `cpux04`). If nothing matches, the error lists the rows that are available.

- `Speedup` of every row is the baseline time divided by the row time, so it also compares backends.
- `Δ ± σ` of a row compares it with its *reference row*: the row of the same backend with the WaterLily ref, Julia version and float type of the baseline. With `--speedup_base=master`, each backend is compared with its own `master` row.

WaterLily refs are named from the clone in `$WATERLILY_DIR`: a local branch at that commit, else a tag, else a remote branch, else the hash. Point it to an up-to-date clone when reading results from another machine.

## Plots

`--plot_dir=<dir>` saves two PDF files per case into `<dir>` (relative to the repository root): the cost per cell and time step, `<case>_cost_<versions>.pdf`, and the speedup, `<case>_benchmark_<versions>.pdf`. The plotting packages live in a separate environment that has to be instantiated once, see [Environments](environments.md).
