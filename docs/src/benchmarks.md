# Running benchmarks

[`benchmark.sh`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/benchmark.sh) runs every combination of Julia versions, WaterLily refs, backends and thread counts it is given. Each combination runs in a new Julia process that times all the requested cases and sizes and writes one JSON file per case. Nothing is shared between the processes but the files.

```sh
./benchmark.sh -v "release 1.11" -w "main my-branch" -b "Array CuArray" -t "1 4" \
               -c "tgv jelly-biotsavart" -p "6,7 5,6" -s 25 -ft Float32
```

runs 2 Julia versions × 2 WaterLily refs × 3 backends (CPUx01, CPUx04, GPU-NVIDIA) = 12 processes, each timing `tgv` at `log2p = 6, 7` and `jelly-biotsavart` at `5, 6`.

## Options

Lists are space-separated and quoted. When an option is given twice, the last value is used.

| Option | Default | Meaning |
|---|---|---|
| `-w`, `--waterlily` | the current checkout | WaterLily refs (branches, tags or commit hashes), checked out in turn in `$WATERLILY_DIR` |
| `-wd`, `--waterlily_dir` | `$WATERLILY_DIR` | the WaterLily.jl clone |
| `-v`, `--versions` | the `julia` on the `PATH` | Julia versions, run as `julia +<version>` (needs juliaup), e.g. `"release 1.11"` |
| `-b`, `--backends` | `"Array CuArray"` | `Array` (CPU), `CuArray` (NVIDIA), `ROCArray` (AMD) |
| `-t`, `--threads` | `4` | CPU thread counts, used by `Array` |
| `-c`, `--cases` | `"tgv jelly-biotsavart"` | cases, see [Cases and checkpoints](cases.md) |
| `-p`, `--log2p` | per case | sizes of each case, comma-separated, e.g. `"6,7 5,6"` |
| `-s`, `--max_steps` | `25` | time steps per run |
| `-ft`, `--float_type` | `Float32` | `Float32` or `Float64`, or `<ftype>+ad<N>` for an AD run with `N` partials, e.g. `Float32+ad4`, see [Forward-mode AD](@ref) |
| `-dev`, `--developed` | `checkpoints` | directory of the developed-flow checkpoints; `""` times the startup transient instead |
| `-r`, `--repeats` | `1` | repetitions of the whole sweep, see [Repetitions](@ref) |
| `-u`, `--update` | `false` | update the environment before each run, see [Updating the environment](@ref) |
| `-dd`, `--data_dir` | `data/benchmark/` | output directory |
| `-f`, `--force` | off | run even if the [Machine check](@ref) fails |
| `-bs`, `--biotsavart` | none | BiotSavartBCs refs paired with `-w`, see [BiotSavartBCs changes](biotsavart.md) |
| `-bsd`, `--biotsavart_dir` | `$BIOTSAVART_DIR` | the BiotSavartBCs.jl clone used by `-bs` |

## Cases, sizes, steps and types

`-p`, `-s` and `-ft` describe each case of `-c`: give either one value per case, in the same order, or a single value for all of them. An omitted option takes its default: the size of each case from the `DEF_LOG2P` map at the top of `benchmark.sh` (the sizes of its checkpoints, see [Cases and checkpoints](cases.md)), 25 steps and `Float32`. So

```sh
./benchmark.sh -c "tgv sphere" -ft Float64
```

runs `tgv` at `6,7` and `sphere` at `3,4`, both in `Float64`. A case missing from `DEF_LOG2P` needs `-p`. `log2p` sets the resolution: the length scale of the case is `2^log2p` cells, e.g. a `tgv` box of `64^3` cells at `log2p = 6`.

Each run starts from a checkpoint of a developed flow, `checkpoints/<case>_<log2p>_<ftype>.jld2`, and a missing checkpoint is an error. Pass `-dev ""` to time the startup transient instead, or `-dev <dir>` to read the checkpoints from another directory.

## Backends and threads

`Array` runs on the CPU once per thread count in `-t`. With 1 thread it uses WaterLily's SIMD backend (labelled `CPUx01`), and with more threads its KernelAbstractions backend (`CPUx04`, ...). `CuArray` and `ROCArray` run on the GPU (`GPU-NVIDIA`, `GPU-AMD`) in their own environments, which `benchmark.sh` makes on their first run; see [Environments](environments.md). `benchmark.sh` writes the backend preference of each run into `LocalPreferences.toml`, so do not edit that file.

## Forward-mode AD

A float type `<ftype>+ad<N>`, e.g. `-ft Float32+ad4`, times its case with [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) `Dual` numbers, as in a simulation differentiated in forward mode: the arrays hold `Dual`s over `<ftype>` with `N` partials. It measures what a derivative costs on each backend, and what a WaterLily change does to that cost. As any `-ft` value it is given once for all cases or once per case, so repeat a case to time it with several numbers of partials next to the plain run:

```sh
./benchmark.sh -w "main my-branch" -b CuArray -c "tgv tgv tgv" -ft "Float32 Float32+ad4 Float32+ad8" -r 3
julia --project compare.jl --speedup_base=main
```

- The Reynolds number of the case is the differentiated input: it carries the first partial, and the other partials are zero. `Dual` arithmetic does the same work whether a partial is zero or not, so the time is that of `N` independent inputs.
- The runs start from the same checkpoints as the plain ones, values only. After the warm-up `benchmark.jl` prints `max|∂u/∂Re|` and stops if it is zero, so a timed run always carries a derivative. In `sphere` and `donut` that derivative grows without bound with WaterLily's `quick` scheme, about 2.5 times per step, and overflows within 150 steps: the `10c-9u` bound of `quick` ([WaterLily.jl#358](https://github.com/WaterLily-jl/WaterLily.jl/issues/358)), which the Koren scheme of [WaterLily.jl#385](https://github.com/WaterLily-jl/WaterLily.jl/pull/385) replaces (the derivative of both cases then stays bounded). It does not happen in `tgv` and `cylinder`. The values and the pressure solver do not see the partials, and a step takes the same time whether they are finite or not, so the timings hold, but the derivative of those two cases is not a sensitivity.
- The pressure solver stops on the residual of the values, so an AD run takes the solver iterations of the plain run (checked over 30 steps of `tgv`, `cylinder`, `sphere` and `donut`), and the time ratio is the cost of the `Dual` arithmetic.
- The same `<ftype>+ad<N>` is the float type of the run in the file names and in the `FP` column of `compare.jl`, where the AD runs are rows of their own, see [Comparing results](compare.md). The files and tables of plain runs do not change.
- The Biot-Savart cases cannot run: BiotSavartBCs.jl updates its residual with an atomic `+=`, which does not take `Dual` numbers, so an AD float type for a `-biotsavart` case stops with an error before anything runs. `jelly-biotsavart` is a default case: give the cases with `-c`, or keep it plain with `-ft "Float32+ad4 Float32"`.
- AD runs are slow on the CPU, where the `Dual` loops are not vectorised: for `tgv` at `log2p = 6` the first partial alone costs 11 `Float32` steps on one thread and 8 on four, and each process warms up for at least 50 steps. Use small sizes and few steps there (`-p`, `-s`), or a GPU, where a partial costs about one `Float32` step (on a laptop GPU, 5 steps with 4 partials and 10 to 15 with 8). The memory is `N + 1` times that of the plain run.
- A laptop that has been idle runs its first processes faster, and the first processes of a sweep are those of the first ref. With AD runs, which take minutes, this can be 10 to 20 % of a step: use `-r 3`, read `Signif`, and when the first repetition stands out compare the others, e.g. `julia --project compare.jl data/benchmark/*/*_r[23].json`.

`Dual` comes from WaterLily's own ForwardDiff, so the environments need no extra package.

## Updating the environment

With `-u true`, each run first develops WaterLily from `$WATERLILY_DIR` into the benchmark environment and updates the other packages (`Pkg.develop` and `Pkg.update`). A new clone does not need it: an environment without a `Manifest.toml` is installed on its first run (`Pkg.develop` only). Use it when the Julia version changes or when the WaterLily refs being compared have different dependencies. Without it, the environment stays as it is: WaterLily is developed from a path, so checking out another ref with `-w` is enough to run that code.

## Repetitions

`-r N` repeats the whole sweep `N` times, each benchmark in a new process, and reverses the order of the Julia versions and WaterLily refs on even repetitions. The files get an `_r<n>` suffix, and `compare.jl` merges the repetitions of a benchmark into one row (the `Reps` column). Use it when the state of the machine can change between processes, which is typical on laptops (clock speed, thermal and power limits); see [Noise and repetitions](@ref).

## Machine check

On Linux, `benchmark.sh` first checks that the machine will not distort the timings. It stops if a CPU is capped below its maximum frequency, giving the command that lifts the cap, or if fewer CPUs are available than the largest `-t` plus one (Julia also runs an interactive thread). It warns if turbo is off or the load average is above 1. `-f` runs anyway, e.g. on shared CI runners.

## Output

Each process writes one JSON file per case into `<data_dir>/<hostname>_<WaterLily hash>/`, named

```
<case>_<log2p>_<max_steps>_<ftype>_<backend>_<WaterLily hash>_<Julia version>[_r<n>].json
```

where `<log2p>` joins the sizes, e.g. `tgv_67_25_Float32_CPUx04_50cb584_1.11.5.json` for `log2p = 6,7`. An AD run has `<ftype>+ad<N>` as its float type, e.g. `tgv_67_25_Float32+ad4_CPUx04_50cb584_1.11.5.json` (see [Forward-mode AD](@ref)). A run with `-bs` uses `<WaterLily hash>+bs<BiotSavartBCs hash>` instead of the WaterLily hash. The files hold `BenchmarkTools` results whose tags describe the run, and `compare.jl` reads the tags, not the file names, so renaming a file is safe for `compare.jl` (but see `environments.toml` below).

Each data directory also keeps the environment of its runs, so that they can be reproduced:

- `Manifest-<Julia version>-<environment>-<hash>.toml`: the resolved `Manifest.toml` of the run's environment (`cpu`, or `CUDA` and `AMDGPU` for the [GPU environments](environments.md)), one file per distinct Manifest. A package developed from a local clone, WaterLily from `$WATERLILY_DIR` or BiotSavartBCs with `-bs`, gets `path = "<package>.jl"` instead of the path on the machine, and the commit it ran from as `clone-commit` (with `clone-changes = true` if the clone had uncommitted changes). So the Manifest says which code ran, without paths of the machine it ran on.
- `environments.toml`: for each JSON file, its Manifest and the WaterLily preferences of the run (`LocalPreferences.toml`, the SIMD or KernelAbstractions backend).

Two sweeps writing into the same data directory at the same time can lose entries of `environments.toml`; sweeps run one after the other are fine.
