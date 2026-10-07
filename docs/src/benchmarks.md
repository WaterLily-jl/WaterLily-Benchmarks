# Running benchmarks

[`benchmark.sh`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/benchmark.sh) runs every combination of Julia versions, WaterLily refs, backends and thread counts it is given. Each combination runs in a new Julia process that times all the requested cases and sizes and writes one JSON file per case. Nothing is shared between the processes but the files.

```sh
./benchmark.sh -v "release 1.11" -w "master my-branch" -b "Array CuArray" -t "1 4" \
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
| `-ft`, `--float_type` | `Float32` | `Float32` or `Float64` |
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

where `<log2p>` joins the sizes, e.g. `tgv_67_25_Float32_CPUx04_50cb584_1.11.5.json` for `log2p = 6,7`. A run with `-bs` uses `<WaterLily hash>+bs<BiotSavartBCs hash>` instead of the WaterLily hash. The files hold `BenchmarkTools` results whose tags describe the run, and `compare.jl` reads the tags, not the file names, so renaming a file is safe.
