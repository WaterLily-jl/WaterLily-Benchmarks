# Clusters and data

## SLURM jobs

[`jobs/`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/tree/main/jobs) has example job scripts for LUMI (`lumi.sh`, AMD GPUs) and MareNostrum 5 (`mn5.sh`, NVIDIA GPUs and CPUs up to 16 threads). They run `benchmark.sh` from the working directory, so submit them from the repository root:

```sh
sbatch jobs/mn5.sh
```

Adapt the account, partition and `WATERLILY_DIR` to yours. A few things to keep in mind on a cluster:

- Give the job at least one CPU more than the largest `-t`: the machine check of `benchmark.sh` stops otherwise (Julia also runs an interactive thread).
- The first run of a new clone installs the benchmark environment, the first GPU run makes the GPU environment (`gpu/CUDA` or `gpu/AMDGPU`), and `-u true` updates them (see [Updating the environment](@ref)). All of these download packages, so they need internet access.
- `mn5.sh` sets its own `JULIA_DEPOT_PATH` for the accelerated partition.

## Published data

The raw data of studies reported in issues, pull requests or papers are kept in [WaterLily-Benchmarks-data](https://github.com/WaterLily-jl/WaterLily-Benchmarks-data): one folder per study, with the JSON files as `benchmark.sh` wrote them, the resolved `Manifest.toml` files, and a README with the hardware, the software versions and the exact commands. `compare.jl` reads a study's `data/` folder directly:

```sh
WATERLILY_DIR=<WaterLily.jl clone> julia --project compare.jl --data_dir=<study>/data --speedup_base="CPUx01,master"
```

Use the WaterLily-Benchmarks commit given in the study's README to get the same tables. The data directories that `benchmark.sh` writes already hold the Manifests of their runs and `environments.toml` (see [Output](@ref)), so a study can keep them as they are.
