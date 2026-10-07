```@raw html
---
layout: home

hero:
  name: WaterLily-Benchmarks
  text: Performance tests for WaterLily.jl
  tagline: Time sim_step! across WaterLily versions, Julia versions, CPU and GPU backends, and compare the results with their noise.
  actions:
    - theme: brand
      text: Get started
      link: /getting-started
    - theme: alt
      text: Pull-request benchmarks
      link: /pull-requests
    - theme: alt
      text: View on GitHub
      link: https://github.com/WaterLily-jl/WaterLily-Benchmarks

features:
  - title: Sweeps
    details: One command runs every combination of WaterLily refs, Julia versions, backends, thread counts, cases and sizes, each in a fresh Julia process.
    link: /benchmarks
  - title: Developed flows
    details: Cases start from stored checkpoints of a developed flow, so the timings measure the steady cost of a step, not the startup transient.
    link: /cases
  - title: Noise-aware comparisons
    details: compare.jl reports the cost difference between refs with its scatter and significance, merging repeated runs in separate processes.
    link: /methodology
  - title: Pull-request benchmarks
    details: A maintainer comments /benchmark on a WaterLily.jl pull request and gets the comparison with master as a comment.
    link: /pull-requests
  - title: CPU and GPU
    details: Serial (SIMD) and multithreaded CPU runs, and NVIDIA (CUDA) and AMD (ROCm) GPUs, each in its own environment.
    link: /environments
  - title: Profiling
    details: NVTX traces of the solver kernels with NVIDIA Nsight Systems and Nsight Compute.
    link: /profiling
---
```

## Quick start

With a clone of [WaterLily.jl](https://github.com/WaterLily-jl/WaterLily.jl) in `$WATERLILY_DIR`, compare a branch with `master` on the CPU:

```sh
git clone https://github.com/WaterLily-jl/WaterLily-Benchmarks && cd WaterLily-Benchmarks
./benchmark.sh -w "master my-branch" -b Array -t "1 4" -r 3
julia --project compare.jl --speedup_base=master
```

`compare.jl` prints one table per case and size, with the time per step, the speedup, and the difference of each row against `master` on the same backend with its significance. See [Getting started](getting-started.md) for the requirements and [Methodology](methodology.md) for what the columns mean.
