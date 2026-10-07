# Cases and checkpoints

Each case is a function in [`src/cases.jl`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/src/cases.jl) that returns a WaterLily `Simulation` for a size `log2p`, a backend and a float type. The `-biotsavart` cases use the open boundaries of [BiotSavartBCs.jl](https://github.com/WaterLily-jl/BiotSavartBCs.jl) instead of WaterLily's default ones, and solve the pressure Poisson equation through its `mom_project!` (see [BiotSavartBCs changes](biotsavart.md)).

| Case | Flow | Grid (cells), `n = 2^log2p` | Default `-p` | Checkpoints |
|---|---|---|---|---|
| `tgv` | Taylor-Green vortex, Re = 1600, symmetry planes | `n × n × n` | `6,7` | 6, 7, 8 |
| `tgv-periodic` | `tgv`, periodic in every direction | `n × n × n` | `6,7` | 6, 7 |
| `sphere` | sphere of diameter `n`, Re = 3700, convective exit | `16n × 6n × 6n` | `3,4` | 3, 4, 5 |
| `sphere-biotsavart` | `sphere` with Biot-Savart boundaries | `16n × 6n × 6n` | `3,4` | 3, 4 |
| `cylinder` | cylinder of diameter `n` heaving across the flow, Re = 1000, periodic span | `9n × 6n × 2n` | `4,5` | 4, 5 |
| `cylinder-biotsavart` | `cylinder` with Biot-Savart boundaries | `9n × 6n × 2n` | `4,5` | 4, 5 |
| `donut` | torus of radius `n/4`, Re = 1000 | `2n × n × n` | `5,6` | 5, 6 |
| `jelly-biotsavart` | swimming jellyfish, Re = 500, a quarter of the domain with two symmetry planes | `n × n × 4n` | `5,6` | 5, 6, 7 |

The bodies of `cylinder`, `cylinder-biotsavart` and `jelly-biotsavart` move, so they are measured again at every step.

```@raw html
<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin: 16px 0">
  <figure style="margin: 0"><img src="./assets/cases/tgv.png" alt="tgv" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>tgv</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/tgv-periodic.png" alt="tgv-periodic" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>tgv-periodic</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/sphere.png" alt="sphere" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>sphere</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/sphere-biotsavart.png" alt="sphere-biotsavart" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>sphere-biotsavart</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/cylinder.png" alt="cylinder" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>cylinder</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/cylinder-biotsavart.png" alt="cylinder-biotsavart" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>cylinder-biotsavart</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/donut.png" alt="donut" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>donut</code></figcaption></figure>
  <figure style="margin: 0"><img src="./assets/cases/jelly-biotsavart.png" alt="jelly-biotsavart" style="width: 100%; border-radius: 6px; background: #fff"/><figcaption style="text-align: center; font-size: 0.85em"><code>jelly-biotsavart</code></figcaption></figure>
</div>
```

*Vorticity of the developed flow in the largest checkpoint of each case.*

## Developed-flow checkpoints

By default every benchmark starts from a *developed* flow, so that it times the steady cost of a step rather than the startup transient. The flows are stored as `checkpoints/<case>_<log2p>_<ftype>.jld2`: each case is advanced to a developed state (to `tU/L = 10` for the `tgv` cases and 100 for the bluff bodies, and for 10 swimming cycles for `jelly-biotsavart`, as set in `develop_time` in `src/util.jl`) and saved. A run with a size or float type that has no checkpoint stops with an error. `benchmark.sh -dev ""` times the transient instead.

The checkpoints are git-LFS files. In a clone made with `GIT_LFS_SKIP_SMUDGE=1`, as in CI, they are small pointers, and `benchmark.sh` downloads the ones a sweep needs with `git lfs pull`.

[`develop.jl`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/develop.jl) makes the checkpoints, and [`visualize.jl`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/blob/main/visualize.jl) renders their vorticity into `checkpoints/viz/` to check them by eye (`Float64` cases are skipped; it uses GLMakie from the plotting environment):

```sh
julia --project develop.jl                                    # every case at its checkpoint sizes, Float32
julia --project develop.jl --cases='["sphere"]' --log2p='[(5,)]'
julia --project visualize.jl --cases='["sphere"]' --log2p='[(5,)]'
julia --project=gpu/CUDA develop.jl --backend=CuArray        # on an NVIDIA GPU
```

Both take `--cases`, `--log2p` and `--ftype` as Julia lists, and `--data_dir` for the checkpoint directory (default `checkpoints/`).

## Adding a case

1. Add a function to `src/cases.jl` that returns the `Simulation`, named as the case with `-` replaced by `_` (the case `tgv-periodic` is the function `tgv_periodic`).
2. In `src/util.jl`, add the case to `all_cases`, and give it entries in `tests_dets` (the domain shape in units of `2^log2p`, for the cost per cell, and the plot title), `develop_time` and `checkpoint_log2p`. If the body moves, add it to `remeasure_case`.
3. Add the same default sizes to `DEF_LOG2P` at the top of `benchmark.sh`.
4. Make the checkpoints with `julia --project develop.jl --cases='["<case>"]'`, check them with `visualize.jl`, and commit them (git-LFS) with their images.
