# MPI scaling

[`mpi/`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/tree/main/mpi) benchmarks the MPI extension of WaterLily (branch [`mpi-igg`](https://github.com/WaterLily-jl/WaterLily.jl/pull/281), on top of [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl)) on CPUs and GPUs: strong and weak scaling, and correctness checks against a run without MPI. It is separate from `benchmark.sh`: each run is one MPI launch of `mpi/scaling.jl`, which prints one line per run instead of writing JSON files.

| File | What it does |
|---|---|
| `mpi/setup.jl` | builds the `mpi/` environment (or another one, `--env`): WaterLily from `$WATERLILY_DIR`, MPI.jl bound to the system MPI, the WaterLily backend |
| `mpi/scaling.jl` | times `sim_step!` of a case on every rank and prints the time per step; `--check` also records global diagnostics |
| `mpi/check.jl` | compares the diagnostics of several runs with a reference run |
| `mpi/cuda_mpi_test.jl` | checks the CUDA-aware MPI alone: correctness on device buffers, and the ping-pong speed of device against host buffers (profile it with Nsight Systems to see how MPI moves device buffers) |
| `mpi/halo_profile.jl` | profiles ImplicitGlobalGrid's `update_halo!` of WaterLily-sized device arrays alone: time per call, the CUDA calls and kernels behind it (`CUDA.@profile`), and where the host spends the time (Julia's profiler); optionally without CUDA.jl's implicit synchronization between streams, or with the routine of WaterLily's `face_comm!` (all split dimensions at once, as in the Poisson iterations since WaterLily `12c339c`) |
| `mpi/halo_trace.jl` | times each phase of ImplicitGlobalGrid's `update_halo!` (packing, posting, waiting, unpacking), per dimension and rank, next to the bare MPI exchange of the same faces from plain, page-locked and device buffers; needs the timers of the ImplicitGlobalGrid fork (branch [`halo-trace`](https://github.com/b-fg/ImplicitGlobalGrid.jl/tree/halo-trace)) and `IGG_TRACE=1`, which also make `scaling.jl` print the time per step spent in `update_halo!`, by array size; with `IGG_FUSED=1` or `2` (fork branch [`halo-fused`](https://github.com/b-fg/ImplicitGlobalGrid.jl/tree/halo-fused)) the fused halo updates of CUDA arrays instead, and the updates split by `update_halo_start!` and `update_halo_finish!` (WaterLily branch `mpi-igg-overlap`, which overlaps the Poisson halos with their stencil loops; A/B in `mpi/jobs/jupiter_overlap_ab.sh`) under signatures of their own |
| `mpi/plot_strong.jl`, `mpi/plot_weak.jl` | plot the scaling curves from the job outputs (with the `plotting` environment) |
| `mpi/jobs/` | the environment and job scripts of each cluster (`jupiter_*`, `snellius_*`) and `submit.sh` |
| `mpi/fused/` | the environment of WaterLily branch `mpi-igg-fused` (its own halos, without ImplicitGlobalGrid) and its jobs, `mpi/fused/jobs/` |

## Setup

On a login node (it needs internet), with a WaterLily.jl clone on branch `mpi-igg`:

```sh
source mpi/jobs/jupiter_env.sh                                # the modules of the jobs, and WATERLILY_DIR
julia mpi/setup.jl --cuda-local                               # GPUs: KernelAbstractions backend, CUDA.jl on the CUDA module
julia mpi/setup.jl --env=mpi/cpu --backend=SIMD --cuda-local  # CPUs, one thread per rank: a second environment
WATERLILY_DIR=$WATERLILY_DIR-fused julia mpi/setup.jl --env=mpi/fused --cuda-local     # branch mpi-igg-fused
```

`jupiter_env.sh` expects Julia from juliaup in `PATH`, loads GCC, ParaStationMPI (or `$WATERLILY_MPI`, e.g. `OpenMPI`) and CUDA, sets `JULIA_CUDA_MEMORY_POOL=none` (CUDA-aware MPI cannot pass memory of CUDA.jl's pool, see the known issues of MPI.jl), and keeps `JULIA_DEPOT_PATH` and `WATERLILY_DIR` when they are set (by default both go under `$PROJECT/$USER`). On a CPU cluster use `julia mpi/setup.jl --backend=SIMD` (one thread per rank). `setup.jl` binds MPI.jl to the MPI library of the loaded module, so run it again whenever the modules change (the preferences go to `mpi/LocalPreferences.toml`, not tracked; Julia 1.12 and later also write the path of WaterLily into `mpi/Project.toml`, a local change not to commit); MPIPreferences does not recognise ParaStation MPI, so `setup.jl` gives it the MPICH ABI (`--mpi-abi` sets another). On a laptop without an MPI module, `--system-mpi=false` keeps MPI.jl's own MPICH. The smoke test that `setup.jl` prints ends with a run on 2 ranks: `mpiexec` on a laptop, `srun` in an allocation on a cluster (JUPITER's login nodes have no `mpiexec`). `--cuda-local` sets CUDA.jl's runtime preference before anything loads CUDA.jl: loaded with its own CUDA libraries first, CUDA.jl mixes them with those of the CUDA module. The backend preference is fixed per environment: GPU runs need `KernelAbstractions`, and `scaling.jl` stops otherwise. So the CPU runs on JUPITER, one thread per rank as on Snellius, have their own environment, `mpi/cpu` (not tracked), with the packages of `mpi/Project.toml` and the `SIMD` backend; `--cuda-local` there keeps CUDA.jl, which it also holds, on the libraries of the CUDA module. WaterLily branch `mpi-igg-fused` has its own environment too, `mpi/fused`, whose `Project.toml` (tracked, which `setup.jl` keeps) is that of `mpi/` without ImplicitGlobalGrid, built from a WaterLily.jl clone of its own (`$WATERLILY_DIR-fused` above, next to the clone of `mpi/`), so that the clone of `mpi/` keeps its branch and the jobs that use it.

## Running `scaling.jl`

```sh
julia --project=mpi mpi/scaling.jl --case=tgv --log2p=7                               # serial
mpiexec -n 4 julia --project=mpi mpi/scaling.jl --case=tgv --log2p=7                  # 4 CPU ranks
srun -n 8 julia --project=mpi mpi/scaling.jl --case=tgv --log2p=8 --backend=CuArray   # 8 GPUs
```

The cases are those of `src/cases.jl`: `tgv` is `tgv_periodic` (triply periodic, a `(2^p)^3` box) and `sphere` is the sphere with a convective exit (a `16D×6D×6D` box, `D = 2^p`). `--log2p` sets the global size; with `--weak` it is the size per rank, and the global grid grows with the number of ranks (each rank of the weak TGV holds one period). `--dims=a,b,c` gives the size in cells instead. Other options: `--backend=Array|CuArray`, `--T=Float32|Float64`, `--steps` and `--warmup` (timed and untimed steps), and `--serial` (no MPI, even under `srun`). Rank 0 prints

```
[ Info: MPI np=8  case=tgv  grid=256x256x256  local=128x128x128  100 steps  per_step=...ms  levels=7  mean_iters=1.0  cost=...ns/DOF/dt  backend=CuArray
```

after a header with the commits of WaterLily and of WaterLily-Benchmarks, the MPI library and the device of each rank. When every rank sees all the GPUs of its node, ImplicitGlobalGrid gives each node-local rank its own GPU; when the launcher binds one GPU per rank (Slurm `--gpus-per-task=1`, the default on JUPITER), each rank keeps the one it sees. Check in the header that the ranks of a node show different GPUs. To test the GPU code paths on a machine with one GPU, let several ranks share it: `CUDA_VISIBLE_DEVICES=0 mpiexec -n 2 ...` (the timings are then meaningless).

## Correctness of the multi-GPU TGV

`--check=k` records the global kinetic energy, enstrophy, `max|u|`, `max|p|`, time step and Poisson iterations every `k` steps, and `--out=file.csv` saves them. [`mpi/jobs/jupiter_tgv_check.sh`](https://github.com/WaterLily-jl/WaterLily-Benchmarks/tree/main/mpi/jobs/jupiter_tgv_check.sh) runs the TGV from its initial condition on one GPU without MPI and with MPI on 1, 4 and 8 GPUs (two nodes), on 8 GPUs with CUDA-aware halos, and on 8 CPU ranks, then compares them all with `mpi/check.jl`:

```sh
sbatch -A <budget> mpi/jobs/jupiter_tgv_check.sh 7 200 10   # log2p, steps, record every 10 steps
```

What to expect, from the same check of a `64^3` TGV over 100 steps on one laptop GPU shared by up to 4 ranks:

- MPI on one GPU is identical to the run without MPI (`identical` in the table), as on CPUs.
- CUDA-aware (`IGG_CUDAAWARE_MPI=1`) and host-staged (`=0`) halos give identical results: only the transport changes.
- With more ranks, the diagnostics differ slightly from the serial run (largest relative difference of the kinetic energy about `4e-4` on 2 and 4 ranks): the halo is one cell wide, so at the faces between ranks the convection term uses the one-sided form of the domain boundaries.
- GPU and CPU runs with the same decomposition differ by FP32 round-off (about `3e-5`).

The halos of the solver (velocity, pressure, multigrid levels) go through ImplicitGlobalGrid, which either passes device buffers to a CUDA-aware MPI (`IGG_CUDAAWARE_MPI=1`) or stages them through host memory (`=0`). `jupiter_env.sh` stages them: on the GH200 nodes the copy to the host is cheap, and with ParaStation MPI a `512^3` TGV took 64.5 against 75.2 ms per step on 4 GPUs and 41.7 against 79.8 ms on 8 GPUs of two nodes. On one node `mpi/halo_profile.jl` finds `update_halo!` slower with device buffers at every face size of that run (239 against 227 µs per call at the 530 KB faces of `512^3`, 542 against 152 µs at 134 KB, 212 against 136 µs at 9 KB), although MPI alone moves a 530 KB face between two GPUs of a node about 3 times faster from device buffers (`mpi/cuda_mpi_test.jl`). ImplicitGlobalGrid can also choose per direction (`IGG_CUDAAWARE_MPI_DIMX`, `_DIMY`, `_DIMZ`), but do not stage x while passing another direction as device buffers: ImplicitGlobalGrid 0.17.3 then never exchanges the x halos. With x and y on device buffers and z, the direction across the nodes (`srun --distribution=cyclic`), staged, 8 GPUs took 63.7 against 51.2 ms. `ForwardDiff.Dual` and `Bool` arrays, which ImplicitGlobalGrid does not take, go through WaterLily's own `MPI.Sendrecv!` on device buffers, so they need a CUDA-aware MPI either way.

WaterLily branch `mpi-igg-fused` exchanges the halos itself, without ImplicitGlobalGrid: one message per neighbour rank (the diagonal ones included), packed and unpacked by one kernel each, for every array (velocity, pressure, multigrid levels, `Dual`, `Bool`), and the large Poisson, divergence and velocity-correction loops run on the cells that read no ghost while the messages travel (loops of at least `WATERLILY_OVERLAP_MIN` cells, default `2^20`). Its GPU halos are staged through page-locked host memory mapped into the device, unless `WATERLILY_GPU_AWARE_MPI` is `1` (every halo device to device, CUDA-aware MPI) or `N > 1` (the halos of at least `N` bytes); `mpi/fused/jobs/jupiter_fused_ab.sh` times these settings and checks that they give identical runs (`--project=mpi/fused`, results in `mpi/fused/data/`, not tracked). On 8 GPUs of 2 nodes the TGV 512³ took 47.6 ms per step there, against 32.9 ms with the ImplicitGlobalGrid fork's fused halos (on one node, 57.5 against 56.4 ms). `mpi/fused/jobs/jupiter_fused_diag.sh` runs both in one allocation, on 8 GPUs and on 4 (`mpi/fused/halo_diag.jl`, a wrapper of `scaling.jl`, times each phase of each halo with `DIAG_LAPS=1`, and stages through host buffers of the fork's kind with `DIAG_BUFFERS=register`). Its first run kept the gap (47.0 against 33.7 and 32.5 ms), also with the fork's buffers (47.2), and showed the host waiting for the pack and unpack kernels: they searched the halo regions per element, 3 ms for a 258³ halo of the 26 regions of 2×2×2 ranks on a laptop GPU, against 13 µs with the index of each element precomputed (WaterLily f38c69a). With that fix the same runs gave 31.8 ms on 8 GPUs, against the fork's 33.5 and 32.8, and a later run 32.4 (32.9 without the overlap, 37.0 device to device) against 34.4 and 33.7, and 54.9 against 56.9 ms on 4 GPUs. With the fix, device to device does not pay (`jupiter_fused_ab.sh`, job 2583225, every check identical): on 4 GPUs staged 55.7, device to device 56.6, from 1 MiB 54.3 ms; on 8 GPUs of 2 nodes 32.1, 37.0 and 33.5 ms. `mpi/fused/jobs/jupiter_fused_nsys.sh` traces both transports with Nsight Systems, to see what the MPI library does with a device buffer: the ping-pong of `mpi/cuda_mpi_test.jl` inside a node and between two, and the TGV with every halo and its phases marked (`halo_diag.jl`, `DIAG_NVTX=1`) on one rank of each node (`mpi/fused/jobs/nsys_ranks.sh`); `mpi/fused/nsys_halos.py` reads the SQLite exports and prints, per halo array and phase or per ping-pong size, the time and the CUDA API calls and memory copies made in it. `IGG_CUDAAWARE_MPI`, `IGG_FUSED` and `IGG_TRACE` do not apply to it, and `mpi/halo_profile.jl` and `mpi/halo_trace.jl` time ImplicitGlobalGrid's halos only. Like ImplicitGlobalGrid's `select_device`, it gives each rank of a node its own GPU and stops when a node has more ranks than GPUs, unless the launcher binds one GPU per rank (one device in `CUDA_VISIBLE_DEVICES`, as with Slurm's `--gpus-per-task=1`).

## Scaling curves

`mpi/jobs/submit.sh` submits one job per node count, each running `mpi/jobs/<cluster>_scaling.sh`:

```sh
bash mpi/jobs/submit.sh jupiter strong tgv 9 "1 2 4 8 16" -- -A <budget>
bash mpi/jobs/submit.sh jupiter weak tgv 8 "1 2 4 8 16 32" -- -A <budget>
bash mpi/jobs/submit.sh jupiter_cpu strong tgv 8 "1 2 4 8 16" -- -A <budget>
bash mpi/jobs/submit.sh snellius weak tgv 6 "1 2 4 8"
```

`WATERLILY_ENV` sets the Julia environment of the runs (by default `mpi`, and `mpi/cpu` for `jupiter_cpu`), and the outputs then go to a directory named after it: `WATERLILY_ENV=mpi/fused bash mpi/jobs/submit.sh jupiter weak tgv 8 "1 2 4 8 16 32" -- -A <budget>` runs WaterLily branch `mpi-igg-fused` and writes to `data/mpi/jupiter_weak_tgv_8_fused/`. The job on one node also runs the points inside it (on JUPITER 1, 2 and 4 GPUs, the first without MPI, which is the reference of the efficiency); the others run one point each, so a point that failed can be submitted again alone (`... weak tgv 6 "32" -- -A <budget>`). Each GPU job prints the cores of the ranks of its first node, which should be those of one Grace CPU each (0-71, 72-143, ...). The outputs go to `data/mpi/<cluster>_<mode>_<case>_<log2p>/`, and the plot scripts read a whole directory and save the figure in it:

```sh
julia --project=plotting mpi/plot_strong.jl data/mpi/jupiter_strong_tgv_9
julia --project=plotting mpi/plot_weak.jl data/mpi/jupiter_weak_tgv_8 --refnp=4   # efficiency against one full node
```

On Snellius, `WATERLILY_RPN` sets the ranks per node (default 64), spread evenly over the CCXs of each node. On JUPITER there is one rank per GPU, 4 per node. `jupiter_cpu` runs `mpi/jobs/jupiter_cpu_scaling.sh` on the CPUs of the JUPITER nodes (4 Grace CPUs of 72 cores): one thread per rank, `WATERLILY_RPN` ranks per node (default 256, which splits the `2^p` grids, unlike all 288 cores), the same number on each Grace CPU, spread evenly over its cores. The job prints the NUMA nodes of the first node, to check that the cores of each Grace CPU are numbered as the binding assumes.
