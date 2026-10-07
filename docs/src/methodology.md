# Methodology

How `benchmark.sh` measures a step and how `compare.jl` turns the measurements into the columns of its tables.

## What is timed

One sample is a single `sim_step!` followed by a backend synchronisation. A *run* is `max_steps` consecutive steps (default `25`), and each benchmark does 5 runs. With developed flows (the default), every run is first reset to the same checkpoint, so all runs time the same steps of the flow; with `-dev ""` the runs march on continuously from the startup transient. Before the runs, the case is warmed up from the same state for at least 50 steps and 2 seconds (JIT, device clocks), and `GC.gc()` is called.

## Reference time

For each run, take the mean of its per-step times, i.e. the run time divided by `max_steps`. The reference, shown as `Mean`, is the **minimum of the run means**, which discards runs slowed down from outside, as BenchmarkTools' minimum does: a step slowed down by an interrupt or a GC pause raises the mean of its run, and the minimum drops that run. Every run times the same steps, so the mean counts all the work of those steps, also when steps do different amounts of work. In `jelly-biotsavart`, for example, a pressure solve needs a varying number of Biot-Savart boundary updates, so its step times come in levels.

The minimum of the run medians is still shown, as `Median`. It describes the typical step and ignores occasional slow steps, but it is not a measure of cost when steps do different amounts of work: when about half the steps sit on each level, one step changing level moves the median by the whole gap between levels, while the total work barely changes. In the example below, `Median` of `jelly-biotsavart` on CPUx01 jumps by 17% from v1.7.0 to v1.8.0 while `Mean` changes by 1%. When all steps do the same work, as in `tgv` or `sphere`, `Mean` and `Median` are close.

## Table columns

An example table from a comparison of WaterLily releases (`jelly-biotsavart` at `log2p = 5`, master as speedup baseline, 3 repetitions):

```
▶ log2p = 5
┌────────────┬───────────┬────────┬─────────┬───────┬────────┬────────┬─────────────┬─────────┬───────┬──────────────┬─────────┬──────┐
│  Backend   │ WaterLily │ Julia  │   FP    │ Alloc │  Mean  │ Median │    Cost     │ Speedup │ Noise │    Δ ± σ     │ Signif  │ Reps │
│            │           │        │         │  [k]  │  [ms]  │  [ms]  │ [ns/DOF/dt] │         │  [%]  │     [%]      │ [|Δ|/σ] │      │
├────────────┼───────────┼────────┼─────────┼───────┼────────┼────────┼─────────────┼─────────┼───────┼──────────────┼─────────┼──────┤
│     CPUx01 │    master │ 1.11.5 │ Float32 │   0.7 │ 157.99 │ 165.25 │     1205.41 │    1.00 │   0.4 │            - │       - │    3 │
│     CPUx01 │    v1.6.1 │ 1.11.5 │ Float32 │   0.8 │ 159.65 │ 143.64 │     1218.04 │    0.99 │   0.6 │  +1.0 ±  0.7 │    1.5  │    3 │
│     CPUx01 │    v1.7.0 │ 1.11.5 │ Float32 │   0.8 │ 160.19 │ 144.18 │     1222.13 │    0.99 │   0.5 │  +1.4 ±  0.6 │    2.2  │    3 │
│     CPUx01 │    v1.8.0 │ 1.11.5 │ Float32 │   0.8 │ 161.27 │ 168.63 │     1230.37 │    0.98 │   0.5 │  +2.1 ±  0.7 │    3.1  │    3 │
│     CPUx04 │    master │ 1.11.5 │ Float32 │  33.2 │  56.61 │  57.74 │      431.90 │    2.79 │   1.6 │            - │       - │    3 │
│     CPUx04 │    v1.6.1 │ 1.11.5 │ Float32 │  51.6 │  63.00 │  57.98 │      480.61 │    2.51 │   1.2 │ +11.3 ±  2.0 │    5.7  │    3 │
│     CPUx04 │    v1.7.0 │ 1.11.5 │ Float32 │  53.6 │  64.39 │  59.71 │      491.28 │    2.45 │   1.1 │ +13.7 ±  1.9 │    7.2  │    3 │
│     CPUx04 │    v1.8.0 │ 1.11.5 │ Float32 │  62.3 │  65.31 │  66.21 │      498.29 │    2.42 │   1.5 │ +15.4 ±  2.2 │    7.0  │    3 │
│ GPU-NVIDIA │    master │ 1.11.5 │ Float32 │  36.0 │   7.67 │   7.80 │       58.55 │   20.59 │   5.6 │            - │       - │    3 │
│ GPU-NVIDIA │    v1.6.1 │ 1.11.5 │ Float32 │  31.9 │   7.81 │   7.25 │       59.61 │   20.22 │   1.4 │  +1.8 ±  5.8 │    0.3  │    3 │
│ GPU-NVIDIA │    v1.7.0 │ 1.11.5 │ Float32 │  34.0 │   7.44 │   6.77 │       56.76 │   21.24 │   4.6 │  -3.1 ±  7.3 │    0.4  │    3 │
│ GPU-NVIDIA │    v1.8.0 │ 1.11.5 │ Float32 │  42.6 │   8.29 │   8.37 │       63.26 │   19.05 │   4.5 │  +8.1 ±  7.2 │    1.1  │    3 │
└────────────┴───────────┴────────┴─────────┴───────┴────────┴────────┴─────────────┴─────────┴───────┴──────────────┴─────────┴──────┘
```

- **Alloc [k]**: allocations per step, averaged over a block of `max_steps` steps, divided by 1000. Only meaningful on the SIMD backend (CPUx01); the KernelAbstractions backends report kernel-launch bookkeeping.
- **GC [%]** (hidden by default; show with `--gc`): GC fraction of the fastest step.
- **Mean [ms]**: the reference time per step, the minimum over runs of the mean step of each run. It is the point estimate used for cost, speedup and Δ.
- **Median [ms]**: the minimum over runs of the median step of each run, i.e. the typical step. It differs from `Mean` when steps do different amounts of work, see [Reference time](@ref).
- **Cost [ns/DOF/dt]**: `Mean` divided by the number of cells, i.e. the cost per cell and per time step.
- **Speedup**: `Mean(speedup_base) / Mean`. The speedup baseline is a single row for the whole table, the first one by default (set it with `--speedup_base`), so it also compares across backends.
- **Noise [%]**: scatter of the row's measurement relative to `Mean` (one standard deviation), i.e. how much `Mean` changes from run to run and from process to process, see below.
- **Δ ± σ [%]**: cost difference against the *reference row*, which is the row of the same backend matching the speedup baseline's `(WaterLily ref, Julia, FP)`, e.g. the master row of that backend when comparing master against a PR. The reference row itself prints `-`. Positive means slower. σ is the scatter of Δ, `σ = sqrt(Noise_row² + Noise_ref²)`, because Δ is the difference of two noisy rows.
- **Signif [|Δ|/σ]**: significance of Δ (it is not Δ divided by the row's own `Noise`). Below about 1 the Δ is indistinguishable from scatter; believe it from about 2 to 3. A `*` marks a value where the row or its reference row has `Reps = 1`: σ then only covers the scatter within a process.
- **Reps**: number of repetitions (separate processes) merged into the row, see [Repetitions](@ref).

## Noise and repetitions

All the runs of a benchmark happen inside one Julia process, so they share any machine state that outlives the process (CPU frequency, thermal or power state). On a power- or thermal-limited machine such as a laptop, two processes measuring the same code a few minutes apart have been seen to differ by 14% while each reported a noise below 1%, and no run of the slow process reached the level of the fast one. A single process cannot measure this scatter, so with `Reps = 1` a small `Noise` and a large `Signif` do not make a Δ trustworthy on such machines. With `--repeats N` each benchmark is measured in `N` separate processes, and:

- `Mean` is the minimum over the processes of their reference times.
- `Noise` with `Reps = 1` is the std of the 5 run means, divided by `Mean`.
- `Noise` with `Reps > 1` is the larger of (a) the std of the per-process reference times and (b) the largest std of the run means of a process, divided by `Mean`. (a) measures how much the reported number changes from one process to the next; (b) is a floor, since (a) is poorly estimated from 2 or 3 processes. Both are standard deviations, so the column means the same with and without repetitions.

Example with 3 repetitions, run means in ms:

```
          Process 1   Process 2   Process 3
  Run 1     36.4        41.3        36.5
  Run 2     36.5        41.4        36.6
  Run 3     36.6        41.5        36.7
  Run 4     36.4        41.3        36.5
  Run 5     36.5        41.4        36.6

  per-process reference times (min of each column): 36.4, 41.3, 36.5  ->  Mean = 36.4
  (a) between processes: std(36.4, 41.3, 36.5) = 2.80 ms
  (b) within a process:  largest std of a column = 0.08 ms
  Noise = max(2.80, 0.08) / 36.4 = 7.7%
```

Three repetitions are a better default than two on a laptop: with two, (a) rests on a single difference, and both processes can land in the same machine state by chance.
