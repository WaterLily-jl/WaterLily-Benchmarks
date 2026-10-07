# BiotSavartBCs changes

The `-biotsavart` cases solve the pressure Poisson equation through [BiotSavartBCs.jl](https://github.com/WaterLily-jl/BiotSavartBCs.jl)'s own `mom_project!` rather than WaterLily's `solver!`. A change that spans both packages, such as the stopping criterion of the Poisson solver, can only be measured by switching BiotSavartBCs together with WaterLily.

By default the benchmark environment follows the `main` branch of BiotSavartBCs (the `[sources]` entry of `Project.toml`). To pair each WaterLily ref with a BiotSavartBCs ref, pass `-bs` a list of BiotSavartBCs branches, tags or hashes, one per `-w` ref, and a local BiotSavartBCs clone with `-bsd` or `$BIOTSAVART_DIR`:

```sh
./benchmark.sh -w "master poisson-rms-tol" -bs "main combined-tol" -bsd ~/BiotSavartBCs.jl -c jelly-biotsavart -b Array -t "1 4" -r 3
```

Each run checks out the WaterLily ref first and then its BiotSavartBCs ref, since a BiotSavartBCs branch can need new WaterLily functions. A change in BiotSavartBCs alone is measured against a fixed WaterLily with `-w "master master" -bs "main my-branch"`.

## How it works

- `-bs` points the `[sources]` entry of `Project.toml` at the local clone and forces `-u true`. On exit, also after an error or an interrupt, the sweep restores `Project.toml`, `Manifest.toml` and the `gpu/` environments as they were, since Pkg would otherwise keep using the clone. On a clone that had no `Manifest.toml` before the sweep, this removes it again: the next `benchmark.sh` run installs it, but `compare.jl` needs `julia --project -e 'using Pkg; Pkg.instantiate()'` first (the action does this).
- A `-bs` run is tagged `<WaterLily hash>+bs<BiotSavartBCs hash>`, in the results and in the file and directory names, so runs that differ only in BiotSavartBCs stay apart.
- `compare.jl` shows such a run as e.g. `master (bs main)`, naming the BiotSavartBCs ref from the clone in `$BIOTSAVART_DIR`, and `--speedup_base` also accepts a BiotSavartBCs ref or hash.

On [pull requests](pull-requests.md), `-bs` works the same way: the action checks out BiotSavartBCs.jl itself, so the refs must exist on GitHub. For example, `/benchmark -w "pr-123 pr-123" -bs "main my-branch"` compares two BiotSavartBCs branches on the pull request's WaterLily.
