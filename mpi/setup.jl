# Build the MPI benchmark environment (mpi/Project.toml) on a login node, which has internet access:
#
#   source mpi/jobs/<cluster>_env.sh        # the MPI (and CUDA) modules used by the jobs
#   julia mpi/setup.jl [--backend=KernelAbstractions|SIMD] [--cuda-local] [--system-mpi=false]
#
# It develops WaterLily from $WATERLILY_DIR (a WaterLily.jl clone on branch mpi-igg), binds MPI.jl to the MPI
# library of the loaded module (MPIPreferences.use_system_binary), sets WaterLily's backend and precompiles.
#   --backend     KernelAbstractions (default) for GPUs and threaded CPUs, SIMD for one thread per rank on CPUs
#   --cuda-local  use the CUDA toolkit of the loaded module instead of the CUDA_jll artifacts, so that CUDA.jl and a
#                 CUDA-aware MPI use the same CUDA runtime
#   --system-mpi=false  keep MPI.jl's own MPICH (a laptop without an MPI module)
# Run it again after loading other modules: MPI.jl has to be bound to the libmpi that the jobs load.
arg(key, default) = (i = findlast(a -> startswith(a, "--$key="), ARGS); isnothing(i) ? default : split(ARGS[i], "="; limit=2)[2])
const backend = arg("backend", "KernelAbstractions")
backend in ("KernelAbstractions", "SIMD") || error("--backend must be KernelAbstractions or SIMD")

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"  # precompile once, at the end, with the preferences below in place
using Pkg
Pkg.activate(@__DIR__)

# WaterLily: the MPI extension lives on branch mpi-igg; print the commit so a wrong checkout shows up here, not in a job
wl_dir = get(ENV, "WATERLILY_DIR", "")
isdir(wl_dir) || error("set WATERLILY_DIR to a WaterLily.jl clone (branch mpi-igg)")
branch = readchomp(`git -C $wl_dir rev-parse --abbrev-ref HEAD`)
@info "WaterLily.jl at $wl_dir: $(readchomp(`git -C $wl_dir log -1 --format="%h %s"`)) ($branch)"
branch == "mpi-igg" || @warn "WaterLily.jl is on branch $branch, not mpi-igg"
Pkg.develop(path=wl_dir)
Pkg.instantiate()

# Preferences before precompiling: MPI.jl and CUDA.jl compile against the libraries they are bound to
if arg("system-mpi", "true") == "true"
    using MPIPreferences
    MPIPreferences.use_system_binary()
end
if any(==("--cuda-local"), ARGS)
    using CUDA
    CUDA.set_runtime_version!(local_toolkit=true)
end
using Preferences, UUIDs
set_preferences!(UUID("ed894a53-35f9-47f1-b17f-85db9237eebd"), "backend" => backend; force=true)  # WaterLily

Pkg.precompile()
Pkg.status()
println("\nWaterLily backend: $backend. Smoke test (serial, then 2 ranks):")
println("  julia --project=mpi mpi/scaling.jl --case=tgv --log2p=5 --steps=10 --warmup=5")
println("  mpiexec -n 2 julia --project=mpi mpi/scaling.jl --case=tgv --log2p=5 --steps=10 --warmup=5")
