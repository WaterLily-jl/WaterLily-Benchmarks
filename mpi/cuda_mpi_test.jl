# CUDA-aware MPI without WaterLily: correctness, then the speed of device against host buffers between two ranks,
# then ImplicitGlobalGrid's halo exchange of arrays of the size WaterLily's would have.
#
#   srun -n 2 julia --project=mpi mpi/cuda_mpi_test.jl [N=128]  # two GPUs of one node (-N 2 -n 2: two nodes)
#
# Rank 0 and 1 send a buffer back and forth (ping-pong) and rank 0 prints the one-way time and bandwidth per size.
# Then all ranks split a periodic N^3 box as WaterLily does (one ghost cell per side) and rank 0 prints the time of
# one update_halo! of a scalar field and of the three velocity components, for device and host arrays, of the MPI
# calls of one such exchange alone (receive from and send to both neighbours of the first split dimension, buffers of
# the face size), and of an Allreduce of one number: per time step WaterLily makes about 60 halo exchanges and 15
# reductions (TGV, 2 ranks).
# A CUDA-aware MPI moves device buffers between the GPUs (NVLink inside a node) at least as fast as host buffers;
# much slower device buffers mean the MPI library stages them through the host. To see how it moves them, profile it:
#
#   srun -n 2 nsys profile --trace=cuda,osrt -o cuda_mpi_%q{SLURM_PROCID} -f true julia --project=mpi mpi/cuda_mpi_test.jl
#   nsys stats --report cuda_gpu_mem_time_sum,cuda_api_sum cuda_mpi_0.nsys-rep
#
# where the memory copies of the MPI library show up as peer-to-peer or device-to-device (direct) or as
# device-to-host and host-to-device (staged). NVTX ranges mark each size and buffer kind of the ping-pong, and each
# halo test, their payload the number of round trips or calls (mpi/fused/jobs/jupiter_fused_nsys.sh).
using MPI, CUDA, ImplicitGlobalGrid, Printf
# NVTX.jl comes with CUDA.jl, not as a dependency of the environment; its ranges are no-ops outside a profiler
const NVTX = Base.loaded_modules[Base.PkgId(Base.UUID("5da4648a-3479-48b8-97b9-01cb529c0a1f"), "NVTX")]
MPI.Init()
const comm = MPI.COMM_WORLD
const me, np = MPI.Comm_rank(comm), MPI.Comm_size(comm)
np >= 2 || error("run on at least 2 ranks")
left, right = mod(me - 1, np), mod(me + 1, np)

device = "$(gethostname()): $(CUDA.name(CUDA.device())) [uuid $(string(CUDA.uuid(CUDA.device()))[1:8])]"
devices = MPI.gather(device, comm)
if me == 0
    println("MPI: $(first(split(MPI.Get_library_version(), '\n'))), $np ranks, has_cuda=$(MPI.has_cuda())")
    foreach(((r, d),) -> println("  rank $(r-1): $d"), enumerate(devices))
end

# Correctness: a ring shift and an Allreduce on device buffers
send = CUDA.fill(Float32(me), 2^20); recv = CUDA.zeros(Float32, 2^20); CUDA.synchronize()
MPI.Sendrecv!(send, recv, comm; dest=right, source=left)
shift_ok = MPI.Allreduce(all(==(left), Array(recv)), &, comm)
x = CUDA.fill(1f0, 2^20); CUDA.synchronize()
MPI.Allreduce!(x, +, comm)
sum_ok = MPI.Allreduce(all(==(np), Array(x)), &, comm)
me == 0 && println("device buffers: Sendrecv! ", shift_ok ? "ok" : "WRONG", ", Allreduce! ", sum_ok ? "ok" : "WRONG")

# Ping-pong between rank 0 and 1; one-way time per message
function pingpong(buf, n)
    for warmup in (true, false)
        MPI.Barrier(comm); t = MPI.Wtime()
        for _ in 1:(warmup ? 10 : n)
            if me == 0
                MPI.Send(buf, comm; dest=1); MPI.Recv!(buf, comm; source=1)
            elseif me == 1
                MPI.Recv!(buf, comm; source=0); MPI.Send(buf, comm; dest=0)
            end
        end
        warmup || return (MPI.Wtime() - t) / 2n
    end
end
me == 0 && @printf("%10s  %12s %10s  %12s %10s\n", "bytes", "device [us]", "[GB/s]", "host [us]", "[GB/s]")
for bytes in (8, 1024, 32 * 1024, 256 * 1024, 2^20, 8 * 2^20, 64 * 2^20)
    n = clamp(2^28 ÷ bytes, 20, 2000)
    bd, bh = CUDA.zeros(UInt8, bytes), zeros(UInt8, bytes)
    td = NVTX.@range "pingpong device $bytes B" payload=n+10 pingpong(bd, n)
    th = NVTX.@range "pingpong host $bytes B" payload=n+10 pingpong(bh, n)
    me == 0 && @printf("%10d  %12.2f %10.2f  %12.2f %10.2f\n", bytes, 1e6td, bytes / td / 1e9, 1e6th, bytes / th / 1e9)
end

# ImplicitGlobalGrid's halo exchange, as WaterLily calls it: arrays of N/dims+2 cells, halos of the split dimensions
N = parse(Int, get(ARGS, 1, "128"))
mpidims = Tuple(Int.(MPI.Dims_create(np, [0, 0, 0])))
nxyz = N .÷ mpidims .+ 2
init_global_grid(nxyz...; dimx=mpidims[1], dimy=mpidims[2], dimz=mpidims[3], periodx=1, periody=1, periodz=1,
                 overlaps=(2, 2, 2), halowidths=(1, 1, 1), init_MPI=false, select_device=false, quiet=true)
splitdims = Tuple(findall(>(1), collect(mpidims)))
function per_call(f, n)
    for _ in 1:10; f(); end
    CUDA.synchronize(); MPI.Barrier(comm); t = MPI.Wtime()
    for _ in 1:n; f(); end
    CUDA.synchronize(); MPI.Barrier(comm); 1e6 * (MPI.Wtime() - t) / n
end
d = first(splitdims); lo, hi = ImplicitGlobalGrid.neighbors(d)  # the neighbours along the first split dimension
face = prod(nxyz[k] for k in 1:3 if k != d)
function exchange!(s, r)  # the MPI calls of one halo exchange of one field along d
    reqs = [MPI.Irecv!(r[1], comm; source=lo, tag=1), MPI.Irecv!(r[2], comm; source=hi, tag=2),
            MPI.Isend(s[1], comm; dest=hi, tag=1), MPI.Isend(s[2], comm; dest=lo, tag=2)]
    MPI.Waitall(reqs)
end
for (name, alloc) in (("device", CUDA.rand), ("host", rand))
    p = alloc(Float32, nxyz...); u = [alloc(Float32, nxyz...) for _ in 1:3]; CUDA.synchronize()
    s, r = [alloc(Float32, face) for _ in 1:2], [alloc(Float32, face) for _ in 1:2]; CUDA.synchronize()
    tp = NVTX.@range "$name update_halo! scalar" payload=210 per_call(() -> update_halo!(p; dims=splitdims), 200)
    tu = NVTX.@range "$name update_halo! velocity" payload=210 per_call(() -> update_halo!(u...; dims=splitdims), 200)
    tm = NVTX.@range "$name MPI only" payload=210 per_call(() -> exchange!(s, r), 200)
    me == 0 && @printf("%-6s %s, split %s: update_halo! scalar %7.1f us, velocity %7.1f us; MPI only %7.1f us\n",
                       name, join(nxyz, "x"), join(mpidims, "x"), tp, tu, tm)
end
ta = per_call(() -> MPI.Allreduce(1f0, +, comm), 1000)
me == 0 && @printf("Allreduce of one Float32: %.1f us (IGG_CUDAAWARE_MPI=%s)\n", ta, get(ENV, "IGG_CUDAAWARE_MPI", ""))
finalize_global_grid(finalize_MPI=false)
MPI.Finalize()
