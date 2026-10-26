# CUDA-aware MPI without WaterLily: correctness, then the speed of device against host buffers between two ranks.
#
#   srun -n 2 julia --project=mpi mpi/cuda_mpi_test.jl          # two GPUs of one node (-N 2 -n 2: two nodes)
#
# Rank 0 and 1 send a buffer back and forth (ping-pong) and rank 0 prints the one-way time and bandwidth per size.
# A CUDA-aware MPI moves device buffers between the GPUs (NVLink inside a node) at least as fast as host buffers;
# much slower device buffers mean the MPI library stages them through the host. To see how it moves them, profile it:
#
#   srun -n 2 nsys profile --trace=cuda,osrt -o cuda_mpi_%q{SLURM_PROCID} -f true julia --project=mpi mpi/cuda_mpi_test.jl
#   nsys stats --report cuda_gpu_mem_time_sum,cuda_api_sum cuda_mpi_0.nsys-rep
#
# where the memory copies of the MPI library show up as peer-to-peer or device-to-device (direct) or as
# device-to-host and host-to-device (staged).
using MPI, CUDA, Printf
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
    td = pingpong(CUDA.zeros(UInt8, bytes), n)
    th = pingpong(zeros(UInt8, bytes), n)
    me == 0 && @printf("%10d  %12.2f %10.2f  %12.2f %10.2f\n", bytes, 1e6td, bytes / td / 1e9, 1e6th, bytes / th / 1e9)
end
MPI.Finalize()
