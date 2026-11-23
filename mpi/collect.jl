# One CSV row per run of mpi/scaling.jl, read from the files that hold its output (the runs of mpi/jobs/jupiter_ad.sh,
# one file each, or a job output with several runs); a directory stands for its *.txt files:
#
#   julia mpi/collect.jl [--out=runs.csv] [--table] DIR_OR_FILE [DIR_OR_FILE ...]
#
# Everything comes from what the run printed (its header, the `Info:` line, `MEMORY` and, with --laps, `HALO`), not from
# file names: the commits, the eltype and its partials, the ranks, the sizes, the time per step, the Poisson
# iterations, the bytes per cell and the halo phases (ms per step). `job` is the Slurm job of a `# run=... job=...`
# line, which the job scripts write before each run. `ms_ratio` and `mem_ratio` compare a run with the mean of the
# Float runs without partials of the same job, case, mode, sizes, ranks, transport and timers (`iters_differ` is 1
# when their Poisson iterations are not those of the run, so that the ratio also holds another amount of work).
# --out writes the CSV to a file instead of the standard output, --table prints the main columns aligned.
# Plain Julia, no packages.
using Printf

const columns = (:file, :job, :waterlily, :benchmarks, :case, :mode, :launch, :np, :T, :partials, :grid, :local, :cells,
    :steps, :ms_per_step, :mean_iters, :levels, :backend, :halo_backend, :gpu_aware, :laps, :halo_ms, :halo_calls,
    :wait_ms, :pack_ms, :post_ms, :work_ms, :wait_recv_ms, :unpack_ms, :wait_send_ms, :laps_sync, :arrays_B_per_cell,
    :halos_B_per_cell, :device_MB, :host_rss_MB, :ms_ratio, :mem_ratio, :iters_differ, :device, :mpi)
keyvals(line) = Dict(String(m[1]) => String(m[2]) for m in eachmatch(r"(\w+)=([^\s,]+)", line))

function runs(file)
    out = Dict{Symbol,Any}[]; job = ""; r = Dict{Symbol,Any}()
    done!() = (haskey(r, :ms_per_step) && push!(out, r); r = Dict{Symbol,Any}(:file => file, :job => job, :laps => 0))
    for line in eachline(file)
        if startswith(line, "# run=")
            job = get(keyvals(line), "job", "")
        elseif (m = match(r"^WaterLily \S+ \((\w+) .*WaterLily-Benchmarks (\w+), .*backend (\w+), (\w+), (\w+)(?:, partials=(\d+))?", line)) !== nothing
            done!()  # the header starts a run
            r[:waterlily], r[:benchmarks], r[:backend], r[:T] = m[1], m[2], m[4], m[5]
            r[:partials] = something(m[6], "0")
        elseif isempty(r)
            continue
        elseif (m = match(r"^MPI: (.*?), has_cuda", line)) !== nothing
            kv = keyvals(line); r[:mpi] = strip(replace(m[1], "=" => "", "," => " "))
            r[:halo_backend] = get(kv, "halo", "igg")
            r[:gpu_aware] = get(kv, "WATERLILY_GPU_AWARE_MPI", get(kv, "IGG_CUDAAWARE_MPI", ""))
        elseif (m = match(r"^  rank 0: (.*)", line)) !== nothing
            r[:device] = replace(strip(m[1]), "," => " ")
        elseif (m = match(r"Info: (SERIAL|MPI np=(\d+))\s+case=", line)) !== nothing
            kv = keyvals(line)
            r[:launch], r[:np] = m[2] === nothing ? ("serial", "1") : ("mpi", m[2])
            r[:case], r[:grid], r[:local] = kv["case"], kv["grid"], kv["local"]
            r[:mode] = get(kv, "mode", "strong"); r[:steps] = match(r"(\d+) steps", line)[1]
            r[:ms_per_step] = chopsuffix(kv["per_step"], "ms"); r[:mean_iters], r[:levels] = kv["mean_iters"], kv["levels"]
            haskey(kv, "partials") && (r[:T] = kv["T"]; r[:partials] = kv["partials"])
        elseif occursin("Info: MEMORY", line)
            kv = keyvals(line); r[:cells] = kv["cells"]
            r[:arrays_B_per_cell], r[:halos_B_per_cell] = chopsuffix(kv["arrays"], "B/cell"), chopsuffix(kv["halos"], "B/cell")
            r[:device_MB], r[:host_rss_MB] = chopsuffix(kv["device"], "MB"), chopsuffix(kv["host_rss"], "MB")
        elseif occursin("Info: HALO", line)
            kv = keyvals(line); r[:laps] = 1
            r[:halo_ms], r[:halo_calls], r[:laps_sync] = chopsuffix(kv["halo"], "ms"), kv["calls"], kv["sync"]
            foreach(p -> r[Symbol(p, :_ms)] = kv[p], ("wait", "pack", "post", "work", "wait_recv", "unpack", "wait_send"))
        end
    end
    done!(); out
end

# The ratios of each run to the Float runs without partials of its point
function ratios!(rows)
    num(r, c) = parse(Float64, r[c]); point(r) = map(c -> get(r, c, ""), (:job, :case, :mode, :grid, :np, :launch, :T, :gpu_aware, :laps, :backend))
    for r in rows
        base = filter(b -> b[:partials] == "0" && point(b) == point(r), rows)
        (isempty(base) || r[:partials] == "0") && continue
        r[:ms_ratio] = @sprintf("%.3f", num(r, :ms_per_step)/(sum(b -> num(b, :ms_per_step), base)/length(base)))
        r[:iters_differ] = Int(any(b -> b[:mean_iters] != r[:mean_iters], base))
        haskey(r, :arrays_B_per_cell) && all(b -> haskey(b, :arrays_B_per_cell), base) &&
            (r[:mem_ratio] = @sprintf("%.3f", num(r, :arrays_B_per_cell)/num(base[1], :arrays_B_per_cell)))
    end
end

paths = filter(a -> !startswith(a, "--"), ARGS)
isempty(paths) && error("usage: julia mpi/collect.jl [--out=runs.csv] [--table] DIR_OR_FILE [...]")
files = reduce(vcat, [isdir(p) ? sort!(filter(endswith(".txt"), readdir(p; join=true))) : [p] for p in paths])
rows = reduce(vcat, runs.(files); init=Dict{Symbol,Any}[])
isempty(rows) && error("no run of mpi/scaling.jl in $(join(paths, ", "))")
ratios!(rows)
sort!(rows, by=r -> (r[:job], r[:mode], r[:case], prod(parse.(Int, split(r[:local], "x"))), parse(Int, r[:np]), get(r, :gpu_aware, ""), r[:laps],
                     parse(Int, r[:partials]), r[:file]))
foreach(r -> r[:file] = basename(r[:file]), rows)

if "--table" in ARGS
    cols = (:file, :np, :T, :partials, :grid, :local, :gpu_aware, :ms_per_step, :ms_ratio, :mean_iters, :halo_ms, :arrays_B_per_cell, :mem_ratio)
    w = [max(length(string(c)), maximum(r -> length(string(get(r, c, ""))), rows)) + 2 for c in cols]
    println(join(rpad.(string.(cols), w)))
    foreach(r -> println(join(rpad.([string(get(r, c, "")) for c in cols], w))), rows)
    any(r -> get(r, :iters_differ, 0) == 1, rows) && println("WARNING: some runs with partials make other Poisson iterations than their Float runs (iters_differ in the CSV)")
end
out = findlast(startswith("--out="), ARGS)
if !isnothing(out) || !("--table" in ARGS)
    io = isnothing(out) ? stdout : open(split(ARGS[out], "="; limit=2)[2], "w")
    println(io, join(columns, ","))
    foreach(r -> println(io, join((get(r, c, "") for c in columns), ",")), rows)
    isnothing(out) || (close(io); println("$(length(rows)) runs written to $(split(ARGS[out], "="; limit=2)[2])"))
end
