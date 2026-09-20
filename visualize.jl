include("cases.jl")
include("util.jl")
using GLMakie  # WaterLily's `viz!` extension

# Render a vorticity image per checkpoint into `checkpoints/viz/` as a visual check. Float64 cases are skipped.
function visualize_checkpoints(cases, log2p, ftype, backend; ckpt_dir="checkpoints/", img_dir="checkpoints/viz/")
    mkpath(img_dir)
    for (case, ps, ft) in zip(cases, log2p, ftype)
        ft === Float64 && (println("skipping $(case) (Float64)"); continue)
        for n in ps
            sim = getf(case)(n, backend; T=ft)
            load!(sim.flow; fname=checkpoint_name(case, n, ft), dir=ckpt_dir); measure!(sim)
            img = joinpath(img_dir, "$(case)_p$(n).png")
            viz!(sim; img=img, hidedecorations=true, verbose=false)
            println("  → $(img)"); flush(stdout)
        end
    end
end

cases, log2p, max_steps, ftype, backend, data_dir = parse_cla(ARGS;
    cases=all_cases, log2p=[(6,7), (3,4), (4,5), (5,6), (5,6)],
    max_steps=fill(25, length(all_cases)), ftype=fill(Float32, length(all_cases)),
    backend=Array, data_dir="checkpoints/"
)
visualize_checkpoints(cases, log2p, ftype, backend; ckpt_dir=data_dir)
