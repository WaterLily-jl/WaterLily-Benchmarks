# Automatic profiling

Similarly to benchmarking, profiling is also automated through [`profile.sh`](profile.sh). It is currently designed to do profiling on NVIDIA GPUs only.The main kernels of the solver, generated by [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), are traced using [NVTX.jl](https://github.com/JuliaGPU/NVTX.jl). Traces are introduced in the [`profiling`](https://github.com/WaterLily-jl/WaterLily.jl/tree/profiling) branch of WaterLily.

The NVIDIA Nsight performance analysis tool is used to track the NVTX traces. It launches the profiling script as: `nsys profile --sample=none --trace=nvtx,cuda --output=$DATA_DIR/$case/$case.nsys-rep --export=sqlite --force-overwrite=true julia $case_arguments`. The post-processing tool used the `nsys stats -r nvtx_gpu_proj_sum` command on the profiling data to obtain the final results. More options are available through the `nsys` API, and the Nsight GUI.

The `profile.sh` script works with the same arguments as the `benchmark.sh` script, see the main [README.md](README.md) file for a detailed explanation. An additional argument is used here: `--run` (or `-r`). `--run=0` assumes that profiling data is already available in `--data_dir`, and it only post-process those results. `--run=1` runs the actual profiling tests, and `--run=2` runs the tests and then performs the data post-processing.
