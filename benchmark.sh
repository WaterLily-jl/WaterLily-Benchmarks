#!/bin/bash
# Paths handed to julia must be native: on Windows (Git Bash/MSYS) julia reads /c/foo as C:\c\foo
native_path () { if command -v cygpath &> /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
THIS_DIR=$(native_path "$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )")
export JULIA_NUM_THREADS="auto"

# Utils
join_array_comma () {
    arr=("$@")
    printf -v joined '%s,' $arr
    echo "[${joined%,}]"
}
join_array_str_comma () {
    arr=("$@")
    printf -v joined '\"%s\",' $arr
    echo "[${joined%,}]"
}
join_array_tuple_comma () {
    arr=("$@")
    printf -v joined '(%s),' $arr
    echo "[${joined%,}]"
}

# Expand provided values ($3..) to one per case into R: a single value broadcasts,
# N values are kept, none (omitted) uses the per-case defaults in $2.
expand () { # $1=arg name (for errors), $2=per-case defaults, $3..=provided values
    local name=$1 defs=$2 i; shift 2
    if   [ $# -eq "$NCASES" ]; then R=("$@")
    elif [ $# -eq 0 ];         then R=($defs)
    elif [ $# -eq 1 ];         then R=(); for ((i=0; i<NCASES; i++)); do R+=("$1"); done
    else echo "ERROR: '$name' has $# value(s) but expected 1 or $NCASES (cases)" >&2; exit 1; fi
}

# Reverse the given values into R (used to flip the version order on even repetitions)
reverse () { local i; R=(); for ((i=$#; i>=1; i--)); do R+=("${!i}"); done; }

# Normalise a boolean-ish string into "true"/"false" (sets the global UPDATE)
set_update () {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        true|1|yes|y) UPDATE=true ;;
        false|0|no|n) UPDATE=false ;;
        *) printf "ERROR: Invalid value '%s' for --update/-u (expected true/false/0/1)\n" "$1" 1>&2; exit 1 ;;
    esac
}

# Check if juliaup exists in environment
check_if_juliaup () {
    if command -v juliaup &> /dev/null
    then # juliaup exists
        return 0
    else # juliaup does not exist
        return 1
    fi
}

# Grep current julia version
julia_version () {
    julia_v=($(julia -v))
    echo "${julia_v[2]}"
}

# Get current WaterLily version
waterlily_version () {
    waterlily_v=($(git -C $WATERLILY_DIR rev-parse --short HEAD))
    echo "${waterlily_v}"
}

# Julia command based on juliaup or not
julia_cmd () {
    if check_if_juliaup && [[ $DEFAULT_VERSION -eq 1 ]]; then
        julia +$version "${full_args[@]}"
    else
        julia "${full_args[@]}"
    fi
}

git_checkout () {
    if $WATERLILY_CHECKOUT; then
        echo "Git checkout to WaterLily $wl_version"
        git -C "$WATERLILY_DIR" checkout $wl_version
    fi
    # BiotSavartBCs after WaterLily: its branch may need new WaterLily symbols
    if [ -n "${bs_version}" ]; then
        echo "Git checkout to BiotSavartBCs $bs_version"
        git -C "$BIOTSAVART_DIR" checkout $bs_version
    fi
}

local_preferences () {
    if [[ $backend == "Array" && $thread == 1 ]]; then
        printf "[WaterLily]\nbackend = \"SIMD\"\n" > "$project/LocalPreferences.toml"
    else
        printf "[WaterLily]\nbackend = \"KernelAbstractions\"\n" > "$project/LocalPreferences.toml"
    fi
}

# Update project environment with new Julia version: Mark WaterLily as a development packag, then update dependencies and precompile.
# A GPU backend runs in its own environment, gpu/<CUDA|AMDGPU>: a copy of this one plus the GPU package, resolved on its
# own so that neither the CPU runs nor the other GPU vendor hold its versions back. It is made on its first run, and made
# again when Project.toml changes (e.g. with -bs) or with -u true.
update_environment () {
    project=$THIS_DIR; local add=""
    if [ "$backend" != "Array" ]; then
        project="$THIS_DIR/gpu/${GPU_PKG[$backend]}"
        if $UPDATE || [ ! -f "$project/Manifest.toml" ] || ! cmp -s "$THIS_DIR/Project.toml" "$project/Project.base.toml"; then
            mkdir -p "$project" && rm -f "$project/Project.base.toml" && cp "$THIS_DIR/Project.toml" "$project/Project.toml"
            add="Pkg.add(\"${GPU_PKG[$backend]}\");"
        fi
    fi
    local_preferences
    if ! $UPDATE && [ -z "$add" ]; then
        return
    fi
    echo "Updating environment $project to Julia $version and compiling WaterLily"
    local up=""; $UPDATE && up="Pkg.update();"
    # With -bs, [sources] already points at $BIOTSAVART_DIR, so Pkg.update resolves the local clone.
    # Pkg is loaded before activating: with --project, a Manifest from Julia <= 1.12 breaks `using Pkg` on 1.13.
    full_args=(-e "using Pkg; Pkg.activate(\"$project\"); Pkg.develop(PackageSpec(path=get(ENV, \"WATERLILY_DIR\", \"\"))); $add $up")
    julia_cmd || { echo "ERROR: updating the environment for WaterLily $wl_version on Julia $version failed." >&2; exit 1; }
    [ -z "$add" ] || cp "$THIS_DIR/Project.toml" "$project/Project.base.toml"  # the Project.toml it was made from
}

run_benchmark () {
    full_args=(--project=${project} --startup-file=no $args)
    echo "Running: julia ${full_args[@]}"
    julia_cmd || { echo "ERROR: the benchmark failed (WaterLily $wl_version, Julia $version, $backend, -t ${thread:-auto}), stopping the sweep." >&2; exit 1; }
}

# Linux only: stop if a CPU is capped or there are too few CPUs for the threads (Julia also runs an
# interactive thread), warn if turbo is off or the machine is busy. -f/--force runs anyway.
check_machine () {
    [ "$(uname -s)" = Linux ] || return 0
    local errors=() warnings=() f t load=$(cut -d ' ' -f 1 /proc/loadavg)
    for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq; do
        [ -r $f/scaling_max_freq ] && (( $(<$f/scaling_max_freq) < $(<$f/cpuinfo_max_freq) )) &&
            errors+=("CPU ${f//[^0-9]/} is capped below its maximum frequency. Fix: sudo cp $f/cpuinfo_max_freq $f/scaling_max_freq")
    done
    [[ " ${BACKENDS[*]} " == *" Array "* ]] && for t in "${THREADS[@]}"; do
        (( t > 1 && $(nproc) <= t )) && errors+=("$(nproc) CPUs are available for -t $t, give at least $((t + 1)) (check taskset)")
    done
    [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)" = 1 ] && warnings+=("turbo is off")
    awk -v l=$load 'BEGIN { exit !(l > 1) }' && warnings+=("load average is $load: other processes are busy")
    (( ${#warnings[@]} )) && printf "WARNING: %s\n" "${warnings[@]}" >&2
    (( ${#errors[@]} )) && { printf "ERROR: %s\n" "${errors[@]}" >&2; $FORCE || { echo "Fix the above, or run anyway with -f/--force." >&2; exit 1; }; }
    return 0
}

# Checkpoints are git-LFS files. Fetch the ones of this sweep that are still LFS pointers, which is the state of a
# clone made with GIT_LFS_SKIP_SMUDGE=1 (as in CI): only the needed files are downloaded, not the whole directory.
fetch_checkpoints () {
    [ -n "$DEVELOPED" ] && [ -d "$DEVELOPED" ] || return 0
    local i n f rel files=()
    for ((i=0; i<NCASES; i++)); do
        for n in ${LOG2P[$i]//,/ }; do
            f="$DEVELOPED/${CASES[$i]}_${n}_${FTYPE[$i]}.jld2"
            [ -f "$f" ] && [ "$(wc -c < "$f")" -lt 1024 ] && grep -q "^version https://git-lfs" "$f" || continue
            rel=$(realpath --relative-to="$THIS_DIR" "$f")
            [[ $rel == ../* ]] && { echo "ERROR: $f is a git-LFS pointer outside this repository: run git lfs pull there." >&2; exit 1; }
            files+=("$rel")
        done
    done
    (( ${#files[@]} )) || return 0
    command -v git-lfs &> /dev/null || { echo "ERROR: git-lfs is needed to fetch the checkpoints: ${files[*]}" >&2; exit 1; }
    echo "Fetching ${#files[@]} checkpoint(s) from git-LFS: ${files[*]}"
    git -C "$THIS_DIR" lfs pull --include="$(IFS=,; echo "${files[*]}")" || { echo "ERROR: git lfs pull failed." >&2; exit 1; }
}

# Print benchamrks info
display_info () {
    echo "--------------------------------------"
    echo "Running benchmark tests for:
 - WaterLily:     ${WL_VERSIONS[@]}
 - WaterLily dir: $WATERLILY_DIR
 - Benchmark dir: $DATA_DIR
 - Julia:         ${VERSIONS[@]}
 - Backends:      ${BACKENDS[@]}"
    [ ${#BS_VERSIONS[@]} -ne 0 ] && echo " - BiotSavartBCs: ${BS_VERSIONS[@]} (dir: ${BIOTSAVART_DIR:-$BS_DIR})"
    if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
        echo " - CPU threads:   ${THREADS[@]}"
    fi
    echo " - Cases:         ${CASES[@]}
 - Size:          ${LOG2P[@]:0:$NCASES}
 - Sim. steps:    ${MAXSTEPS[@]:0:$NCASES}
 - Data type:     ${FTYPE[@]:0:$NCASES}
 - Developed:     ${DEVELOPED:-(transient)}
 - Update env:    $UPDATE
 - Repeats:       $REPEATS"
    echo "--------------------------------------"; echo
}

# Default backends
JULIA_USER_VERSION=$(julia_version)
VERSIONS=()
DEFAULT_VERSION=0
WL_DIR=""
BS_DIR=""
DATA_DIR="data/benchmark/"
WL_VERSIONS=()
BS_VERSIONS=()                                            # -bs: BiotSavartBCs versions, paired 1:1 with -w
BACKENDS=('Array' 'CuArray')
declare -A GPU_PKG=([CuArray]=CUDA [ROCArray]=AMDGPU)      # GPU backend -> its package (and gpu/ environment)
THREADS=('4')
UPDATE=false
REPEATS=1                                                 # -r: repeat the whole sweep in new processes
FORCE=false                                               # -f: run even if check_machine finds a problem
DEVELOPED="checkpoints"                                   # -dev <dir>: developed-flow checkpoints; "" times the transient
# Default sweep (run when -c is omitted) and per-case defaults for omitted -p/-s/-ft.
CASES=('tgv' 'jelly-biotsavart')
LOG2P=(); MAXSTEPS=(); FTYPE=()                            # provided -p/-s/-ft (empty => default)
declare -A DEF_LOG2P=([tgv]=6,7 [tgv-periodic]=6,7 [jelly-biotsavart]=5,6 [sphere]=3,4 [sphere-biotsavart]=3,4 [cylinder]=4,5 [cylinder-biotsavart]=4,5 [donut]=5,6)  # default size per case; add cases here and to checkpoint_log2p in src/util.jl
DEF_MAXSTEPS=25; DEF_FTYPE=Float32                         # default steps/type (uniform)

# Parse arguments
while [ $# -gt 0 ]; do
case "$1" in
    --waterlily_dir|-wd)
    WL_DIR=($2)
    shift
    ;;
    --waterlily|-w)
    WL_VERSIONS=($2)
    shift
    ;;
    --biotsavart|-bs)
    BS_VERSIONS=($2)
    shift
    ;;
    --biotsavart_dir|-bsd)
    BS_DIR=($2)
    shift
    ;;
    --versions|-v)
    VERSIONS=($2)
    shift
    ;;
    --backends|-b)
    BACKENDS=($2)
    shift
    ;;
    --threads|-t)
    THREADS=($2)
    shift
    ;;
    --cases|-c)
    CASES=($2)
    shift
    ;;
    --log2p|-p)
    LOG2P=($2)
    shift
    ;;
    --max_steps|-s)
    MAXSTEPS=($2)
    shift
    ;;
    --float_type|-ft)
    FTYPE=($2)
    shift
    ;;
    --data_dir|-dd)
    DATA_DIR=($2)
    shift
    ;;
    --developed|-dev)
    DEVELOPED=($2)
    shift
    ;;
    --update|-u)
    set_update "$2"
    shift
    ;;
    --repeats|-r)
    REPEATS=$2
    shift
    ;;
    --force|-f)
    FORCE=true
    ;;
    *)
    printf "ERROR: Invalid argument %s\n" "${1}" 1>&2
    exit 1
esac
shift
done

# Assert "--threads" argument is not empy if "Array" backend is present
if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
    if [ "${#THREADS[@]}" == 0 ]; then
        echo "ERROR: Backend 'Array' is present, but '--threads' argument is empty."
        exit 1
    fi
fi

# Assert the backends are known
for b in "${BACKENDS[@]}"; do
    [ "$b" == "Array" ] || [ -n "${GPU_PKG[$b]+x}" ] || { echo "ERROR: Invalid backend '$b' (expected Array, CuArray or ROCArray)" >&2; exit 1; }
done

# Assert "--repeats" is a positive integer
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: Invalid value '$REPEATS' for --repeats/-r (expected a positive integer)" >&2; exit 1
fi

# Stop before touching anything if the machine would distort the timings
check_machine

# Expand case args: single value broadcasts, N kept, omitted uses per-case defaults.
NCASES=${#CASES[@]}
dP=; dS=; dFT=
for c in "${CASES[@]}"; do
    [ ${#LOG2P[@]} -ne 0 ] || [ -n "${DEF_LOG2P[$c]+x}" ] || { echo "ERROR: case '$c' has no default -p; add it to DEF_LOG2P in benchmark.sh or pass -p explicitly" >&2; exit 1; }
    dP+="${DEF_LOG2P[$c]} "; dS+="$DEF_MAXSTEPS "; dFT+="$DEF_FTYPE "
done
expand -p  "$dP"  "${LOG2P[@]}";    LOG2P=("${R[@]}")
expand -s  "$dS"  "${MAXSTEPS[@]}"; MAXSTEPS=("${R[@]}")
expand -ft "$dFT" "${FTYPE[@]}";    FTYPE=("${R[@]}")
fetch_checkpoints  # only the checkpoints of this sweep, if they are git-LFS pointers

# Check WATERLILY_DIR is set and functional
if [ -z $WL_DIR ]; then # --waterlily-dir argument not passed
    if [ -z $WATERLILY_DIR ]; then # WATERLILY_DIR not set
        printf "WATERLILY_DIR environmental variable must be set.\nEither export it globally or pass it using: --waterlily-dir=foo/bar/\n"
        exit 1
    fi
else
    export WATERLILY_DIR=$WL_DIR
fi
export WATERLILY_DIR=$(native_path "$(realpath -e $WATERLILY_DIR)")
if [[ ! -d $WATERLILY_DIR && -L $WATERLILY_DIR ]]; then # check WATERLILY_DIR path exists
  echo "WaterLily path $WATERLILY_DIR does not exist."
fi

# Check if specific WaterLily version have been specified
if (( ${#WL_VERSIONS[@]} != 0 )); then
    WATERLILY_CHECKOUT=true
else
    WATERLILY_CHECKOUT=false
    WL_VERSIONS=($(waterlily_version))
fi

# Paired BiotSavartBCs versions (optional): one per WaterLily version, from a local clone (-bsd or $BIOTSAVART_DIR)
if (( ${#BS_VERSIONS[@]} != 0 )); then
    [ -n "$BS_DIR" ] && export BIOTSAVART_DIR=$BS_DIR
    if [ -z "${BIOTSAVART_DIR:-}" ]; then
        printf "ERROR: --biotsavart/-bs needs a local BiotSavartBCs clone via --biotsavart_dir/-bsd or \$BIOTSAVART_DIR.\n" 1>&2; exit 1
    fi
    export BIOTSAVART_DIR=$(native_path "$(realpath -e "$BIOTSAVART_DIR")")
    if (( ${#BS_VERSIONS[@]} != ${#WL_VERSIONS[@]} )); then
        printf "ERROR: --biotsavart has ${#BS_VERSIONS[@]} value(s) but must match --waterlily (${#WL_VERSIONS[@]}).\n" 1>&2; exit 1
    fi
    # Pkg.develop cannot override a [sources] pin: repoint it at the local clone and force -u true. Pkg then keeps the
    # clone's path in the Manifests even after Project.toml is restored, so all the environments are restored on exit.
    BS_SNAP=$(mktemp -d)
    cp -a "$THIS_DIR/Project.toml" "$BS_SNAP/"
    [ -f "$THIS_DIR/Manifest.toml" ] && cp -a "$THIS_DIR/Manifest.toml" "$BS_SNAP/"
    [ -d "$THIS_DIR/gpu" ] && cp -a "$THIS_DIR/gpu" "$BS_SNAP/"
    trap 'rm -rf "$THIS_DIR/Manifest.toml" "$THIS_DIR/gpu"; cp -a "$BS_SNAP/." "$THIS_DIR/"; rm -rf "$BS_SNAP"' EXIT
    # match the [sources] entry (`= {...}`), not the [deps] UUID (`= "..."`)
    sed -i "s|^BiotSavartBCs = {.*|BiotSavartBCs = {path = \"$BIOTSAVART_DIR\"}|" "$THIS_DIR/Project.toml"
    UPDATE=true
    echo "Note: -bs repointed [sources] BiotSavartBCs -> $BIOTSAVART_DIR and forced -u true (environments restored on exit)."
fi

# Check if Julia versions have been specified, and if so check that juliaup is installed
if (( ${#VERSIONS[@]} != 0 )); then
    if ! check_if_juliaup; then
        printf "Versions ${WL_VERSIONS[@]} were requested, but juliaup is not found.\n"
        exit 1
    fi
    DEFAULT_VERSION=1
else
    VERSIONS=($JULIA_USER_VERSION)
fi

# Display information
display_info

# Join arrays
CASES=$(join_array_str_comma "${CASES[*]}")
LOG2P=$(join_array_tuple_comma "${LOG2P[*]}")
MAXSTEPS=$(join_array_comma "${MAXSTEPS[*]}")
FTYPE=$(join_array_comma "${FTYPE[*]}")
args_cases="--cases=$CASES --log2p=$LOG2P --max_steps=$MAXSTEPS --ftype=$FTYPE --data_dir=$DATA_DIR"
args_cases="$args_cases --developed=$DEVELOPED"  # always forwarded, so -dev "" reaches src/benchmark.jl

# Benchmarks. With -r N the sweep runs N times in new processes, reversing the version order on even
# repetitions, to sample a machine state that outlives a process. compare.jl merges the repetitions.
for rep in $(seq 1 $REPEATS) ; do
    args_rep=""
    V_IDX=("${!VERSIONS[@]}"); W_IDX=("${!WL_VERSIONS[@]}")
    if (( REPEATS > 1 )); then
        echo "Repetition $rep of $REPEATS"
        args_rep="--rep=$rep"
        if (( rep % 2 == 0 )); then
            reverse "${V_IDX[@]}"; V_IDX=("${R[@]}")
            reverse "${W_IDX[@]}"; W_IDX=("${R[@]}")
        fi
    fi
    for vi in "${V_IDX[@]}" ; do
        version="${VERSIONS[$vi]}"
        echo "Running with Julia version $version from $( which julia )"
        for i in "${W_IDX[@]}" ; do
            wl_version="${WL_VERSIONS[$i]}"
            bs_version="${BS_VERSIONS[$i]:-}"
            git_checkout
            for backend in "${BACKENDS[@]}" ; do
                if [ "${backend}" == "Array" ]; then
                    for thread in "${THREADS[@]}" ; do
                        args="-t $thread ${THIS_DIR}/src/benchmark.jl --backend=$backend $args_cases $args_rep"
                        update_environment
                        run_benchmark
                    done
                else
                    args="${THIS_DIR}/src/benchmark.jl --backend=$backend $args_cases $args_rep"
                    update_environment
                    run_benchmark
                fi
            done
        done
    done
done

echo "All done!"
exit 0
