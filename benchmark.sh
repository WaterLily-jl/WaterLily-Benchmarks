#!/bin/bash
THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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
    if [[ check_if_juliaup && DEFAULT_VERSION -eq 1 ]]; then
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
    # Paired BiotSavartBCs checkout AFTER WaterLily, so branches needing new WaterLily
    # symbols (e.g. combined-tol needs l2n_tol) resolve against the right WaterLily.
    if [ -n "${biot_version}" ]; then
        echo "Git checkout to BiotSavartBCs $biot_version"
        git -C "$BIOTSAVART_DIR" checkout $biot_version
    fi
}

local_preferences () {
    if [[ $backend == "Array" && $thread == 1 ]]; then
        printf "[WaterLily]\nbackend = \"SIMD\"\n" > LocalPreferences.toml
    else
        printf "[WaterLily]\nbackend = \"KernelAbstractions\"\n" > LocalPreferences.toml
    fi
}

# Update project environment with new Julia version: Mark WaterLily as a development packag, then update dependencies and precompile.
update_environment () {
    local_preferences
    if ! $UPDATE; then
        return
    fi
    echo "Updating environment to Julia $version and compiling WaterLily"
    # For paired Biot runs, [sources] has already been repointed at $BIOTSAVART_DIR (see below),
    # so Pkg.update resolves BiotSavartBCs from the local clone; git_checkout picks the branch.
    full_args=(--project=$THIS_DIR -e "using Pkg; Pkg.develop(PackageSpec(path=get(ENV, \"WATERLILY_DIR\", \"\"))); Pkg.update();")
    julia_cmd
}

run_benchmark () {
    full_args=(--project=${THIS_DIR} --startup-file=no $args)
    echo "Running: julia ${full_args[@]}"
    julia_cmd
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
    [ ${#BIOT_VERSIONS[@]} -ne 0 ] && echo " - BiotSavartBCs: ${BIOT_VERSIONS[@]} (dir: ${BIOTSAVART_DIR:-$BS_DIR})"
    if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
        echo " - CPU threads:   ${THREADS[@]}"
    fi
    echo " - Cases:         ${CASES[@]}
 - Size:          ${LOG2P[@]:0:$NCASES}
 - Sim. steps:    ${MAXSTEPS[@]:0:$NCASES}
 - Data type:     ${FTYPE[@]:0:$NCASES}
 - Developed:     ${DEVELOPED:-(transient)}
 - Update env:    $UPDATE"
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
BIOT_VERSIONS=()                                          # --biotsavart/-wb: BiotSavartBCs branches, paired 1:1 with --waterlily
BACKENDS=('Array' 'CuArray')
THREADS=('4')
UPDATE=false
DEVELOPED="checkpoints"                                   # --developed=<dir>: time from developed-flow checkpoints (default); -dev "" for transient
# Default sweep (run when -c is omitted) and per-case defaults for omitted -p/-s/-ft.
CASES=('tgv' 'jelly')
LOG2P=(); MAXSTEPS=(); FTYPE=()                            # provided -p/-s/-ft (empty => default)
declare -A DEF_LOG2P=([tgv]=6,7 [jelly]=5,6 [sphere]=3,4 [cylinder]=4,5)  # default size per case; add cases here
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
    --biotsavart|-wb)
    BIOT_VERSIONS=($2)
    shift
    ;;
    --biotsavart_dir|-wbd)
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

# Check WATERLILY_DIR is set and functional
if [ -z $WL_DIR ]; then # --waterlily-dir argument not passed
    if [ -z $WATERLILY_DIR ]; then # WATERLILY_DIR not set
        printf "WATERLILY_DIR environmental variable must be set.\nEither export it globally or pass it using: --waterlily-dir=foo/bar/\n"
        exit 1
    fi
else
    export WATERLILY_DIR=$WL_DIR
fi
export WATERLILY_DIR=$(realpath -e $WATERLILY_DIR)
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

# Paired BiotSavartBCs versions (optional). Needs a local clone (--biotsavart_dir/-wbd or
# $BIOTSAVART_DIR) and one Biot branch per WaterLily version. Used to benchmark criterion
# changes that span both packages (e.g. jelly: WaterLily master+Biot main vs poisson-rms-tol+combined-tol).
if (( ${#BIOT_VERSIONS[@]} != 0 )); then
    [ -n "$BS_DIR" ] && export BIOTSAVART_DIR=$BS_DIR
    if [ -z "${BIOTSAVART_DIR:-}" ]; then
        printf "ERROR: --biotsavart/-wb needs a local BiotSavartBCs clone via --biotsavart_dir/-wbd or \$BIOTSAVART_DIR.\n" 1>&2; exit 1
    fi
    export BIOTSAVART_DIR=$(realpath -e "$BIOTSAVART_DIR")
    if (( ${#BIOT_VERSIONS[@]} != ${#WL_VERSIONS[@]} )); then
        printf "ERROR: --biotsavart has ${#BIOT_VERSIONS[@]} value(s) but must match --waterlily (${#WL_VERSIONS[@]}).\n" 1>&2; exit 1
    fi
    # Repoint [sources] at the local clone so its branch is switchable per run (Pkg.develop
    # cannot override a [sources] pin). Force an environment update to re-resolve, and restore
    # the original Project.toml on exit.
    cp "$THIS_DIR/Project.toml" "$THIS_DIR/Project.toml.wbbak"
    trap 'mv -f "$THIS_DIR/Project.toml.wbbak" "$THIS_DIR/Project.toml" 2>/dev/null' EXIT
    # match only the [sources] dict entry (`= {...}`), NOT the [deps] UUID string (`= "..."`)
    sed -i "s|^BiotSavartBCs = {.*|BiotSavartBCs = {path = \"$BIOTSAVART_DIR\"}|" "$THIS_DIR/Project.toml"
    UPDATE=true
    echo "Note: -wb repointed [sources] BiotSavartBCs -> $BIOTSAVART_DIR and forced -u true (restored on exit)."
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
args_cases="$args_cases --developed=$DEVELOPED"  # always forwarded so -dev "" reaches benchmark.jl (transient)

# Benchmarks
for version in "${VERSIONS[@]}" ; do
    echo "Running with Julia version $version from $( which julia )"
    for i in "${!WL_VERSIONS[@]}" ; do
        wl_version="${WL_VERSIONS[$i]}"
        biot_version="${BIOT_VERSIONS[$i]:-}"
        git_checkout
        for backend in "${BACKENDS[@]}" ; do
            if [ "${backend}" == "Array" ]; then
                for thread in "${THREADS[@]}" ; do
                    args="-t $thread ${THIS_DIR}/benchmark.jl --backend=$backend $args_cases"
                    update_environment
                    run_benchmark
                done
            else
                args="${THIS_DIR}/benchmark.jl --backend=$backend $args_cases"
                update_environment
                run_benchmark
            fi
        done
    done
done

echo "All done!"
exit 0
