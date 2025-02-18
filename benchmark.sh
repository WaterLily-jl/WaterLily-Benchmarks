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
        cd $WATERLILY_DIR
        git checkout $wl_version
        cd $THIS_DIR
    fi
}

local_preferences () {
    if [[ $backend == "Array" && $thread == 1 ]]; then
        printf "[WaterLily]\nbackend = \"SIMD\"" > LocalPreferences.toml
    else
        printf "[WaterLily]\nbackend = \"KernelAbstractions\"" > LocalPreferences.toml
    fi
}

# Update project environment with new Julia version: Mark WaterLily as a development packag, then update dependencies and precompile.
update_environment () {
    local_preferences
    echo "Updating environment to Julia $version and compiling WaterLily"
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
    if [[ " ${BACKENDS[*]} " =~ [[:space:]]'Array'[[:space:]] ]]; then
        echo " - CPU threads:   ${THREADS[@]}"
    fi
    echo " - Cases:         ${CASES[@]}
 - Size:          ${LOG2P[@]:0:$NCASES}
 - Sim. steps:    ${MAXSTEPS[@]:0:$NCASES}
 - Data type:     ${FTYPE[@]:0:$NCASES}"
    echo "--------------------------------------"; echo
}

# Default backends
JULIA_USER_VERSION=$(julia_version)
VERSIONS=()
DEFAULT_VERSION=0
WL_DIR=""
DATA_DIR="data/benchmark/"
WL_VERSIONS=()
BACKENDS=('Array' 'CuArray')
THREADS=('4')
# Default cases. Arrays below must be same length (specify each case individually)
CASES=('tgv' 'jelly')
LOG2P=('6,7' '5,6')
MAXSTEPS=('100' '100')
FTYPE=('Float32' 'Float32')

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

# Assert all case arguments have equal size
NCASES=${#CASES[@]}
NLOG2P=${#LOG2P[@]}
NMAXSTEPS=${#MAXSTEPS[@]}
NFTYPE=${#FTYPE[@]}
st=0
for i in $NLOG2P $NMAXSTEPS $NFTYPE; do
    [ "$NCASES" = "$i" ]
    st=$(( $? + st ))
done
if [ $st != 0 ]; then
    echo "ERROR: Case arguments are arrays of different sizes."
    exit 1
fi

# Check WATERLILY_DIR is set and functional
if [ -z $WL_DIR ]; then # --waterlily-dir argument not passed
    if [ -z $WATERLILY_DIR ]; then # WATERLILY_DIR not set
        printf "WATERLILY_DIR environmental variable must be set.\nEither export it globally or pass it using: --waterlily-dir=foo/bar/"
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

# Check if Julia versions have been specified, and if so check that juliaup is installed
if (( ${#VERSIONS[@]} != 0 )); then
    if ! check_if_juliaup; then
        printf "Versions ${WL_VERSIONS[@]} were requested, but juliaup is not found."
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

# Benchmarks
for version in "${VERSIONS[@]}" ; do
    echo "Running with Julia version $version from $( which julia )"
    for wl_version in "${WL_VERSIONS[@]}" ; do
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
