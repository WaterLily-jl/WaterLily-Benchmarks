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
    if [ command -v juliaup ] &> /dev/null || [ ! $JULIAUP ]
    then # juliaup does not exist or $JULIAUP is false
        return 1
    else # run with juliaup
        return 0
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

# Update project environment with new Julia version: Mark WaterLily as a development packag, then update dependencies and precompile.
update_environment () {
    if $WATERLILY_CHECKOUT; then
        echo "Git checkout to WaterLily $wl_version"
        cd $WATERLILY_DIR
        git checkout $wl_version
        cd $THIS_DIR
    fi
    if check_if_juliaup; then
        echo "Updating environment to Julia $version and compiling WaterLily"
        julia +${version} --project=$THIS_DIR -e "using Pkg; Pkg.develop(PackageSpec(path=get(ENV, \"WATERLILY_DIR\", \"\"))); Pkg.update();"
    fi
}

run_benchmark () {
    if check_if_juliaup; then
        full_args=(+${version} --project=${THIS_DIR} --startup-file=no $args)
    else
        full_args=(--project=${THIS_DIR} --startup-file=no $args)
    fi

    echo "Running: julia ${full_args[@]}"
    julia "${full_args[@]}"
}

# Print benchamrks info
display_info () {
    echo "--------------------------------------"
    echo "Running benchmark tests for:
 - WaterLily:     ${WL_VERSIONS[@]}
 - WaterLily dir: $WATERLILY_DIR
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
JULIAUP=true
JULIA_USER_VERSION=$(julia_version)
WL_DIR=""
WL_VERSIONS=()
BACKENDS=('Array' 'CuArray')
THREADS=('1' '4')
# Default cases. Arrays below must be same length (specify each case individually)
CASES=('tgv' 'jelly')
LOG2P=('6,7' '5,6')
MAXSTEPS=('100' '100')
FTYPE=('Float32' 'Float32')

# Parse arguments
while [ $# -gt 0 ]; do
case "$1" in
    --juliaup|-ju)
    JULIAUP=($2)
    shift
    ;;
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

# Display information
display_info

# Join arrays
CASES=$(join_array_str_comma "${CASES[*]}")
LOG2P=$(join_array_tuple_comma "${LOG2P[*]}")
MAXSTEPS=$(join_array_comma "${MAXSTEPS[*]}")
FTYPE=$(join_array_comma "${FTYPE[*]}")
args_cases="--cases=$CASES --log2p=$LOG2P --max_steps=$MAXSTEPS --ftype=$FTYPE"

# Benchmarks
for version in "${VERSIONS[@]}" ; do
    if check_if_juliaup; then
        echo "Julia $version benchmarks"
    else
        echo "Running with default Julia version $( julia_version ) from $( which julia )"
    fi
    for wl_version in "${WL_VERSIONS[@]}" ; do
        update_environment
        for backend in "${BACKENDS[@]}" ; do
            if [ "${backend}" == "Array" ]; then
                for thread in "${THREADS[@]}" ; do
                    args="-t $thread ${THIS_DIR}/benchmark.jl --backend=$backend $args_cases"
                    run_benchmark
                done
            else
                args="${THIS_DIR}/benchmark.jl --backend=$backend $args_cases"
                run_benchmark
            fi
        done
    done
    if ! check_if_juliaup; then
        break
    fi # if no juliaup, we only test default Julia version
done

echo "All done!"
exit 0
