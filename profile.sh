#!/bin/bash
## Usage example with --run=0: postproc only, --run=1: run only, --run=2: run and postproc

## sh profile.sh -ns "nsys" -c "tgv sphere cylinder" -p "8 5 6" -s 1000 -r 1
## sh profile.sh -ns "ncu" -k "327 355" -c "cylinder" -p "4,5,6" -s 20

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

## Utils
## Grep current julia version
julia_version () {
    julia_v=($(julia -v))
    echo "${julia_v[2]}"
}
## Get current WaterLily version
waterlily_version () {
    waterlily_v=($(git -C $WATERLILY_DIR rev-parse --short HEAD))
    echo "${waterlily_v}"
}
## Grep current julia version
waterlily_profile_branch () {
    cd $WATERLILY_DIR
    git checkout profiling
    julia --project -e "using Pkg; Pkg.update();"
    cd $THIS_DIR
}
## Update environment
update_environment () {
    echo "Updating environment to Julia $version"
    julia --project=$THIS_DIR -e "using Pkg; Pkg.develop(PackageSpec(path=get(ENV, \"WATERLILY_DIR\", \"\"))); Pkg.update();"
}
## Run profiling nsys
run_profiling_nsys () {
    full_args=(--project=${THIS_DIR} --startup-file=no $args)
    echo "Running NSYS profiling: nsys profile --sample=none --trace=nvtx,cuda --output=$DATA_DIR/$case/$case.nsys-rep --export=sqlite --force-overwrite=true julia ${full_args[@]}"
    nsys profile --sample=none --trace=nvtx,cuda --output=$DATA_DIR/$case/$case.nsys-rep --export=sqlite --force-overwrite=true julia "${full_args[@]}"
}
## Run postprocessing
run_postprocessing_nsys () {
    full_args=(--project=${THIS_DIR} --startup-file=no $args)
    echo "Running postprocessing: julia ${full_args[@]}"
    julia "${full_args[@]}"
}
## Run profiling ncu
run_profiling_ncu () {
    full_args=(--project=${THIS_DIR} --startup-file=no $args)
    echo "Running NCU profiling: ncu --set full --kernel-name regex:$kernels_reg --launch-skip 100 --launch-count 100 -o $DATA_DIR/$case/$kernels_o-p$pp -f julia ${full_args[@]}"
    ncu --set full --kernel-name regex:$kernels_reg --launch-skip 100 --launch-count 100 -o $DATA_DIR/$case/$kernels_o-p$pp -f julia ${full_args[@]}
}
# Make a regex for the kernels
join_with_pipe() {
  arr=("$@")
  echo "${arr// /|}"
}
join_with_underscore() {
  arr=("$@")
  echo "${arr// /_}"
}
## Print benchamrks info
display_info () {
    echo "--------------------------------------"
    echo "Running profiling tests for:
 - WaterLily:     ${WL_VERSIONS[@]}
 - WaterLily dir: $WATERLILY_DIR
 - Data dir:      $DATA_DIR
 - Julia:         $VERSION
 - Backends:      $BACKEND
 - Cases:         ${CASES[@]}
 - Size:          ${LOG2P[@]:0:$NCASES}
 - Sim. steps:    $MAXSTEPS
 - Data type:     $FTYPE
 - File:          $FILE
 - Profiling:     ${NSIGHT[@]}
 - Kernels:       ${KERNELS[@]}"
    echo "--------------------------------------"; echo
}

## Default backends
WL_DIR=""
DATA_DIR="data/profiling"
PLOT_DIR="plots/profiling"
WL_VERSIONS='profiling'
JULIA_USER_VERSION=$(julia_version)
VERSION=$JULIA_USER_VERSION
BACKEND='CuArray'
NSIGHT=('nsys')
KERNELS=()
## Default cases. Arrays below must be same length (specify each case individually)
CASES=() # ('tgv' 'sphere' 'cylinder')
LOG2P=() # ('8' '5' '6')
MAXSTEPS='1000'
FTYPE='Float32'
RUN='0'
FILE='profile.jl'

## Parse arguments
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
    VERSION=($2)
    shift
    ;;
    --backends|-b)
    BACKEND=($2)
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
    --run|-r)
    RUN=($2)
    shift
    ;;
    --file|-f)
    FILE=($2)
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
    --plot_dir|-pd)
    PLOT_DIR=($2)
    shift
    ;;
    --nsight|-ns)
    NSIGHT=($2)
    shift
    ;;
    --kernels|-k)
    KERNELS=($2)
    shift
    ;;
    *)
    printf "ERROR: Invalid argument %s\n" "${1}" 1>&2
    exit 1
esac
shift
done

## Assert all case arguments have equal size
NCASES=${#CASES[@]}
NLOG2P=${#LOG2P[@]}
st=0
for i in $NLOG2P; do
    [ "$NCASES" = "$i" ]
    st=$(( $? + st ))
done
if [ $st != 0 ]; then
    echo "ERROR: cases and log2p arrays of different sizes."
    exit 1
fi

## Check WATERLILY_DIR is set and functional
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

## Check if specific WaterLily version have been specified
if (( ${#WL_VERSIONS[@]} != 0 )); then
    WATERLILY_CHECKOUT=true
else
    WATERLILY_CHECKOUT=false
    WL_VERSIONS=($(waterlily_version))
fi

## Checkout to WaterLily profiling branch and update it
waterlily_profile_branch

## Display information
display_info

## Update this environment
update_environment

## Profiling
args_cases="--backend=$BACKEND --max_steps=$MAXSTEPS --ftype=$FTYPE --data_dir=$DATA_DIR --plot_dir=$PLOT_DIR"
for ((i = 0; i < ${#CASES[@]}; ++i)); do
    case=${CASES[$i]}
    mkdir -p $DATA_DIR/$case
    if printf '%s\0' "${NSIGHT[@]}" | grep -Fxqz -- 'nsys'; then
        if [ $RUN -gt 0 ]; then
            args="${THIS_DIR}/${FILE} --case=$case --log2p=${LOG2P[$i]} $args_cases --run=1"
            run_profiling_nsys
        fi
        if [ $RUN -ne 1 ]; then
            args="${THIS_DIR}/${FILE} --case=$case --log2p=${LOG2P[$i]} $args_cases --run=0"
            run_postprocessing_nsys
        fi
    fi
    if printf '%s\0' "${NSIGHT[@]}" | grep -Fxqz -- 'ncu'; then
        if (( ${#KERNELS[@]} != 0 )); then
            mapfile -t LOG2Ps < <(echo "${LOG2P[$i]}" | tr ',' '\n')
            kernels_reg=$(join_with_pipe "${KERNELS[*]}")
            kernels_o=$(join_with_underscore "${KERNELS[*]}")
            for pp in "${LOG2Ps[@]}" ; do
                args="${THIS_DIR}/${FILE} --case=$case --log2p=$pp $args_cases --run=1"
                run_profiling_ncu
            done
        else
            printf "NCU profiling requested, but no kernels were passed. Use -k or --kernels to pass the list of kernels to profile."
            exit 1
        fi
    fi
done
echo "All done!"
exit 0
