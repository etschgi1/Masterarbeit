#!/usr/bin/env bash

##
# source this script from a jupyter terminal or notebook cell
# to unset all jupyter related env variables and functions
##

DEBUG="$1"

function debug_print() {
    if [ -z "$DEBUG" ]; then
        return
    fi
    echo "$@"
}


if conda -V >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"

    for i in $(seq ${CONDA_SHLVL}); do
        conda deactivate
    done

    debug_print "Deactivated all conda envs ..."
else
    debug_print "No conda found."
fi

PREVIOUS_IFS="$IFS"
IFS=$'\n'
SLURM_VARS=$( env | sort | grep -E "^SLURM_.*=" | sed "s/=.*//g" )
for var in $SLURM_VARS; do
    unset $var
done
debug_print "Unset all SLURM_* env variables ..."
IFS="$PREVIOUS_IFS"

spack unload
debug_print "Unloaded all spack packages ..."

module purge
debug_print "Unloaded all modules ..."

# sanitize LD_LIBRARY_PATH by removing all paths from spack base
spack_base=$( readlink -f "$( dirname $( which spack ) )/../" )
library_path=${LD_LIBRARY_PATH//:/ }
new_library_path=
for path in $library_path; do
    if [[ $path =~ $spack_base ]]; then
        continue
    fi
    if [[ $new_library_path =~ $path ]]; then
        continue
    fi
    if [ -z "$new_library_path" ]; then
        new_library_path="$path"
    else
        new_library_path="$new_library_path:$path"
    fi
done
export LD_LIBRARY_PATH="$new_library_path"
export LIBRARY_PATH=
debug_print "Removed all spack library paths ..."

echo "Jupyter env (conda, slurm & spack) unloaded."

