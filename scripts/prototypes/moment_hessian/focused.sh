#!/bin/bash
#SBATCH --partition=all_debug
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'printf "removing scratch: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
export TMPDIR="$scratch"
tree=$1
revision=$2
run=$3
label=$4
shift 4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
command=("$python" -m pytest -p no:cacheprovider -m 'slow or not slow' -q --tb=short --basetemp "$scratch/basetemp" "$@")
(
    cd "$tree" || exit 2
    export PYTHONPATH="$tree:$tree/tests/biot" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
    export GIT_DIR
    GIT_DIR=$(git -C "$tree" rev-parse --git-common-dir)
    printf 'revision=%s tree=%s command=' "$revision" "$tree"
    printf '%q ' "${command[@]}"
    printf '\n'
    "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import jax,os,nova; from nova.biot import polygonanalytic; assert jax.config.jax_enable_x64; print("nova="+nova.__file__, "measurement_module="+polygonanalytic.__file__, "cwd="+os.path.realpath(os.getcwd()), flush=True)'
    "${command[@]}"
    code=$?
    echo "EXIT=$code"
    exit "$code"
) > "$run/$label.log" 2>&1
exit $?
