#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=03:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'printf "scratch contents before removal: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
export TMPDIR="$scratch"
tree=$1
base_revision=$2
head_revision=$3
run=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir "$scratch/base"
git -C "$tree" archive "$base_revision" nova tests scripts input data pyproject.toml uv.lock | tar -xf - -C "$scratch/base" || exit 2
git_directory=$(git -C "$tree" rev-parse --git-common-dir)

run_arm() {
    local root=$1 revision=$2 log=$3
    local targets=("$root/tests/biot")
    local file
    while IFS= read -r file; do
        if [[ "$file" != "$root/tests/biot/"* ]]; then
            targets+=("$file")
        fi
    done < <(rg -l 'polygonanalytic|packed_analytic_moments' "$root/tests" | sort)
    local command=("$python" -m pytest -p no:cacheprovider -m 'slow or not slow' -q --tb=short "${targets[@]}")
    (
        cd "$root" || exit 2
        export PYTHONPATH="$root:$root/tests/biot" JAX_PLATFORMS=cpu GIT_DIR="$git_directory"
        printf 'revision=%s tree=%s command=' "$revision" "$root"
        printf '%q ' "${command[@]}"
        printf '\n'
        "$python" -c 'import os, nova; from nova.biot import polygonanalytic; print("nova="+nova.__file__, "measurement_module="+polygonanalytic.__file__, "cwd="+os.path.realpath(os.getcwd()), flush=True)'
        "${command[@]}"
        code=$?
        echo "EXIT=$code"
        exit "$code"
    ) > "$log" 2>&1
    return $?
}

run_arm "$scratch/base" "$base_revision" "$run/baseline-suite.log"
base_exit=$?
run_arm "$tree" "$head_revision" "$run/after-suite.log"
head_exit=$?
echo "BASE_EXIT=$base_exit HEAD_EXIT=$head_exit"
if (( head_exit != 0 )); then exit "$head_exit"; fi
exit 0
