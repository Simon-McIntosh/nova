#!/bin/bash
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
out=$1
source_tree=$2
base_revision=$3
head_revision=$4
profiler="$source_tree/scripts/prototypes/moment_hessian"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$out/base" "$out/head"
trap 'find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch" > "$out/scratch-removal.log"' EXIT

mkdir "$scratch/base"
git -C "$source_tree" archive "$base_revision" nova scripts tests benchmarks pyproject.toml | tar -xf - -C "$scratch/base" || exit 2

measure() {
    local arm=$1 tree=$2 revision=$3
    local arm_out="$out/$arm" dumps="$scratch/dumps-$arm"
    mkdir -p "$dumps"
    (
        cd "$tree" || exit 2
        export PYTHONPATH="$tree" JAX_ENABLE_COMPILATION_CACHE=false
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export XLA_FLAGS="--xla_dump_to=$dumps --xla_dump_hlo_pass_re=.* --xla_dump_hlo_as_text --xla_dump_include_timestamp"
        printf 'revision=%s tree=%s command=%s -u %s --program point_jet --size 4 --revision %s --output %s\n' "$revision" "$tree" "$python" "$profiler/profile.py" "$revision" "$arm_out/profile.json"
        "$python" -u "$profiler/profile.py" --program point_jet --size 4 --revision "$revision" --output "$arm_out/profile.json"
        code=$?
        if [[ "$code" -eq 0 ]]; then
            "$python" -u "$profiler/analyse.py" --dumps "$dumps" --receipt "$arm_out/profile.json" --output "$arm_out/attribution.json"
            code=$?
        fi
        echo "EXIT=$code"
        exit "$code"
    ) > "$arm_out/arm.log" 2>&1
}

measure base "$scratch/base" "$base_revision"
base_exit=$?
measure head "$source_tree" "$head_revision"
head_exit=$?
printf 'PROFILE_SUMMARY base_exit=%s head_exit=%s\n' "$base_exit" "$head_exit"
if [[ "$base_exit" -eq 0 && "$head_exit" -eq 0 ]]; then
    echo EXIT=0
    exit 0
fi
echo EXIT=1
exit 1
