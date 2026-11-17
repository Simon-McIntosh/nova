#!/usr/bin/env bash
# Run the two suite arms for this node as one job.
#
# The baseline arm runs at the base revision over that revision's own tree: the
# base is exported with git archive because this checkout's git metadata is
# read-only to the worker. The after arm runs in this checkout at its head. Each
# arm records the command exactly as resolved over its own tree, so promotion can
# compare it with the template resolved the same way.
#
#   env -u SLURM_JOB_ID -u SLURM_JOBID TMPDIR=/tmp /usr/bin/sbatch \
#     --partition=rigel --time=02:00:00 --cpus-per-task=8 --mem=110G \
#     --job-name=icm-suite-arms \
#     --output=/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/logs/icm-arms-%j.log \
#     scripts/iron_field_source/run_arms.sh
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
base=9b016126111a5acaebca064c07432042b5a2628c
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
out=/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/arms
scratch=$out/base-tree
tests=tests/test_magnetisation.py

export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH || true

mkdir -p "$out"
rm -rf "$scratch"
mkdir -p "$scratch"
git -C "$root" archive "$base" | tar -x -C "$scratch" --exclude='docs/figures/*'
echo "BASE-EXPORTED $base into $scratch"

command_for() {
    echo "PYTHONPATH=. JAX_PLATFORMS=cpu $python -m pytest -p no:cacheprovider -q $tests"
}

run_arm() {
    name=$1
    directory=$2
    log=$out/$3.log
    started=$(date +%s)
    set +e
    (cd "$directory" && PYTHONPATH=. JAX_PLATFORMS=cpu "$python" -m pytest \
        -p no:cacheprovider -q "$tests") >"$log" 2>&1
    status=$?
    set -e
    finished=$(date +%s)
    summary=$(grep -E "^[0-9]+ (passed|failed)|passed|failed|error" "$log" | tail -1)
    printf '%s\t%s\t%s\t%s\n' "$name" "$status" "$((finished - started))" "$summary" |
        tee -a "$out/summary.tsv"
}

run_arm baseline "$scratch" baseline_arm
run_arm after "$root" after_arm

{
    printf '{\\n'
    printf '  "baseline_command": "%s",\\n' "$(command_for)"
    printf '  "after_command": "%s",\\n' "$(command_for)"
    printf '  "base_sha": "%s",\\n' "$base"
    printf '  "head_sha": "%s",\\n' "$(git -C "$root" rev-parse HEAD)"
    printf '  "baseline_log": "%s",\\n' "$out/baseline_arm.log"
    printf '  "after_log": "%s"\\n' "$out/after_arm.log"
    printf '}\\n'
} >"$out/commands.json"
echo "ARMS-COMPLETE $(cat "$out/summary.tsv" | tr '\\t' ' ')"