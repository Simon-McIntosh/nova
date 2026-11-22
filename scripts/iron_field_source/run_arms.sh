#!/usr/bin/env bash
# Run the two suite arms for this node as one job.
#
# The baseline arm runs at the base revision over that revision's own tree: the
# base is a shared clone checked out at that revision, because the tree needs its
# git objects too (one suite test shells out to git show) and this checkout's git
# metadata is read-only to the worker. The after arm runs in this checkout at its
# head. Each arm records the command exactly as resolved over its own tree, so
# promotion can compare it with the template resolved the same way.
#
#   export ICM_ROOT=$PWD
#   sbatch --partition=rigel --time=03:00:00 --cpus-per-task=8 --mem=110G \
#     --job-name=icm-suite-arms \
#     --output=/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/logs/icm-arms-%j.log \
#     scripts/iron_field_source/run_arms.sh
set -euo pipefail

# SLURM copies the batch script into its spool directory, so the script cannot
# find the checkout from its own path; the submitter names the checkout.
root=${ICM_ROOT:?export ICM_ROOT with the checkout path before submitting}
base=9b016126111a5acaebca064c07432042b5a2628c
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
out=/work/projects/imas_gpu/nova/evidence/iron-core-magnetisation/arms
# TMPDIR is set before anything reads it: a submit environment may export the
# login node's /run/user/<uid>, which does not exist on a compute node, and a
# mktemp that inherited it would fail under set -e before either arm ran.
export TMPDIR=/tmp
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH || true

# The base tree is scratch: it lives on node-local disk and the job removes it
# through its own EXIT trap. Nothing under /work is ever removed, not even a
# directory this job created.
scratch=$(mktemp -d "$TMPDIR/icm-base-tree-XXXXXX")
trap 'rm -rf "$scratch"' EXIT
# The gate the rounds record: the fence's three files at this head. The base
# revision predates test_iron_field_source.py, so its arm names the two files
# that exist there.
tests="tests/test_imas_iron_read.py tests/test_iron_field_source.py tests/test_magnetisation.py"
base_tests="tests/test_imas_iron_read.py tests/test_magnetisation.py"

mkdir -p "$out"

# Prepared only by an arm that needs it, and under a job-unique path: two arms
# sharing one scratch directory delete each other's clone mid-checkout.
prepare_base() {
    git clone --shared --no-checkout --quiet "$root" "$scratch"
    git -C "$scratch" checkout --quiet --detach "$base"
    echo "BASE-CHECKOUT $base in $scratch ($(git -C "$scratch" rev-parse HEAD))"
}

command_for() {
    echo "PYTHONPATH=. JAX_PLATFORMS=cpu $python -m pytest -p no:cacheprovider -q $1"
}

run_arm() {
    name=$1
    directory=$2
    list=$3
    # One log per job and arm: a cancelled rerun must not overwrite the log a manifest
    # cites for a job that completed.
    log=$out/$name-arm-${SLURM_JOB_ID:-manual}.log
    started=$(date +%s)
    set +e
    (cd "$directory" && PYTHONPATH=. JAX_PLATFORMS=cpu "$python" -m pytest \
        -p no:cacheprovider -q $list) >"$log" 2>&1
    status=$?
    set -e
    finished=$(date +%s)
    summary=$(grep -E "^[0-9]+ (passed|failed)|passed|failed|error" "$log" | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$((finished - started))" "$summary" "$log" |
        tee -a "$out/summary.tsv"
}

arm=${1:-both}
case "$arm" in
    baseline)
        prepare_base
        run_arm baseline "$scratch" "$base_tests"
        ;;
    after) run_arm after "$root" "$tests" ;;
    both)
        prepare_base
        run_arm baseline "$scratch" "$base_tests"
        run_arm after "$root" "$tests"
        ;;
    *)
        echo "unknown arm: $arm" >&2
        exit 2
        ;;
esac

{
    printf '{\\n'
    printf '  "baseline_command": "%s",\\n' "$(command_for "$base_tests")"
    printf '  "after_command": "%s",\\n' "$(command_for "$tests")"
    printf '  "base_sha": "%s",\\n' "$base"
    printf '  "head_sha": "%s",\\n' "$(git -C "$root" rev-parse HEAD)"
    printf '  "baseline_log": "%s",\\n' "$out/baseline-arm-${SLURM_JOB_ID:-manual}.log"
    printf '  "after_log": "%s"\\n' "$out/after-arm-${SLURM_JOB_ID:-manual}.log"
    printf '}\\n'
} >"$out/commands.json"
echo "ARMS-COMPLETE $(cat "$out/summary.tsv" | tr '\\t' ' ')"