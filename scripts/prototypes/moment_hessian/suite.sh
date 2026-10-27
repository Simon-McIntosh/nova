#!/bin/bash
#SBATCH --partition=sirius
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'printf "removing scratch: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
export TMPDIR="$scratch"
tree=$1
base_revision=$2
head_revision=$3
run=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
mkdir -p "$scratch/base" "$run/suite-files"
git -C "$tree" archive "$base_revision" nova tests scripts input data pyproject.toml uv.lock | tar -xf - -C "$scratch/base" || exit 2
git_directory=$(git -C "$tree" rev-parse --git-common-dir)
results="$run/suite-files/results.jsonl"
: > "$results"

run_arm() {
    local root=$1 revision=$2 arm=$3
    local arm_log="$run/suite-files/$arm/arm.log"
    mkdir -p "$run/suite-files/$arm"
    mapfile -t targets < <(
        {
            rg --files "$root/tests/biot" -g 'test_*.py'
            rg -l 'polygonanalytic|packed_analytic_moments' "$root/tests" -g 'test_*.py'
        } | sort -u
    )
    printf 'revision=%s tree=%s command=%s\n' "$revision" "$root" \
        "${BASH_SOURCE[0]} $tree $base_revision $head_revision $run" > "$arm_log"
    printf 'nova=%s measurement_module=%s cwd=%s\n' \
        "$root/nova/__init__.py" "$root/nova/biot/polygonanalytic.py" "$root" >> "$arm_log"
    local arm_exit=0
    local file relative slug log code
    for file in "${targets[@]}"; do
        relative=${file#"$root"/}
        slug=${relative//\//__}
        slug=${slug%.py}
        log="$run/suite-files/$arm/$slug.log"
        command=("$python" -m pytest -p no:cacheprovider -m 'slow or not slow' -q --tb=short "$file")
        (
            cd "$root" || exit 2
            export PYTHONPATH="$root:$root/tests/biot" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 GIT_DIR="$git_directory"
            printf 'revision=%s tree=%s command=' "$revision" "$root"
            printf '%q ' "${command[@]}"
            printf '\n'
            "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import jax,os,nova; from nova.biot import polygonanalytic; assert jax.config.jax_enable_x64; print("nova="+nova.__file__, "measurement_module="+polygonanalytic.__file__, "cwd="+os.path.realpath(os.getcwd()), flush=True)'
            "${command[@]}"
            code=$?
            echo "EXIT=$code"
            exit "$code"
        ) > "$log" 2>&1
        code=$?
        if (( code != 0 )); then arm_exit=1; fi
        "$python" - "$log" "$arm" "$relative" "$code" "$results" "$arm_log" <<'PY'
import json
from pathlib import Path
import re
import sys

log, arm, relative, code, results, arm_log = sys.argv[1:]
lines = Path(log).read_text(errors="replace").splitlines()
summaries = [line for line in lines if re.search(r"\bin \d+(?:\.\d+)?s(?:\s|$)", line)]
failures = []
for line in lines:
    if line.startswith(("FAILED ", "ERROR ")):
        identifier = line.split()[1]
        if "/tests/" in identifier:
            identifier = "tests/" + identifier.split("/tests/", 1)[1]
        failures.append(identifier)
if int(code) and not failures:
    failures.append(relative + "::<exit-" + code + ">")
row = {
    "arm": arm,
    "file": relative,
    "exit_status": int(code),
    "completed": bool(summaries),
    "failure_ids": sorted(set(failures)),
    "summary": summaries[-1] if summaries else None,
    "log_path": log,
}
with Path(results).open("a") as stream:
    stream.write(json.dumps(row) + "\n")
message = "FILE " + json.dumps(row, sort_keys=True)
with Path(arm_log).open("a") as stream:
    stream.write(message + "\n")
print(message, flush=True)
PY
    done
    printf 'ARM_SUMMARY arm=%s files=%s exit=%s\nEXIT=%s\n' \
        "$arm" "${#targets[@]}" "$arm_exit" "$arm_exit" >> "$arm_log"
    return "$arm_exit"
}

run_arm "$scratch/base" "$base_revision" base
base_exit=$?
run_arm "$tree" "$head_revision" head
head_exit=$?

(
    printf 'revision=%s tree=%s command=%s\n' "$head_revision" "$tree" \
        "${BASH_SOURCE[0]} $tree $base_revision $head_revision $run"
    printf 'nova=%s measurement_module=%s cwd=%s\n' \
        "$tree/nova/__init__.py" "$tree/nova/biot/polygonanalytic.py" "$tree"
    "$python" - "$results" "$run/suite-files/comparison.json" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
arms = {name: [row for row in rows if row["arm"] == name] for name in ("base", "head")}
failures = {
    name: sorted({item for row in group for item in row["failure_ids"]})
    for name, group in arms.items()
}
incomplete = {name: [row["file"] for row in group if not row["completed"]]
              for name, group in arms.items()}
added = sorted(set(failures["head"]) - set(failures["base"]))
receipt = {
    "files": {name: len(group) for name, group in arms.items()},
    "failure_ids": failures,
    "incomplete_files": incomplete,
    "added_failure_ids": added,
    "passed_delta": bool(arms["base"] and arms["head"] and
                         not incomplete["base"] and not incomplete["head"] and not added),
}
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2) + "\n")
print("SUITE_SUMMARY " + json.dumps(receipt, sort_keys=True), flush=True)
raise SystemExit(0 if receipt["passed_delta"] else 1)
PY
    code=$?
    echo "EXIT=$code"
    exit "$code"
) > "$run/suite-files/comparison.log" 2>&1
comparison_exit=$?
echo "BASE_EXIT=$base_exit HEAD_EXIT=$head_exit COMPARISON_EXIT=$comparison_exit"
exit "$comparison_exit"
