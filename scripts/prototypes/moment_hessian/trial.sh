#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:59:00
set -uo pipefail

export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/$USER-${SLURM_JOB_ID}-XXXX")
trap 'printf "scratch contents before removal: %s\n" "$scratch"; find "$scratch" -mindepth 1 -maxdepth 1 -print; rm -rv "$scratch"' EXIT
export TMPDIR="$scratch"
source_tree=$1
revision=$2
run=$3
previous=$4
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
export NOVA_OPERAND_CACHE="$HOME/.cache/nova-moment-hessian"
export NOVA_COMPILATION_CACHE_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm
mkdir -p "$NOVA_OPERAND_CACHE"
export MEASURE_RUN="$run" MEASURE_PREVIOUS="$previous" MEASURE_DETAIL=1

for arm in split integration; do
    tree="$scratch/$arm"
    mkdir "$tree"
    git -C "$source_tree" archive "$revision" nova scripts tests benchmarks pyproject.toml | tar -xf - -C "$tree" || exit 2
    export PYTHONPATH="$tree:$source_tree/scripts/prototypes/moment_hessian" MEASURE_ARM="$arm"
    export MEASURE_CARRIER
    MEASURE_CARRIER=$("$python" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); a=sys.argv[2]; print(Path(json.loads((r/(a+"-setup.json")).read_text())["carrier_cache"]["store"])/"fixture.nc")' "$previous" "$arm")
    "$python" - "$tree" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "scripts/prototypes/fast_map_production/solver.py"
text = path.read_text()
marker = "    machine = certified_machine(carrier, kind, args.cells, exact)"
assert text.count(marker) == 1
text = text.replace(marker, "    from trial_fixture import pinned_carrier\n    machine = pinned_carrier()")
marker = "    started = perf_counter()\n    map_program = jax.jit(mapped).lower(reference, *operands).compile()"
assert text.count(marker) == 1
text = text.replace(marker, "    from trial_probe import measure\n    measure(locals())\n    return\n\n" + marker)
path.write_text(text)
PY
    command=("$python" -m scripts.prototypes.fast_map_production.solver --kind limited --cells 550 --method gmres --max-steps 1 --line-search --no-panel --output "$run/$arm-setup.json")
    (
        cd "$tree" || exit 2
        printf 'revision=%s tree=%s command=' "$revision" "$tree"
        printf '%q ' "${command[@]}"
        printf '\n'
        "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import os,nova; from nova.biot import polygonanalytic; print("nova="+nova.__file__, "measurement_module="+polygonanalytic.__file__, "cwd="+os.path.realpath(os.getcwd()), flush=True)'
        "${command[@]}"
        code=$?
        echo "EXIT=$code"
        exit "$code"
    ) > "$run/$arm-trial.log" 2>&1
    code=$?
    echo "ARM=$arm EXIT=$code"
    if (( code != 0 )); then exit "$code"; fi
done
echo EXIT=0
