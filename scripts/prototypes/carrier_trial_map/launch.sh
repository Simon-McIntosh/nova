#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=00:40:00
set -uo pipefail
export TMPDIR=/tmp
scratch=$(mktemp -d "$TMPDIR/carrier-measure-${SLURM_JOB_ID}-XXXX")
trap 'printf "removing scratch %s\n" "$scratch"; rm -rf "$scratch"' EXIT
export TMPDIR="$scratch"
py=/home/ITER/mcintos/Code/nova/.venv/bin/python
source_tree=$1
export MEASURE_RUN=$2 MEASURE_BANK=$3
shift 3
for arm in "$@"; do
  revision=1fd48c0fa32641a918ba3787c4a1400d5779a692
  if [ "$arm" = split ]; then revision=a7ea79849; fi
  tree="$scratch/$arm"
  mkdir "$tree"
  git -C "$source_tree" archive "$revision" nova scripts tests benchmarks pyproject.toml | tar -x -C "$tree" || exit 2
  if [ "$arm" = topology ]; then
    git -C "$source_tree" show a7ea79849:nova/equilibrium/topology.py > "$tree/nova/equilibrium/topology.py"
  fi
  export PYTHONPATH="$tree:$source_tree/scripts/prototypes/carrier_trial_map" MEASURE_ARM="$arm"
  "$py" - "$tree" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]) / 'scripts/prototypes/fast_map_production/solver.py'
text = path.read_text()
marker = '    started = perf_counter()\n    map_program = jax.jit(mapped).lower(reference, *operands).compile()'
assert text.count(marker) == 1
text = text.replace(marker, '    from probe import measure\n    measure(locals())\n    return\n\n' + marker)
path.write_text(text)
PY
  command="$py -m scripts.prototypes.fast_map_production.solver --kind limited --cells 550 --method gmres --max-steps 1 --line-search --no-panel --output $MEASURE_RUN/$arm-setup.json"
  (
    cd "$tree" || exit 2
    printf 'revision=%s tree=%s command=%s\n' "$revision" "$tree" "$command"
    printf '# command: %s\n' "$command"
    "$py" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import nova, os; from nova.equilibrium import topology; print("nova="+nova.__file__, "measurement_module="+topology.__file__, "cwd="+os.getcwd(), flush=True)'
    $command
    code=$?
    echo "EXIT=$code"
    exit "$code"
  ) > "$MEASURE_RUN/$arm-stages.log" 2>&1
  code=$?
  echo "ARM=$arm EXIT=$code"
  if [ "$code" != 0 ]; then echo "EXIT=$code"; exit "$code"; fi
done
echo EXIT=0
