#!/bin/bash
#SBATCH --job-name=mast-22086-pinned-cache
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -uo pipefail

shadow=/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s18-hexgrid/hdg-cache-replay-shadow
python_bin=/home/ITER/mcintos/Code/nova/.venv/bin/python
export TMPDIR=/tmp
export PYTHONPATH="$shadow"
export JAX_PLATFORMS=cuda,cpu

echo "WRAP_SENTINEL before_python utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) pid=$$"

heartbeat() {
    while true; do
        echo "PAYLOAD_HEARTBEAT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) wrapper_pid=$$"
        sleep 30
    done
}

heartbeat &
heartbeat_pid=$!
set +e
"$python_bin" -u - "$shadow" <<'PY'
from pathlib import Path
import runpy
import sys

shadow = Path(sys.argv[1])
print(f"PYTHON_SENTINEL interpreter_started executable={sys.executable}", flush=True)
import jax

print(f"PYTHON_SENTINEL devices={jax.devices()}", flush=True)
script = shadow / "docs/figures/topology-visual-corroboration/generate_topology_visuals.py"
sys.argv = [str(script), "--cache-only", "--regenerate-mast-shot", "22086"]
runpy.run_path(str(script), run_name="__main__")
PY
status=$?
set -e
kill "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true
echo "WRAP_SENTINEL after_python status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
