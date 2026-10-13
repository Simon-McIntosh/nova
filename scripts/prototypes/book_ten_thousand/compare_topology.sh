#!/usr/bin/env bash
set -euo pipefail

run_dir="$1"
root="$PWD"
base_sha=6fce3ebfc2f6b46fb1a53db17c31d293699455cd
head_sha="$(git rev-parse HEAD)"
python=/home/ITER/mcintos/Code/nova/.venv/bin/python
base="$(mktemp -d -p "$TMPDIR" book-ten-thousand-base.XXXXXX)"
git archive "$base_sha" nova tests scripts benchmarks conftest.py pyproject.toml | tar -x -C "$base"
mkdir -p "$base/bin"
cat > "$base/bin/git" <<EOF
#!/bin/sh
if [ "\$1" = rev-parse ] && [ "\$2" = HEAD ] && [ "\$#" -eq 2 ]; then
    printf '%s\\n' $base_sha
else
    exec /usr/bin/git "\$@"
fi
EOF
chmod +x "$base/bin/git"

base_log="$run_dir/topology-h200-base.log"
head_log="$run_dir/topology-h200-head.log"
base_command="env TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PATH=$base/bin:/usr/bin:/bin PYTHONPATH=$base NOVA_TOPOLOGY_EVIDENCE_DIR=$run_dir/topology-h200-base-evidence $python -m pytest -p no:cacheprovider --basetemp=$base/pytest --rootdir=$base $base/tests/equilibrium/test_topology_read.py -q"
head_command="env TMPDIR=/tmp JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$root NOVA_TOPOLOGY_EVIDENCE_DIR=$run_dir/topology-h200-head-evidence $python -m pytest -p no:cacheprovider --basetemp=$TMPDIR/book-ten-thousand-h200-head-pytest --rootdir=$root $root/tests/equilibrium/test_topology_read.py -q"
printf 'revision=%s tree=%s command=%s\n' "$base_sha" "$base" "$base_command" > "$base_log"
(cd "$base" && env JAX_PLATFORMS=cuda,cpu PYTHONPATH="$base" "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import nova.equilibrium.topology as m; import pathlib; print("module_path="+m.__file__+" cwd="+str(pathlib.Path.cwd().resolve()))') >> "$base_log" 2>&1
set +e
(cd "$base" && eval "$base_command") >> "$base_log" 2>&1
base_status=$?
set -e
printf 'EXIT=%s\n' "$base_status" >> "$base_log"
printf 'revision=%s tree=%s command=%s\n' "$head_sha" "$root" "$head_command" > "$head_log"
env JAX_PLATFORMS=cuda,cpu PYTHONPATH="$root" "$python" -c 'from nova.jax.config import configure_dtypes; configure_dtypes(); import nova.equilibrium.topology as m; import pathlib; print("module_path="+m.__file__+" cwd="+str(pathlib.Path.cwd().resolve()))' >> "$head_log" 2>&1
set +e
eval "$head_command" >> "$head_log" 2>&1
head_status=$?
set -e
printf 'EXIT=%s\n' "$head_status" >> "$head_log"
printf 'BASE_LOG=%s BASE_EXIT=%s HEAD_LOG=%s HEAD_EXIT=%s\n' "$base_log" "$base_status" "$head_log" "$head_status"
