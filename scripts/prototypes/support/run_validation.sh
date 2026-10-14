#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --job-name=moment-validation
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda,cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
export NOVA_ANALYTIC_ORACLE_BUILDER=device
export NOVA_MEASUREMENT_REVISION="$(git rev-parse HEAD)"
run_dir="$1"
cache_dir="$2"
mkdir -p "$run_dir/diagnostics" "$run_dir/newton" "$run_dir/jets"
run_stage() {
    local log="$1"
    shift
    {
        printf 'revision=%s TREE=%s COMMAND=' "$NOVA_MEASUREMENT_REVISION" "$PWD"
        printf '%q ' "$@"
        printf '\n'
        /home/ITER/mcintos/Code/nova/.venv/bin/python -c 'import os,nova; print("NOVA_MODULE="+nova.__file__+" CWD="+os.getcwd(), flush=True)'
        "$@"
        result=$?
        echo "EXIT=$result"
        return "$result"
    } > "$log" 2>&1
}
run_stage "$run_dir/forward-gates.log" /home/ITER/mcintos/Code/nova/.venv/bin/python \
    -m scripts.prototypes.support.gate_rows \
    --base 2be7601fdd73557e87fce380a859aaf58d184209 --out "$run_dir/gates" \
    tests/test_solve_receipt_topology.py tests/test_solve_trips_are_a_scan.py \
    tests/test_solve_path_census_inputs.py tests/test_forward_constraints.py \
    tests/test_forward_secondary_null.py tests/test_forward_census.py \
    tests/test_compiled_slice_cache.py tests/test_production_solver_receipt_trip_alignment.py \
    tests/test_solve_program_size.py tests/test_equilibrium_forward_solve.py &
gate_pid=$!
failed=0
run_stage "$run_dir/diagnostics-limited-110.log" /home/ITER/mcintos/Code/nova/.venv/bin/python \
    -m scripts.prototypes.support.solve_rows --kind limited --cells 110 \
    --out "$run_dir/diagnostics" --cache "$cache_dir" --require-cache \
    --jet-cache "$run_dir/jets" --diagnostics-only || failed=1
for row in "diverted 2000" "limited 2000" "limited 5000"; do
    read -r kind cells <<< "$row"
    run_stage "$run_dir/dense-$kind-$cells.log" /home/ITER/mcintos/Code/nova/.venv/bin/python \
        -m scripts.prototypes.support.solve_rows --kind "$kind" --cells "$cells" \
        --out "$run_dir/newton" --cache "$cache_dir" --require-cache \
        --jet-cache "$run_dir/jets" --methods dense || failed=1
done
run_stage "$run_dir/support-diverted-5000.log" /home/ITER/mcintos/Code/nova/.venv/bin/python \
    -m scripts.prototypes.support.measure --kind diverted --cells 5000 \
    --out "$run_dir/support" --cache "$run_dir/support/cache" --prepare-only || failed=1
run_stage "$run_dir/dense-diverted-5000.log" /home/ITER/mcintos/Code/nova/.venv/bin/python \
    -m scripts.prototypes.support.solve_rows --kind diverted --cells 5000 \
    --out "$run_dir/newton" --cache "$run_dir/support/cache" --require-cache \
    --jet-cache "$run_dir/jets" --methods dense || failed=1
wait "$gate_pid" || failed=1
echo "EXIT=$failed"
exit "$failed"
