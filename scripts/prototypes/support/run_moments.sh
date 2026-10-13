#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:28:00
#SBATCH --job-name=moment-newton
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
run_dir="$1"
cache_dir="$2"
failed=0
for row in "limited 2000" "diverted 2000" "limited 5000" "diverted 5000" "limited 10000" "diverted 10000"; do
    read -r kind cells <<< "$row"
    log="$run_dir/moments-$kind-$cells.log"
    timeout --verbose 260s /home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.support.solve_rows \
        --kind "$kind" --cells "$cells" --out "$run_dir/moment-rows" --cache "$cache_dir" > "$log" 2>&1
    result=$?
    echo "EXIT=$result" >> "$log"
    echo "ROW=$kind-$cells EXIT=$result LOG=$log"
    if [ "$result" -ne 0 ]; then failed=1; fi
done
echo "EXIT=$failed"
exit "$failed"
