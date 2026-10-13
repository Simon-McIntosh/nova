#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --job-name=fixed-point-latency
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE"
export PYTHONPATH="$PWD"
run_dir="$1"
for cells in 2000 5000 10000; do
    for kind in diverted limited; do
        /home/ITER/mcintos/Code/nova/.venv/bin/python -m scripts.prototypes.fixed_points.latency_gate --kind "$kind" --cells "$cells" --out "$run_dir/rows" > "$run_dir/latency-$kind-$cells.log" 2>&1
        result=$?
        echo EXIT=$result >> "$run_dir/latency-$kind-$cells.log"
        echo ROW=$kind-$cells EXIT=$result
        if [ "$result" -ne 0 ]; then exit "$result"; fi
    done
done
echo EXIT=0
