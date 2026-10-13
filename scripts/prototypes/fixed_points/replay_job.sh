#!/bin/bash
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --job-name=fixed-point-replay
export TMPDIR=/tmp XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export XLA_FLAGS="--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,CONDITIONAL,WHILE"
export PYTHONPATH="$PWD"
python_bin=/home/ITER/mcintos/Code/nova/.venv/bin/python
printf 'REVISION=%s TREE=%s COMMAND=%s -m scripts.prototypes.fixed_points.capture_replay --output %s\n' "$(git rev-parse HEAD)" "$PWD" "$python_bin" "$1"
"$python_bin" -c 'import nova; from pathlib import Path; print("NOVA_MODULE="+nova.__file__, "CWD="+str(Path.cwd().resolve()))'
"$python_bin" -m scripts.prototypes.fixed_points.capture_replay --output "$1"
result=$?
echo EXIT=$result
exit "$result"
