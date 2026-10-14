"""Measure dense factor cost at the moment-equation dimensions."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.scipy.linalg as dense
import numpy as np
import nova


def run(kind, cells, bias_row, out):
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    certificate = json.loads(bias_row.read_text())
    assert certificate["kind"] == kind and certificate["requested_cells"] == cells
    realised = certificate["realised_cells"]
    dimension = 3 * realised
    index = jnp.arange(dimension, dtype=jnp.float64)
    interaction = jnp.sin(0.013 * (index[:, None] + 1) * (index[None, :] + 1))
    matrix = jnp.eye(dimension, dtype=jnp.float64) + interaction / (100 * dimension)
    jax.block_until_ready(matrix)
    start = perf_counter()
    factor = jax.jit(dense.lu_factor).lower(matrix).compile()
    compile_seconds = perf_counter() - start
    jax.block_until_ready(factor(matrix))
    walls = []
    for _ in range(3):
        before = perf_counter()
        jax.block_until_ready(factor(matrix))
        walls.append((perf_counter() - before) * 1000)
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=realised,
        moment_dimension=dimension,
        matrix_bytes=matrix.nbytes,
        matrix_basis="dimension-matched diagonally dominant dense surrogate",
        actual_structured_jacobian=False,
        compile_seconds=compile_seconds,
        factor_median_ms=float(np.median(walls)),
        factor_min_ms=float(min(walls)),
        factor_walls_ms=walls,
        gpu_memory=jax.devices()[0].memory_stats(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}-{cells}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--bias-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} CWD={Path.cwd().resolve()}", flush=True)
    print(f"DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.bias_row, args.out)


if __name__ == "__main__":
    main()
