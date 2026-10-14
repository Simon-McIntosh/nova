"""Test damped exact-support iterations from nearby analytic moments."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from live_iterations import _setup

import jax
import jax.numpy as jnp
import numpy as np
import nova
from nova.equilibrium import gs


def run(kind, cells, cache, bias_row, newton_row, out):
    if jax.default_backend() != "gpu" or jax.config.jax_enable_x64 is not True:
        raise RuntimeError("live-map measurement requires H200 double precision")
    (
        initial,
        operands,
        cache_window,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        provenance,
    ) = _setup(kind, cells, cache, bias_row, newton_row)
    scale = jnp.maximum(jnp.max(jnp.abs(initial[:, 0])), 1e-30) * jnp.stack(
        (jnp.ones_like(operands.geometry.pitch),) + (operands.geometry.pitch,) * 2,
        axis=-1,
    )

    @jax.jit
    def one_step(state, operands, window):
        residual, (support, booked) = gs.moment_residual(
            state,
            operands,
            window,
            roots,
            convention,
            policy,
            profile,
            quadrature,
        )
        return residual, support.valid & booked.valid, support.stationary.status

    rows = {}
    for relaxation in (0.25, 0.1, 0.02):
        state = initial * 1.001
        history = []
        validity = []
        statuses = []
        walls = []
        for _ in range(40):
            start = perf_counter()
            residual, valid, status = one_step(state, operands, cache_window)
            jax.block_until_ready((residual, valid, status))
            walls.append((perf_counter() - start) * 1000)
            error = float(jnp.max(jnp.abs(residual / scale)))
            history.append(error)
            validity.append(bool(valid))
            statuses.append(np.asarray(status).tolist())
            if not np.isfinite(error) or not bool(valid) or error < 1e-12:
                break
            state = state - relaxation * residual
        rows[str(relaxation)] = dict(
            iterations=len(history),
            residual_history=history,
            valid_history=validity,
            stationary_status=statuses,
            iteration_wall_ms=walls,
            converged=bool(validity[-1] and np.isfinite(error) and error < 1e-12),
        )
        print(f"ROW {kind} {relaxation} {rows[str(relaxation)]}", flush=True)
    result = dict(
        kind=kind,
        requested_cells=cells,
        realised_cells=len(initial),
        provenance=provenance,
        seed_basis="analytic moments scaled by 1.001",
        rows=rows,
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
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--bias-row", type=Path, required=True)
    parser.add_argument("--newton-row", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd().resolve()}", flush=True)
    print(f"NOVA_MODULE={nova.__file__} GS_MODULE={gs.__file__}", flush=True)
    print(f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()}", flush=True)
    run(args.kind, args.cells, args.cache, args.bias_row, args.newton_row, args.out)


if __name__ == "__main__":
    main()
