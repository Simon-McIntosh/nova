"""Measure device batches of the local booking tangent on one P100."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_step_speed.parse_trace import main as parse_trace


def main():
    root = Path(__file__).resolve().parents[3]
    print(
        "HEADER",
        json.dumps(
            {
                "revision": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "tree": str(root),
                "cwd": str(Path.cwd().resolve()),
                "nova": nova.__file__,
                "module": support_read.__file__,
                "backend": jax.default_backend(),
            }
        ),
        flush=True,
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(
        lambda state: support_read.read(
            bank, cache, state, seeds, roots, convention, policy
        )
    )(moments)
    booked = jax.jit(
        lambda state, read: support_read.step_residual(
            bank, state, state, read, profile, quadrature
        )
    )(moments, support).booked

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    reference = None
    for width in (4, 8, 16, 32):
        selected = replace(quadrature, cut_batch_width=width)
        compiled = jax.jit(
            lambda b, m, s, p: support_read._local_booking_sensitivities(
                b, m, s, profile, gradient, selected, p
            )
        )
        result = compiled(bank, moments, support, booked)
        jax.block_until_ready(result)
        walls = []
        for _ in range(3):
            started = perf_counter()
            result = compiled(bank, moments, support, booked)
            jax.block_until_ready(result)
            walls.append(1e3 * (perf_counter() - started))
        values = np.asarray(result)
        if reference is None:
            reference = values
        relative = np.linalg.norm(values - reference) / max(
            np.linalg.norm(reference), 1
        )
        print(
            "WIDTH",
            json.dumps(
                {
                    "width": width,
                    "wall_ms": sorted(walls)[1],
                    "relative_to_four": float(relative),
                }
            ),
            flush=True,
        )
        assert relative <= 1e-10
        trace = Path(os.environ["BANK_STEP_SWEEP_TRACE_ROOT"]) / str(width)
        jax.profiler.start_trace(str(trace), create_perfetto_link=False)
        with jax.profiler.TraceAnnotation(f"stage:booking_width_{width}"):
            jax.block_until_ready(compiled(bank, moments, support, booked))
        jax.profiler.stop_trace()
        parse_trace(next(trace.rglob("*.trace.json.gz")))


if __name__ == "__main__":
    main()
