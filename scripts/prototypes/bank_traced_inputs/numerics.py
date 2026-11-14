"""Measure cold centroid-pin numerics and full-program reuse without pytest."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read as bank
from nova.equilibrium.constraint import VerticalPositionPin
from tests.test_bank_traced_inputs import small, _table

assert jax.config.jax_enable_x64 is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    operands, cache, roots, convention, policy, quadrature, state, _ = (
        small.__wrapped__()
    )
    source = _table()
    target = jnp.asarray([0.005, -0.008], dtype=jnp.float64)
    pin = VerticalPositionPin(target, source_scale=1e-3, tolerance=1e-8)
    moments = tuple(jnp.broadcast_to(value, (2, value.size)) for value in state)
    seeds = jnp.broadcast_to(roots, (2, *roots.shape))

    @jax.jit
    def program(op, window, state, amps, pressure, ff, target):
        return bank.solve_batch(
            op,
            window,
            state,
            seeds,
            seeds,
            convention,
            policy,
            lambda lane: current.TabulatedProfile(pressure[lane], ff[lane]),
            None,
            quadrature,
            max_steps=4,
            freeze_active_set=False,
            coil_currents=amps,
            vertical_pin=replace(pin, target_height=target),
        )

    events = []

    class Capture(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            events.append(message)
            if "jit(program)" in message or "tracing program" in message:
                print("PROGRAM_LOG", message, flush=True)

    logger = logging.getLogger("jax")
    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    pressure = jnp.stack((source.p_prime, source.p_prime * 1.1))
    ff = jnp.stack((source.ff_prime, source.ff_prime))
    report = dict(
        revision=args.revision, cwd=str(Path.cwd().resolve()), module=bank.__file__
    )

    def record(result, target):
        heights = jax.vmap(lambda *x: bank._centroid_height(operands.geometry, x))(
            *result.moments
        )
        return dict(
            target=np.asarray(target).tolist(),
            height=np.asarray(heights).tolist(),
            displacement=np.asarray(heights - target).tolist(),
            pin_error=np.asarray(result.pin_error).tolist(),
            amplitude=np.asarray(result.pin_amplitude).tolist(),
            residual=np.asarray(result.residual).tolist(),
            converged=np.asarray(result.converged).tolist(),
            steps=np.asarray(result.steps).tolist(),
            reads=np.asarray(result.reads).tolist(),
        )

    with jax.log_compiles(True):
        started = time.perf_counter()
        first = program(
            operands, cache, moments, jnp.asarray([[0.0], [0.2]]), pressure, ff, target
        )
        jax.block_until_ready(first)
        report["cold_seconds"] = time.perf_counter() - started
        report["first"] = record(first, target)
        report["cold_compiles"] = sum("Compiling jit(program)" in x for x in events)
        report["cold_events"] = [
            x for x in events if "jit(program)" in x or "tracing program" in x
        ]
        args.output.write_text(json.dumps(report, indent=2))
        # Calling the same dispatcher, rather than only the compiled executable,
        # makes any cache miss observable in the second slice's compile log.
        events.clear()
        started = time.perf_counter()
        following = program(
            operands,
            cache,
            moments,
            jnp.asarray([[0.1], [-0.1]]),
            pressure * 1.02,
            ff,
            target + 0.001,
        )
        jax.block_until_ready(following)
        report["second_seconds"] = time.perf_counter() - started
        report["second"] = record(following, target + 0.001)
        report["changed_compiles"] = sum("Compiling jit(program)" in x for x in events)
    report["pin_tolerance"] = 1e-8
    report["residual_tolerance"] = 1e-10
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, sort_keys=True), flush=True)
    assert report["cold_compiles"] == 1
    assert report["changed_compiles"] == 0
    for result in (report["first"], report["second"]):
        assert all(result["converged"])
        assert max(abs(x) for x in result["displacement"]) <= 1e-8
        assert max(result["residual"]) <= 1e-10


if __name__ == "__main__":
    main()
