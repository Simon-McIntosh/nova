"""Count construction sites and staged equations of the centroid-pin program."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp

from nova.equilibrium import current, support_read as bank
from nova.equilibrium.constraint import VerticalPositionPin
from tests.test_bank_traced_inputs import small, _table

assert jax.config.jax_enable_x64 is True


def inventory(graph):
    primitives = Counter()
    seen = set()

    def visit(value):
        if hasattr(value, "eqns"):
            if id(value) in seen:
                return
            seen.add(id(value))
            for equation in value.eqns:
                primitives[equation.primitive.name] += 1
                visit(equation.params)
        elif hasattr(value, "jaxpr") and value.jaxpr is not value:
            visit(value.jaxpr)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(graph)
    return dict(
        unique_jaxprs=len(seen),
        unique_equations=sum(primitives.values()),
        primitives=dict(primitives),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    control = inventory(jax.make_jaxpr(lambda x: x * x + 1)(jnp.ones(2)))
    assert control["primitives"].get("mul", 0) == 1
    assert control["primitives"].get("add", 0) == 1
    print("INVENTORY_CONTROL", json.dumps(control, sort_keys=True), flush=True)
    operands, cache, roots, convention, policy, quadrature, state, _ = (
        small.__wrapped__()
    )
    source = _table()
    target = jnp.asarray([0.005, -0.008], dtype=jnp.float64)
    pin = VerticalPositionPin(target, source_scale=1e-3, tolerance=1e-8)
    moments = tuple(jnp.broadcast_to(value, (2, value.size)) for value in state)
    seeds = jnp.broadcast_to(roots, (2, *roots.shape))
    counts = Counter()

    def instrument(module, name):
        original = getattr(module, name)

        def counted(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        setattr(module, name, counted)

    for name in ("solve_batch", "read_batch", "newton_step", "step_jacobian"):
        instrument(bank, name)
    if hasattr(bank, "_booking_jacobian"):
        instrument(bank, "_booking_jacobian")
    for name in ("book", "book_tangent_channels"):
        instrument(current, name)

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

    pressure = jnp.stack((source.p_prime, source.p_prime * 1.1))
    ff = jnp.stack((source.ff_prime, source.ff_prime))
    started = time.perf_counter()
    graph = jax.make_jaxpr(program)(
        operands, cache, moments, jnp.asarray([[0.0], [0.2]]), pressure, ff, target
    )
    seconds = time.perf_counter() - started
    result = dict(
        revision=args.revision,
        cwd=str(Path.cwd().resolve()),
        module=bank.__file__,
        trace_seconds=seconds,
        construction_sites=dict(counts),
    )
    args.output.write_text(json.dumps(result, indent=2))
    result.update(inventory(graph))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, sort_keys=True), flush=True)
    assert counts["newton_step"] > 0 and counts["step_jacobian"] > 0


if __name__ == "__main__":
    main()
