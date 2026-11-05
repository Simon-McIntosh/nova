"""Measure the production lane mapper and its booking work on shared seeds."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from scripts.prototypes.bank_solve_cycle import measure as cycle
from scripts.prototypes.bank_step_speed import profile as timer

sr = cycle.support_read


def production_mapping(function, values):
    """Use the lane mapper present at the fresh-factor solve call site."""
    source = inspect.getsource(sr.solve_batch)
    if "lambda _: jax.vmap(one_fresh)((state, step_support))" in source:
        return jax.vmap(function)(values)
    assert "lambda _: jax.lax.map(one_fresh, (state, step_support))" in source
    return jax.lax.map(function, values)


def inputs(lanes, machine):
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    scales = jnp.linspace(2e-5, 6e-5, lanes)
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    support = jax.jit(
        lambda b, state: sr.read_batch(
            b,
            cache,
            state,
            jnp.broadcast_to(seeds, (lanes, *seeds.shape)),
            jnp.broadcast_to(roots, (lanes, *roots.shape)),
            convention,
            policy,
        )
    )(bank, states)
    booked = jax.jit(
        lambda b, state, reads: jax.lax.map(
            lambda items: (
                sr.step_residual(
                    b, items[0], items[0], items[1], profile, quadrature
                ).booked
            ),
            (state, reads),
        )
    )(bank, states, support)
    return states, support, booked


def jacobians(
    bank, states, support, booked, profile, quadrature, mapping=production_mapping
):
    return mapping(
        lambda items: sr.step_jacobian(
            bank,
            *items[:2],
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            items[2],
            fresh_geometry=True,
        ),
        (states, support, booked),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--matrices", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    cycle.reference.emit(
        "HEADER",
        revision=args.revision,
        nova=cycle.reference.nova.__file__,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
        backend=jax.default_backend(),
    )
    assert jax.config.jax_enable_x64
    machine = cycle.reference.machine("diverted")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    output = {}
    matrices = {}
    for lanes in (1, 16):
        states, support, booked = inputs(lanes, machine)

        def emit(kind, **data):
            cycle.reference.emit(kind, lanes=lanes, **data)
            if kind == "STAGE":
                output[f"{lanes}:{data['name']}"] = data["median_ms"]

        with patch.object(timer, "emit", emit):
            def function(b, m, s, p):
                return jacobians(b, m, s, p, profile, quadrature)
            result = timer.measure("assembly", function, bank, states, support, booked)
            assert np.isfinite(np.asarray(result)).all()
            if args.matrices:
                matrices[f"matrix_{lanes}"] = np.asarray(result)
            if args.profile and lanes == 1:
                from scripts.prototypes.bank_step_jacobian_split.measure import parts

                functions = parts(
                    profile, quadrature, args.run / f"{args.arm}-extracted.py"
                )
                for mapping, mapper in (
                    ("map", jax.lax.map),
                    ("vmap", lambda f, x: jax.vmap(f)(x)),
                ):
                    for name in ("local", "geometry"):
                        def function(b, m, s, p, f=functions[name], mapper=mapper):
                            return mapper(lambda x: f(b, *x), (m, s, p))
                        timer.measure(
                            f"{name}_{mapping}", function, bank, states, support, booked
                        )
                        program = (
                            jax.jit(function)
                            .lower(bank, states, support, booked)
                            .compile()
                        )
                        (args.run / f"{args.arm}-{name}-{mapping}.hlo").write_text(
                            program.as_text()
                        )
                timer.measure(
                    "assembly_vmap",
                    lambda b, m, s, p: jacobians(
                        b, m, s, p, profile, quadrature, lambda f, x: jax.vmap(f)(x)
                    ),
                    bank,
                    states,
                    support,
                    booked,
                )
        (args.run / f"{args.arm}-timings.json").write_text(json.dumps(output, indent=2))
    if matrices:
        np.savez(args.run / f"{args.arm}-matrices.npz", **matrices)
    if args.solve:
        with (
            patch.object(cycle.reference, "machine", lambda kind: machine),
            patch.object(
                sys,
                "argv",
                [
                    "measure",
                    "--output",
                    str(args.run / f"{args.arm}-solve.json"),
                    "--revision",
                    args.revision,
                ],
            ),
        ):
            cycle.main()


if __name__ == "__main__":
    main()
