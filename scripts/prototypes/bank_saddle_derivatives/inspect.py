"""Separate nodal input rounding from Bell reconstruction rounding at a saddle."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from decimal import Decimal, localcontext
import json
from math import factorial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import fixed_points
from scripts.prototypes.bank_residual_noise_floor.measure import nudged
from scripts.prototypes.bank_residual_noise_floor.propagate import field_receipt
from scripts.prototypes.bank_solve_profile import fixture


def decimal(value):
    return Decimal.from_float(float(value))


def exact_coefficients(jets, vertices, scale, transform):
    """Evaluate the stored Bell operator with 80-digit arithmetic throughout."""
    factors = [Decimal(1), scale, scale, scale**2, scale**2, scale**2]
    scaled = [[v * factor for v, factor in zip(row, factors)] for row in jets]
    f, gx, gy, hxx, hxy, hyy = scaled[0]
    residual = []
    for values, vertex in zip(scaled, vertices):
        x, y = map(decimal, vertex)
        taylor = [
            f + gx * x + gy * y + hxx * x * x / 2 + hxy * x * y + hyy * y * y / 2,
            gx + hxx * x + hxy * y,
            gy + hxy * x + hyy * y,
            hxx,
            hxy,
            hyy,
        ]
        residual.extend(a - b for a, b in zip(values, taylor))
    coefficients = [
        sum(decimal(a) * b for a, b in zip(row, residual)) for row in transform
    ]
    for index, value in enumerate([f, gx, gy, hxx / 2, hxy, hyy / 2]):
        coefficients[index] += value
    return coefficients


def exact_derivatives(coefficients, point, origin, scale, rank):
    xy = [(decimal(p) - decimal(o)) / scale for p, o in zip(point, origin)]
    values = []
    for indices in np.ndindex((2,) * rank):
        dx, dy = indices.count(0), indices.count(1)
        value = Decimal(0)
        for coefficient, (a, b) in zip(coefficients, fixed_points.POWERS):
            if a >= dx and b >= dy:
                multiplier = (
                    factorial(a)
                    // factorial(a - dx)
                    * factorial(b)
                    // factorial(b - dy)
                )
                value += (
                    coefficient * multiplier * xy[0] ** (a - dx) * xy[1] ** (b - dy)
                )
        values.append(value / scale**rank)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            dict(
                revision=args.revision,
                nova=nova.__file__,
                module=fixed_points.__file__,
                cwd=str(Path.cwd()),
                device=str(jax.devices()),
                x64=jax.config.jax_enable_x64,
            )
        ),
        flush=True,
    )
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"
    batch = fixture.build(16)
    bank = batch.operands
    saved = np.load(args.source / "gpu-states.npz")
    prior = np.load(args.source / "propagation.npz")
    worst = json.loads((args.source / "propagation.json").read_text())["worst"]
    lane, sample = worst["lane"], worst["sample"]
    state = tuple(jnp.asarray(saved[f"moment_{i}"][lane]) for i in range(3))
    shifted = tuple(nudged(v, sample + 1, 1 + sample % 4) for v in state)
    point = jnp.asarray(prior["field_0_point"])
    capture = jax.jit(field_receipt)
    observations = [
        jax.tree.map(np.asarray, capture(bank, value, point))
        for value in (state, shifted)
    ]
    records = []
    derivatives = []
    with localcontext() as context:
        context.prec = 80
        for arm, (value, observed) in enumerate(zip((state, shifted), observations)):
            cell, triangle = int(observed["cell"]), int(observed["triangle"])
            slots = np.asarray(bank.mesh.triangles[cell, triangle])
            scale = decimal(bank.mesh.scales[cell, triangle])
            vertices = np.asarray(bank.mesh.local_vertices[cell, triangle])
            origin = np.asarray(bank.mesh.origins[cell, triangle])
            transform = np.asarray(bank.mesh.transforms[cell, triangle])
            flat = [decimal(v) for v in np.stack(value, axis=-1).reshape(-1)]
            rows = np.asarray(bank.rows[slots]).reshape(3, -1, 6)
            exterior = np.asarray(bank.exterior[slots])
            exact_jets = [
                [
                    sum(decimal(a) * b for a, b in zip(rows[n, :, q], flat))
                    + decimal(exterior[n, q])
                    for q in range(6)
                ]
                for n in range(3)
            ]
            rounded_jets = [[decimal(v) for v in row] for row in observed["jets"]]
            ideal = exact_coefficients(exact_jets, vertices, scale, transform)
            rounded = exact_coefficients(rounded_jets, vertices, scale, transform)
            channels = {}
            exact = {}
            for rank, name in ((3, "third"), (4, "fourth")):
                ideal_d = exact_derivatives(ideal, point, origin, scale, rank)
                rounded_d = exact_derivatives(rounded, point, origin, scale, rank)
                actual = [decimal(v) for v in observed[name].reshape(-1)]
                channels[name] = dict(
                    actual=np.asarray(observed[name]).tolist(),
                    exact_from_rounded_jets=[str(v) for v in rounded_d],
                    exact_from_bank_operands=[str(v) for v in ideal_d],
                    input_error=float(
                        max(abs(a - b) for a, b in zip(rounded_d, ideal_d))
                    ),
                    reconstruction_error=float(
                        max(abs(a - b) for a, b in zip(actual, rounded_d))
                    ),
                    total_error=float(max(abs(a - b) for a, b in zip(actual, ideal_d))),
                )
                exact[name] = (ideal_d, rounded_d)
            derivatives.append(exact)
            records.append(
                dict(
                    arm=arm,
                    cell=cell,
                    triangle=triangle,
                    scale=float(scale),
                    **channels,
                )
            )
        movement = {}
        for name in ("third", "fourth"):
            ideal = [
                b - a for a, b in zip(derivatives[0][name][0], derivatives[1][name][0])
            ]
            rounded = [
                b - a for a, b in zip(derivatives[0][name][1], derivatives[1][name][1])
            ]
            actual = observations[1][name] - observations[0][name]
            movement[name] = dict(
                actual_max=float(np.max(np.abs(actual))),
                exact_bank_max=float(max(abs(v) for v in ideal)),
                exact_rounded_input_max=float(max(abs(v) for v in rounded)),
                input_rounding_change=float(
                    max(abs(a - b) for a, b in zip(rounded, ideal))
                ),
            )
    result = dict(
        revision=args.revision,
        worst=worst,
        point=np.asarray(point).tolist(),
        rows=records,
        movement=movement,
        discrete_triangle_unchanged=all(
            (r["cell"], r["triangle"]) == (records[0]["cell"], records[0]["triangle"])
            for r in records
        ),
    )
    args.output.write_text(json.dumps(result, indent=2))
    print("REPRODUCTION " + json.dumps(movement, sort_keys=True), flush=True)
    print("COMPLETE " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
