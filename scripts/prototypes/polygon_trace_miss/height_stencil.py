"""Measure a stable packed height column with a sixth-order corner stencil."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from collections import Counter
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot import polygonanalytic as kernel
from nova.equilibrium.topology import BiotMomentCoupling
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT, STATES

assert jax.config.jax_enable_x64 is True
jax.config.update("jax_enable_compilation_cache", False)


def describe(name, value):
    value = np.asarray(value)
    print(
        "ARRAY",
        name,
        "shape",
        value.shape,
        "finite",
        bool(np.isfinite(value).all()),
        flush=True,
    )
    assert np.isfinite(value).all()


def calibrate(measured, output):
    """Check a height column against the qualified independent reference."""
    with np.load(ROOT / "node-0.npz") as archive:
        known = archive["hessian"][:, :, 1]
    describe("calibration_measured_height_column", measured)
    describe("calibration_independent_height_column", known)
    scale = np.linalg.norm(known, axis=1)
    assert np.all(scale > 0)
    relative = np.linalg.norm(measured - known, axis=1) / scale
    perturbation = np.linalg.norm(1.000001 * measured - known, axis=1) / scale
    describe("calibration_relative_error", relative)
    describe("calibration_perturbation_error", perturbation)
    print("HEIGHT_INDEPENDENT_CALIBRATION", relative.tolist(), flush=True)
    print("HEIGHT_CALIBRATION_PERTURBATION", perturbation.tolist(), flush=True)
    assert max(relative) <= 1e-8
    assert min(perturbation) > 1e-8
    (output / "calibration-independent.json").write_text(
        json.dumps(
            dict(
                passed=True,
                budget=1e-8,
                relative=[float(v) for v in relative],
                perturbation_relative=[float(v) for v in perturbation],
                reference=str(ROOT / "node-0.npz"),
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="height-stencil")
    parser.add_argument("--point-budget", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("MEASUREMENT_MODULE", kernel.__file__, flush=True)
    print("MEASUREMENT_CWD", Path.cwd().resolve(), flush=True)
    print("DEVICES", jax.devices(), "x64", jax.config.jax_enable_x64, flush=True)
    geometry = dict(np.load(ROOT / "h200/geometry.npz"))
    polygons = tuple(
        p[:n]
        for p, n in zip(geometry["vertices"], geometry["vertex_count"], strict=True)
    )
    selected = [
        row["node"]
        for row in json.loads((ROOT / "split.json").read_text())["nodes"]
        if row["interior"]
    ]
    assert len(selected) == 64
    archives = [dict(np.load(ROOT / f"diagnosis/accuracy-{s}.npz")) for s in STATES]
    points = archives[0]["points"][selected]
    moments = jnp.asarray(np.stack([row["moments"] for row in archives]))
    coupling = BiotMomentCoupling.from_polygons(polygons)
    coefficients = jax.vmap(lambda m: coupling.coefficients(tuple(m.T)).T)(moments)
    describe("points", points)
    describe("moments", moments)
    describe("coefficients", coefficients)
    describe("packed_edges", coupling.edge)

    entries = Counter()

    def packed(point, height=None, purpose="sample"):
        entries[purpose] += 1
        return jnp.stack(
            kernel._packed_analytic_moments_primal(
                jnp,
                jnp.broadcast_to(point[0], coupling.norm.shape),
                jnp.broadcast_to(point[1], coupling.norm.shape)
                if height is None
                else height,
                coupling.edge,
                coupling.weight,
                coupling.norm,
                coupling.centre.T,
                coupling.centre.T,
                coupling.reflection_axis,
                coupling.reflection_partner,
                stable_poles=True,
            )
        )

    def contract(rows, radius):
        radial = jnp.sum(rows[None, 3:6, :] * coefficients, axis=(1, 2))
        vertical = jnp.sum(rows[None, 6:9, :] * coefficients, axis=(1, 2))
        return 2 * jnp.pi * radius * jnp.stack((vertical, -radial), axis=-1)

    partial = args.output / "height-partial.npz"
    values = []
    if partial.exists():
        with np.load(partial) as prior:
            np.testing.assert_array_equal(
                prior["points"], points[: len(prior["points"])]
            )
            describe("prior_height_derivative", prior["height_derivative"])
            values = list(prior["height_derivative"].swapaxes(0, 1))
    if values:
        calibrate(values[0], args.output)
    start = len(values)
    stop = min(64, start + args.point_budget) if args.point_budget else 64
    shifts = (1e-3, -1e-3, 5e-4, -5e-4, 2.5e-4, -2.5e-4)
    for index in range(start, stop):
        point = jnp.asarray(points[index])
        nearest = jnp.minimum(
            jnp.min(jnp.abs(coupling.edge[:, 1] - point[1]), axis=0),
            jnp.min(jnp.abs(coupling.edge[:, 3] - point[1]), axis=0),
        )
        corner = nearest <= 1e-7
        above = (
            jnp.maximum(
                jnp.max(coupling.edge[:, 1], axis=0),
                jnp.max(coupling.edge[:, 3], axis=0),
            )
            + 1.0
        )
        held_height = jnp.where(corner, above, point[1])
        _, analytic_rows = jax.jvp(
            lambda height: packed(point, height, purpose="height_jvp"),
            (held_height,),
            (jnp.where(corner, 0.0, 1.0),),
        )
        describe("corner_mask", corner)
        describe("held_height", held_height)
        describe("noncorner_height_derivative_rows", analytic_rows)
        samples = jnp.stack(
            [packed(point + jnp.asarray((0.0, shift))) for shift in shifts]
        )
        describe("packed_height_samples", samples)
        coarse = (samples[0] - samples[1]) / (2 * shifts[0])
        middle = (samples[2] - samples[3]) / (2 * shifts[2])
        fine = (samples[4] - samples[5]) / (2 * shifts[4])
        derivative_rows = (64 * fine - 20 * middle + coarse) / 45
        derivative_rows = jnp.where(corner[None, :], derivative_rows, analytic_rows)
        describe("height_derivative_rows", derivative_rows)
        value = np.asarray(contract(derivative_rows, point[0]))
        describe("height_column", value)
        values.append(value)
        temporary = partial.with_suffix(".pending.npz")
        np.savez(
            temporary,
            height_derivative=np.asarray(values).swapaxes(0, 1),
            points=points[: index + 1],
        )
        temporary.replace(partial)
        print("HEIGHT_NODE", index, flush=True)
        if index == 0:
            calibrate(value, args.output)
    data = np.asarray(values).swapaxes(0, 1)
    describe("height_derivative", data)
    receipt = dict(
        completed=stop == 64,
        count=stop,
        states=list(STATES),
        steps=list(shifts),
        per_cell=True,
        stable_poles=True,
        corner_mask=True,
        expression_entries=dict(entries),
        shape=list(data.shape),
        finite=bool(np.isfinite(data).all()),
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print("HEIGHT_EXPRESSION_ENTRIES", json.dumps(dict(entries)), flush=True)
    if stop == 64:
        print("MEASUREMENT_COMPLETE height-stencil", flush=True)
    else:
        print("MEASUREMENT_CHECKPOINT height-stencil", stop, "/64", flush=True)


if __name__ == "__main__":
    main()
