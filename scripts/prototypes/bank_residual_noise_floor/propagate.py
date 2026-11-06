"""Locate amplification by holding one fresh-read output fixed at a time."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import fixed_points, support_read as sr, topology
from scripts.prototypes.bank_residual_noise_floor.attribute import root_receipt
from scripts.prototypes.bank_residual_noise_floor.measure import nudged
from scripts.prototypes.bank_booking_determinism.measure import emit
from scripts.prototypes.bank_solve_profile import fixture

GROUPS = (
    "none",
    "saddle_form",
    "position",
    "direction",
    "curvature",
    "cubic",
    "edge_interval",
    "field_coefficients",
    "fluxes",
)


def field_receipt(bank, moments, point):
    """Observe the actual Bell construction on the saddle's selected triangle."""
    captured = {}

    def trace(frame, event, arg):
        if frame.f_code is not fixed_points.field_from_jets.__code__:
            return None
        if event == "return":
            captured.update(
                {
                    key: frame.f_locals[key]
                    for key in ("scaled", "taylor", "residual", "coefficients")
                }
            )
        return trace

    jets = sr.nodal_jets(bank, moments)
    sys.settrace(trace)
    try:
        field = fixed_points.field_from_jets(jets, bank.sample_index, bank.mesh)
    finally:
        sys.settrace(None)
    locator = fixed_points._CachedField(
        bank.mesh,
        jnp.arange(len(bank.nodes)),
        bank.rows,
        jnp.stack(moments, axis=-1),
        bank.exterior,
    )
    cell, triangle, slots = locator._location(point)
    captured = {k: v[cell, triangle] for k, v in captured.items()}
    captured.update(
        point=point,
        cell=cell,
        triangle=triangle,
        jets=jets[slots],
        scale=bank.mesh.scales[cell, triangle],
        transform=bank.mesh.transforms[cell, triangle],
    )
    third, fourth = topology._curvature_derivatives(
        field, point, jnp.median(bank.geometry.pitch)
    )
    captured.update(third=third, fourth=fourth)
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    emit(
        "HEADER",
        revision=args.revision,
        nova=nova.__file__,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    batch = fixture.build(16)
    bank, cache = batch.operands, batch.cache
    data = np.load(args.run / "gpu-states.npz")
    state = tuple(jnp.asarray(data[f"moment_{i}"]) for i in range(3))
    read = jax.jit(
        lambda bank, cache, state: sr.read_batch(
            bank,
            cache,
            state,
            batch.seeds_batch,
            batch.previous_batch,
            batch.convention,
            batch.policy,
        )
    )
    original = read(bank, cache, state)

    def evaluate(bank, cache, state, reference, group):
        fresh = sr.read_batch(
            bank,
            cache,
            state,
            batch.seeds_batch,
            batch.previous_batch,
            batch.convention,
            batch.policy,
        )

        def choose(actual, prior, name):
            return jax.tree.map(
                lambda x, y: jnp.where(group == GROUPS.index(name), y, x), actual, prior
            )

        reading = fresh.reading
        form = choose(reading.saddle_form, reference.saddle_form, "saddle_form")
        form = form._replace(
            **{
                name: choose(
                    getattr(form, name), getattr(reference.saddle_form, name), name
                )
                for name in GROUPS[2:6]
            }
        )
        reading = reading._replace(
            saddle_form=form,
            edge_interval=choose(
                reading.edge_interval, reference.edge_interval, "edge_interval"
            ),
            field_coefficients=choose(
                reading.field_coefficients,
                reference.field_coefficients,
                "field_coefficients",
            ),
            axis_flux=choose(reading.axis_flux, reference.axis_flux, "fluxes"),
            boundary_flux=choose(
                reading.boundary_flux, reference.boundary_flux, "fluxes"
            ),
        )
        selected = fresh._replace(reading=reading)
        booked = jax.vmap(
            lambda m, s: sr.step_residual(
                bank, m, m, s, batch.profile, batch.quadrature
            )
        )(state, selected)
        return jnp.concatenate(booked.residual, axis=-1), fresh

    program = jax.jit(evaluate)
    record = {"groups": {}, "movement": {}}
    arrays = {}
    moments = []
    supports = []
    for group, name in enumerate(GROUPS):
        baseline, fresh = jax.block_until_ready(
            program(bank, cache, state, original.reading, group)
        )
        rows = []
        for index in range(64):
            shifted = jax.tree.map(lambda x: nudged(x, index + 1, 1 + index % 4), state)
            residual, fresh = jax.block_until_ready(
                program(bank, cache, shifted, original.reading, group)
            )
            assert np.all(fresh.valid)
            rows.append(np.asarray(residual))
            if group == 0:
                moments.append(shifted)
                supports.append(jax.tree.map(np.asarray, fresh))
        values = np.stack(rows)
        arrays[name] = values
        arrays[name + "_baseline"] = np.asarray(baseline)
        record["groups"][name] = dict(
            maximum_change=float(np.max(np.abs(values - baseline))),
            maximum_spread=float(np.max(np.ptp(values, axis=0))),
        )
        emit("FROZEN", name=name, **record["groups"][name])
        (args.run / "propagation.json").write_text(json.dumps(record, indent=2))
    for name in ("position", "direction", "curvature", "cubic"):
        values = np.stack([getattr(s.reading.saddle_form, name) for s in supports])
        baseline = np.asarray(getattr(original.reading.saddle_form, name))
        movement = np.abs(values - baseline)
        ulp = np.abs(np.spacing(baseline))
        record["movement"][name] = dict(
            maximum=float(movement.max()),
            ulps=float(np.max(movement / np.where(ulp > 0, ulp, 1))),
        )
        arrays[name + "_values"] = values
        arrays[name + "_reference"] = baseline
    worst = np.unravel_index(
        np.argmax(np.abs(arrays["none"] - arrays["none_baseline"])),
        arrays["none"].shape,
    )
    sample, lane, component = map(int, worst)
    inspect_field = jax.jit(field_receipt)
    snapshots = []
    for m, point in (
        (state, original.reading.saddle_form.position),
        (moments[sample], supports[sample].reading.saddle_form.position),
    ):
        snapshots.append(
            jax.tree.map(
                np.asarray,
                inspect_field(
                    bank, tuple(x[lane] for x in m), jnp.asarray(point[lane])
                ),
            )
        )
    for index, snapshot in enumerate(snapshots):
        for key, value in snapshot.items():
            arrays[f"field_{index}_{key}"] = value
    record["worst"] = dict(sample=sample, lane=lane, component=component)
    captured, roots = root_receipt(
        bank, jax.tree.map(lambda x: x[lane], original.reading), batch.quadrature, 184
    )
    for key, value in captured.items():
        arrays["root_" + key] = value
    record["roots"] = roots
    np.savez_compressed(args.run / "propagation.npz", **arrays)
    (args.run / "propagation.json").write_text(json.dumps(record, indent=2))
    emit("MOVEMENT", values=record["movement"], worst=record["worst"])
    emit("ROOT_ERROR", maximum=max(abs(row["error"]) for row in roots))
    emit("COMPLETE", output=str(args.run / "propagation.json"))


if __name__ == "__main__":
    main()
