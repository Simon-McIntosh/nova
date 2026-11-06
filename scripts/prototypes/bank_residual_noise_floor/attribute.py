"""Resolve saddle-geometry sensitivity and inspect the roots it actually uses."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from contextlib import contextmanager
from decimal import Decimal, localcontext
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current, support_read as sr, topology
from scripts.prototypes.bank_residual_noise_floor.measure import nudged
from scripts.prototypes.bank_booking_determinism.measure import emit
from scripts.prototypes.bank_solve_profile import fixture

STAGES = (
    "none",
    "saddle_position",
    "saddle_direction",
    "saddle_curvature",
    "saddle_cubic",
    "edge_intervals",
    "normal_vertices",
    "read_roots",
    "booking_roots",
    "depressed_linear",
    "depressed_constant",
    "cubic_discriminant",
    "cubic_angle",
)


@contextmanager
def injections(stage, seed, hits):
    def bump(x, name):
        hits.add(name)
        return jnp.where(stage == STAGES.index(name), nudged(x, seed), x)

    original_book, original_rings = current.book, current._normal_rings
    original_roots = topology._cubic_parameters
    inside = False

    def book(psi, reading, *args):
        form = reading.saddle_form
        form = form._replace(
            **{
                key: bump(getattr(form, key), "saddle_" + key)
                for key in ("position", "direction", "curvature", "cubic")
            }
        )
        reading = reading._replace(
            saddle_form=form,
            edge_interval=bump(reading.edge_interval, "edge_intervals"),
        )
        return original_book(psi, reading, *args)

    def rings(*args):
        nonlocal inside
        inside = True
        vertices, counts = original_rings(*args)
        inside = False
        return bump(vertices, "normal_vertices"), counts

    def roots(*args):
        return bump(original_roots(*args), "booking_roots" if inside else "read_roots")

    source, start = inspect.getsourcelines(original_roots)
    anchors = {
        "q = 2 *": ("p", "depressed_linear"),
        "discriminant = q": ("q", "depressed_constant"),
        "multiple = discriminant": ("discriminant", "cubic_discriminant"),
        "angles = angle": ("angle", "cubic_angle"),
    }
    lines = {
        start + i: item
        for i, line in enumerate(source)
        for anchor, item in anchors.items()
        if line.strip().startswith(anchor)
    }
    assert len(lines) == len(anchors)

    def trace(frame, event, arg):
        if frame.f_code is not original_roots.__code__:
            return None
        if event == "line" and inside and frame.f_lineno in lines:
            name, stage_name = lines[frame.f_lineno]
            frame.f_locals[name] = bump(frame.f_locals[name], stage_name)
        return trace

    with (
        patch.object(current, "book", book),
        patch.object(current, "_normal_rings", rings),
        patch.object(topology, "_cubic_parameters", roots),
    ):
        sys.settrace(trace)
        try:
            yield
        finally:
            sys.settrace(None)


def root_receipt(bank, reading, quadrature, cell):
    """Capture the actual root intermediates, then refine used roots in Decimal."""
    saved = {}
    root_names = (
        "cubic",
        "quadratic",
        "linear",
        "constant",
        "b",
        "c",
        "d",
        "p",
        "q",
        "discriminant",
        "amplitude",
        "angle",
        "three",
        "root",
    )

    def trace(frame, event, arg):
        if frame.f_code is topology._cubic_parameters.__code__:
            if event == "return":
                saved.update({k: frame.f_locals[k] for k in root_names})
                saved["roots"] = arg
            return trace
        if frame.f_code is current._normal_rings.__code__:
            if event == "return":
                saved.update(
                    {k: frame.f_locals[k] for k in ("hit", "active", "parameters")}
                )
            return trace
        return None

    def inspect_rings(vertices, count, centre, form, selected, intervals):
        sys.settrace(trace)
        try:
            current._normal_rings(
                vertices, count, centre, form, selected, intervals, quadrature.segments
            )
        finally:
            sys.settrace(None)
        return saved

    captured = jax.jit(inspect_rings)(
        bank.geometry.vertices[cell],
        bank.geometry.vertex_count[cell],
        bank.geometry.centre[cell],
        reading.saddle_form,
        reading.fragment_selected[cell],
        reading.edge_interval[cell],
    )
    captured = jax.tree.map(np.asarray, captured)
    rows = []
    with localcontext() as ctx:
        ctx.prec = 80
        for ray, edge, slot in np.argwhere(captured["hit"]):
            values = [
                Decimal.from_float(float(captured[k][ray, edge]))
                for k in ("cubic", "quadratic", "linear", "constant")
            ]
            a, b, c, d = values
            approximate = float(captured["roots"][ray, edge, slot])
            root = Decimal.from_float(approximate)
            for _ in range(12):
                root -= (((a * root + b) * root + c) * root + d) / (
                    (3 * a * root + 2 * b) * root + c
                )
            error = Decimal.from_float(approximate) - root
            offset = float(captured["b"][ray, edge] / 3)
            rows.append(
                dict(
                    ray=int(ray),
                    edge=int(edge),
                    slot=int(slot),
                    root=approximate,
                    refined=str(root),
                    error=float(error),
                    offset=offset,
                    cancellation=float(abs(offset) / max(abs(approximate), 1e-300)),
                    polynomial=[str(v) for v in values],
                )
            )
    return captured, rows


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
    loaded = np.load(args.run / "gpu-states.npz")
    state = tuple(jnp.asarray(loaded[f"moment_{i}"]) for i in range(3))

    def evaluate(bank, cache, state, seeds, previous, stage, seed):
        support = sr.read_batch(
            bank, cache, state, seeds, previous, batch.convention, batch.policy
        )
        result = jax.vmap(
            lambda m, s: sr.step_residual(
                bank, m, m, s, batch.profile, batch.quadrature
            )
        )(state, support)
        return jnp.concatenate(result.residual, axis=-1), support

    inputs = (
        bank,
        cache,
        state,
        batch.seeds_batch,
        batch.previous_batch,
        jnp.asarray(0),
        jnp.asarray(0),
    )
    ordinary, support = jax.block_until_ready(jax.jit(evaluate)(*inputs))
    hits = set()

    def instrumented(*inputs):
        with injections(inputs[-2], inputs[-1], hits):
            return evaluate(*inputs)

    program = jax.jit(instrumented).lower(*inputs).compile()
    baseline, _ = jax.block_until_ready(program(*inputs))
    assert hits == set(STAGES[1:]), hits
    record = dict(
        instrumentation_delta=np.max(np.abs(baseline - ordinary), axis=-1).tolist(),
        stages={},
        normal_cells=np.flatnonzero(
            np.asarray(support.reading.normal_form_cells[0])
        ).tolist(),
    )
    arrays = {"baseline": np.asarray(baseline)}
    for stage, name in enumerate(STAGES[1:], 1):
        values = []
        for seed in range(1, 65):
            residual, fresh = jax.block_until_ready(program(*inputs[:-2], stage, seed))
            assert np.all(fresh.valid) and np.all(np.isfinite(residual))
            values.append(np.asarray(residual))
        values = np.stack(values)
        delta = np.max(np.abs(values - baseline), axis=-1)
        arrays[name] = values
        record["stages"][name] = dict(
            median=np.median(delta, axis=0).tolist(),
            maximum=delta.max(axis=0).tolist(),
            location=np.argmax(
                np.max(np.abs(values - baseline), axis=0), axis=-1
            ).tolist(),
        )
        (args.run / "geometry.json").write_text(json.dumps(record, indent=2))
        emit("STAGE", name=name, **record["stages"][name])
    np.savez_compressed(args.run / "geometry-residuals.npz", **arrays)
    captured, roots = root_receipt(
        bank, jax.tree.map(lambda x: x[0], support.reading), batch.quadrature, 184
    )
    np.savez_compressed(args.run / "root-intermediates.npz", **captured)
    record["used_roots"] = roots
    (args.run / "geometry.json").write_text(json.dumps(record, indent=2))
    emit("ROOTS", roots=roots)
    emit("COMPLETE", output=str(args.run / "geometry.json"))


if __name__ == "__main__":
    main()
