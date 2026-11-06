"""Measure fresh-read sensitivity by injecting at existing stage boundaries."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import current, support_read as sr
from scripts.prototypes.bank_booking_determinism.measure import emit, fingerprint
from scripts.prototypes.bank_read_count_sensitivity.capture import masks, signatures
from scripts.prototypes.bank_solve_profile import fixture

STAGES = (
    "none",
    "field_coefficients",
    "nodal_jets",
    "stationary_fluxes",
    "whole_cells",
    "conic_cells",
    "normal_cells",
    "normalization",
    "subtraction",
    "signed_coefficients",
    "conic_vertices",
    "normalized_coefficients",
    "raw_current",
    "fit_coefficients",
)


def nudged(value, seed, steps=1):
    """Move each nonzero finite entry by signed representable steps."""
    slot = jnp.arange(value.size, dtype=jnp.uint32).reshape(value.shape)
    bits = (slot + jnp.asarray(seed, jnp.uint32) * 747796405) * 2891336453
    bits = (bits ^ (bits >> 16)) * 2246822519
    target = jnp.where((bits & 1) == 0, jnp.inf, -jnp.inf)
    moved = value
    for _ in range(steps):
        moved = jnp.nextafter(moved, target)
    return jnp.where(jnp.isfinite(value) & (value != 0), moved, value)


@contextmanager
def injections(stage, seed, hits):
    """Observe CPython's live locals; no arithmetic owner is duplicated."""

    def bump(value, name):
        hits.add(name)
        return jax.tree.map(
            lambda x: jnp.where(stage == STAGES.index(name), nudged(x, seed), x),
            value,
        )

    source, start = inspect.getsourcelines(current.book)
    # Each anchor is the first statement after the requested output exists.
    anchors = {
        "signed = quadrature.sigma": ("normalized", "normalized_coefficients"),
        "span = read.boundary_flux": ("coefficients", "fit_coefficients"),
        "def cell_field": ("signed", "signed_coefficients"),
        "batch_width = min": ("whole", "whole_cells"),
        "normal = _integrate_bucket": ("conic", "conic_cells"),
        "cuts = jax.tree.map": ("normal", "normal_cells"),
        "valid = read.valid": ("raw", "raw_current"),
        "return Booking": ("normalization", "normalization"),
    }
    lines = {}
    for offset, line in enumerate(source):
        for anchor, item in anchors.items():
            if line.strip().startswith(anchor):
                lines[start + offset] = item
    assert len(lines) == len(anchors), lines

    def trace(frame, event, arg):
        if frame.f_code is not current.book.__code__:
            return None
        if event == "line" and frame.f_lineno in lines:
            variable, name = lines[frame.f_lineno]
            frame.f_locals[variable] = bump(frame.f_locals[variable], name)
        return trace

    original_jets, original_book = sr.nodal_jets, current.book
    original_rings, original_residual = current._conic_rings, sr.step_residual

    def jets(*args):
        return bump(original_jets(*args), "nodal_jets")

    def book(psi, reading, *args):
        reading = reading._replace(
            field_coefficients=bump(reading.field_coefficients, "field_coefficients"),
            axis_flux=bump(reading.axis_flux, "stationary_fluxes"),
            boundary_flux=bump(reading.boundary_flux, "stationary_fluxes"),
        )
        return original_book(psi, reading, *args)

    def rings(*args):
        vertices, counts, valid = original_rings(*args)
        return bump(vertices, "conic_vertices"), counts, valid

    def residual(*args):
        result = original_residual(*args)
        return result._replace(residual=bump(result.residual, "subtraction"))

    # Match the unwrapped function's code even while its public name is patched.
    code = original_book.__code__

    def local_trace(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "line" and frame.f_lineno in lines:
            variable, name = lines[frame.f_lineno]
            frame.f_locals[variable] = bump(frame.f_locals[variable], name)
        return local_trace

    with (
        patch.object(sr, "nodal_jets", jets),
        patch.object(current, "book", book),
        patch.object(current, "_conic_rings", rings),
        patch.object(sr, "step_residual", residual),
    ):
        sys.settrace(local_trace)
        try:
            yield
        finally:
            sys.settrace(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    arm = "cpu" if args.cpu else "gpu"
    emit(
        "HEADER",
        revision=args.revision,
        nova=nova.__file__,
        module=sr.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == ("cpu" if args.cpu else "gpu")
    batch = fixture.build(16)
    bank, cache = batch.operands, batch.cache
    seeds, previous = batch.seeds_batch, batch.previous_batch
    state = batch.states
    if args.cpu:
        saved = np.load(args.run / "gpu-states.npz")
        state = tuple(jnp.asarray(saved[f"moment_{i}"][[0, 10]]) for i in range(3))
        seeds, previous = seeds[[0, 10], ...], previous[[0, 10], ...]
    else:

        def solve(bank, cache, state, seeds, previous):
            return sr.solve_batch(
                bank,
                cache,
                state,
                seeds,
                previous,
                batch.convention,
                batch.policy,
                batch.profile,
                fixture.gradient,
                batch.quadrature,
                max_steps=60,
                reuse_factor=False,
            )

        result = jax.block_until_ready(
            jax.jit(solve)(bank, cache, state, seeds, previous)
        )
        emit(
            "SOLVE",
            reads=np.asarray(result.reads).tolist(),
            residual=np.asarray(result.residual).tolist(),
            converged=np.asarray(result.converged).tolist(),
        )
        assert np.all(result.converged)
        state = result.moments
        np.savez(
            args.run / "gpu-states.npz",
            **{f"moment_{i}": np.asarray(v) for i, v in enumerate(state)},
        )

    def evaluate(bank, cache, state, seeds, previous, stage, seed):
        support = sr.read_batch(
            bank, cache, state, seeds, previous, batch.convention, batch.policy
        )
        result = jax.vmap(
            lambda m, s: sr.step_residual(
                bank, m, m, s, batch.profile, batch.quadrature
            )
        )(state, support)
        residual = jnp.concatenate(result.residual, axis=-1)
        return residual, support, result.booked

    inputs = (bank, cache, state, seeds, previous, jnp.asarray(0), jnp.asarray(0))
    plain = jax.jit(evaluate)
    reference, support, booked = jax.block_until_ready(plain(*inputs))
    hits = set()

    def instrumented(bank, cache, state, seeds, previous, stage, seed):
        with injections(stage, seed, hits):
            return evaluate(bank, cache, state, seeds, previous, stage, seed)

    program = jax.jit(instrumented).lower(*inputs).compile()
    assert hits == set(STAGES[1:]), hits
    baseline, support, booked = jax.block_until_ready(program(*inputs))
    record = dict(
        revision=args.revision,
        arm=arm,
        lanes=[0, 10] if args.cpu else list(range(16)),
        state=[fingerprint(v) for v in state],
        hooks=sorted(hits),
        plain_norm=np.max(np.abs(reference), axis=-1).tolist(),
        baseline_norm=np.max(np.abs(baseline), axis=-1).tolist(),
        instrumentation_delta=np.max(np.abs(baseline - reference), axis=-1).tolist(),
        perturbations=[],
        stages={},
        masks=[signatures(m) for m in masks(support.reading)],
    )
    output = args.run / f"{arm}.json"
    arrays = {"baseline": np.asarray(baseline), "plain": np.asarray(reference)}

    def persist():
        output.write_text(json.dumps(record, indent=2))
        np.savez_compressed(args.run / f"{arm}-residuals.npz", **arrays)

    persist()
    original_masks = [np.asarray(m) for m in masks(support.reading)]
    for index in range(64):
        shifted = jax.tree.map(lambda x: nudged(x, index + 1, 1 + index % 4), state)
        residual, fresh, _ = jax.block_until_ready(
            program(bank, cache, shifted, seeds, previous, 0, 0)
        )
        assert np.all(fresh.valid)
        changed = [
            int(np.count_nonzero(np.asarray(a) != b))
            for a, b in zip(masks(fresh.reading), original_masks)
        ]
        record["perturbations"].append(
            dict(
                norm=np.max(np.abs(residual), axis=-1).tolist(),
                delta=np.max(np.abs(residual - baseline), axis=-1).tolist(),
                mask_changes=changed,
            )
        )
        arrays[f"perturbation_{index}"] = np.asarray(residual)
    persist()
    emit(
        "PERTURBATIONS",
        arm=arm,
        baseline=record["baseline_norm"],
        median=np.median([v["norm"] for v in record["perturbations"]], axis=0).tolist(),
    )
    for stage, name in enumerate(STAGES[1:], 1):
        rows = []
        for seed in range(1, 65):
            residual, fresh, _ = jax.block_until_ready(
                program(bank, cache, state, seeds, previous, stage, seed)
            )
            assert np.all(fresh.valid) and np.all(np.isfinite(residual))
            rows.append(np.asarray(residual))
        values = np.stack(rows)
        delta = np.max(np.abs(values - baseline), axis=-1)
        record["stages"][name] = dict(
            median=np.median(delta, axis=0).tolist(),
            maximum=delta.max(axis=0).tolist(),
            minimum=delta.min(axis=0).tolist(),
            norm_median=np.median(np.max(np.abs(values), axis=-1), axis=0).tolist(),
        )
        arrays[name] = values
        persist()
        emit("STAGE", name=name, **record["stages"][name])
    # The map from booked moments to residual must detect a known one-ulp change.
    assert max(record["stages"]["whole_cells"]["maximum"]) > 0
    assert max(record["stages"]["conic_cells"]["maximum"]) > 0
    assert max(record["stages"]["normalization"]["maximum"]) > 0
    zeros = jax.tree.map(jnp.zeros_like, tuple(v[0] for v in state))
    jets, field = sr._directional_field(bank, zeros)
    assert np.count_nonzero(np.asarray(jets)) == 0
    assert np.count_nonzero(np.asarray(field.sampled_coefficient(bank.geometry))) == 0
    record["zero_direction_exact"] = True
    record["span"] = np.asarray(
        support.reading.boundary_flux - support.reading.axis_flux
    ).tolist()
    record["moment_max"] = [float(np.max(np.abs(v))) for v in state]
    record["normalization"] = np.asarray(booked.normalization).tolist()
    persist()
    emit("COMPLETE", arm=arm, output=str(output), positive_controls=True)


if __name__ == "__main__":
    main()
