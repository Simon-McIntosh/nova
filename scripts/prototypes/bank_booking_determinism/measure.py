"""Fingerprint eager booking stages and measure the unchanged batched solve."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

import nova
from nova.equilibrium import current, support_read
from scripts.prototypes.bank_solve_profile import fixture
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from tests.equilibrium.test_topology_read import _analytic_inputs


def emit(label, **row):
    print(label, json.dumps(row), flush=True)


def fingerprint(value):
    array = np.asarray(value)
    return dict(
        shape=list(array.shape),
        dtype=str(array.dtype),
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--booking-only", action="store_true")
    args = parser.parse_args()
    emit(
        "HEADER",
        revision=args.revision,
        nova=nova.__file__,
        module=current.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    arrays, stages, seen, arguments = {}, [], {}, {}
    names = (
        "psi",
        "read",
        "geometry",
        "coefficients",
        "normalized",
        "signed",
        "full",
        "cut",
        "whole",
        "cut_count",
        "cut_index",
        "cuts",
        "moments",
        "raw",
        "normalization",
    )

    def trace(frame, event, arg):
        if frame.f_code != current.book.__code__:
            return None
        for name in names:
            value = frame.f_locals.get(name)
            if value is None or seen.get(name) is value:
                continue
            leaves = jax.tree_util.tree_flatten_with_path(value)[0]
            if any(isinstance(v, jax.core.Tracer) for _, v in leaves):
                continue
            seen[name] = value
            for path, leaf in leaves:
                key = f"{len(stages):03d}:{name}{jax.tree_util.keystr(path)}"
                array = np.asarray(leaf)
                arrays[key] = array.copy()
                stages.append(dict(key=key, line=frame.f_lineno, **fingerprint(array)))
        if event == "return":
            for name in ("psi", "read", "geometry", "profile", "quadrature"):
                arguments[name] = frame.f_locals[name]
        return trace

    sys.settrace(trace)
    batch = fixture.build(16)
    sys.settrace(None)
    initial = np.stack([np.asarray(v) for v in batch.moments])
    arrays["initial_moments"] = initial
    record = dict(
        revision=args.revision,
        module=current.__file__,
        cwd=str(Path.cwd().resolve()),
        stages=stages,
        initial=fingerprint(initial),
        calls=[],
    )

    def persist():
        args.output.with_suffix(".json").write_text(json.dumps(record, indent=2))
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)

    persist()
    emit("BOOKING", fingerprint=record["initial"], stages=len(stages))
    if args.booking_only:
        return
    inputs = (
        batch.operands,
        batch.cache,
        batch.states,
        batch.seeds_batch,
        batch.previous_batch,
    )

    def solve(bank, cache, state, seeds, previous):
        return support_read.solve_batch(
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
            contraction_limit=0.5,
        )

    start = time.perf_counter()
    program = jax.jit(solve).lower(*inputs).compile()
    emit("COMPILE", seconds=time.perf_counter() - start)
    jax.block_until_ready(program(*inputs))
    for call in range(2):
        start = time.perf_counter()
        result = jax.block_until_ready(program(*inputs))
        row = dict(call=call, wall_ms=1000 * (time.perf_counter() - start))
        row.update(
            {
                key: np.asarray(getattr(result, key)).tolist()
                for key in ("steps", "reads", "refactors", "residual", "converged")
            }
        )
        record["calls"].append(row)
        emit("SOLVE", **row)
        persist()
        assert all(row["converged"]) and max(row["residual"]) <= 1e-10
    _, analytic, wall, _, _ = _analytic_inputs("diverted")
    reading = result.support.reading
    oracle, roots = analytic_oracle(
        analytic, batch.roots, jax.tree.map(lambda x: x[0], reading), wall
    )
    flux = jax.vmap(lambda state: support_read.nodal_jets(batch.operands, state)[:, 0])(
        result.moments
    )
    record["precision"] = dict(
        axis=np.asarray(reading.axis).tolist(),
        saddle=np.asarray(reading.boundary).tolist(),
        boundary_flux=np.asarray(reading.boundary_flux).tolist(),
        flux=np.asarray(flux).tolist(),
        span=oracle["flux_span"],
        minor_radius=oracle["minor_radius"],
        oracle=oracle,
        oracle_roots=np.asarray(roots).tolist(),
    )
    persist()
    emit("COMPLETE", output=str(args.output))


if __name__ == "__main__":
    main()
