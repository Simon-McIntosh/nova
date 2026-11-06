"""Measure fresh solve repeats and retain wall outputs at their actual read inputs."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.bank_solve_profile import fixture
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from tests.equilibrium.test_topology_read import _analytic_inputs


def emit(label, **row):
    print(label, json.dumps(row), flush=True)


def counters(result):
    return {
        key: np.asarray(getattr(result, key)).tolist()
        for key in ("steps", "reads", "refactors", "residual", "converged")
    }


def fingerprints(tree):
    result = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(value)
        result[jax.tree_util.keystr(path)] = dict(
            shape=list(array.shape),
            dtype=str(array.dtype),
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        )
    return result


def difference(left, right):
    finite = np.isfinite(left) & np.isfinite(right)
    equal = (left == right) | (np.isnan(left) & np.isnan(right))
    mismatch = ~finite & ~equal
    delta = np.abs(np.subtract(left, right, out=np.zeros_like(left), where=finite))
    axes = tuple(range(1, delta.ndim))
    return dict(
        absolute=delta.max(axis=axes).tolist(),
        relative=(
            delta.max(axis=axes)
            / np.maximum(
                np.where(np.isfinite(right), np.abs(right), 0).max(axis=axes),
                np.finfo(float).tiny,
            )
        ).tolist(),
        nonfinite_mismatch=mismatch.sum(axis=axes).tolist(),
        bit_differences=(left.view(np.uint64) != right.view(np.uint64))
        .sum(axis=axes)
        .tolist(),
    )


def callback_control():
    """Require the CPU callback device while keeping computation on the GPU."""
    assert jax.default_backend() == "gpu"
    positive = []
    jax.jit(
        jax.vmap(
            lambda lane: jax.debug.callback(
                lambda x: positive.append(int(x)), lane, ordered=True
            )
        )
    )(jnp.arange(16))
    jax.effects_barrier()
    assert positive == list(range(16)), positive
    emit("LANE_ORDER_CONTROL", observed=positive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    args = parser.parse_args()
    emit(
        "HEADER",
        revision=args.revision,
        nova=nova.__file__,
        module=topology.__file__,
        cwd=str(Path.cwd().resolve()),
        device=jax.devices()[0].device_kind,
        x64=bool(jax.config.jax_enable_x64),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    batch = fixture.build(16)
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

    record = dict(
        arm=args.arm,
        repeat=args.repeat,
        revision=args.revision,
        calls=[],
        inputs=fingerprints(inputs),
    )

    def persist():
        args.output.with_suffix(".json").write_text(json.dumps(record, indent=2))

    emit("COMPILE", mode="unaltered")
    program = jax.jit(solve).lower(*inputs).compile()
    jax.block_until_ready(program(*inputs))
    for call in range(2):
        start = time.perf_counter()
        result = jax.block_until_ready(program(*inputs))
        wall = 1000 * (time.perf_counter() - start)
        row = dict(call=call, wall_ms=wall, **counters(result))
        row["per_read_ms"] = (wall / np.asarray(result.reads)).tolist()
        row["state"] = fingerprints(result.moments)
        row["residual_path"] = np.asarray(result.residual_path).tolist()
        assert all(row["converged"]) and max(row["residual"]) <= 1e-10
        record["calls"].append(row)
        emit("UNALTERED", **row)
        persist()
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

    original = topology._wall_events
    source = inspect.getsource(original)
    sequential = "jax.lax.map(lambda pair: one(*pair), (start, edge))"
    batched = "jax.vmap(one)(start, edge)"
    assert (sequential in source) != (batched in source)
    other_source = (
        source.replace(sequential, batched)
        if sequential in source
        else source.replace(batched, sequential)
    )
    namespace = dict(vars(topology))
    exec(compile(other_source, "<paired-wall-polish>", "exec"), namespace)
    alternative = namespace["_wall_events"]
    observed = []
    callback_control()

    def capture(coefficients, points, values):
        observed.append(
            tuple(np.asarray(x).copy() for x in (coefficients, points, values))
        )

    def observed_wall(field, geometry, policy, sigma):
        points, values = original(field, geometry, policy, sigma)
        jax.debug.callback(capture, field.coefficients, points, values, ordered=True)
        return points, values

    topology._wall_events = observed_wall
    emit("COMPILE", mode="observed")
    traced = jax.jit(lambda *values: solve(*values)).lower(*inputs).compile()
    observed_result = jax.block_until_ready(traced(*inputs))
    jax.effects_barrier()
    topology._wall_events = original
    assert observed and len(observed) % 16 == 0
    count = len(observed) // 16
    assert np.all(np.asarray(observed_result.reads) == count), (
        count,
        counters(observed_result),
    )
    coefficient, points, values = (
        np.stack([row[index] for row in observed]).reshape(
            count, 16, *observed[0][index].shape
        )
        for index in range(3)
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        coefficients=coefficient,
        points=points,
        values=values,
    )
    record["observed"] = counters(observed_result)
    record["observed"]["wall_calls"] = count
    record["observed"]["matches_unaltered_counters"] = all(
        record["observed"][key] == record["calls"][-1][key]
        for key in ("steps", "reads")
    )
    emit("OBSERVED", **record["observed"])
    persist()

    dummy_jets = jnp.zeros_like(batch.operands.exterior)
    template = fixed_points.BellField(
        jnp.asarray(coefficient[0, 0]),
        batch.operands.mesh,
        dummy_jets,
        batch.operands.sample_index,
    )

    def wall_batch(function):
        return jax.jit(
            lambda coefficients, mesh, geometry: jax.vmap(
                lambda c: function(
                    replace(template, coefficients=c, mesh=mesh),
                    geometry,
                    batch.policy,
                    batch.convention.sigma,
                )
            )(coefficients)
        )

    own_program, other_program = wall_batch(original), wall_batch(alternative)
    pairs = []
    for index, coefficients in enumerate(coefficient):
        own = tuple(
            np.asarray(x)
            for x in jax.block_until_ready(
                own_program(
                    jnp.asarray(coefficients),
                    batch.operands.mesh,
                    batch.operands.geometry,
                )
            )
        )
        other = tuple(
            np.asarray(x)
            for x in jax.block_until_ready(
                other_program(
                    jnp.asarray(coefficients),
                    batch.operands.mesh,
                    batch.operands.geometry,
                )
            )
        )
        row = dict(
            read=index,
            same_input_points=difference(own[0], other[0]),
            same_input_values=difference(own[1], other[1]),
            own_replay_points=difference(points[index], own[0]),
            own_replay_values=difference(values[index], own[1]),
        )
        pairs.append(row)
    record["wall_comparisons"] = pairs
    persist()
    emit("TRAJECTORY_COMPLETE", output=str(args.output), reads=count)


if __name__ == "__main__":
    main()
