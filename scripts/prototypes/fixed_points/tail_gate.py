"""Replay the full-moment diverted workload with interleaved latency controls."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import fields
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points as fixed
from scripts.prototypes.fixed_points import measure


_compile = measure.compile_solver
_latency = measure.latency
_adaptive = set()


def compile_solver(field, seeds, pitch, adaptive=False):
    result = _compile(field, seeds, pitch, adaptive)
    if adaptive:
        _adaptive.add(id(result[0]))
    return result


def latency(executable, operands, sweep, pitch, calls):
    if id(executable) not in _adaptive:
        return _latency(executable, operands, sweep, pitch, calls)
    caches = [
        fixed.RowCache(
            fixed.BellMesh(
                *(getattr(field.mesh, item.name) for item in fields(fixed.BellMesh))
            ),
            jnp.zeros((len(field.lookup), 2)),
            field.lookup,
            field.rows,
            field.exterior,
        )
        for field in operands
    ]
    seeds = [jnp.asarray(row["warm_seed"]) for row in sweep]
    sign = jnp.asarray(-1.0)
    compiled = fixed.solve.lower(
        caches[0], operands[0].moments, seeds[0], seeds[0], pitch, sign
    ).compile()
    prepared = [
        (cache, field.moments, seed, seed, pitch, sign)
        for cache, field, seed in zip(caches, operands, seeds)
    ]
    jax.block_until_ready(prepared)
    for index, args in enumerate(prepared):
        result = compiled(*args)
        jax.block_until_ready(result)
        assert np.all(np.asarray(result.valid)), (index, result)
        np.testing.assert_allclose(
            result.position, sweep[index]["positions"], atol=2e-9, rtol=0
        )
    times, production = [], []
    for index in range(max(calls, 1000)):
        state = index % len(operands)
        order = (
            ("prototype", "production") if index % 2 else ("production", "prototype")
        )
        for arm in order:
            start = perf_counter()
            result = (
                compiled(*prepared[state])
                if arm == "production"
                else executable(operands[state], seeds[state], pitch)
            )
            jax.block_until_ready(result)
            (production if arm == "production" else times).append(
                perf_counter() - start
            )
    result = dict(
        calls=len(times),
        median_seconds=float(np.median(times)),
        p99_seconds=float(np.percentile(times, 99)),
        seconds=times,
        production_median_seconds=float(np.median(production)),
        production_p99_seconds=float(np.percentile(production, 99)),
        production_seconds=production,
    )
    assert result["production_p99_seconds"] < 0.001, result["production_p99_seconds"]
    return result


if __name__ == "__main__":
    measure.compile_solver = compile_solver
    measure.latency = latency
    measure.main()
