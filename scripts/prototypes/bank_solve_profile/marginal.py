"""Time the converged solve at a fixed step count to read off its marginal step.

The stage split sums to about twice the whole solve's per-iterate wall. This
receipt times the same seed with the loop's step ceiling pinned below and at
convergence, so the difference between two arms is the cost of the steps
between them, with no division by a step count.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from statistics import median
from time import perf_counter

import jax
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile import fixture


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    emit(
        "HEADER",
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        nova=nova.__file__,
        repeats=args.repeats,
    )
    batch = fixture.build(1, perturb=False)
    operands = batch.operands
    cache = batch.cache
    states = batch.states
    seeds_batch = batch.seeds_batch
    previous_batch = batch.previous_batch
    convention = batch.convention
    policy = batch.policy
    profile = batch.profile
    quadrature = batch.quadrature
    gradient = fixture.gradient

    for ceiling in (1, 2, 4, 6):
        solve = jax.jit(
            lambda state, ceiling=ceiling: support_read.solve_batch(
                operands,
                cache,
                state,
                seeds_batch,
                previous_batch,
                convention,
                policy,
                profile,
                gradient,
                quadrature,
                max_steps=ceiling,
                reuse_factor=False,
                contraction_limit=0.5,
            )
        )
        result = solve(states)
        jax.block_until_ready(result)
        samples = []
        for _ in range(args.repeats):
            started = perf_counter()
            result = solve(states)
            jax.block_until_ready(result)
            samples.append(1e3 * (perf_counter() - started))
        emit(
            "MARGINAL",
            lanes=1,
            ceiling=ceiling,
            median_ms=median(samples),
            min_ms=min(samples),
            steps=int(np.max(np.asarray(result.steps))),
            converged=bool(np.all(np.asarray(result.converged))),
        )


if __name__ == "__main__":
    main()
