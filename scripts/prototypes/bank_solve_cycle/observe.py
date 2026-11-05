"""Observe the actual staged solve loop without reproducing its arithmetic."""

import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


def observe_updates(monkeypatch):
    """Print each completed update, including fresh reads spent on halving."""
    original = jax.lax.while_loop
    started = perf_counter()

    def report(limit, steps, residual, releases, trials, active):
        active = np.asarray(active, dtype=int)
        trials = np.asarray(trials, dtype=int)
        print(
            "UPDATE",
            json.dumps(
                {
                    "limit": limit,
                    "elapsed_seconds": perf_counter() - started,
                    "steps": np.asarray(steps).tolist(),
                    "fresh_residual": np.asarray(residual).tolist(),
                    "releases": np.asarray(releases).tolist(),
                    "trials": trials.tolist(),
                    "backtracks": (trials - active).tolist(),
                }
            ),
            flush=True,
        )

    def observed_loop(condition, body, initial):
        if body.__qualname__ != "solve_batch.<locals>.advance":
            return original(condition, body, initial)
        limit = initial[17].shape[1]

        def observed_advance(carried):
            following = body(carried)
            jax.debug.callback(
                lambda *values: report(limit, *values),
                following[11],
                following[2],
                following[15] - carried[15],
                following[13] - carried[13],
                (~carried[3]).astype(jnp.int32),
                ordered=True,
            )
            return following

        return original(condition, observed_advance, initial)

    monkeypatch.setattr(jax.lax, "while_loop", observed_loop)
