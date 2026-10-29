"""Refresh the connected support between tracked bank Newton solves."""

# ruff: noqa: E402

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def norm(residual):
    return float(jnp.max(jnp.abs(jnp.concatenate(residual.residual))))


def run(max_outer, output):
    operands, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def fresh(operands, cache, current):
        return support_read.read(
            operands, cache, current, seeds, roots, convention, policy
        )

    @jax.jit
    def residual(operands, base, current, support):
        return support_read.step_residual(
            operands, base, current, support, profile, quadrature
        )

    @jax.jit
    def step(operands, current, support):
        return support_read.newton_step(
            operands,
            current,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
        )

    current = moments
    warm_walls = []
    for outer in range(max_outer):
        support = fresh(operands, cache, current)
        initial = residual(operands, current, current, support)
        before = norm(initial)
        emit("OUTER", index=outer, fresh_residual=before, valid=bool(support.valid))
        if before <= 1e-10:
            break
        start_moments = current
        for inner in range(3):
            tracked = residual(operands, current, current, support)
            tracked_norm = norm(tracked)
            if tracked_norm <= 1e-11:
                break
            started = perf_counter()
            update = step(operands, current, support)
            jax.block_until_ready(update)
            wall_ms = 1e3 * (perf_counter() - started)
            warm_walls.append(wall_ms)
            if not bool(update.converged):
                raise AssertionError("bank linear update refused")
            candidate = tuple(
                value + delta
                for value, delta in zip(current, update.update, strict=True)
            )
            candidate_read = residual(operands, current, candidate, support)
            if not bool(candidate_read.valid):
                raise AssertionError("tracked support became invalid")
            emit(
                "INNER",
                outer=outer,
                index=inner,
                before=tracked_norm,
                after=norm(candidate_read),
                wall_ms=wall_ms,
            )
            current = candidate
            support = support._replace(reading=candidate_read.reading)
        tracked_norm = norm(residual(operands, current, current, support))
        accepted = False
        for half in range(12):
            length = 2.0**-half
            candidate = tuple(
                start + length * (end - start)
                for start, end in zip(start_moments, current, strict=True)
            )
            refreshed = fresh(operands, cache, candidate)
            fresh_result = residual(operands, candidate, candidate, refreshed)
            trial_norm = norm(fresh_result)
            if (
                bool(refreshed.valid)
                and np.isfinite(trial_norm)
                and trial_norm < before
            ):
                current = candidate
                accepted = True
                emit(
                    "REFRESH",
                    index=outer,
                    residual=trial_norm,
                    valid=True,
                    tracked=tracked_norm,
                    length=length,
                )
                break
        if not accepted:
            current = start_moments
            emit("NO_DESCENT", index=outer, residual=before, tracked=tracked_norm)
            break
    final_support = fresh(operands, cache, current)
    final = residual(operands, current, current, final_support)
    final_norm = norm(final)
    emit(
        "FINAL",
        outer_count=outer + 1,
        fresh_residual=final_norm,
        converged=bool(final_support.valid) and final_norm <= 1e-10,
        warm_step_ms=float(np.median(warm_walls[1:])) if len(warm_walls) > 1 else None,
    )
    if output:
        np.savez_compressed(
            output,
            moments=np.stack(tuple(np.asarray(x) for x in current)),
            converged=final_norm <= 1e-10,
            residual_sup=final_norm,
        )
    if final_norm > 1e-10:
        raise AssertionError(f"fresh residual {final_norm} exceeds 1e-10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-outer", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(root),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
    )
    run(args.max_outer, args.output)
