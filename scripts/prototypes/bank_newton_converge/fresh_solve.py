"""Solve the published bank equation with a newly read support at each step."""

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
from nova.equilibrium import support_read, topology
from scripts.prototypes.bank_newton_step.measure import machine


def emit(name, **values):
    print(name + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def magnitude(result):
    return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))


def run(state_path, max_steps):
    (
        operands,
        cache,
        roots,
        convention,
        policy,
        profile,
        quadrature,
        moments,
        analytic,
    ) = machine("diverted")
    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)

    @jax.jit
    def read(bank, window, state):
        return support_read.read(bank, window, state, seeds, roots, convention, policy)

    @jax.jit
    def residual(bank, base, candidate, support):
        return support_read.step_residual(
            bank, base, candidate, support, profile, quadrature
        )

    @jax.jit
    def step(bank, state, support):
        return support_read.newton_step(
            bank,
            state,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=True,
        )

    warm_walls = []
    for index in range(max_steps):
        support = read(operands, cache, moments)
        current = residual(operands, moments, moments, support)
        before = magnitude(current)
        emit("STEP", index=index, fresh_residual=before, valid=bool(support.valid))
        if before <= 1e-10 and bool(support.valid):
            break
        started = perf_counter()
        correction = step(operands, moments, support)
        jax.block_until_ready(correction)
        wall_ms = 1e3 * (perf_counter() - started)
        warm_walls.append(wall_ms)
        emit(
            "LINEAR",
            index=index,
            wall_ms=wall_ms,
            linear_residual=float(correction.linear_residual),
            converged=bool(correction.converged),
        )
        if index == 0:
            predicted = support_read.step_jacobian_vector(
                operands,
                moments,
                correction.update,
                support,
                profile,
                lambda radius, flux: jnp.zeros_like(radius),
                quadrature,
                current.booked,
                fresh_geometry=True,
            )
            remainder = tuple(
                a + b for a, b in zip(current.residual, predicted, strict=True)
            )
            norm = jnp.linalg.norm(jnp.concatenate(current.residual))
            emit(
                "DENSE_DIRECTION",
                ratio=float(jnp.linalg.norm(jnp.concatenate(remainder)) / norm),
                update_sup=float(jnp.max(jnp.abs(jnp.concatenate(correction.update)))),
            )
            best = float("inf")
            for length in (1e-4, 1e-5, 1e-6, 1e-7):
                high = tuple(
                    value + length * change
                    for value, change in zip(moments, correction.update, strict=True)
                )
                low = tuple(
                    value - length * change
                    for value, change in zip(moments, correction.update, strict=True)
                )
                high_support = read(operands, cache, high)
                low_support = read(operands, cache, low)
                high_result = residual(operands, high, high, high_support)
                low_result = residual(operands, low, low, low_support)
                central = tuple(
                    (a - b) / (2 * length)
                    for a, b in zip(
                        high_result.residual, low_result.residual, strict=True
                    )
                )
                remainder = tuple(
                    a + b for a, b in zip(current.residual, central, strict=True)
                )
                ratio = float(jnp.linalg.norm(jnp.concatenate(remainder)) / norm)
                emit(
                    "FRESH_LINEARIZATION",
                    length=length,
                    ratio=ratio,
                    high_valid=bool(high_support.valid),
                    low_valid=bool(low_support.valid),
                )
                best = min(best, ratio)
            assert best < 1e-3
        accepted = False
        for half in range(13):
            length = 2.0**-half
            candidate = tuple(
                value + length * change
                for value, change in zip(moments, correction.update, strict=True)
            )
            tracked = residual(operands, moments, candidate, support)
            next_support = read(operands, cache, candidate)
            refreshed = residual(operands, candidate, candidate, next_support)
            fresh_norm = magnitude(refreshed)
            if (
                bool(next_support.valid)
                and np.isfinite(fresh_norm)
                and fresh_norm < before
            ):
                emit(
                    "ACCEPT",
                    index=index,
                    length=length,
                    tracked_residual=magnitude(tracked),
                    fresh_residual=fresh_norm,
                )
                moments = candidate
                accepted = True
                break
        if not accepted:
            emit("NO_DESCENT", index=index, residual=before)
            break

    support = read(operands, cache, moments)
    result = residual(operands, moments, moments, support)
    terminal = magnitude(result)
    converged = bool(support.valid) and terminal <= 1e-10
    emit(
        "FINAL",
        steps=index,
        fresh_residual=terminal,
        converged=converged,
        warm_step_ms=float(np.median(warm_walls[1:])) if len(warm_walls) > 1 else None,
    )
    np.savez_compressed(
        state_path,
        moments=np.stack(tuple(np.asarray(value) for value in moments)),
        axis=np.asarray(support.reading.axis),
        boundary=np.asarray(support.reading.boundary),
        axis_flux=np.asarray(support.reading.axis_flux),
        boundary_flux=np.asarray(support.reading.boundary_flux),
        converged=np.asarray(converged),
        residual_sup=np.asarray(terminal),
    )
    if not converged:
        raise AssertionError(f"fresh residual {terminal} exceeds 1e-10")

    reference = jax.jit(topology.read)(analytic, operands.geometry, convention, policy)
    from tests.equilibrium import test_topology_read as fixture

    _, _, wall, _, analytic_saddle = fixture._analytic_inputs("diverted")
    radius = 0.5 * (
        float(jnp.max(jnp.asarray(wall)[:, 0]))
        - float(jnp.min(jnp.asarray(wall)[:, 0]))
    )
    axis_error = float(jnp.linalg.norm(support.reading.axis - reference.axis))
    saddle_error = float(
        jnp.linalg.norm(support.reading.boundary - jnp.asarray(analytic_saddle))
    )
    span = float(jnp.abs(reference.boundary_flux - reference.axis_flux))
    boundary_error = float(
        jnp.abs(support.reading.boundary_flux - reference.boundary_flux)
    )
    emit(
        "ORACLE",
        axis_error=axis_error,
        minor_radius=radius,
        axis_ratio=axis_error / radius,
        saddle_error=saddle_error,
        saddle_ratio=saddle_error / radius,
        boundary_flux_error=boundary_error,
        flux_span=span,
        boundary_ratio=boundary_error / span,
    )
    active = int(operands.active_nodes)
    nodes = operands.nodes[:active]
    np.savez_compressed(
        state_path,
        moments=np.stack(tuple(np.asarray(value) for value in moments)),
        coordinates=np.asarray(nodes),
        solved_flux=np.asarray(support_read.nodal_jets(operands, moments)[:active, 0]),
        analytic_flux=np.asarray(jax.vmap(analytic.value)(nodes)),
        wall=np.asarray(wall),
        solved_roots=np.asarray(
            jnp.stack((support.reading.axis, support.reading.boundary))
        ),
        analytic_roots=np.asarray(jnp.stack((reference.axis, reference.boundary))),
        converged=np.asarray(converged),
        residual_sup=np.asarray(terminal),
        axis_ratio=np.asarray(axis_error / radius),
        saddle_ratio=np.asarray(saddle_error / radius),
        boundary_ratio=np.asarray(boundary_error / span),
    )
    assert axis_error / radius <= 5e-4
    assert saddle_error / radius <= 5e-4
    assert boundary_error / span <= 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
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
    assert jax.config.jax_enable_x64
    run(args.state, args.max_steps)
