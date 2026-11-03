"""Measure staged bank solves and the existing per-step prototype on one device."""

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
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from scripts.prototypes.bank_newton_step.measure import machine
from tests.equilibrium import test_topology_read as fixture


ROOT = Path(__file__).resolve().parents[3]


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def magnitude(result):
    return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))


def prototype(
    operands, cache, roots, convention, policy, profile, quadrature, seed, seeds
):
    """The prototype's fresh-read, fresh-factor, backtracked host loop."""

    @jax.jit
    def read(state):
        return support_read.read(
            operands, cache, state, seeds, roots, convention, policy
        )

    @jax.jit
    def residual(state, support):
        return support_read.step_residual(
            operands, state, state, support, profile, quadrature
        )

    @jax.jit
    def step(state, support):
        return support_read.newton_step(
            operands,
            state,
            support,
            profile,
            lambda radius, flux: jnp.zeros_like(radius),
            quadrature,
            fresh_geometry=True,
        )

    support = read(seed)
    jax.block_until_ready(support)
    jax.block_until_ready(residual(seed, support))
    jax.block_until_ready(step(seed, support))
    started = perf_counter()
    state = seed
    reads = 0
    for index in range(60):
        support = read(state)
        reads += 1
        before = magnitude(residual(state, support))
        if bool(support.valid) and before <= 1e-10:
            break
        correction = step(state, support)
        accepted = False
        for half in range(13):
            length = 2.0**-half
            candidate = tuple(
                value + length * change
                for value, change in zip(state, correction.update, strict=True)
            )
            fresh = read(candidate)
            reads += 1
            after = magnitude(residual(candidate, fresh))
            if bool(fresh.valid) and np.isfinite(after) and after < before:
                state = candidate
                accepted = True
                break
        if not accepted:
            break
    wall = 1e3 * (perf_counter() - started)
    emit(
        "PROTOTYPE",
        wall_ms=wall,
        steps=index,
        refactors=index,
        reads=reads,
        fresh_residual=before,
        converged=bool(support.valid) and before <= 1e-10,
    )
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", choices=("none", "half", "tenth"), required=True)
    args = parser.parse_args()
    reuse = args.reuse != "none"
    contraction_limit = 0.1 if args.reuse == "tenth" else 0.5
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    emit(
        "HEADER",
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        reuse=args.reuse,
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    @jax.jit
    def single(state):
        return support_read.solve(
            bank,
            cache,
            state,
            seeds,
            roots,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=reuse,
            contraction_limit=contraction_limit,
        )

    @jax.jit
    def batch(state):
        return support_read.solve_batch(
            bank,
            cache,
            state,
            jnp.broadcast_to(seeds, (16, *seeds.shape)),
            jnp.broadcast_to(roots, (16, *roots.shape)),
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=reuse,
            contraction_limit=contraction_limit,
        )

    single(moments)
    jax.block_until_ready(single(moments))
    started = perf_counter()
    one = single(moments)
    jax.block_until_ready(one)
    single_wall = 1e3 * (perf_counter() - started)
    emit(
        "SINGLE",
        wall_ms=single_wall,
        steps=int(one.steps),
        refactors=int(one.refactors),
        reads=int(one.reads),
        freezes=int(one.freezes),
        unfreezes=int(one.unfreezes),
        fresh_residual=float(one.residual),
        converged=bool(one.converged),
    )
    assert bool(one.converged) and float(one.residual) <= 1e-10

    _, _, wall, _, _ = fixture._analytic_inputs("diverted")
    oracle, reference_roots = analytic_oracle(
        analytic, roots, one.support.reading, wall
    )
    emit("ORACLE", **oracle)
    assert oracle["axis_ratio"] <= 5e-4
    assert oracle["saddle_ratio"] <= 5e-4
    assert oracle["boundary_ratio"] <= 1e-4

    prototype(
        bank, cache, roots, convention, policy, profile, quadrature, moments, seeds
    )
    scales = jnp.linspace(2e-5, 6e-5, 16)
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    batch(states)
    jax.block_until_ready(batch(states))
    started = perf_counter()
    many = batch(states)
    jax.block_until_ready(many)
    batch_wall = 1e3 * (perf_counter() - started)
    emit(
        "BATCH",
        wall_ms=batch_wall,
        per_slice_ms=batch_wall / 16,
        steps=np.asarray(many.steps).tolist(),
        refactors=np.asarray(many.refactors).tolist(),
        reads=np.asarray(many.reads).tolist(),
        freezes=np.asarray(many.freezes).tolist(),
        unfreezes=np.asarray(many.unfreezes).tolist(),
        fresh_residual=np.asarray(many.residual).tolist(),
        converged=np.asarray(many.converged).tolist(),
    )
    assert bool(jnp.all(many.converged))
    span = abs(float(one.support.reading.boundary_flux - one.support.reading.axis_flux))
    radius = oracle["minor_radius"]
    reference_boundary_flux = float(analytic.value(jnp.asarray(reference_roots[1])))
    for lane in range(16):
        own = single(tuple(value[lane] for value in states))
        jax.block_until_ready(own)
        assert bool(own.converged)
        flux = support_read.nodal_jets(bank, own.moments)[:, 0]
        batched_flux = support_read.nodal_jets(
            bank, tuple(value[lane] for value in many.moments)
        )[:, 0]
        flux_gap = float(jnp.max(jnp.abs(flux - batched_flux))) / span
        axis_gap = (
            float(
                jnp.linalg.norm(
                    own.support.reading.axis - many.support.reading.axis[lane]
                )
            )
            / radius
        )
        saddle_gap = (
            float(
                jnp.linalg.norm(
                    own.support.reading.boundary - many.support.reading.boundary[lane]
                )
            )
            / radius
        )
        boundary_gap = (
            abs(
                float(
                    own.support.reading.boundary_flux
                    - many.support.reading.boundary_flux[lane]
                )
            )
            / span
        )
        oracle_axis = (
            float(jnp.linalg.norm(many.support.reading.axis[lane] - reference_roots[0]))
            / radius
        )
        oracle_saddle = (
            float(
                jnp.linalg.norm(
                    many.support.reading.boundary[lane] - reference_roots[1]
                )
            )
            / radius
        )
        oracle_boundary = (
            abs(
                float(many.support.reading.boundary_flux[lane])
                - reference_boundary_flux
            )
            / oracle["flux_span"]
        )
        emit(
            "LANE",
            lane=lane,
            flux_ratio=flux_gap,
            axis_ratio=axis_gap,
            saddle_ratio=saddle_gap,
            boundary_ratio=boundary_gap,
            oracle_axis_ratio=oracle_axis,
            oracle_saddle_ratio=oracle_saddle,
            oracle_boundary_ratio=oracle_boundary,
            single_residual=float(own.residual),
            batch_residual=float(many.residual[lane]),
        )
        assert (
            flux_gap <= 1e-8
            and axis_gap <= 5e-4
            and saddle_gap <= 5e-4
            and boundary_gap <= 1e-4
            and oracle_axis <= 5e-4
            and oracle_saddle <= 5e-4
            and oracle_boundary <= 1e-4
        )


if __name__ == "__main__":
    main()
