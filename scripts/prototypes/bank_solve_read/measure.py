"""Measure one staged bank solve against the analytic oracle on one device.

The same script runs in a base tree and in the head tree: every field the head
adds to ``SolveResult`` is read with ``getattr``, so a base arm reports the
counters it has and ``null`` for the ones it does not.
"""

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
    print(label + " " + json.dumps(values, sort_keys=True, default=_default), flush=True)


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    return float(value)


def counter(result, name):
    value = getattr(result, name, None)
    return None if value is None else float(np.asarray(value).ravel()[0])


def run_gate(state, seeds, roots, bank, cache, convention, policy, profile, gradient, quadrature):
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
        reuse_factor=False,
        contraction_limit=0.5,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = "unversioned-archive"
    emit(
        "HEADER",
        arm=args.arm,
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        machine("diverted")
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    lanes = 16
    scales = jnp.linspace(2e-5, 6e-5, lanes)
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds_batch = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
    roots_batch = jnp.broadcast_to(roots, (lanes, *roots.shape))

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    @jax.jit
    def single(state):
        return run_gate(
            state, seeds, roots, bank, cache, convention, policy, profile,
            gradient, quadrature,
        )

    @jax.jit
    def batch(state):
        return support_read.solve_batch(
            bank,
            cache,
            state,
            seeds_batch,
            roots_batch,
            convention,
            policy,
            profile,
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=False,
            contraction_limit=0.5,
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
        steps=counter(one, "steps"),
        refactors=counter(one, "refactors"),
        reads=counter(one, "reads"),
        tracked_reads=counter(one, "tracked_reads"),
        rebuild_reads=None
        if counter(one, "tracked_reads") is None
        else counter(one, "reads") - counter(one, "tracked_reads"),
        certificates=counter(one, "certificates"),
        freezes=counter(one, "freezes"),
        unfreezes=counter(one, "unfreezes"),
        fresh_residual=float(one.residual),
        converged=bool(one.converged),
    )
    assert bool(one.converged) and float(one.residual) <= 1e-10

    _, _, wall, _, _ = fixture._analytic_inputs("diverted")
    oracle, reference_roots = analytic_oracle(analytic, roots, one.support.reading, wall)
    emit("ORACLE", **oracle)
    assert oracle["axis_ratio"] <= 5e-4
    assert oracle["saddle_ratio"] <= 5e-4
    assert oracle["boundary_ratio"] <= 1e-4

    batch(states)
    jax.block_until_ready(batch(states))
    started = perf_counter()
    many = batch(states)
    jax.block_until_ready(many)
    batch_wall = 1e3 * (perf_counter() - started)
    emit(
        "BATCH",
        wall_ms=batch_wall,
        per_slice_ms=batch_wall / lanes,
        lanes=lanes,
        steps=np.asarray(many.steps).tolist(),
        refactors=np.asarray(many.refactors).tolist(),
        reads=np.asarray(many.reads).tolist(),
        tracked_reads=None
        if counter(many, "tracked_reads") is None
        else np.asarray(many.tracked_reads).tolist(),
        certificates=None
        if counter(many, "certificates") is None
        else np.asarray(many.certificates).tolist(),
        fresh_residual=np.asarray(many.residual).tolist(),
        converged=np.asarray(many.converged).tolist(),
    )
    assert bool(jnp.all(many.converged))
    span = abs(float(one.support.reading.boundary_flux - one.support.reading.axis_flux))
    radius = oracle["minor_radius"]
    reference_boundary_flux = float(analytic.value(jnp.asarray(reference_roots[1])))
    for lane in range(lanes):
        own = single(tuple(value[lane] for value in states))
        jax.block_until_ready(own)
        assert bool(own.converged)
        flux = support_read.nodal_jets(bank, own.moments)[:, 0]
        batched_flux = support_read.nodal_jets(
            bank, tuple(value[lane] for value in many.moments)
        )[:, 0]
        emit(
            "LANE",
            lane=lane,
            flux_ratio=float(jnp.max(jnp.abs(flux - batched_flux))) / span,
            axis_ratio=float(
                jnp.linalg.norm(own.support.reading.axis - many.support.reading.axis[lane])
            )
            / radius,
            saddle_ratio=float(
                jnp.linalg.norm(
                    own.support.reading.boundary - many.support.reading.boundary[lane]
                )
            )
            / radius,
            boundary_ratio=abs(
                float(
                    own.support.reading.boundary_flux
                    - many.support.reading.boundary_flux[lane]
                )
            )
            / span,
            oracle_axis_ratio=float(
                jnp.linalg.norm(many.support.reading.axis[lane] - reference_roots[0])
            )
            / radius,
            oracle_saddle_ratio=float(
                jnp.linalg.norm(
                    many.support.reading.boundary[lane] - reference_roots[1]
                )
            )
            / radius,
            oracle_boundary_ratio=abs(
                float(many.support.reading.boundary_flux[lane]) - reference_boundary_flux
            )
            / oracle["flux_span"],
            single_residual=float(own.residual),
            batch_residual=float(many.residual[lane]),
        )


if __name__ == "__main__":
    main()