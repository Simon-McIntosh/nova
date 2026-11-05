"""Split the converged bank solve's per-iterate wall into stages on one device.

Each stage is measured as its own jitted call at the solve's shapes, warmed,
with ``block_until_ready`` and a median over repeats. Nothing is timed inside
the solve's fused program: a timer there syncs the device and reports the
barrier rather than the stage.

The stages follow the call boundaries the product code already has. The read's
discrete decision and its continuous geometry are produced together inside
``topology.read_at_stationary_points``, so the continuous part is measured as
the terminal ``_support_at_level`` call it makes and the decision is the
remainder of that read; the fragment names the composition.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg

import nova
from nova.equilibrium import fixed_points, support_read, topology
from scripts.prototypes.bank_newton_step.measure import machine
from tests.equilibrium import test_topology_read as fixture

ROOT = Path(__file__).resolve().parents[3]
ITERATIONS = 12


def emit(label, **values):
    print(
        label + " " + json.dumps(values, sort_keys=True, default=_default), flush=True
    )


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    return float(value)


def timed(name, lanes, function, *args, repeats):
    value = function(*args)
    jax.block_until_ready(value)
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        value = function(*args)
        jax.block_until_ready(value)
        samples.append(1e3 * (perf_counter() - started))
    emit(
        "STAGE",
        lanes=lanes,
        name=name,
        median_ms=median(samples),
        min_ms=min(samples),
        samples_ms=samples,
    )
    return value


def counter(result, name):
    value = getattr(result, name, None)
    return None if value is None else float(np.asarray(value).ravel()[0])


def lane(view, index):
    return jax.tree.map(lambda leaf: leaf[index], view)


def profile_lanes(lanes, repeats):
    operands, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        machine("diverted")
    )

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    seeds = roots + 0.03 * jnp.median(operands.geometry.pitch)
    scales = jnp.linspace(2e-5, 6e-5, 16)[:lanes]
    states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    seeds_batch = jnp.broadcast_to(seeds, (lanes, *seeds.shape))
    previous_batch = jnp.broadcast_to(roots, (lanes, *roots.shape))
    axis_sign = -1.0

    @jax.jit
    def census(state):
        return jax.vmap(
            lambda one, seed, warm: support_read.stationary_stencil(
                operands,
                cache,
                one,
                seed,
                warm,
                convention,
                axis_sign,
                iterations=ITERATIONS,
            )
        )(state, seeds_batch, previous_batch)

    @jax.jit
    def fit(state):
        def one(inner):
            jets = support_read.nodal_jets(operands, inner)
            return fixed_points.field_from_jets(
                jets, operands.sample_index, operands.mesh
            )

        return jax.vmap(one)(state)

    @jax.jit
    def topology_read(field, stencil):
        return jax.vmap(
            lambda f, s: topology.read_at_stationary_points(
                f, operands.geometry, convention, policy, s
            )
        )(field, stencil.solved)

    @jax.jit
    def support_geometry(coefficient, level, axis, form, diverted):
        return jax.vmap(
            lambda c, level, a, f, d: topology._support_at_level(
                operands.geometry, c, convention.sigma, level, a, f, d, policy
            )
        )(coefficient, level, axis, form, diverted)

    @jax.jit
    def residual(state, support):
        return jax.vmap(
            lambda one, read: support_read.step_residual(
                operands, one, one, read, profile, quadrature
            )
        )(state, support)

    @jax.jit
    def assembly(state, support, booked):
        return jax.vmap(
            lambda one, read, bk: support_read.step_jacobian(
                operands,
                one,
                read,
                profile,
                gradient,
                quadrature,
                booked=bk,
                fresh_geometry=True,
            )
        )(state, support, booked)

    @jax.jit
    def factor(matrix, state, booked):
        def one(single, moments_lane, bk):
            balanced, row, column = support_read._equilibrate(single, moments_lane, bk)
            lu, pivots = linalg.lu_factor(balanced)
            return balanced, row, column, lu, pivots

        return jax.vmap(one)(matrix, state, booked)

    @jax.jit
    def backsolve(matrix, factored, right):
        _, row, column, lu, pivots = factored

        def one(single, lu_matrix, pivots_lane, row_norm, c, r):
            solution = c * linalg.lu_solve((lu_matrix, pivots_lane), r / row_norm)
            linear = jnp.linalg.norm(single @ solution - r) / jnp.linalg.norm(r)
            return solution, linear

        return jax.vmap(one)(matrix, lu, pivots, row, column, right)

    @jax.jit
    def solve_batch(state):
        return support_read.solve_batch(
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
            max_steps=60,
            reuse_factor=False,
            contraction_limit=0.5,
        )

    stencil = timed("census", lanes, census, states, repeats=repeats)
    field = timed("fit", lanes, fit, states, repeats=repeats)
    reading = timed("topology", lanes, topology_read, field, stencil, repeats=repeats)
    timed(
        "support_geometry",
        lanes,
        support_geometry,
        reading.field_coefficients,
        reading.boundary_flux,
        reading.axis,
        reading.saddle_form,
        reading.boundary_class == int(topology.TopologyClass.DIVERTED),
        repeats=repeats,
    )

    # One warm read and booking supplies the shapes the step stages consume.
    support = jax.jit(
        lambda state: support_read.read_batch(
            operands,
            cache,
            state,
            seeds_batch,
            previous_batch,
            convention,
            policy,
            axis_sign,
            iterations=ITERATIONS,
        )
    )(states)
    step = timed("residual", lanes, residual, states, support, repeats=repeats)
    matrix = timed(
        "assembly", lanes, assembly, states, support, step.booked, repeats=repeats
    )
    factored = timed(
        "factor", lanes, factor, matrix, states, step.booked, repeats=repeats
    )
    right = jax.vmap(
        lambda *values: -jnp.concatenate(tuple(jnp.ravel(v) for v in values))
    )(*step.residual)
    timed("backsolve", lanes, backsolve, matrix, factored, right, repeats=repeats)

    result = timed("whole_solve", lanes, solve_batch, states, repeats=repeats)
    steps = np.asarray(result.steps).tolist()
    emit(
        "SOLVE",
        lanes=lanes,
        steps=steps,
        max_steps=int(np.max(steps)),
        median_steps=float(np.median(steps)),
        refactors=np.asarray(result.refactors).tolist(),
        reads=np.asarray(result.reads).tolist(),
        converged=np.asarray(result.converged).tolist(),
        fresh_residual=np.asarray(result.residual).tolist(),
        terminal_residual=counter(result, "residual"),
    )
    if lanes == 1:

        @jax.jit
        def single(state):
            return support_read.solve(
                operands,
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

        one = single(moments)
        jax.block_until_ready(one)
        emitted = timed("single_solve", 1, single, moments, repeats=repeats)
        emit(
            "SINGLE",
            steps=counter(emitted, "steps"),
            refactors=counter(emitted, "refactors"),
            reads=counter(emitted, "reads"),
            fresh_residual=float(emitted.residual),
            converged=bool(emitted.converged),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    try:
        import subprocess

        revision = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        revision = "unversioned-archive"
    _, _, wall, _, _ = fixture._analytic_inputs("diverted")
    emit(
        "HEADER",
        revision=revision,
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64),
        repeats=args.repeats,
        wall_points=int(len(wall)),
    )
    assert jax.config.jax_enable_x64
    for lanes in (1, 16):
        profile_lanes(lanes, args.repeats)


if __name__ == "__main__":
    main()
