"""Compare the prototype and carried-read paths at identical Newton iterates."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_newton_step.measure import machine


ROOT = Path(__file__).resolve().parents[3]


def emit(label, **values):
    print(label + " " + json.dumps(values, sort_keys=True, default=float), flush=True)


def difference(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return float(np.max(np.abs(a - b)))


def residual(bank, state, reading, profile, quadrature):
    result = support_read.step_residual(
        bank, state, state, reading, profile, quadrature
    )
    return float(jnp.max(jnp.abs(jnp.concatenate(result.residual))))


def main():
    emit(
        "HEADER",
        revision=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        tree=str(ROOT),
        cwd=str(Path.cwd().resolve()),
        nova=nova.__file__,
        module=support_read.__file__,
        backend=jax.default_backend(),
        x64=jax.config.jax_enable_x64,
    )
    assert jax.config.jax_enable_x64
    bank, cache, roots, convention, policy, profile, quadrature, state, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    @jax.jit
    def fresh(one):
        return support_read.read(bank, cache, one, seeds, roots, convention, policy)

    @jax.jit
    def tracked(one, previous, tracking):
        return support_read.read_batch(
            bank,
            cache,
            tuple(value[None] for value in one),
            seeds[None],
            previous[None],
            convention,
            policy,
            tracking=tracking,
        )

    @jax.jit
    def step(one, reading):
        return support_read.newton_step(
            bank,
            one,
            reading,
            profile,
            gradient,
            quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )

    prior = None
    previous = roots
    for index in range(4):
        baseline_read = fresh(state)
        if prior is None:
            carried_read = support_read.read_batch(
                bank,
                cache,
                tuple(value[None] for value in state),
                seeds[None],
                previous[None],
                convention,
                policy,
            )
        else:
            carried_read = tracked(state, previous, prior)
        carried = jax.tree.map(lambda value: value[0], carried_read)
        baseline_norm = residual(bank, state, baseline_read, profile, quadrature)
        carried_norm = residual(bank, state, carried, profile, quadrature)
        baseline_step = step(state, baseline_read)
        carried_step = step(state, carried)
        emit(
            "STEP",
            index=index,
            fresh_residual=baseline_norm,
            carried_residual=carried_norm,
            update_gap=max(
                difference(a, b)
                for a, b in zip(baseline_step.update, carried_step.update, strict=True)
            ),
            membership_gap=difference(
                baseline_read.reading.membership, carried.reading.membership
            ),
            coefficient_gap=difference(
                baseline_read.reading.field_coefficients,
                carried.reading.field_coefficients,
            ),
            edge_gap=difference(
                baseline_read.reading.edge_interval, carried.reading.edge_interval
            ),
            position_gap=difference(
                baseline_read.stationary.position, carried.stationary.position
            ),
            rebuilt=bool(carried.rebuilt),
        )
        prior = support_read.tracking_of(carried_read)
        previous = carried.stationary.position
        state = tuple(
            value + change
            for value, change in zip(state, baseline_step.update, strict=True)
        )


if __name__ == "__main__":
    main()
