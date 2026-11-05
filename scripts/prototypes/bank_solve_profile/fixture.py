"""One definition of the profile fixture: the seed, the lane perturbations, the batch.

Every receipt in this directory measures the same diverted bank on the same
seed, so the recipe lives here once. It restates what
``scripts/prototypes/bank_solve_read/measure.py`` builds inline in its gate —
that module runs a whole measurement on import of its ``main``, so its recipe
cannot be imported without running it — and this is the single copy every
receipt here reads.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from scripts.prototypes.bank_newton_step.measure import machine

LANES = 16
SCALE_LOW = 2e-5
SCALE_HIGH = 6e-5
SEED_OFFSET = 0.03


@dataclass(frozen=True)
class Batch:
    """The machine, the seed and one perturbed lane per batch member."""

    operands: object
    cache: object
    convention: object
    policy: object
    profile: object
    quadrature: object
    roots: object
    moments: object
    seeds: object
    states: tuple
    seeds_batch: object
    previous_batch: object
    lanes: int


def gradient(radius, flux):
    return jnp.zeros_like(radius)


def build(lanes, perturb=True):
    """The batch of ``lanes`` states the receipts time.

    ``perturb`` off returns the unperturbed seed, which converges in fewer
    steps than any perturbed lane: a receipt that pins the loop's step ceiling
    needs a state whose convergence sits above that ceiling.
    """
    operands, cache, roots, convention, policy, profile, quadrature, moments, _ = (
        machine("diverted")
    )
    seeds = roots + SEED_OFFSET * jnp.median(operands.geometry.pitch)
    scales = jnp.linspace(SCALE_LOW, SCALE_HIGH, LANES)[:lanes]
    if perturb:
        states = tuple(value[None] * (1 + scales[:, None]) for value in moments)
    else:
        states = tuple(value[None] for value in moments)
    return Batch(
        operands=operands,
        cache=cache,
        convention=convention,
        policy=policy,
        profile=profile,
        quadrature=quadrature,
        roots=roots,
        moments=moments,
        seeds=seeds,
        states=states,
        seeds_batch=jnp.broadcast_to(seeds, (lanes, *seeds.shape)),
        previous_batch=jnp.broadcast_to(roots, (lanes, *roots.shape)),
        lanes=lanes,
    )
