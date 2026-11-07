"""Sixteen lanes of the diverted certificate on one shared drive trajectory.

A slice's drive is the plasma profile amplitude: the certificate's source is a
constant-gradient profile, so scaling ``p_prime`` and ``ff_prime`` together is
one physical knob. ``STEPS`` are the relative amplitude changes per slice.

``support_read.solve_batch`` takes one ``profile`` and one operator bank for
the whole batch, so every lane of a batched call sits at the same drive. The
sixteen lanes are therefore sixteen distinct cold seeds carried along one
shared ramp, which is the batch shape the entry point can express today.
"""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import jax

from nova.equilibrium import topology

from scripts.prototypes.bank_newton_step.measure import machine
from scripts.prototypes.bank_solve_profile import fixture as profile_fixture

LANES = profile_fixture.LANES
SLICES = 8

# Relative change of the profile amplitude per slice. Chosen against the
# certificate's plasma current: at a 1 MA plasma and a 1 ms slice boundary a
# fast ramp moves about 1 kA per slice (1e-3); a flattop moves about 10 A
# (1e-5); a deliberately coarse time grid moves about 10 kA (1e-2).
STEPS = (1e-5, 1e-3, 1e-2)
STEP_ORIGIN = {
    1e-5: "slow flattop slice: about 10 A of plasma current",
    1e-3: "fast ramp slice: about 1 kA of plasma current",
    1e-2: "coarse time grid: about 10 kA of plasma current",
}


@dataclass(frozen=True)
class Machine:
    operands: object
    cache: object
    roots: object
    convention: object
    policy: object
    profile: object
    quadrature: object
    moments: object
    analytic: object
    minor_radius: float
    analytic_axis: object
    analytic_saddle: object
    analytic_axis_flux: object
    analytic_boundary_flux: object
    cold_states: tuple
    cold_seeds: object


def scaled_profile(profile, amplitude):
    """The same profile source with both amplitudes scaled together."""
    return replace(
        profile,
        p_prime=lambda psi: amplitude * profile.p_prime(psi),
        ff_prime=lambda psi: amplitude * profile.ff_prime(psi),
    )


def build():
    """The machine, the sixteen cold seeds and the analytic oracle at unit drive."""
    batch = profile_fixture.build(LANES)
    operands, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        machine("diverted")
    )
    analytic_read = jax.jit(topology.read)(analytic, operands.geometry, convention, policy)
    from tests.equilibrium import test_topology_read as fixture

    oracle, _, _, _, _ = fixture._analytic_inputs("diverted")
    return Machine(
        operands=operands,
        cache=cache,
        roots=roots,
        convention=convention,
        policy=policy,
        profile=profile,
        quadrature=quadrature,
        moments=moments,
        analytic=analytic,
        minor_radius=float(oracle.minor_radius),
        analytic_axis=jnp.asarray(roots[0]),
        analytic_saddle=jnp.asarray(roots[1]),
        analytic_axis_flux=analytic_read.axis_flux,
        analytic_boundary_flux=analytic_read.boundary_flux,
        cold_states=batch.states,
        cold_seeds=batch.seeds_batch,
    )


def amplitudes(step):
    """The shared drive at each slice: ``1 + k * step``."""
    return 1.0 + step * jnp.arange(SLICES)