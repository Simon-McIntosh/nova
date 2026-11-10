"""Pass every machine-sized bank array to compiled predictor comparisons."""

import jax
from types import SimpleNamespace

from nova.equilibrium import support_read
from scripts.prototypes.bank_solve_profile.fixture import gradient
from scripts.prototypes.bank_warm_start.streams import scaled_profile


def programs(machine, tangent):
    """Compile once with a runtime bank and a traced drive amplitude."""
    inputs = (
        machine.operands,
        machine.cache,
        machine.convention,
        machine.policy,
        machine.quadrature,
    )
    profile = machine.profile

    @jax.jit
    def solve(
        operands,
        cache,
        convention,
        policy,
        quadrature,
        moments,
        seeds,
        previous,
        amplitude,
    ):
        return support_read.solve_batch(
            operands,
            cache,
            moments,
            seeds,
            previous,
            convention,
            policy,
            scaled_profile(profile, amplitude),
            gradient,
            quadrature,
            max_steps=60,
            reuse_factor=False,
            contraction_limit=0.5,
            tolerance=1e-10,
        )

    @jax.jit
    def predict(
        operands,
        cache,
        convention,
        policy,
        quadrature,
        moments,
        stationary,
        drive,
        target,
    ):
        carrier = SimpleNamespace(
            operands=operands,
            cache=cache,
            convention=convention,
            policy=policy,
            quadrature=quadrature,
            profile=profile,
        )
        return tangent(carrier, moments, stationary, drive, target)

    return inputs, solve, predict
