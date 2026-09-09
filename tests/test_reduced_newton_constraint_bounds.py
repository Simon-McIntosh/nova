"""Terminal admission of constrained reduced-route compensating currents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import reduced_newton
    from nova.equilibrium.constraint import (
        CircuitCurrentUnknown,
        ConstraintBinding,
        ConstraintPair,
        CurrentCentroidConstraint,
    )
    from nova.equilibrium.forward_operator import PrescribedCurrentField
    from nova.equilibrium.observation import MomentIntegralSupport

    from tests.test_reduced_newton import machine  # noqa: F401


SOLVE_TOLERANCE = 1.0e-8
NEWTON_STEPS = 24
CENTROID_MOVE = 1.0e-2
CEILING_A = 1.0e-3
IN_BOUND_CEILING_A = 1.0e6
RECEIPT_PATH = (
    Path(__file__).parents[1]
    / "docs/figures/constraint-augmented-newton-krylov/compensator-bound/receipt.json"
)


def _prescribed(profile):
    """Attach the fixture's conductor response as the prescribed field."""
    operator = profile.operator
    response = jnp.concatenate(
        (
            jnp.asarray(operator.grid.source_target),
            jnp.asarray(operator.wall.source_target),
        )
    )
    operator.prescribed_field = PrescribedCurrentField(
        response=response, current=jnp.zeros(response.shape[1])
    )


def _centroid(profile, flux) -> float:
    """Read the constrained vertical current centroid."""
    return float(
        np.asarray(
            profile.current_moment_observation(
                jnp.asarray(flux), support=MomentIntegralSupport.ALL_DOMAIN
            ).centroid_z
        )
    )


def _centroid_pair(profile, flux, target):
    """Return a circuit-current row selected from the local response matrix."""
    scale = float(np.ptp(np.asarray(profile.lattice.height)))
    seeded = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",), support=MomentIntegralSupport.ALL_DOMAIN
        ),
        unknown=CircuitCurrentUnknown(
            direction=jnp.ones(
                (profile.operator.prescribed_current_field.current.size, 1)
            ),
            ampere_scale=jnp.ones(1),
        ),
        binding=ConstraintBinding(
            target=jnp.asarray([target]),
            tolerance=jnp.asarray([1.0e-6]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
        ),
    )
    pairs, _selection = reduced_newton.derive_reduced_constraint_pairs(
        profile, (seeded,), flux
    )
    return pairs[0]


@pytest.fixture(scope="module")
def prepared(machine):
    """Return one converged Solov'ev state and a reachable centroid command."""
    profile, seed = machine
    _prescribed(profile)
    free = reduced_newton.solve_reduced_newton(
        profile.operator,
        seed,
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
    )
    assert free.converged
    target = _centroid(profile, free.state) + CENTROID_MOVE
    return profile, free, _centroid_pair(profile, free.state, target)


def _solve(profile, free, pair, ceiling):
    """Run one route configuration from the same terminal free equilibrium."""
    return reduced_newton.solve_constrained_reduced_newton(
        profile,
        free.state,
        constraint_pairs=(pair,),
        tolerance=SOLVE_TOLERANCE,
        newton_steps=NEWTON_STEPS,
        constraint_current_ceiling=ceiling,
    )


def _records_identical(left, right) -> bool:
    """Return whether two terminal row receipts agree exactly."""
    if len(left) != len(right):
        return False
    for left_record, right_record in zip(left, right, strict=True):
        for field in left_record._fields:
            first = getattr(left_record, field)
            second = getattr(right_record, field)
            if first is None or second is None:
                if first is not second:
                    return False
            elif not np.array_equal(np.asarray(first), np.asarray(second)):
                return False
    return True


def _write_receipt(over_ceiling, unbounded, in_bound) -> None:
    """Persist both admission outcomes alongside the constrained-route evidence."""
    receipt = {
        "receipt": "constrained reduced-route compensator admission",
        "source": {
            "backend": jax.default_backend(),
            "x64": bool(jax.config.jax_enable_x64),
            "fixture": "bootstrapped Solovev free-boundary equilibrium",
        },
        "over_ceiling": {
            "ceiling_a": CEILING_A,
            "maximum_compensating_current_a": float(
                np.max(np.abs(np.asarray(over_ceiling.compensating_current)))
            ),
            "converged": over_ceiling.converged,
            "qualified": over_ceiling.qualified,
            "refusal_reason": over_ceiling.refusal_reason,
            "row_qualified": np.asarray(over_ceiling.constraints[0].qualified)
            .astype(bool)
            .tolist(),
        },
        "in_bound_identity": {
            "ceiling_a": IN_BOUND_CEILING_A,
            "unbounded_converged": unbounded.converged,
            "bounded_converged": in_bound.converged,
            "state_bit_identical": bool(
                np.array_equal(np.asarray(unbounded.state), np.asarray(in_bound.state))
            ),
            "compensating_current_bit_identical": bool(
                np.array_equal(
                    np.asarray(unbounded.compensating_current),
                    np.asarray(in_bound.compensating_current),
                )
            ),
            "records_bit_identical": _records_identical(
                unbounded.constraints, in_bound.constraints
            ),
        },
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def test_compensator_ceiling_refuses_without_changing_an_in_bound_receipt(prepared):
    """A physical ceiling rejects excess current while safe commands are identical."""
    profile, free, pair = prepared
    unbounded = _solve(profile, free, pair, None)
    in_bound = _solve(profile, free, pair, IN_BOUND_CEILING_A)
    over_ceiling = _solve(profile, free, pair, CEILING_A)
    _write_receipt(over_ceiling, unbounded, in_bound)

    assert unbounded.converged
    assert unbounded.qualified
    assert in_bound.converged
    assert in_bound.qualified
    assert np.array_equal(np.asarray(unbounded.state), np.asarray(in_bound.state))
    assert np.array_equal(
        np.asarray(unbounded.compensating_current),
        np.asarray(in_bound.compensating_current),
    )
    assert _records_identical(unbounded.constraints, in_bound.constraints)
    assert np.max(np.abs(np.asarray(over_ceiling.compensating_current))) > CEILING_A
    assert not over_ceiling.converged
    assert not over_ceiling.qualified
    assert over_ceiling.refusal_reason == "compensating-current-ceiling-exceeded"
    assert not bool(np.asarray(over_ceiling.constraints[0].qualified)[0])


def test_missing_axis_is_a_terminal_refusal(prepared, monkeypatch):
    """A closed row cannot qualify after its magnetic-axis admission is lost."""
    profile, free, pair = prepared
    result = _solve(profile, free, pair, IN_BOUND_CEILING_A)
    original = profile.operator._fixed_design_read

    def no_axis(*arguments, **keywords):
        masks, topology, connected, _admitted = original(*arguments, **keywords)
        return masks, topology, connected, jnp.asarray(False)

    monkeypatch.setattr(profile.operator, "_fixed_design_read", no_axis)
    qualified, refusal_reason = reduced_newton._constraint_qualification(
        profile.operator,
        result.state,
        None,
        result.compensating_current,
        IN_BOUND_CEILING_A,
    )

    assert not qualified
    assert refusal_reason == "no-qualified-magnetic-axis"
