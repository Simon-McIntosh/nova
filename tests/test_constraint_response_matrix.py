"""Matrix-led compensating directions on a machine-free circuit fixture.

The fixture is four flux nodes over four prescribed circuits: an antisymmetric
pair that is the only vertical authority in the machine, a symmetric solenoid,
and a symmetric shaping circuit.  It is the situation the vertical centroid row
was hand-wired for, so a derived direction that does not land exactly on the
antisymmetric pair here is wrong for a reason a bank row would never show.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    CompensatorRule,
    ConstraintBinding,
    ConstraintPair,
    compensator_rule_name,
    constraint_response_matrix,
    derive_circuit_compensators,
    select_compensating_directions,
)
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import PrescribedCurrentField
from nova.jax.config import configure_dtypes


HEIGHT = np.asarray([-1.5, -0.5, 0.5, 1.5])
GAP_WEIGHT = np.asarray([-1.0, -0.2, 0.2, 1.4])
MOMENT_WEIGHT = 4.0
UPPER = 0
LOWER = 1
RESPONSE = np.asarray(
    [
        [0.0, 1.5, 1.0, 0.8],
        [0.0, 0.5, 1.0, 0.2],
        [0.5, 0.0, 1.0, 0.2],
        [1.5, 0.0, 1.0, 0.8],
    ]
)
PAIR_DIRECTION = np.asarray([1.0, -1.0, 0.0, 0.0])
POSITION_SCALE = 0.5
SEED = np.asarray([0.4, 0.2, 0.2, 0.4])
AMPERE_SCALE = 2000.0


@dataclass(frozen=True)
class _WeightedHeight:
    """A linear placement observation reading one weighted flux moment."""

    weight: tuple[float, ...]

    @property
    def row_count(self) -> int:
        """Contribute exactly one residual row."""
        return 1

    def observed(self, _profile, context, _payload):
        """Return the weighted moment of the flux state."""
        weight = jnp.asarray(self.weight, dtype=context.flux.dtype)
        return jnp.atleast_1d(weight @ context.flux / MOMENT_WEIGHT)

    def residual(self, profile, context, _unknown, payload, target, scale):
        """Return the scale-normalised placement error."""
        return (self.observed(profile, context, payload) - target) / scale

    def dual_flux_image(self, profile, context, payload):
        """Return the flux gradient of the observation as a column."""
        jacobian = jax.jacrev(
            lambda flux: self.observed(profile, context._replace(flux=flux), payload)
        )(context.flux)
        return jnp.moveaxis(jacobian, 0, -1)


class _CircuitOperator:
    """A flux map that only ever returns what the prescribed circuits put in."""

    def __init__(self) -> None:
        self.prescribed_current_field = PrescribedCurrentField(
            response=jnp.asarray(RESPONSE),
            current=jnp.zeros(RESPONSE.shape[1]),
        )
        self.source = SimpleNamespace(closure_degrees=0)

    def flux_map(self, *_args):
        """Return the trivial map whose root is the circuit contribution alone."""
        return lambda flux: jnp.zeros_like(flux)

    def flux_map_with_shadow(self, *_args):
        """Return the shadowed form of the same trivial map."""
        return lambda flux, shadow: jnp.where(shadow, flux, jnp.zeros_like(flux))

    def residual_shadow_mask(self, flux, _requested=None, previous_shadow=None):
        """Keep every node active; this fixture has no free boundary."""
        del previous_shadow
        return jnp.zeros_like(flux, dtype=bool)


def _profile() -> ForwardProfile:
    """Assemble the smallest profile the augmented solve path accepts."""
    profile = object.__new__(ForwardProfile)
    profile.operator = _CircuitOperator()
    profile.newton_steps = 8

    def receipt(flux, history, *_args, constraints=(), **_kwargs):
        return SimpleNamespace(flux=flux, fixed_point=history, constraints=constraints)

    profile._receipt = receipt
    return profile


def _pair(weight, *, direction, target, scale=POSITION_SCALE):
    """Bind one weighted-height row to an explicitly named circuit direction."""
    return ConstraintPair(
        functional=_WeightedHeight(tuple(float(item) for item in weight)),
        unknown=CircuitCurrentUnknown(
            direction=jnp.asarray(direction),
            ampere_scale=jnp.asarray([AMPERE_SCALE]),
        ),
        binding=ConstraintBinding(
            target=jnp.atleast_1d(jnp.asarray(target)),
            tolerance=jnp.asarray([1.0e-8]),
            scale=jnp.asarray([scale]),
            initial_unknown=jnp.asarray([0.0]),
            payload=None,
            policy="imposed",
        ),
    )


def _solve(profile, pairs, seed):
    """Run the augmented solve on this fixture with a settled active set."""
    return profile._solve_augmented_constraints(
        jnp.asarray(seed),
        None,
        constraint_pairs=tuple(pairs),
        warmup=0,
        gmres_iterations=6,
        active_set_steps=2,
        stop_on_active_set_settlement=False,
    )


def test_response_matrix_matches_central_circuit_differences() -> None:
    """The autodiff-and-carrier matrix equals a difference over each circuit."""
    configure_dtypes()
    profile = _profile()
    pair = _pair(HEIGHT, direction=PAIR_DIRECTION, target=0.0)
    flux = jnp.asarray([0.3, -0.2, 0.7, 1.1])
    matrix = np.asarray(constraint_response_matrix(profile, (pair,), flux))

    step = 1.0e-4
    field = profile.operator.prescribed_current_field
    differenced = np.zeros_like(matrix)
    for circuit in range(RESPONSE.shape[1]):
        unit = np.zeros(RESPONSE.shape[1])
        unit[circuit] = step
        delta = np.asarray(field.flux_delta(jnp.asarray(unit)))
        context = pair.functional
        forward = np.asarray(context.observed(profile, _context(flux + delta), None))
        backward = np.asarray(context.observed(profile, _context(flux - delta), None))
        differenced[:, circuit] = (forward - backward) / (2.0 * step)

    np.testing.assert_allclose(matrix, differenced, rtol=1.0e-9, atol=1.0e-12)
    expected = (HEIGHT @ RESPONSE) / MOMENT_WEIGHT
    np.testing.assert_allclose(matrix[0], expected, rtol=1.0e-12, atol=1.0e-14)


def _context(flux):
    """Return a bare constraint context around one flux state."""
    from nova.equilibrium.constraint import ConstraintContext

    return ConstraintContext(jnp.asarray(flux), None, None, None)


def test_derived_direction_reproduces_the_named_pair_on_the_centroid_row() -> None:
    """One placement row lands on the antisymmetric pair and solves identically."""
    configure_dtypes()
    profile = _profile()
    seed = jnp.asarray(SEED)
    target = 0.35
    fixed = _pair(HEIGHT, direction=PAIR_DIRECTION, target=target)
    (derived,), selection = derive_circuit_compensators(profile, (fixed,), seed)

    assert selection.rule is CompensatorRule.DOMINANT_AUTHORITY
    assert not selection.competing
    np.testing.assert_allclose(
        selection.directions[:, 0], PAIR_DIRECTION, rtol=0.0, atol=1.0e-12
    )
    assert selection.leading_circuits(0) == (UPPER, LOWER)
    row = np.asarray(selection.response)[0]
    expected_spectrum = np.linalg.norm(row / POSITION_SCALE)
    np.testing.assert_allclose(
        selection.singular_values, [expected_spectrum], rtol=1.0e-12
    )

    fixed_result = _solve(profile, (fixed,), seed)
    derived_result = _solve(profile, (derived,), seed)
    fixed_record = fixed_result.constraints[0]
    derived_record = derived_result.constraints[0]

    tolerance = float(np.asarray(fixed.binding.tolerance)[0])
    np.testing.assert_allclose(
        np.asarray(derived_record.physical_unknown),
        np.asarray(fixed_record.physical_unknown),
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        np.asarray(derived_result.flux), np.asarray(fixed_result.flux), atol=1.0e-9
    )
    np.testing.assert_allclose(np.asarray(derived_record.observed), [target], atol=1e-9)
    assert bool(derived_record.qualified[0])
    assert compensator_rule_name(derived_record.compensator_rule) == (
        "dominant_authority"
    )
    np.testing.assert_allclose(
        np.asarray(derived_record.compensator_singular_values),
        [expected_spectrum],
        rtol=1.0e-12,
    )
    assert compensator_rule_name(fixed_record.compensator_rule) == "explicit"
    assert fixed_record.compensator_singular_values is None


def test_a_contaminated_row_still_names_the_pair_and_a_floor_recovers_it() -> None:
    """A weakly asymmetric machine keeps the pair leading and truncates cleanly."""
    configure_dtypes()
    profile = _profile()
    seed = jnp.asarray(SEED)
    pair = _pair(GAP_WEIGHT, direction=PAIR_DIRECTION, target=0.2)
    (_derived,), selection = derive_circuit_compensators(profile, (pair,), seed)
    assert selection.leading_circuits(0, count=2) == (UPPER, LOWER)
    assert np.max(np.abs(selection.directions[2:, 0])) < 0.25

    (_floored,), truncated = derive_circuit_compensators(
        profile, (pair,), seed, participation_floor=0.5
    )
    np.testing.assert_allclose(
        truncated.directions[:, 0],
        [1.0, np.asarray(selection.directions)[LOWER, 0], 0.0, 0.0],
        atol=1.0e-12,
    )


def test_two_competing_rows_receive_distinct_decoupled_directions() -> None:
    """Rows reading the same circuits are distributed by the matrix, not shared."""
    configure_dtypes()
    profile = _profile()
    seed = jnp.asarray(SEED)
    pairs = (
        _pair(HEIGHT, direction=PAIR_DIRECTION, target=0.30),
        _pair(GAP_WEIGHT, direction=PAIR_DIRECTION, target=0.18),
    )
    derived, selection = derive_circuit_compensators(profile, pairs, seed)

    assert selection.competing
    assert selection.rule is CompensatorRule.SINGULAR_DISTRIBUTION
    assert selection.row_coupling[0, 1] > 0.9
    assert selection.singular_values.shape == (2,)
    assert selection.singular_values[0] > selection.singular_values[1] > 0.0

    columns = selection.directions
    assert np.max(np.abs(columns[:, 0] - columns[:, 1])) > 0.5
    closed = selection.authority @ columns
    np.testing.assert_allclose(np.diag(closed), selection.direction_authority)
    off_diagonal = closed - np.diag(np.diag(closed))
    assert np.max(np.abs(off_diagonal)) < 1.0e-10

    shared = select_compensating_directions(
        selection.authority, rule=CompensatorRule.DOMINANT_AUTHORITY
    )
    contested = shared.authority @ shared.directions
    cross_talk = np.max(np.abs(contested - np.diag(np.diag(contested))))
    assert cross_talk / np.min(np.abs(np.diag(contested))) > 0.5

    result = _solve(profile, derived, seed)
    for record, pair in zip(result.constraints, pairs, strict=True):
        np.testing.assert_allclose(
            np.asarray(record.observed),
            np.asarray(pair.binding.target),
            atol=1.0e-8,
        )
        assert compensator_rule_name(record.compensator_rule) == (
            "singular_distribution"
        )


def test_the_dominant_rule_can_be_named_against_the_competition_default() -> None:
    """A caller that states the rule overrides the competition test."""
    configure_dtypes()
    authority = np.asarray([[2.5, -2.5, 0.0, 0.0], [2.2, -1.6, 0.4, 0.32]])
    automatic = select_compensating_directions(authority)
    named = select_compensating_directions(
        authority, rule=CompensatorRule.DOMINANT_AUTHORITY
    )
    assert automatic.rule is CompensatorRule.SINGULAR_DISTRIBUTION
    assert named.rule is CompensatorRule.DOMINANT_AUTHORITY
    np.testing.assert_allclose(
        named.directions[:, 0], [1.0, -1.0, 0.0, 0.0], atol=1.0e-12
    )
    assert np.all(named.direction_authority > 0.0)
    np.testing.assert_allclose(
        named.singular_values, automatic.singular_values, rtol=1.0e-12
    )


def test_a_row_without_circuit_authority_is_refused() -> None:
    """A constraint no circuit can move cannot be given a derived direction."""
    with pytest.raises(ValueError, match="non-zero circuit authority"):
        select_compensating_directions(np.zeros((1, 4)))


if __name__ == "__main__":
    pytest.main([__file__])
