"""Continuity contracts for the declared-anchor parity source."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.efit_forward_parity_slice import (
    DeclaredAnchorOperator,
    _profile_function,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes


def _declared_operator(node_count: int) -> DeclaredAnchorOperator:
    nodes = np.linspace(0.0, 1.0, 65)
    p_prime = -1.1 + 0.35 * nodes - 0.08 * nodes**2
    ff_prime = -0.2 + 0.04 * nodes
    operator = object.__new__(DeclaredAnchorOperator)
    operator.grid = SimpleNamespace(
        node_number=node_count,
        coordinate=jnp.c_[
            jnp.linspace(0.8, 1.4, node_count),
            jnp.zeros(node_count),
        ],
    )
    operator.source = ForwardSource(
        core=DomainProfile(
            p_prime=_profile_function(nodes, p_prime),
            ff_prime=_profile_function(nodes, ff_prime),
        )
    )
    operator.area = jnp.linspace(0.01, 0.025, node_count)
    operator.declared_axis_flux = -0.4
    operator.declared_boundary_flux = 0.6
    operator.declared_support = jnp.ones(node_count, dtype=bool)
    return operator


def _flux_state(psi_norm: jax.Array) -> jax.Array:
    return -0.4 + jnp.asarray(psi_norm)


def _expanded_closure_values(
    nodes: np.ndarray, values: np.ndarray, coordinate: np.ndarray
) -> np.ndarray:
    """Evaluate the algebraically expanded two-edge closure."""
    edge_width = nodes[1] - nodes[0]
    lower_slope = (values[1] - values[0]) / edge_width
    upper_slope = (values[-1] - values[-2]) / edge_width
    lower_parameter = (coordinate - nodes[0] + edge_width) / edge_width
    lower_value_basis = (
        10.0 * lower_parameter**3 - 15.0 * lower_parameter**4 + 6.0 * lower_parameter**5
    )
    lower_slope_basis = (
        -4.0 * lower_parameter**3 + 7.0 * lower_parameter**4 - 3.0 * lower_parameter**5
    )
    lower = lower_value_basis * values[0] + lower_slope_basis * edge_width * lower_slope
    upper_parameter = (coordinate - nodes[-1]) / edge_width
    upper_value_basis = (
        1.0
        - 10.0 * upper_parameter**3
        + 15.0 * upper_parameter**4
        - 6.0 * upper_parameter**5
    )
    upper_slope_basis = (
        upper_parameter
        - 6.0 * upper_parameter**3
        + 8.0 * upper_parameter**4
        - 3.0 * upper_parameter**5
    )
    upper = (
        upper_value_basis * values[-1] + upper_slope_basis * edge_width * upper_slope
    )
    interior = np.interp(coordinate, nodes, values)
    return np.where(
        coordinate < nodes[0] - edge_width,
        0.0,
        np.where(
            coordinate < nodes[0],
            lower,
            np.where(
                coordinate <= nodes[-1],
                interior,
                np.where(coordinate <= nodes[-1] + edge_width, upper, 0.0),
            ),
        ),
    )


def test_shared_horner_closure_matches_expanded_edges() -> None:
    """The smaller traced expression preserves the declared profile exactly."""
    configure_dtypes()
    nodes = np.linspace(0.0, 1.0, 65)
    values = -1.1 + 0.35 * nodes - 0.08 * nodes**2
    coordinate = np.linspace(-0.2, 1.2, 1401)

    expected = _expanded_closure_values(nodes, values, coordinate)
    observed = np.asarray(_profile_function(nodes, values)(jnp.asarray(coordinate)))

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-12)


def test_declared_anchor_tangent_matches_centered_difference_at_both_edges() -> None:
    """Endpoint crossings retain the derivative represented by the JVP."""
    configure_dtypes()
    operator = _declared_operator(6)
    state = _flux_state(jnp.asarray([0.0, 1.0, 0.23, 0.41, 0.67, 0.82]))
    directions = (
        jnp.asarray([0.8, -0.6, 0.3, -0.2, 0.4, -0.1]),
        jnp.asarray(np.random.default_rng(17).normal(size=6)),
        jnp.asarray(np.random.default_rng(29).normal(size=6)),
    )

    for evaluate in (
        operator.cell_current_moments,
        jax.jit(operator.cell_current_moments),
    ):
        for direction in directions:
            scale = 1.0e-6 / float(jnp.max(jnp.abs(direction)))
            tangent = jax.jvp(
                lambda flux: evaluate(flux).cell_current,
                (state,),
                (direction,),
            )[1]
            centered = (
                evaluate(state + scale * direction).cell_current
                - evaluate(state - scale * direction).cell_current
            ) / (2.0 * scale)
            relative_error = float(
                jnp.linalg.norm(centered - tangent)
                / jnp.maximum(jnp.linalg.norm(centered), 1.0)
            )
            assert relative_error < 1.0e-4


def test_vanishing_crossing_does_not_flip_declared_current_population() -> None:
    """The fixed declared cohort stays nonzero on both sides of each anchor."""
    configure_dtypes()
    operator = _declared_operator(4)
    centre = jnp.asarray([0.0, 1.0, 0.35, 0.7])
    displacement = jnp.asarray([1.0, -1.0, 0.0, 0.0]) * 1.0e-10

    populations = []
    for psi_norm in (centre - displacement, centre + displacement):
        current = operator.cell_current_moments(_flux_state(psi_norm)).cell_current
        populations.append(int(jnp.count_nonzero(current)))

    assert populations == [4, 4]
    edge_width = 1.0 / 64.0
    closed = operator.cell_current_moments(
        _flux_state(
            jnp.asarray(
                [
                    -edge_width,
                    1.0 + edge_width,
                    -2.0 * edge_width,
                    1.0 + 2.0 * edge_width,
                ]
            )
        )
    ).cell_current
    np.testing.assert_array_equal(closed, jnp.zeros(4))


def test_target_current_normalisation_is_unchanged_in_profile_interior() -> None:
    """Interior source values and their physical current target remain exact."""
    configure_dtypes()
    operator = _declared_operator(4)
    psi_norm = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    state = _flux_state(psi_norm)
    nodes = jnp.linspace(0.0, 1.0, 65)
    reference = DomainProfile(
        p_prime=lambda coordinate: jnp.interp(
            coordinate, nodes, -1.1 + 0.35 * nodes - 0.08 * nodes**2
        ),
        ff_prime=lambda coordinate: jnp.interp(coordinate, nodes, -0.2 + 0.04 * nodes),
    )
    expected = reference.current_density(operator.radius, psi_norm) * operator.area
    target_current = 8.0e5

    moments = operator.cell_current_moments(state)
    normalised, amplitude = operator.normalised_current_moments(state, target_current)

    np.testing.assert_array_equal(moments.cell_current, expected)
    np.testing.assert_allclose(
        amplitude,
        target_current / jnp.sum(expected),
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        jnp.sum(normalised.cell_current),
        target_current,
        rtol=1.0e-14,
    )
