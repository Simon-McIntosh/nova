"""Current centroid rows retain within-cell first moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import fixture_positional_stiffness as stiffness
from benchmarks import centroid_constrained_fixture_receipt as fixture_receipt
from nova.equilibrium.constraint import ConstraintContext
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.observation import (
    MomentIntegralSupport,
    observe_current_moments,
    recover_physical_first_moments,
)
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.centroid_row import centroid_constraint_pair

configure_dtypes()


def test_within_cell_moments_move_both_centroids_and_keep_a_tangent():
    second = jnp.asarray(((2.0, 3.0, 0.5), (4.0, 5.0, -0.25)))
    radial = jnp.asarray((0.2, -0.1))
    vertical = jnp.asarray((0.3, 0.4))
    physical_r, physical_z = recover_physical_first_moments(second, radial, vertical)
    expected_r = second[:, 0] * radial + second[:, 2] * vertical
    expected_z = second[:, 2] * radial + second[:, 1] * vertical
    np.testing.assert_allclose(physical_r, expected_r)
    np.testing.assert_allclose(physical_z, expected_z)

    def centroid(coupled_radial):
        first_r, first_z = recover_physical_first_moments(
            second, coupled_radial, vertical
        )
        return observe_current_moments(
            jnp.asarray((2.0, 3.0)),
            jnp.asarray(((1.0, 0.0), (2.0, 1.0))),
            radial_moment=first_r,
            vertical_moment=first_z,
            core_mask=jnp.asarray((True, False)),
            support=MomentIntegralSupport.CONFINED_CORE,
        ).centroid_r

    np.testing.assert_allclose(centroid(radial), 1.0 + expected_r[0] / 2.0)
    np.testing.assert_allclose(jax.grad(centroid)(radial), (1.0, 0.0))


@pytest.fixture(scope="module")
def analytic_row():
    context = stiffness._build_context(
        "weak-rotation-reactor-static", -110, clip_mode="exact"
    )
    profile = ForwardProfile(
        context["operator"],
        StencilMesh(
            context["machine"].node,
            context["machine"].stencil,
            context["machine"].area,
        ),
    )
    pair = centroid_constraint_pair(
        context["current_centroid"][:1],
        pitch=float(np.sqrt(np.median(np.asarray(context["machine"].area)))),
        components=("centroid_r",),
        analytic_profile=profile,
        analytic_flux=context["analytic"],
        requested_class=stiffness.REQUESTED_CLASS,
        target_current=context["target_current"],
    )
    return context, profile, pair


def test_analytic_first_moment_centroid_has_derived_tolerance(analytic_row):
    context, profile, pair = analytic_row
    current, *_ = profile._integral_state(
        context["analytic"],
        requested_class=stiffness.REQUESTED_CLASS,
        target_current=context["target_current"],
    )
    assert len(np.asarray(current.cell_current)) == 135
    centre_only = np.sum(
        np.asarray(current.cell_current)
        * np.asarray(profile.operator.grid.coordinate)[:, 0]
    ) / np.sum(np.asarray(current.cell_current))
    target = float(np.asarray(pair.binding.target)[0])
    tolerance = float(np.asarray(pair.binding.tolerance)[0])
    analytic = float(np.asarray(pair.binding.payload)[0])
    analytic_observed = pair.functional.observed(
        profile,
        ConstraintContext(
            flux=jnp.asarray(context["analytic"]),
            requested_class=stiffness.REQUESTED_CLASS,
            target_current=context["target_current"],
            shadow=None,
        ),
        pair.binding.payload,
    )
    np.testing.assert_allclose(
        analytic_observed, pair.binding.payload, rtol=0.0, atol=0.0
    )
    assert abs(centre_only - target) > 6.0e-3
    assert 5.0e-5 < abs(analytic - target) < 1.0e-4
    np.testing.assert_allclose(
        tolerance, abs(analytic - target), rtol=0.0, atol=1.0e-14
    )
    assert abs(float(np.asarray(analytic_observed)[0]) - target) <= tolerance

    displacement = np.asarray((2.0 * np.sign(analytic - target) * tolerance, 0.0))
    displaced = stiffness._translated_state(context, displacement)
    observed = pair.functional.observed(
        profile,
        ConstraintContext(
            flux=jnp.asarray(displaced),
            requested_class=stiffness.REQUESTED_CLASS,
            target_current=context["target_current"],
            shadow=None,
        ),
        pair.binding.payload,
    )
    assert abs(float(np.asarray(observed)[0]) - target) > tolerance


def test_certificate_pair_carries_analytic_first_moment_tolerance():
    context = fixture_receipt._context(
        "weak-rotation-reactor-static", -110, clip_mode="exact"
    )
    (pair,) = fixture_receipt._certificate_pairs(context, level=False)
    assert len(np.asarray(context["profile"].operator.grid.coordinate)) == 135
    analytic = pair.functional.observed(
        context["profile"],
        ConstraintContext(
            flux=jnp.asarray(context["analytic"]),
            requested_class=context["requested_class"],
            target_current=context["target_current"],
            shadow=None,
        ),
        pair.binding.payload,
    )
    np.testing.assert_allclose(pair.binding.payload, analytic, rtol=0.0, atol=0.0)
    residual = np.abs(np.asarray(analytic) - np.asarray(pair.binding.target))
    tolerance = np.asarray(pair.binding.tolerance)
    np.testing.assert_allclose(tolerance[0], residual[0], rtol=0.0, atol=1.0e-14)
    assert 5.0e-5 < tolerance[0] < 1.0e-4
