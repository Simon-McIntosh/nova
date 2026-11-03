"""Pinned scaled residual values for the registered constraint rows."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.constraint import (
    ConstraintContext,
    CurrentCentroidConstraint,
    ExternalShafranovConstraint,
    FieldComponentConstraint,
    FluxLevelConstraint,
    IsofluxConstraint,
    WallGapConstraint,
    WallGapTarget,
    XPointConstraint,
)
from nova.equilibrium.observation import CurrentMomentObservation, MomentIntegralSupport
from nova.jax.config import configure_dtypes


@pytest.fixture(scope="module")
def residual_inputs():
    """Return seven base-state rows with the payloads their observations read."""
    configure_dtypes()
    radius = np.linspace(1.0, 2.0, 7)
    height = np.linspace(-1.0, 1.0, 7)
    radius_grid, height_grid = np.meshgrid(radius, height, indexing="ij")
    lattice = SimpleNamespace(
        shape=(7, 7),
        radius=radius,
        height=height,
        radial_step=radius[1] - radius[0],
        vertical_step=height[1] - height[0],
        node_count=49,
    )
    coordinates = np.column_stack((radius_grid.ravel(), height_grid.ravel()))
    flux = jnp.asarray(2.0 * coordinates[:, 0] + 3.0 * coordinates[:, 1])
    profile = SimpleNamespace(lattice=lattice)
    observation = CurrentMomentObservation(
        plasma_current=jnp.asarray(1.0e6),
        centroid_r=jnp.asarray(1.5),
        centroid_z=jnp.asarray(0.25),
        support=MomentIntegralSupport.ALL_DOMAIN,
    )
    profile.current_moment_observation = lambda *_args, **_kwargs: observation
    context = ConstraintContext(flux, None, None, None)

    cases = {
        "centroid": (
            CurrentCentroidConstraint(("centroid_r", "centroid_z")),
            jnp.asarray([4.0, -2.0]),
            jnp.asarray([True, False]),
            jnp.asarray([1.0, -0.25]),
            jnp.asarray([0.5, 2.5]),
            np.asarray([1.0, -2.0]),
        ),
        "flux_level": (
            FluxLevelConstraint(point_count=2),
            jnp.zeros(2),
            jnp.asarray([[1.25, -0.25], [1.5, 0.5]]),
            jnp.asarray([2.0, 4.0]),
            jnp.asarray([0.5, 2.0]),
            np.asarray([-0.5, 0.25]),
        ),
        "isoflux": (
            IsofluxConstraint(point_count=1, reference="reference_point"),
            jnp.zeros(1),
            jnp.asarray([[1.25, -0.25], [1.5, 0.25]]),
            jnp.asarray([0.5]),
            jnp.asarray([2.0]),
            np.asarray([-1.25]),
        ),
        "xpoint": (
            XPointConstraint(),
            jnp.zeros(2),
            jnp.asarray([1.25, -0.25]),
            jnp.asarray([1.0, 1.0]),
            jnp.asarray([2.0, 4.0]),
            np.asarray([0.5, 0.5]),
        ),
        "field": (
            FieldComponentConstraint(("radial", "vertical")),
            jnp.zeros(2),
            jnp.asarray([[1.25, -0.25], [1.5, 0.25]]),
            jnp.asarray([0.0, 0.1]),
            jnp.asarray([0.2, 0.05]),
            np.asarray([-1.9098593171027447, 2.2441318157838754]),
        ),
        "wall_gap": (
            WallGapConstraint(reference="reference_point"),
            jnp.zeros(1),
            WallGapTarget(
                origin=jnp.asarray([1.5, 0.0]),
                direction=jnp.asarray([-1.0, 0.0]),
                gap=jnp.asarray(0.25),
                reference_point=jnp.asarray([1.0, 0.0]),
            ),
            jnp.asarray([0.1]),
            jnp.asarray([0.2]),
            np.asarray([2.0]),
        ),
        "shafranov": (
            ExternalShafranovConstraint(minor_radius=0.4),
            jnp.zeros(1),
            (jnp.asarray(0.6 * coordinates[:, 0]), jnp.asarray(0.4)),
            jnp.asarray([-2.5]),
            jnp.asarray([0.4]),
            np.asarray([-0.8903176005338165]),
        ),
    }
    return profile, context, cases


@pytest.mark.parametrize(
    "case_name",
    ["centroid", "flux_level", "isoflux", "xpoint", "field", "wall_gap", "shafranov"],
)
def test_residual_matches_base_value_and_traces_over_batch(residual_inputs, case_name):
    """Residual values match the base and retain their row axis under JAX transforms."""
    profile, context, cases = residual_inputs
    functional, unknown, payload, target, scale, expected = cases[case_name]

    def residual_for(flux):
        return functional.residual(
            profile,
            context._replace(flux=flux),
            unknown,
            payload,
            target,
            scale,
        )

    actual = residual_for(context.flux)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    transformed = jax.jit(jax.vmap(residual_for))(
        jnp.stack((context.flux, context.flux))
    )
    assert transformed.shape == (2, *expected.shape)
    np.testing.assert_allclose(
        transformed,
        np.broadcast_to(expected, transformed.shape),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
