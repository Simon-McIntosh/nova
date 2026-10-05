"""The analytic limiter encloses the zero-flux plasma on every straight edge."""

# Configure precision before modules that construct numerical defaults.
# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import os
import jax
import numpy as np
import pytest
from scripts.analytic_oracle_fixtures.measure import limiter_contour
from tests.rotating_equilibrium_references import reference_cases

assert jax.config.jax_enable_x64 is True


def _chord_wall(case, points=121, clearance=0.12):
    """Sample the offset curve without compensating straight-edge sagitta."""
    inboard, outboard = case.boundary_midplane_radii()
    angle = 2 * np.pi * (np.arange(points) + 0.5) / points
    radius = (inboard + outboard) / 2 - (outboard - inboard) / 2 * np.cos(angle)
    height = np.sign(np.sin(angle)) * np.sqrt(
        np.clip(case.flux(radius, 0.0) / case.field_coefficient, 0.0, None)
    )
    offset = 1 + clearance * 0.5 * (1 + np.cos(angle))
    return np.column_stack(
        (case.major_radius + offset * (radius - case.major_radius), offset * height)
    )


def _edge_flux(case, wall):
    parameter = np.linspace(0, 1, 129)
    samples = (
        wall[:, None]
        + parameter[None, :, None] * (np.roll(wall, -1, axis=0) - wall)[:, None]
    )
    flux = 2 * np.pi * case.flux(samples[..., 0], samples[..., 1])
    return samples, flux


def _assert_confined(case, wall):
    samples, flux = _edge_flux(case, wall)
    maximum = float(flux.max())
    assert maximum <= 1e-12, f"wall enters plasma: maximum={maximum:.17g} Wb"
    assert abs(maximum) <= 1e-12
    index = np.unravel_index(np.argmax(flux), flux.shape)
    contact = np.asarray((case.boundary_midplane_radii()[1], 0.0))
    np.testing.assert_allclose(samples[index], contact, rtol=0, atol=1e-12)
    edge = wall[[index[0], (index[0] + 1) % len(wall)]]
    np.testing.assert_allclose(edge[:, 0], contact[0], rtol=0, atol=1e-12)
    assert edge[0, 1] * edge[1, 1] < 0
    print(
        f"LIMITER_INVARIANT points={len(wall)} maximum_wb={maximum:.17g} "
        f"contact_edge={index[0]}"
    )


@pytest.mark.parametrize("points", (9, 31, 121))
@pytest.mark.parametrize("kind", ("static", "weak", "moderate"))
def test_limiter_wall_confines_zero_flux_region(kind, points):
    references = reference_cases()
    case = references[
        "moderate-rotation-conventional"
        if kind == "moderate"
        else "weak-rotation-reactor"
    ]
    if kind == "static":
        case = case.static_limit()
    builder = (
        _chord_wall
        if os.environ.get("NOVA_LIMITER_CHORD_MUTATION") == "1"
        else limiter_contour
    )
    wall = builder(case, points=points)
    assert wall.shape == (points, 2)
    _assert_confined(case, wall)


def test_chord_wall_fails_whole_edge_invariant():
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    wall = _chord_wall(case)
    _, flux = _edge_flux(case, wall)
    assert 0.0400 < flux.max() < 0.0402
    with pytest.raises(AssertionError, match="wall enters plasma"):
        _assert_confined(case, wall)
    print(f"CHORD_REFUSAL maximum_wb={flux.max():.17g}")
