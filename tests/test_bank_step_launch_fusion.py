"""Contracts for the bank step's shared boundary integration."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import inspect

from nova.equilibrium import clip_quadrature, current, support_read


def test_geometry_tangent_uses_shared_boundary_integral():
    assert (
        current.moving_polygon_current_moment_tangent
        is clip_quadrature.moving_polygon_current_moment_tangent
    )
    source = inspect.getsource(support_read._normal_geometry_booking_tangent)
    assert "current.moving_polygon_current_moment_tangent(" in source
    assert "_UNIT_NODE" not in source
    assert "_UNIT_WEIGHT" not in source
