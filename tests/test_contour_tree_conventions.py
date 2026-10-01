"""Topology conventions are explicit rather than inferred from flux extrema."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from nova.equilibrium.contour_tree import (
    ContourTreeResult,
    FluxCurrentSignError,
    sigma_for_state,
)
from nova.io.cocos import convention, transform_factor
from nova.jax.config import configure_dtypes

_CARRIER_SIZE = 81
_RADIAL_SPAN = (0.7, 1.3)
_VERTICAL_SPAN = (-0.3, 0.3)
_VERTICAL_WEIGHT = 1.4
_AXIS_RADIUS = 1.0
_AXIS_HEIGHT = 0.0


def _solovev_carrier(size: int = _CARRIER_SIZE):
    """Return the coarse cell carrier a topology read never rasterises onto."""

    radial = np.linspace(*_RADIAL_SPAN, size)
    vertical = np.linspace(*_VERTICAL_SPAN, size)
    radius, height = np.meshgrid(radial, vertical, indexing="ij")
    return radius, height


def _solovev_flux(radius, height):
    """Return analytic Solov'ev-like poloidal flux with an extremum at the axis."""

    distance = (radius - _AXIS_RADIUS) ** 2 + _VERTICAL_WEIGHT * (
        height - _AXIS_HEIGHT
    ) ** 2
    return -distance


def _raw_flux_on_carrier(cocos: int, plasma_current: float, radius, height):
    """Return one physical flux written in the state's own convention.

    The analytic extremum is expressed in the declared convention through the
    COCOS poloidal-flux factor between conventions, then flipped with the current
    direction, so the raw ordering is fixed by physics rather than by the reader
    under test.
    """

    convention_factor = transform_factor("psi_like", source=17, target=cocos)
    current_sign = 1 if plasma_current > 0.0 else -1
    return current_sign * convention_factor * _solovev_flux(radius, height)


def _state(cocos: int, plasma_current: float) -> SimpleNamespace:
    """Return a sign-consistent raw-flux state for one declared convention."""

    sigma_bp = convention(cocos).sigma_bp
    flux_step = (1.0 if plasma_current > 0.0 else -1.0) * sigma_bp
    return SimpleNamespace(
        cocos=cocos,
        plasma_current=plasma_current,
        psi_magnetic_axis=0.0,
        boundary_psi=flux_step,
    )


@pytest.mark.parametrize(
    ("cocos", "plasma_current"),
    [(11, 3.0), (11, -3.0), (17, 3.0), (17, -3.0)],
)
def test_sigma_orients_axis_as_signed_flux_maximum(cocos, plasma_current):
    """The axis is a local maximum of sigma*psi on the physical carrier."""

    configure_dtypes()
    radius, height = _solovev_carrier()
    psi = _raw_flux_on_carrier(cocos, plasma_current, radius, height)

    axis = np.unravel_index(np.argmax(_solovev_flux(radius, height)), psi.shape)
    state = SimpleNamespace(
        cocos=cocos,
        plasma_current=plasma_current,
        psi_magnetic_axis=float(psi[axis]),
        boundary_psi=float(psi[0, 0]),
    )

    sigma = sigma_for_state(state)
    signed_axis = sigma * float(psi[axis])
    for step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = (axis[0] + step[0], axis[1] + step[1])
        assert signed_axis > sigma * float(psi[neighbour])


def test_sigma_refuses_current_flux_sign_disagreement():
    state = _state(17, 3.0)
    state.plasma_current *= -1.0

    with pytest.raises(FluxCurrentSignError, match="current sign"):
        sigma_for_state(state)


def test_result_round_trips_declared_dd_fields():
    result = ContourTreeResult(
        node=((2, 6.2, 0.0, 1.2), (1, 5.8, -1.1, 0.7)),
        edges=((0, 1),),
        boundary_type=1,
        boundary_psi=0.7,
        psi_magnetic_axis=1.2,
        closest_wall_point=(5.5, -1.4),
    )

    assert tuple(field.name for field in fields(result)) == (
        "node",
        "edges",
        "boundary_type",
        "boundary_psi",
        "psi_magnetic_axis",
        "closest_wall_point",
    )
    assert result.node_capacity == 256
    assert result.edge_capacity == 255
    payload = {field.name: getattr(result, field.name) for field in fields(result)}
    assert ContourTreeResult(**payload) == result
