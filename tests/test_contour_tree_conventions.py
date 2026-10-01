"""Topology conventions are explicit rather than inferred from flux extrema."""

from dataclasses import fields
from types import SimpleNamespace

import pytest

from nova.equilibrium.contour_tree import (
    ContourTreeResult,
    FluxCurrentSignError,
    sigma_for_state,
)
from nova.io.cocos import convention


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
def test_sigma_uses_declared_cocos_and_plasma_current(cocos, plasma_current):
    state = _state(cocos, plasma_current)

    current_sign = 1 if plasma_current > 0.0 else -1
    assert sigma_for_state(state) == -current_sign * convention(cocos).sigma_bp


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
