"""Polarity coverage for the forward-match boundary helpers.

A solved flux field is extremal at the magnetic axis, but which extremum that
is depends on the sign convention the field was written in: with the boundary
flux above the axis flux the axis is the grid minimum, and with the boundary
flux below it the axis is the grid maximum.  Both orderings are live upstream
(``nova/equilibrium/topology.py`` resolves ``axis_flux >= boundary_flux`` in
either direction), so a helper that only handles one of them locates a grid
corner for the other and silently returns an empty boundary where a closed
one exists.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_forward_gs_match.py"
SPEC = importlib.util.spec_from_file_location("diiid_forward_gs_match", MODULE_PATH)
match = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = match
SPEC.loader.exec_module(match)

RADIUS = np.linspace(3.0, 7.0, 129)
HEIGHT = np.linspace(-2.0, 2.0, 129)
AXIS_RZ = np.array([5.0, 0.0])

# (field sign, closed-contour level, axis value) for the two live orderings.
POLARITIES = ((1.0, 0.9, 0.0), (-1.0, -0.9, 0.0))


def _analytic_field(sign: float) -> np.ndarray:
    """Return a field that is extremal at ``AXIS_RZ`` with the given sign."""

    grid_r, grid_z = np.meshgrid(RADIUS, HEIGHT, indexing="ij")
    return sign * ((grid_r - AXIS_RZ[0]) ** 2 + grid_z**2)


def test_grid_axis_is_located_for_both_flux_polarities() -> None:
    for sign, level, axis_value in POLARITIES:
        located = match._grid_axis_rz(
            RADIUS, HEIGHT, _analytic_field(sign), axis_value, level
        )
        assert located.shape == (2,)
        assert np.array_equal(located, AXIS_RZ), (
            f"sign {sign}: located {located} instead of the axis {AXIS_RZ}"
        )


def test_separatrix_returns_a_closed_boundary_for_both_flux_polarities() -> None:
    for sign, level, axis_value in POLARITIES:
        boundary = match._separatrix(
            RADIUS, HEIGHT, _analytic_field(sign), axis_value, level
        )
        assert boundary.shape[1] == 2, (
            f"sign {sign}: boundary shape {boundary.shape} is not an (N, 2) loop"
        )
        assert len(boundary) > 4, (
            f"sign {sign}: boundary has {len(boundary)} vertices where a closed "
            "contour exists"
        )
        assert np.all(np.isfinite(boundary))


def test_separatrix_returns_the_empty_shape_when_the_level_is_never_reached() -> None:
    boundary = match._separatrix(RADIUS, HEIGHT, _analytic_field(1.0), 0.0, 100.0)
    assert boundary.shape == (0, 2)
