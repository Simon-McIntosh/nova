"""Sheet flux of a conic toroidal current sheet against the ring reference.

The reference is ``scripts/edge_sheet/reference.py``: a Gauss-Legendre integral
of the exact ring Green's function along the segment, panel-bisected until two
successive refinements agree to 1e-13 relative at the target.  It is imported by
path because it is a prototype script rather than part of the package.  The
target set is the section's own: half a cell width, one cell width, one segment
length and ten segment lengths away from the middle and from each end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nova.biot.sheet import SheetFlux, sheet_flux

REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "edge_sheet" / "reference.py"
)

_spec = importlib.util.spec_from_file_location("edge_sheet_reference", REFERENCE_PATH)
reference = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reference
_spec.loader.exec_module(reference)

TOLERANCE = 1e-10
"""The section's relative accuracy floor for the closed-form sheet flux."""

CENTRE = np.array([1.5, 0.0])
LENGTH = 0.1
CELL_WIDTH = 0.02
DISTANCES = (0.5 * CELL_WIDTH, CELL_WIDTH, LENGTH, 10.0 * LENGTH)
FAR = DISTANCES[3:]
"""The section's four standoffs: half a cell, a cell, a segment, ten segments."""

SLOPES = (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2)
"""Inclination of the segment to the radial direction: flat to cylindrical."""

SLOPED = SLOPES[1:-1]
STATIONS = ("middle", "start", "end")
TARGET_STATION = "middle"

SLOPED_PARAMS = tuple(
    pytest.param(angle, station, distance)
    for angle in SLOPED
    for distance in DISTANCES
    for station in STATIONS
)


def frame(angle):
    """The segment's endpoints and the section's target set at ``angle``."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([direction[1], -direction[0]])
    start = CENTRE - 0.5 * LENGTH * direction
    end = CENTRE + 0.5 * LENGTH * direction
    targets = reference.target_set(
        CENTRE, direction, normal, LENGTH, cell_width=CELL_WIDTH, distances=DISTANCES
    )
    return start, end, targets


def measured(angle, target):
    """The closed form and the reference at one fixed target."""
    start, end, _ = frame(angle)
    converged = reference.sheet_reference(target, start, end)
    assert converged.converged
    return sheet_flux(target.r, target.z, start, end), converged.psi


def holds_the_floor(angle, target):
    """Whether the closed form meets the floor, judged relative only.

    ``pytest.approx`` carries a default absolute tolerance of 1e-12, which on
    fluxes of order 1e-7 is five orders of magnitude looser than the relative
    floor; the comparison is therefore written out.
    """
    got, expected = measured(angle, target)
    return abs(got - expected) <= TOLERANCE * abs(expected)


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_sloped_flux_matches_reference(angle, station, distance):
    """A sloped segment against the ring reference at the section's floor."""
    _, _, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    got, expected = measured(angle, target)
    relative = abs(got - expected) / abs(expected)
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor at this standoff"
    )


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_flat_pancake_form_matches_reference(station, distance):
    """The flat annulus, Part II eq (26), at every standoff."""
    _, _, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    assert holds_the_floor(0.0, target)


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_vertical_cylinder_form_matches_reference(station, distance):
    """The ``b_1 = 0`` Snow form, Part II eq (24), at every standoff."""
    _, _, targets = frame(np.pi / 2)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    assert holds_the_floor(np.pi / 2, target)


@pytest.mark.parametrize("angle", SLOPES)
def test_flipped_sheet_current_fails_the_comparison(angle):
    """Negative control: the flux is linear in the sheet current's sign."""
    start, end, targets = frame(angle)
    target = next(
        t for t in targets if t.distance == FAR[0] and t.station == TARGET_STATION
    )
    converged = reference.sheet_reference(target, start, end)
    flipped = sheet_flux(target.r, target.z, start, end, sheet_current=-1.0)
    assert flipped != pytest.approx(converged.psi, rel=TOLERANCE)


def test_flux_is_linear_in_sheet_current():
    start, end, _ = frame(np.pi / 4)
    unit = SheetFlux(start, end).flux(1.5, 0.5)
    doubled = SheetFlux(start, end, sheet_current=2.0).flux(1.5, 0.5)
    assert doubled == pytest.approx(2.0 * unit, rel=1e-14)


@pytest.mark.parametrize("angle", SLOPES)
def test_segment_order_does_not_change_the_flux(angle):
    start, end, _ = frame(angle)
    forward = sheet_flux(1.4, 0.3, start, end)
    backward = sheet_flux(1.4, 0.3, end, start)
    assert backward == pytest.approx(forward, rel=1e-14)
