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

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.sheet import MU0, SheetFlux, sheet_field, sheet_flux

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


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_sloped_flux_closed_form_matches_reference(angle, station, distance):
    """The pole-reduced azimuthal integral against the ring reference."""
    start, end, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    converged = reference.sheet_reference(target, start, end)
    assert converged.converged
    got = sheet_flux(target.r, target.z, start, end, closed=True)
    relative = abs(got - converged.psi) / abs(converged.psi)
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor at this standoff"
    )


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_closed_form_agrees_with_the_graded_route(angle, station, distance):
    """The two independent evaluations of one antiderivative agree."""
    start, end, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    closed = sheet_flux(target.r, target.z, start, end, closed=True)
    routed = sheet_flux(target.r, target.z, start, end)
    assert abs(closed - routed) <= TOLERANCE * abs(routed)


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


@pytest.mark.parametrize("angle", SLOPES)
def test_flipped_sheet_current_fails_the_closed_form(angle):
    """Negative control on the pole-reduced route, as on the graded one."""
    start, end, targets = frame(angle)
    target = next(
        t for t in targets if t.distance == FAR[0] and t.station == TARGET_STATION
    )
    converged = reference.sheet_reference(target, start, end)
    flipped = sheet_flux(
        target.r, target.z, start, end, sheet_current=-1.0, closed=True
    )
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


def measured_field(angle, target):
    """The closed-form field and the reference at one fixed target."""
    start, end, _ = frame(angle)
    converged = reference.sheet_reference(target, start, end)
    assert converged.converged
    return sheet_field(target.r, target.z, start, end), (converged.br, converged.bz)


def field_relative(angle, target):
    """The largest component error, as a fraction of the larger component.

    A component the symmetry extinguishes is held out: the axisymmetric sheet
    puts no radial field on the mid-plane, so there the reference returns a
    rounding residual rather than a value to divide by.
    """
    (got_r, got_z), (ref_r, ref_z) = measured_field(angle, target)
    scale = max(abs(ref_r), abs(ref_z))
    worst = 0.0
    for got, expected in ((got_r, ref_r), (got_z, ref_z)):
        if abs(expected) <= reference.DEGENERATE * scale:
            assert abs(got) <= reference.DEGENERATE * scale
            continue
        worst = max(worst, abs(got - expected) / abs(expected))
    return worst


def snow_end(r, z, rp, zz):
    """Part II eq (25): one end of the thin sheet cylinder, per unit length."""
    gamma = zz - z
    a = np.sqrt((r + rp) ** 2 + gamma**2)
    complement = 1.0 - 4.0 * r * rp / a**2
    first, second = complete_kind(complement)
    pole = (r - rp) ** 2 / (r + rp) ** 2
    through = complete_pole(pole, complement)
    radial = a / (2.0 * r) * ((1.0 + complement) * first - 2.0 * second)
    height = gamma / a * (first + (rp - r) / (rp + r) * through)
    return radial, height


def snow_cylinder(r, z, start, end):
    """Part II eq (25): the thin sheet cylinder's field at ``(r, z)``."""
    scale = MU0 / (2.0 * np.pi)
    far = snow_end(r, z, float(end[0]), float(end[1]))
    near = snow_end(r, z, float(start[0]), float(start[1]))
    return scale * (far[0] - near[0]), scale * (far[1] - near[1])


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_sloped_field_matches_reference(angle, station, distance):
    """A sloped segment's field against the ring reference at the floor."""
    _, _, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    relative = field_relative(angle, target)
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor at this standoff"
    )


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_flat_pancake_field_matches_reference(station, distance):
    """The flat annulus field at every standoff."""
    _, _, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    assert field_relative(0.0, target) <= TOLERANCE


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_cylinder_field_matches_snow_form(station, distance):
    """The ``b_1 = 0`` field against Part II eq (25), an independent anchor."""
    start, end, targets = frame(np.pi / 2)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    got = sheet_field(target.r, target.z, start, end)
    expected = snow_cylinder(target.r, target.z, start, end)
    for value, reference_value in zip(got, expected, strict=True):
        assert value == pytest.approx(reference_value, rel=TOLERANCE, abs=1e-18)


@pytest.mark.parametrize("angle", SLOPES)
def test_flipped_sheet_current_fails_the_field_comparison(angle):
    """Negative control: a flipped sheet current misses reference and eq (25)."""
    start, end, targets = frame(angle)
    target = next(
        t for t in targets if t.distance == FAR[0] and t.station == TARGET_STATION
    )
    converged = reference.sheet_reference(target, start, end)
    expected = (converged.br, converged.bz)
    flipped = sheet_field(target.r, target.z, start, end, sheet_current=-1.0)
    scale = max(abs(component) for component in expected)
    for value, reference_value in zip(flipped, expected, strict=True):
        if abs(reference_value) <= reference.DEGENERATE * scale:
            continue
        assert abs(value - reference_value) > TOLERANCE * abs(reference_value)
    if angle == np.pi / 2:
        snow = snow_cylinder(target.r, target.z, start, end)
        snow_scale = max(abs(component) for component in snow)
        for value, snow_value in zip(flipped, snow, strict=True):
            if abs(snow_value) <= reference.DEGENERATE * snow_scale:
                continue
            assert abs(value - snow_value) > TOLERANCE * abs(snow_value)
