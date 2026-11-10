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
from functools import lru_cache
import sys
from pathlib import Path

import numpy as np
import pytest

from nova.biot.completeelliptic import complete_kind, complete_pole
from nova.biot.sheet import MU0, SheetFlux, sheet_field, sheet_flux

import nova.biot.sheet as sheet_module

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
    got = sheet_flux(target.r, target.z, start, end, route="closed")
    relative = abs(got - converged.psi) / abs(converged.psi)
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor at this standoff"
    )


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_closed_form_agrees_with_the_graded_route(angle, station, distance):
    """The two routes agree on ``R``, the one block they do not share.

    ``A``, the arsinh remainder the paper leaves to quadrature, is taken from
    ``_arsinh_gradient`` by both routes, so this comparison cannot fail on it and
    says nothing about it: it checks the closed ``R`` block against the graded
    one, and nothing else.
    """
    start, end, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    closed = sheet_flux(target.r, target.z, start, end, route="closed")
    routed = sheet_flux(target.r, target.z, start, end, route="quadrature")
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
def test_flat_pancake_quadrature_matches_reference(station, distance):
    """The independent panelised radial rule against the ring reference.

    The radial cross-check is held to the reference in its own right as well
    as through its agreement with eq (26).
    """
    start, end, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    converged = reference.sheet_reference(target, start, end)
    routed = sheet_flux(target.r, target.z, start, end, route="quadrature")
    assert abs(routed - converged.psi) <= TOLERANCE * abs(converged.psi)


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_flat_pancake_closed_form_agrees_with_the_quadrature(station, distance):
    """Snow's pancake form and the surface-current integral agree at the floor."""
    start, end, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    closed_form = sheet_flux(target.r, target.z, start, end)
    routed = sheet_flux(target.r, target.z, start, end, route="quadrature")
    relative = abs(closed_form - routed) / abs(routed)
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor"
    )


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_flat_pancake_field_quadrature_matches_reference(station, distance):
    """The kept surface-current integral's field against the reference."""
    _, _, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    assert field_relative(0.0, target, route="quadrature") <= TOLERANCE


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_flat_pancake_field_agrees_with_the_quadrature(station, distance):
    """Eq (27) and the surface-current integral agree on both components."""
    start, end, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    closed_form = sheet_field(target.r, target.z, start, end)
    routed = sheet_field(target.r, target.z, start, end, route="quadrature")
    scale = max(abs(component) for component in routed)
    for got, expected in zip(closed_form, routed, strict=True):
        if abs(expected) <= reference.DEGENERATE * scale:
            continue
        relative = abs(got - expected) / abs(expected)
        assert relative <= TOLERANCE, (
            f"measured {relative:.2e} against the {TOLERANCE:.0e} floor"
        )


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("station", STATIONS)
def test_one_edge_flipped_fails_the_pancake_comparison(station, distance, monkeypatch):
    """Negative control: the two ends' contributions carry opposite signs.

    The sheet's current runs azimuthally, so reversing one edge's sign is the
    mutation that must not reproduce the flux.  The flip is applied to the
    antiderivative as ``_pancake_flux`` calls it, so the assembly under test is
    the module's own two-ends difference rather than a copy of it in this file:
    a regression that adds the ends instead of subtracting them turns this
    control into minus the reference value and fails it.
    """
    start, end, targets = frame(0.0)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    converged = reference.sheet_reference(target, start, end)
    sheet = SheetFlux(start, end)
    _, upper = sheet._pancake_ends()
    original = SheetFlux.pancake_antiderivative

    def flipped(self, gamma, r, rp):
        value = original(self, gamma, r, rp)
        return -value if rp == upper else value

    with monkeypatch.context() as patch:
        patch.setattr(SheetFlux, "pancake_antiderivative", flipped)
        got = sheet.flux(target.r, target.z)
    assert got != pytest.approx(converged.psi, rel=TOLERANCE)


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
        target.r, target.z, start, end, sheet_current=-1.0, route="closed"
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


def measured_field(angle, target, *, route="closed"):
    """The field and the reference at one fixed target."""
    start, end, _ = frame(angle)
    converged = reference.sheet_reference(target, start, end)
    assert converged.converged
    got = sheet_field(target.r, target.z, start, end, route=route)
    return got, (converged.br, converged.bz)


def field_relative(angle, target, *, route="closed"):
    """The largest component error, as a fraction of the larger component.

    A component the symmetry extinguishes is held out: the axisymmetric sheet
    puts no radial field on the mid-plane, so there the reference returns a
    rounding residual rather than a value to divide by.
    """
    (got_r, got_z), (ref_r, ref_z) = measured_field(angle, target, route=route)
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
    relative = field_relative(angle, target, route="quadrature")
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor at this standoff"
    )


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_sloped_closed_form_field_matches_reference(angle, station, distance):
    """The closed form's own field against the ring reference at the floor."""
    _, _, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    relative = field_relative(angle, target, route="closed")
    assert relative <= TOLERANCE, (
        f"measured {relative:.2e} against the {TOLERANCE:.0e} floor at this standoff"
    )


@pytest.mark.parametrize(("angle", "station", "distance"), SLOPED_PARAMS)
def test_closed_form_field_agrees_with_the_graded_route(angle, station, distance):
    """The two routes agree on ``R``, the one block they do not share.

    ``A`` comes from ``_arsinh_gradient`` on both routes, so a disagreement here
    can only come from the closed ``R`` block's own value or its partials; the
    shared term cancels identically and this test says nothing about it.
    """
    start, end, targets = frame(angle)
    target = next(t for t in targets if t.distance == distance and t.station == station)
    closed = sheet_field(target.r, target.z, start, end, route="closed")
    routed = sheet_field(target.r, target.z, start, end, route="quadrature")
    scale = max(abs(component) for component in routed)
    for got, expected in zip(closed, routed, strict=True):
        if abs(expected) <= reference.DEGENERATE * scale:
            continue
        relative = abs(got - expected) / abs(expected)
        assert relative <= TOLERANCE, (
            f"measured {relative:.2e} against the {TOLERANCE:.0e} floor"
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


_INSTRUMENTED = (
    "antiderivative",
    "antiderivative_closed",
    "_vertical_flux",
    "_pancake_flux",
    "_flat_flux",
    "_sloped_field",
    "_sloped_field_closed",
    "_pancake_field",
    "_flat_field",
)
"""The evaluators that make up the two route families, by name."""

CLOSED_FAMILY = {
    0.0: {"_pancake_flux", "_pancake_field"},
    np.pi / 4: {"antiderivative_closed", "_sloped_field_closed"},
    np.pi / 2: {"_vertical_flux", "_sloped_field_closed"},
}
"""The closed form's own evaluators, by slope."""

QUADRATURE_FAMILY = {
    0.0: {"_flat_flux", "_flat_field"},
    np.pi / 4: {"antiderivative", "_sloped_field"},
    np.pi / 2: {"antiderivative", "_sloped_field"},
}
"""The surface-current cross-check's evaluators, by slope."""


def record_calls(monkeypatch, recorded):
    """Record which of the route evaluators the next call reaches."""
    for name in _INSTRUMENTED:
        original = getattr(SheetFlux, name)

        def recorder(self, *args, _name=name, _original=original, **kwargs):
            recorded.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(SheetFlux, name, recorder)


@pytest.mark.parametrize("angle", (0.0, np.pi / 4, np.pi / 2))
def test_route_flag_selects_one_family_at_every_slope(angle, monkeypatch):
    """The flag names one family, and it names the same one at every slope.

    A flat sheet takes Snow's pancake form, a vertical one eq (24) and a sloped
    one the pole-reduced antiderivative; all three are the closed form, and the
    cross-check is the surface-current quadrature at all three.
    """
    start, end, _ = frame(angle)
    sheet = SheetFlux(start, end)
    recorded = []
    record_calls(monkeypatch, recorded)

    sheet.flux(1.35, 0.4, route="closed")
    sheet.field(1.35, 0.4, route="closed")
    assert set(recorded) == CLOSED_FAMILY[angle]

    recorded.clear()
    sheet.flux(1.35, 0.4, route="quadrature")
    sheet.field(1.35, 0.4, route="quadrature")
    assert set(recorded) == QUADRATURE_FAMILY[angle]


@pytest.mark.parametrize("angle", SLOPES)
def test_route_flag_defaults_to_the_closed_form(angle):
    """The unqualified call is the closed route, whatever the slope."""
    start, end, _ = frame(angle)
    sheet = SheetFlux(start, end)
    assert sheet.flux(1.35, 0.4) == sheet.flux(1.35, 0.4, route="closed")
    assert sheet.field(1.35, 0.4) == sheet.field(1.35, 0.4, route="closed")
    assert sheet_flux(1.35, 0.4, start, end) == sheet.flux(1.35, 0.4, route="closed")


@pytest.mark.parametrize("angle", SLOPES)
def test_an_unknown_route_is_refused(angle):
    """A misspelt route is refused rather than silently resolved to a default."""
    start, end, _ = frame(angle)
    sheet = SheetFlux(start, end)
    with pytest.raises(ValueError):
        sheet.flux(1.35, 0.4, route="graded")
    with pytest.raises(ValueError):
        sheet.field(1.35, 0.4, route="graded")


@pytest.mark.parametrize("angle", SLOPES)
def test_closed_flux_route_evaluates_the_elliptic_form(angle, monkeypatch):
    """Guard: the closed flux route must evaluate the complete elliptic kinds.

    The two routes agree to the floor by design, so values alone cannot tell
    them apart; a closed route replaced by the graded one returns the same
    numbers.  This fails on that substitution at every slope.
    """
    start, end, _ = frame(angle)
    sheet = SheetFlux(start, end)
    calls = []
    original = sheet_module.complete_kind

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sheet_module, "complete_kind", spy)
    sheet.flux(1.35, 0.4, route="closed")
    assert calls, "the closed flux route did not evaluate the elliptic kinds"
    calls.clear()
    sheet.flux(1.35, 0.4, route="quadrature")
    assert not calls, "the quadrature flux route reached the closed form"


@pytest.mark.parametrize("angle", SLOPES)
def test_closed_field_route_evaluates_the_closed_block(angle, monkeypatch):
    """Guard: the closed field route must evaluate the closed block itself.

    A flat sheet's closed field is eq (27) through ``_pancake_field``; a
    non-flat one takes the complete elliptic ``R`` block's own partials through
    ``_closed_root_gradient``.  Replacing either body with the graded route's
    leaves both agreeing to the floor and neither marker reached, so this is the
    assertion that reddens on that substitution.
    """
    start, end, _ = frame(angle)
    sheet = SheetFlux(start, end)
    if angle == 0.0:
        marker, holder, target = "_pancake_field", SheetFlux, "_pancake_field"
    else:
        marker, holder, target = (
            "_closed_root_gradient",
            sheet_module,
            "_closed_root_gradient",
        )
    calls = []
    original = getattr(holder, target)

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(holder, target, spy)
    sheet.field(1.35, 0.4, route="closed")
    assert calls, f"the closed field route did not evaluate {marker}"
    calls.clear()
    sheet.field(1.35, 0.4, route="quadrature")
    assert not calls, f"the quadrature field route reached {marker}"


def test_closed_root_is_the_elliptic_block():
    """``R`` equals eq (22a) formed independently from the complete kinds."""
    height, radius, source = 0.37, 1.61, 1.55
    span = height**2 + (radius + source) ** 2
    k2 = 4.0 * radius * source / span
    first, second = complete_kind(1.0 - k2)
    expected = (
        2.0
        * np.sqrt(span)
        * (2.0 * (1.0 - k2) * float(first) + (k2 - 2.0) * float(second))
        / (3.0 * k2)
    )
    got = float(sheet_module._closed_root(height, radius, source))
    assert got == pytest.approx(expected, rel=1e-14)


def beyond_targets(angle):
    """Targets past each end of the segment and on either side of its line.

    The extension stations of ``target_set`` lie ON the supporting line, where
    the closed form's antiderivative is regular; the far side of that line is
    where ``r_1 + r b_1^2`` turns over.  These offset the extension point
    broadside by each of the section's standoffs in both directions, so the set
    covers both sides of the line beyond both ends.
    """
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([direction[1], -direction[0]])
    start = CENTRE - 0.5 * LENGTH * direction
    end = CENTRE + 0.5 * LENGTH * direction
    targets = []
    for distance in DISTANCES:
        for lateral in (-distance, distance):
            for anchor, sign in ((end, 1.0), (start, -1.0)):
                point = anchor + sign * distance * direction + lateral * normal
                targets.append(
                    reference.Target(
                        distance=float(distance),
                        station="beyond",
                        r=float(point[0]),
                        z=float(point[1]),
                        foot=0.5,
                    )
                )
    return targets


def sheet_pair(angle, target, order="forward", **kwargs):
    """The closed-form flux and field at one target, in the given order."""
    start, end, _ = frame(angle)
    if order == "reversed":
        start, end = end, start
    return (
        sheet_flux(target.r, target.z, start, end, **kwargs),
        sheet_field(target.r, target.z, start, end, **kwargs),
    )


GRID_RADII = tuple(np.linspace(0.05, 3.0, 7))
GRID_HEIGHTS = tuple(np.linspace(-1.0, 1.0, 9))
"""A coarse box around the segment, reaching well past both of its ends."""

GRID_PARAMS = tuple(
    pytest.param(angle, float(r), float(z))
    for angle in SLOPED
    for r in GRID_RADII
    for z in GRID_HEIGHTS
)

BEYOND_PARAMS = tuple(
    pytest.param(angle, side, lateral, distance, order)
    for angle in SLOPED
    for side in ("end", "start")
    for lateral in (-1.0, 1.0)
    for distance in DISTANCES
    for order in ("forward", "reversed")
)
"""Beyond an end on either side of the line, at every standoff and orientation."""

NEAR_FIELD_FLOOR = 1e-9
"""The closed route's floor at the half-cell standoff, where the two ends cancel."""


def beyond_target(angle, side, lateral, distance):
    """A target past one end, offset broadside by ``distance`` to either side."""
    direction = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([direction[1], -direction[0]])
    start = CENTRE - 0.5 * LENGTH * direction
    end = CENTRE + 0.5 * LENGTH * direction
    anchor, sign = (end, 1.0) if side == "end" else (start, -1.0)
    point = anchor + sign * distance * direction + lateral * distance * normal
    return reference.Target(
        distance=float(distance),
        station="beyond",
        r=float(point[0]),
        z=float(point[1]),
        foot=0.5,
    )


def assert_field_matches(got, expected):
    """Both components at the floor, measured against the vector's own scale.

    The field is one vector, so each component is held to ``TOLERANCE`` of the
    larger one.  Dividing a component by itself instead would turn a
    cancellation residue -- the radial component where it is a thousandth of
    the vertical, or an exact symmetry zero -- into an unbounded relative
    requirement on two independently rounded numbers.
    """
    scale = max(abs(component) for component in expected)
    for value, reference_value in zip(got, expected, strict=True):
        assert np.isfinite(value)
        error = abs(value - reference_value)
        assert error <= TOLERANCE * scale, (
            f"measured {error / scale:.2e} of the field scale "
            f"against the {TOLERANCE:.0e} floor"
        )


@pytest.mark.parametrize(
    ("angle", "side", "lateral", "distance", "order"), BEYOND_PARAMS
)
def test_beyond_the_ends_matches_reference(angle, side, lateral, distance, order):
    """Beyond an end, on either side of the line: finite and at the floor.

    The fixed target set places its beyond-the-end targets ON the supporting
    line, where the antiderivative is regular.  Off that line the phi = 0
    curvature scale ``r (r_1 + r b_1^2)`` turns negative, and an unsigned
    square root of it returns NaN for the whole target; the closed form must
    stay finite and correct there on both parameterisations of the segment.
    """
    start, end, _ = frame(angle)
    target = beyond_target(angle, side, lateral, distance)
    converged = reference.sheet_reference(target, start, end)
    assert converged.converged, "the reference did not converge at a beyond-end target"
    flux, (b_r, b_z) = sheet_pair(angle, target, order)
    assert np.isfinite(flux)
    # The antiderivative is differenced between the two ends, and at the
    # half-cell standoff that difference cancels five digits: the pole-reduced
    # route then meets 1e-9 while the graded cross-check still meets 1e-10.
    floor = TOLERANCE if distance > DISTANCES[0] else NEAR_FIELD_FLOOR
    assert abs(flux - converged.psi) <= floor * abs(converged.psi)
    assert_field_matches((b_r, b_z), (converged.br, converged.bz))


@pytest.mark.parametrize(("angle", "r", "z"), GRID_PARAMS)
def test_grid_matches_reference(angle, r, z):
    """The closed form over a box around the segment: finite and at the floor.

    The box reaches from just off the axis to three metres out and past both
    ends of the segment on all four sides, so it crosses the region a cell
    interaction matrix samples rather than only the section's twelve fixed
    targets.
    """
    start, end, _ = frame(angle)
    target = reference.Target(0.0, "grid", r, z, 0.5)
    converged = reference.sheet_reference(target, start, end)
    assert converged.converged, "the reference did not converge on the grid"
    flux = sheet_flux(target.r, target.z, start, end)
    field = sheet_field(target.r, target.z, start, end)
    assert np.isfinite(flux)
    assert abs(flux - converged.psi) <= TOLERANCE * abs(converged.psi)
    assert_field_matches(field, (converged.br, converged.bz))


@pytest.mark.parametrize(
    ("gamma", "radius", "source", "expected"),
    (
        (4.4364, 0.09, 0.53, -5.313724027690613e-05),
        (1.0, 1.0, 0.1, -0.0002774170152049587),
        (0.001, 1.0, 0.1, -1.5886497294809025e-06),
        (1.0, 1.0, 4.0, -1.1576070212467167),
        (0.001, 1.0, 4.0, -3.1382961822532587),
        (1.0, 0.09, 1.0, -0.04961699534410705),
        (4.4, 0.09, 2.0, -0.002274859438413864),
        (4.4, 0.09, 5.0, -0.013589152235584235),
        (0.9999499987499375, 0.01, 0.799999999999, -0.0038294326793218585),
        (0.9999499987499375, 0.01, 1.25000000001, -0.007479483847414431),
        (
            0.0010000000000143778,
            0.999999499999875,
            0.799999999999,
            -0.002480158050361592,
        ),
        (0.0010000000000143778, 0.999999499999875, 1.25000000001, -3.134503422810057),
    ),
)
def test_pancake_endpoint_series_matches_high_precision(
    gamma, radius, source, expected
):
    """Endpoint fields against 70-digit elliptic values, including near the plane.

    The NaN fallback makes this assert that an analytic series actually ran.
    The constants were evaluated from the two complete poles and first kind
    at high precision, independently of the source and near-axis series.
    """
    got, _ = sheet_module._pancake_end_series(np, gamma, radius, source, np.nan, np.nan)
    assert np.isfinite(got)
    assert abs(got - expected) <= 5e-15 * abs(expected)


@pytest.mark.parametrize(
    ("gamma", "radius", "source", "slope", "expected"),
    (
        (4.4, 0.09, 0.1, 2.4, -2.7835585637336792e-06),
        (4.4, 0.09, 0.53, 21.72, -0.02612550735129679),
        (-4.4, 0.09, 0.53, 21.72, 0.025837480921348403),
        (4.4, 0.09, 0.53, -21.72, 0.025837480921348403),
        (1.0, 0.09, 0.1, 0.4, -0.0005666167421711974),
        (-1.0, 0.09, 0.1, -0.4, -0.0005666167421711974),
        (0.1, 0.001, 0.01, 2.4, -0.0001182281393935999),
        (1.0, 0.09, 0.1, -2.4, 0.0009603169702279289),
    ),
)
def test_sloped_endpoint_series_matches_high_precision(
    gamma, radius, source, slope, expected
):
    """The combined derivative against a 60-digit angular-integral reference."""
    got = sheet_module._source_gradient_series(
        np, gamma, radius, source, slope, (np.nan, np.nan, np.nan)
    )[1]
    assert np.isfinite(got)
    assert abs(got - expected) <= 5e-15 * abs(expected)


@pytest.mark.parametrize(
    ("gamma", "radius", "source", "slope"),
    ((0.01, 0.09, 0.001, 2.4), (1.0, 0.09, 1.0, 2.4), (1.0, 0.09, 0.1, 0.0)),
)
def test_sloped_endpoint_series_refuses_an_unqualified_branch(
    gamma, radius, source, slope
):
    """Interior, unconverged-series and cylindrical cases retain their regular form."""
    regular = 3.25
    assert (
        sheet_module._source_gradient_series(
            np, gamma, radius, source, slope, (regular, regular, regular)
        )[1]
        == regular
    )


def test_packed_pancake_retains_the_surface_limit():
    """The endpoint's prescribed surface limit also masks its analytic series."""
    _, radial, _ = sheet_module._sheet_pancake_terms(
        np, np.asarray([5e-13]), np.asarray([1.0]), np.asarray([3.0])
    )
    assert radial[0] == 0.0


@pytest.mark.parametrize(
    ("gamma", "radius", "source", "expected"),
    (
        (4.4364, 0.09, 0.53, 0.0017607374722479433),
        (1.0, 1.0, 0.1, 9.323579255191544e-05),
        (0.001, 1.0, 0.1, -0.0005271571736625841),
        (1.0, 0.09, 0.13, 0.0022131621116677413),
        (0.5, 0.5, 0.1, 0.0007619931080212963),
    ),
)
def test_pancake_height_series_matches_high_precision(gamma, radius, source, expected):
    """The zeta-minus-first-kind endpoint against its 70-digit angular integral."""
    _, height = sheet_module._pancake_end_series(
        np, gamma, radius, source, np.nan, np.nan
    )
    assert abs(height - expected) <= 5e-15 * abs(expected)


@pytest.mark.parametrize(
    ("gamma", "radius", "source", "expected"),
    (
        (0.0337, 0.09, 4.9565, 0.021364957290125615),
        (-0.0463, 0.09, 4.984, -0.029190417369569316),
        (0.001, 1.0, 2.0, 0.0019566157278257212),
        (1.0, 1.0, 2.0, 1.5904821201104593),
        (4.4, 0.09, 0.53, 3.1190603443383114),
        (1e-8, 1.0, 0.79, -2.88509515615868511446967726e-8),
        (-1e-8, 1.0, 0.79, 2.88509515615868511446967726e-8),
        (0.001, 1.0, 0.01, -1.57097070286330553469883048e-7),
        (0.03, 0.1, 0.079, -0.487153254288254873083469693),
        (0.01, 1.0, 0.801, -0.0311944426093918782142417924),
        (0.1, 1.0, 1.2, 0.637341574282294800577587285),
    ),
)
def test_cylinder_height_endpoint_matches_high_precision(
    gamma, radius, source, expected
):
    """Cylinder endpoints against complete elliptic values at 60 to 70 digits."""
    height = sheet_module._sheet_cylinder_height(np, gamma, radius, source)
    assert abs(height - expected) <= 5e-15 * abs(expected)


AXIS_WITNESSES = (
    ((0.0, -0.5), (0.0, -0.4), (0.04913072, -0.44836919)),
    ((0.0, -0.5), (0.003067942, -0.499990588), (0.04913072, -0.44836919)),
    ((0.2, -0.5), (0.3, -0.4), (0.0, 0.2)),
    ((0.0, -0.5), (0.3, -0.5), (0.1, 0.2)),
    ((0.2, 0.2), (0.3, 0.3), (0.0, 0.0)),
)


@lru_cache(maxsize=1)
def _compiled_axis_pair():
    import jax
    import jax.numpy as jnp
    from nova.biot.sheet import packed_sheet_field

    def evaluate(values):
        return jnp.stack(packed_sheet_field(jnp, *[x[None] for x in values])).ravel()

    return jax.jit(evaluate), jax.jit(jax.jacfwd(evaluate))


@pytest.mark.parametrize("start,end,point", AXIS_WITNESSES)
def test_axis_sheet_limits_match_graded_reference(start, end, point):
    """Zero source radii and axis targets take their finite, disjoint limits."""
    import jax.numpy as jnp

    target = reference.Target(0.0, "axis", *point, 0.5)
    expected = reference.sheet_reference(target, np.array(start), np.array(end))
    assert expected.converged
    wanted = np.array([expected.br, expected.bz])
    scale = np.linalg.norm(wanted)
    scalar = np.array(sheet_field(*point, start, end))
    packed = np.asarray(_compiled_axis_pair()[0](jnp.array((*point, *start, *end))))
    assert np.isfinite(scalar).all() and np.isfinite(packed).all()
    for got in (scalar, packed):
        assert np.linalg.norm(got - wanted) <= TOLERANCE * scale
    flux = sheet_flux(*point, start, end)
    assert abs(flux - expected.psi) <= TOLERANCE * abs(expected.psi)
    if start[0] == end[0] == 0:
        np.testing.assert_array_equal(scalar, 0)
        np.testing.assert_array_equal(packed, 0)
        assert flux == 0
    if point[0] == 0:
        assert scalar[0] == packed[0] == flux == 0


@pytest.mark.parametrize("radius", [1e-3, 1e-5, 1e-7])
@pytest.mark.parametrize("limit", ["axis_edge", "axis_end", "axis_target"])
def test_axis_approaches_match_graded_reference(radius, limit, capsys):
    """The finite-radius approach preserves small fields and fluxes."""
    import jax.numpy as jnp

    if limit == "axis_edge":
        start, end, point = (radius, -0.5), (radius, -0.4), (0.04913072, -0.44836919)
    elif limit == "axis_end":
        start, end, point = (radius, -0.5), (0.3, -0.4), (0.04913072, -0.44836919)
    else:
        start, end, point = (0.2, -0.5), (0.3, -0.4), (radius, 0.2)
    expected = reference.sheet_reference(
        reference.Target(0.0, "approach", *point, 0.5), np.array(start), np.array(end)
    )
    assert expected.converged
    wanted = np.array([expected.br, expected.bz])
    reference_flux = expected.psi
    if limit == "axis_edge":
        precise = reference.graded_precise_ring_reference(start, end, point)
        reference_flux, *components = precise
        wanted = np.array(components)
    scalar = np.array(sheet_field(*point, start, end))
    packed = np.asarray(_compiled_axis_pair()[0](jnp.array((*point, *start, *end))))
    for got in (scalar, packed):
        assert np.linalg.norm(got - wanted) <= TOLERANCE * np.linalg.norm(wanted)
    flux = sheet_flux(*point, start, end)
    assert abs(flux - reference_flux) <= TOLERANCE * abs(reference_flux)
    scalar_error = np.linalg.norm(scalar - wanted) / np.linalg.norm(wanted)
    packed_error = np.linalg.norm(packed - wanted) / np.linalg.norm(wanted)
    with capsys.disabled():
        print(
            f"\n{limit} R={radius:g}: "
            f"scalar_relative={scalar_error:.3e}, "
            f"packed_relative={packed_error:.3e}, "
            f"flux_relative={abs(flux / reference_flux - 1):.3e}",
            flush=True,
        )


def test_axis_and_off_axis_endpoints_share_the_singular_domain():
    """Both endpoint approaches diverge outside the positive-distance contract."""
    from shapely import LineString, Point

    for radius, coefficient in ((0.0, MU0 / 2), (0.2, MU0 / (2 * np.pi))):
        start, end = (radius, 0.0), (radius + 0.3, 0.0)
        edge = LineString([start, end])
        assert Point(start).distance(edge) == 0
        values = []
        for distance in (1e-3, 1e-4, 1e-5):
            point = (radius, distance)
            assert Point(point).distance(edge) > 0
            values.append(sheet_field(*point, start, end)[1])
        assert np.isfinite(values).all()
        assert np.all(np.diff(values) > 0)
        measured = (values[-1] - values[-2]) / np.log(10)
        assert abs(measured / coefficient - 1) < 0.01


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("start,end,point", AXIS_WITNESSES[:4])
def test_axis_limits_keep_inactive_gradient_paths_finite(start, end, point):
    """Both branches of packed axis selections remain traceable."""
    import jax.numpy as jnp

    values = jnp.array((*point, *start, *end))
    assert np.isfinite(_compiled_axis_pair()[1](values)).all()


@pytest.mark.timeout(900)
def test_shell_oblique_radial_magnetisation():
    """A distant oblique probe resolves the weak radial cell response."""
    import jax.numpy as jnp
    from nova.biot.tiledassembly import cell_edges

    polygon = np.array(
        [
            [0.04, 0.9600000000000017],
            [0.0, 0.9600000000000017],
            [0.0, 1.0],
            [0.003067956762966115, 0.9999952938095762],
            [0.0061358846491540485, 0.9999811752826011],
            [0.009203754782059715, 0.9999576445519639],
            [0.012271538285720144, 0.9999247018391445],
            [0.015339206284987753, 0.9998823474542126],
            [0.018406729905804796, 0.9998305817958234],
            [0.021474080275468918, 0.9997694053512153],
            [0.02454122852291202, 0.9996988186962042],
            [0.027608145778965795, 0.9996188224951786],
            [0.030674803176636116, 0.9995294175010931],
            [0.0337411718513774, 0.9994306045554617],
            [0.036807222941358964, 0.9993223845883495],
            [0.03987292758773938, 0.9992047586183639],
            [0.04, 0.9991994925876344],
        ]
    )
    edges = cell_edges((polygon * 1024,))
    point = (3.0, 1.0)
    scalar = np.zeros(2)
    packed = np.zeros(2)
    wanted = np.zeros(2)
    for i, coefficient in enumerate(edges.coefficient_r[0]):
        start = np.array([edges.start_r[i], edges.start_z[i]]) / 1024
        end = np.array([edges.end_r[i], edges.end_z[i]]) / 1024
        expected = reference.graded_precise_ring_reference(start, end, point)[1:]
        got = np.array(sheet_field(*point, start, end))
        traced = np.asarray(_compiled_axis_pair()[0](jnp.array((*point, *start, *end))))
        for value in (got, traced):
            np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-28)
        scalar += coefficient * got
        packed += coefficient * traced
        wanted += coefficient * expected
    assert wanted[0] == pytest.approx(9.80715345e-15, rel=1e-9)
    np.testing.assert_allclose(scalar, wanted, rtol=1e-10, atol=1e-28)
    np.testing.assert_allclose(packed, wanted, rtol=1e-10, atol=1e-28)

    derivative = _compiled_axis_pair()[1](jnp.array((*point, *start, *end)))
    assert np.isfinite(derivative).all()


def test_sheet_remainder_resolves_both_azimuthal_layers():
    """The ring-distance layer and the opposite pole both remain resolved."""
    import jax.numpy as jnp

    point = (0.10058128716025194, -0.9071429134838774)
    start = (0.0964551824609375, -0.8948164045078125)
    end = (0.09919998656445313, -0.8945162730019531)
    wanted = reference.graded_precise_ring_reference(start, end, point)[1:]
    scalar = np.array(sheet_field(*point, start, end))
    packed = np.asarray(_compiled_axis_pair()[0](jnp.array((*point, *start, *end))))
    for value in (scalar, packed):
        np.testing.assert_allclose(value, wanted, rtol=1e-10, atol=1e-25)
