"""Sheet flux of a conic toroidal current sheet against the ring reference.

The reference is ``scripts/edge_sheet/reference.py``: a Gauss-Legendre integral
of the exact ring Green's function along the segment, panel-bisected until two
successive refinements agree to 1e-13 relative at the target.  It is imported by
path because it is a prototype script rather than part of the package.
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

CENTRE = np.array([1.0, -0.5])
"""Segments are unit length and centred here, clear of every target."""

SLOPES = (0.0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2)
"""Inclination of the segment to the radial direction: flat to cylindrical."""

TARGETS = (
    (0.6, 0.0),
    (1.05, 0.0),
    (1.35, 0.25),
    (0.3, 0.4),
    (2.0, 0.7),
)
"""Fixed targets: inside, just outside, well outside, and far away."""


def segment(angle):
    """A unit-length segment through ``CENTRE`` inclined at ``angle``."""
    direction = 0.5 * np.array([np.cos(angle), np.sin(angle)])
    return CENTRE - direction, CENTRE + direction


def _reference_flux(start, end, r, z):
    target = reference.Target(distance=abs(z), station="middle", r=r, z=z, foot=0.5)
    return reference.sheet_reference(target, start, end)


@pytest.mark.parametrize("angle", SLOPES)
@pytest.mark.parametrize(("r", "z"), TARGETS)
def test_sloped_flux_matches_reference(angle, r, z):
    """Every inclination reproduces the ring reference to the section's floor."""
    start, end = segment(angle)
    converged = _reference_flux(start, end, r, z)
    assert converged.converged
    measured = sheet_flux(r, z, start, end)
    assert measured == pytest.approx(converged.psi, rel=TOLERANCE)


@pytest.mark.parametrize("angle", SLOPES)
@pytest.mark.parametrize(("r", "z"), TARGETS)
def test_flipped_sheet_current_fails_the_comparison(angle, r, z):
    """Negative control: the flux is linear in the sheet current's sign."""
    start, end = segment(angle)
    converged = _reference_flux(start, end, r, z)
    flipped = sheet_flux(r, z, start, end, sheet_current=-1.0)
    assert flipped != pytest.approx(converged.psi, rel=TOLERANCE)


def test_cylinder_snow_form_matches_reference():
    """The ``b_1 = 0`` Snow form, Part II eq (24), against the reference."""
    start, end = np.array([1.2, -0.5]), np.array([1.2, 0.5])
    for r, z in TARGETS:
        converged = _reference_flux(start, end, r, z)
        assert converged.converged
        assert sheet_flux(r, z, start, end) == pytest.approx(
            converged.psi, rel=TOLERANCE
        )


def test_pancake_form_matches_reference():
    """The flat annulus, Part II eq (26), against the reference."""
    start, end = np.array([0.5, -0.5]), np.array([1.5, -0.5])
    for r, z in TARGETS:
        converged = _reference_flux(start, end, r, z)
        assert converged.converged
        assert sheet_flux(r, z, start, end) == pytest.approx(
            converged.psi, rel=TOLERANCE
        )


def test_flux_is_linear_in_sheet_current():
    start, end = segment(np.pi / 4)
    unit = SheetFlux(start, end).flux(1.05, 0.0)
    doubled = SheetFlux(start, end, sheet_current=2.0).flux(1.05, 0.0)
    assert doubled == pytest.approx(2.0 * unit, rel=1e-14)


@pytest.mark.parametrize("angle", SLOPES)
def test_segment_order_does_not_change_the_flux(angle):
    start, end = segment(angle)
    forward = sheet_flux(1.05, 0.0, start, end)
    backward = sheet_flux(1.05, 0.0, end, start)
    assert backward == pytest.approx(forward, rel=1e-14)
