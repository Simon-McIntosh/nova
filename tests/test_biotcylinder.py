"""Validate the dataclass cylinder kernel and Solve assembly through the store.

These drive the columnar frame directly (Source/Target built from dict inputs,
no CoilSet construction tier) so they run without the machine-description
layer.  They pin that the frame-integrated assembly reproduces the canonical
functional kernel :func:`nova.biot.greens.cylinder_greens` to tolerance -- the
analytic check that the operator-assembly path and the standalone Green's
function stay one and the same kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.biot.biotframe import Source, Target
from nova.biot.cylinder import Cylinder
from nova.biot.greens import cylinder_greens
from nova.biot.solve import Solve

_TR = np.array([1.3, 1.1, 0.7, 1.6, 0.95])
_TZ = np.array([0.3, 0.4, -0.2, 0.0, 0.1])


def _ring_source(a=0.9, z0=0.1, da=0.1, dz=0.1):
    """Return a single rectangular-section ring Source and its frame geometry."""
    source = Source(
        {"x": [a], "z": [z0], "dx": da, "dz": dz},
        segment="cylinder",
        section="rectangle",
        nturn=1,
    )
    geometry = (
        float(source["x"][0]),
        float(source["z"][0]),
        float(source["dx"][0]),
        float(source["dz"][0]),
    )
    return source, geometry


def test_cylinder_kernel_matches_greens():
    """The dataclass cylinder kernel reproduces cylinder_greens to 1e-9."""
    source, (a, z0, da, dz) = _ring_source()
    target = Target({"x": _TR, "z": _TZ}, available=[])
    kernel = Cylinder(source, target, turns=[False, False], reduce=[False, False])
    psi, br, bz = cylinder_greens(_TR, _TZ, a, z0, da, dz)
    np.testing.assert_allclose(np.asarray(kernel.Psi).ravel(), psi, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(kernel.Br).ravel(), br, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(kernel.Bz).ravel(), bz, rtol=1e-9)


def test_solve_assembly_matches_greens():
    """The Solve ensemble assembles per-source columns matching the kernel.

    Two rings, non-reduced: each Psi column must equal the single-ring
    Green's-function flux for that source.
    """
    source = Source(
        {"x": [0.9, 1.2], "z": [0.1, 0.2], "dx": 0.1, "dz": 0.1},
        segment="cylinder",
        section="rectangle",
        nturn=1,
    )
    target = Target({"x": _TR, "z": _TZ}, available=[])
    solve = Solve(source, target, turns=[False, False], reduce=[False, False])
    psi = np.asarray(solve.data["Psi"])  # (target, source)
    assert psi.shape == (_TR.size, 2)
    for col in range(2):
        a = float(source["x"][col])
        z0 = float(source["z"][col])
        da = float(source["dx"][col])
        dz = float(source["dz"][col])
        psi_g, _, _ = cylinder_greens(_TR, _TZ, a, z0, da, dz)
        np.testing.assert_allclose(psi[:, col], psi_g, rtol=1e-9)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("area", 0.0),
        ("area", -1.0),
        ("area", np.nan),
        ("area", np.inf),
        ("dx", 0.0),
        ("dx", -1.0),
        ("dx", np.nan),
        ("dx", np.inf),
        ("dz", 0.0),
        ("dz", -1.0),
        ("dz", np.nan),
        ("dz", np.inf),
    ],
)
def test_cylinder_rejects_invalid_section_geometry(attr, value):
    source, _ = _ring_source()
    source.loc[:, attr] = value
    target = Target({"x": [1.3], "z": [0.3]}, available=[])
    with pytest.raises(ValueError, match=rf"{attr} must be finite and positive"):
        Cylinder(source, target, turns=[False, False], reduce=[False, False])


@pytest.mark.parametrize(
    ("da", "dz", "target_r", "target_z"),
    [
        (0.1, 0.1, 0.95, 0.1),
        (0.1, 0.1, 0.95, 0.15),
        (1e-6, 2e-6, 4.0, 3.0),
    ],
    ids=["section-face", "section-corner", "thin-section-far-field"],
)
def test_cylinder_is_finite_on_section_boundaries_and_scale_separation(
    da, dz, target_r, target_z
):
    source, _ = _ring_source(da=da, dz=dz)
    target = Target({"x": [target_r], "z": [target_z]}, available=[])
    kernel = Cylinder(source, target, turns=[False, False], reduce=[False, False])
    assert np.all(np.isfinite([kernel.Psi, kernel.Br, kernel.Bz]))
