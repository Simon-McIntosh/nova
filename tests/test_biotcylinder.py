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
from nova.biot.greens import MU0, cylinder_greens
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


def test_cylinder_rejects_an_authored_hexagon_section():
    """A rectangle kernel cannot consume moments from an authored hexagon."""
    source = Source(
        {"x": [0.9], "z": [0.1], "dx": [0.1], "dz": [0.1]},
        segment="cylinder",
        section="hexagon",
        nturn=1,
    )
    target = Target({"x": [1.3], "z": [0.3]}, available=[])
    with pytest.raises(ValueError, match="require an axis-aligned rectangle"):
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


def test_cylinder_reaches_the_symmetry_axis():
    """An axis target is the section's own loop limit rather than a nan.

    The corner antiderivative divides by the TARGET radius, so an element that stacks
    the section corners itself has no value to return on the axis and fills the whole
    axis column with nan.  Taking the fields from the canonical kernel picks up its
    axis expansion, where the flux and the radial field carry their leading powers of
    the target radius and are therefore exactly zero, and the vertical field is the
    section's area mean of the textbook on-axis loop field.

    That mean is checked here against a rule four times finer than the kernel's own,
    which is an independent quadrature of the same smooth integrand -- a converged
    figure the kernel's fixed rule has to reach on its own.
    """
    source, (a, z0, da, dz) = _ring_source()
    radius = np.array([0.0, 0.0, 0.35])
    level = np.array([z0, z0 + 0.6, z0])
    target = Target({"x": radius, "z": level}, available=[])
    kernel = Cylinder(source, target, turns=[False, False], reduce=[False, False])
    for field in (kernel.Psi, kernel.Br, kernel.Bz, kernel.Aphi):
        assert np.all(np.isfinite(np.asarray(field)))
    on_axis = radius == 0.0
    for field in (kernel.Psi, kernel.Br, kernel.Aphi):
        assert np.all(np.asarray(field).ravel()[on_axis] == 0.0)

    node, weight = np.polynomial.legendre.leggauss(24)
    source_r = a + 0.5 * da * node[:, None]
    source_z = z0 + 0.5 * dz * node[None, :]
    area_weight = 0.25 * weight[:, None] * weight[None, :]
    for index in np.flatnonzero(on_axis):
        gap = level[index] - source_z
        refined = np.sum(
            area_weight * MU0 * source_r**2 / (2.0 * (source_r**2 + gap**2) ** 1.5)
        )
        np.testing.assert_allclose(
            np.asarray(kernel.Bz).ravel()[index], refined, rtol=1e-13
        )


def test_cylinder_flux_and_vector_potential_share_one_permeability():
    """``Aphi`` inverts ``Psi`` rather than being integrated beside it.

    The two are one field through ``Phi = 2 pi R A_phi``, so they must carry the same
    constant.  The base class holds the measured CODATA permeability and the kernel
    holds ``4 pi x 1e-7``; those differ by a part in ten billion, which as a fixed
    offset between two views of one quantity is not something a finer rule removes.
    """
    assert Cylinder.mu_0 == MU0
    source, _ = _ring_source()
    target = Target({"x": _TR, "z": _TZ}, available=[])
    kernel = Cylinder(source, target, turns=[False, False], reduce=[False, False])
    radius = np.asarray(kernel.target("r"))
    recovered = 2.0 * np.pi * Cylinder.mu_0 * radius * np.asarray(kernel.Aphi)
    np.testing.assert_allclose(
        recovered, np.asarray(kernel.Psi), rtol=4.0 * np.finfo(float).eps, atol=0.0
    )
