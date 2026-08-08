"""What a poloidal coil's section treatment is, and what routes it.

A coil is meshed into subframe filaments and every filament is coupled through the
element its ``segment`` column names. Two elements integrate the current over the
filament's true section -- :class:`nova.biot.polysection.PolySection` over an
arbitrary polygon and :class:`nova.biot.cylinder.Cylinder` over an axis-aligned
rectangle -- and a third, :class:`nova.biot.circle.Circle`, carries a point filament
outside a band of four section radii. These tests pin which one a coil gets, that
the two exact ones are one quantity, and the two properties that follow from
integrating the section on every pair rather than banding it.

The reference for the last of those is the uniform-current DOUBLE integral over each
coil's whole undivided section, built from
:func:`nova.biot.sectionaverage.averaged_greens`. It shares no code path with the
elements and no assumption about how either coil is meshed, so it can say which lane
moved. :mod:`benchmarks.coil_section_cost` is the same comparison across coil count,
mesh refinement and every operator, with the assembly cost beside it.
"""

import numpy as np
import pytest

from nova.biot.cylinder import Cylinder
from nova.biot.polysection import PolySection
from nova.biot.sectionaverage import averaged_greens
from nova.frame.coilset import CoilSet

PAIR = [
    # x, z, dx, dz, nturn, name -- two ITER winding-pack outlines, near enough that
    # neither sits wholly inside the other's filament band nor wholly outside it
    (3.9431, 7.5641, 0.9590, 0.9841, 250.0, "PF1"),
    (1.722, 5.313, 0.719, 2.075, 250.0, "CS3U"),
]


def rectangle(x, z, dx, dz):
    """Return the ``(4, 2)`` r-z corners of an axis-aligned section."""
    return np.array(
        [
            [x - dx / 2, z - dz / 2],
            [x + dx / 2, z - dz / 2],
            [x + dx / 2, z + dz / 2],
            [x - dx / 2, z + dz / 2],
        ]
    )


def pair(dcoil, segment=None):
    """Return the coil pair meshed at ``dcoil``, optionally forced onto a segment."""
    attrs = {} if segment is None else {"segment": segment, "ifttt": False}
    coilset = CoilSet(dcoil=dcoil)
    for x, z, dx, dz, nturn, name in PAIR:
        coilset.coil.insert(x, z, dx, dz, nturn=nturn, name=name, **attrs)
    return coilset


def reduced(dcoil, segment=None):
    """Return the reduced coil-coil inductance matrix [H].

    The target frame carries no section of its own, so the target-side area average
    is carried by that frame's subdivision; zero puts one target on every turn,
    which is the finest the lane offers.
    """
    coilset = pair(dcoil, segment)
    coilset.inductance.solve(0)
    return np.asarray(coilset.inductance.Psi)


@pytest.fixture(name="reference")
def fixture_reference():
    """Return the uniform-current double integral over the whole sections [H]."""
    section = [rectangle(*coil[:4]) for coil in PAIR]
    nturn = np.array([coil[4] for coil in PAIR])
    matrix = np.empty((len(PAIR), len(PAIR)))
    for column, source in enumerate(section):
        matrix[:, column] = averaged_greens(section, source)[0] * nturn * nturn[column]
    return matrix


@pytest.mark.parametrize("dcoil", [-2, -5, -20])
def test_a_meshed_coil_couples_through_its_own_section(dcoil):
    """Every filament of a meshed coil takes the polygon kernel."""
    assert np.array_equal(np.unique(pair(dcoil).subframe.segment), ["polysection"])


@pytest.mark.parametrize("section", ["rectangle", "skin", "disc"])
def test_the_section_shape_does_not_change_the_element(section):
    """The polygon kernel reads each filament's own vertices, so no shape is special."""
    coilset = CoilSet(dcoil=-4)
    coilset.coil.insert(3.9, 7.5, 0.96, 0.98, nturn=20, section=section, name="PF")
    assert np.array_equal(np.unique(coilset.subframe.segment), ["polysection"])


def test_an_undivided_rectangle_takes_the_corner_rule():
    """One filament spanning a rectangular coil routes to the four-corner rule."""
    assert np.array_equal(np.unique(pair(-1).subframe.segment), ["cylinder"])


def test_the_corner_rule_and_the_polygon_kernel_are_one_quantity():
    """Both integrate the same current over the same box, so they cannot disagree.

    This is what makes the undivided-rectangle route a cost choice and not a physics
    one: the corner rule reaches the section integral in four antiderivatives where
    the polygon kernel spends a corner evaluation apiece.

    The one entry that separates them is the coincident one, where the target sits
    inside its own source section and each reduction is evaluated on its own singular
    geometry: the flux there is 1.3e-10 apart. Every other entry holds round-off.
    """
    frame = pair(-1, segment="cylinder").subframe
    corner = Cylinder(frame, frame, reduce=[False, False])
    polygon = PolySection(frame, frame, reduce=[False, False])
    scale = np.max(np.abs(np.asarray(corner.Psi)))
    for name in ("Psi", "Br", "Bz"):
        want = np.asarray(getattr(corner, name))
        got = np.asarray(getattr(polygon, name))
        assert got == pytest.approx(want, rel=1e-8, abs=1e-8 * scale), name


def test_the_reduced_inductance_does_not_move_with_the_mesh():
    """Summing the section integral over a tiling returns the whole section's own.

    The pair sum IS the area integral split up, so refining a coil's mesh cannot
    move a reduced coil-coil term. A banded lane has no such identity: the filament
    it substitutes beyond the band is a different function of where the sub-sections
    land, so its reduced terms drift as the mesh changes.
    """
    coarse = reduced(-2)
    for dcoil in (-5, -20):
        assert reduced(dcoil) == pytest.approx(coarse, rel=1e-9)
    banded = [reduced(dcoil, segment="circle") for dcoil in (-2, -5, -20)]
    drift = max(
        np.max(np.abs(value - banded[0]) / np.abs(banded[0])) for value in banded
    )
    assert drift > 1e-4


def test_the_section_lane_lands_closer_to_the_double_integral(reference):
    """Against the quantity both lanes approximate, the exact section wins.

    Only the mutual terms separate them. The diagonal is dominated by the target
    frame's own subdivision residual, which both lanes carry identically, so it
    cannot discriminate and is not asserted on here; the same residual sets the
    floor the exact lane's mutual sits on, which is why it is bounded rather than
    driven to round-off.

    Read at a mesh coarse enough that the two coils sit outside each other's
    filament band, which is the configuration the band was placed to exclude. As the
    mesh refines the banded lane's own error falls towards this floor -- it is a
    function of where the sub-sections land, not a bound.
    """
    off = ~np.eye(len(PAIR), dtype=bool)
    scale = np.abs(reference)[off]
    section = np.abs(reduced(-2) - reference)[off] / scale
    banded = np.abs(reduced(-2, segment="circle") - reference)[off] / scale
    assert section.max() < 5e-4
    assert banded.max() > 5 * section.max()
