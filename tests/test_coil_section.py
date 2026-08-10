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
import shapely.geometry

from nova.biot.cylinder import Cylinder
from nova.biot.polysection import PolySection
from nova.biot.sectionaverage import averaged_greens
from nova.biot.target import TargetQuadraturePolicy, linked_flux_target
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


def pair(dcoil, segment=None, target_policy=None):
    """Return the coil pair meshed at ``dcoil``, optionally forced onto a segment."""
    attrs = {} if segment is None else {"segment": segment, "ifttt": False}
    coilset = CoilSet(dcoil=dcoil, inductance_target_policy=target_policy or "")
    for x, z, dx, dz, nturn, name in PAIR:
        coilset.coil.insert(x, z, dx, dz, nturn=nturn, name=name, **attrs)
    return coilset


def reduced(dcoil, segment=None, order=3):
    """Return the reduced coil-coil inductance matrix [H].

    Positive target nodes integrate the actual material within each existing dcoil
    cell, then contract to that cell before its parent turns and electrical links.
    """
    coilset = pair(dcoil, segment, TargetQuadraturePolicy(order=order))
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


def test_the_reduced_inductance_converges_with_the_mesh():
    """Positive cell quadrature converges to the whole-section double integral.

    The source integral sums exactly across a tiling. The target integral is a
    fixed-order positive rule on each original dcoil cell, so its small residual
    converges as the material cells refine instead of changing their identities.
    """
    section = [reduced(dcoil) for dcoil in (-2, -5, -20)]
    drift = [
        np.max(np.abs(value - section[-1]) / np.abs(section[-1]))
        for value in section[:-1]
    ]
    assert drift[1] < drift[0] < 5e-5
    banded = [reduced(dcoil, segment="circle") for dcoil in (-2, -5, -20)]
    drift = max(
        np.max(np.abs(value - banded[0]) / np.abs(banded[0])) for value in banded
    )
    assert drift > 1e-4


@pytest.mark.parametrize("dcoil", [-2, -5, -20])
def test_target_expansion_preserves_every_dcoil_cell_and_parent_turn_sum(dcoil):
    """Kernel nodes add no conducting cells and do not alter turn ownership."""
    coilset = pair(dcoil)
    quadrature = linked_flux_target(coilset.frame, coilset.subframe)
    assert quadrature.logical.index.tolist() == coilset.subframe.index.tolist()
    assert quadrature.physical_index == tuple(coilset.frame.index)
    for name in coilset.frame.index:
        positions = np.asarray(coilset.subframe.frame) == name
        assert np.sum(np.asarray(quadrature.logical.nturn)[positions]) == pytest.approx(
            coilset.frame.at[name, "nturn"]
        )


def test_linked_source_reduces_columns_but_keeps_physical_target_rows():
    """Electrical current links act on sources, not distinct conductor targets."""
    factor = -0.25
    baseline = pair(-2)
    baseline.inductance.solve(1)
    physical = np.asarray(baseline.inductance.Psi)

    linked = pair(-2)
    linked.linkframe(["PF1", "CS3U"], factor)
    quadrature = linked_flux_target(linked.frame, linked.subframe)
    assert quadrature.physical_index == ("PF1", "CS3U")
    for name in linked.frame.index:
        positions = np.flatnonzero(np.asarray(quadrature.logical.frame) == name)
        assert quadrature.logical.link[positions[0]] == ""
        assert np.all(quadrature.logical.factor[positions] == 1.0)
        assert np.all(quadrature.logical.link[positions[1:]] == name)

    linked.inductance.solve(1)
    assert linked.inductance.data.target.values.tolist() == ["PF1", "CS3U"]
    assert linked.inductance.data.source.values.tolist() == ["PF1"]
    assert linked.inductance.Psi.shape == (2, 1)
    circuit = np.array([1.0, factor])
    expected = physical @ circuit
    np.testing.assert_allclose(linked.inductance.Psi[:, 0], expected, rtol=2e-12)


def test_inductance_rejects_target_policy_mutation_after_construction():
    """A cached method cannot change target quadrature without changing its owner."""
    coilset = pair(-2)
    coilset.inductance.target_policy = TargetQuadraturePolicy(order=4)
    with pytest.raises(ValueError, match="fixed by its CoilSet constructor"):
        coilset.inductance.solve(1)


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


def test_target_order_sweep_converges_to_raw_reciprocity():
    """Positive target quadrature restores mutual-inductance reciprocity by order."""
    matrices = [reduced(-2, order=order) for order in (1, 2, 3, 4)]
    asymmetry = [np.max(np.abs(matrix - matrix.T)) for matrix in matrices]
    assert np.all(np.diff(asymmetry) < 0)
    assert asymmetry[-1] < 1e-9 * np.max(np.abs(matrices[-1]))
    change = [
        np.max(np.abs(later - earlier))
        for earlier, later in zip(matrices, matrices[1:])
    ]
    assert np.all(np.diff(change) < 0)


def test_hollow_target_nodes_stay_in_actual_parent_material():
    """A void never receives positive target weight, even on a fine dcoil grid."""
    coilset = CoilSet(dcoil=0)
    coilset.coil.insert(
        {"box": [3.0, 0.0, 0.8, 0.2]},
        nturn=20,
        name="Hollow",
        turn="rectangle",
        ifttt=False,
    )
    quadrature = linked_flux_target(coilset.frame, coilset.subframe)
    parent = coilset.frame.poly[0].poly
    void = shapely.geometry.Polygon(parent.interiors[0])
    nodes = np.column_stack([quadrature.nodes.x, quadrature.nodes.z])
    assert not any(void.covers(shapely.geometry.Point(node)) for node in nodes)
    assert quadrature.logical.index.tolist() == coilset.subframe.index.tolist()
    assert np.all(quadrature.weights > 0)
    np.testing.assert_allclose(
        np.add.reduceat(quadrature.weights, quadrature.offsets), 1.0
    )


def test_composite_and_wall_clipped_targets_preserve_logical_cells():
    """Disconnected parents and clipped plasma cells keep their source-cell identity."""
    composite = shapely.geometry.MultiPolygon(
        [
            shapely.geometry.box(2.7, -0.2, 2.9, 0.2),
            shapely.geometry.box(3.1, -0.2, 3.3, 0.2),
        ]
    )
    coilset = CoilSet(dcoil=-4, dplasma=-20, tplasma="hex")
    coilset.coil.insert(composite, nturn=10, name="Composite")
    coilset.firstwall.insert({"circle": [4.5, 0.0, 0.8]}, Ic=1.0)
    quadrature = linked_flux_target(coilset.frame, coilset.subframe)
    assert quadrature.logical.index.tolist() == coilset.subframe.index.tolist()
    assert quadrature.physical_index == tuple(coilset.frame.index)
    assert len(quadrature.logical) == len(coilset.subframe)
    assert len(quadrature.nodes) > len(quadrature.logical)
    for parent_name in coilset.frame.index:
        parent = coilset.frame.at[parent_name, "poly"].poly
        positions = np.flatnonzero(np.asarray(quadrature.logical.frame) == parent_name)
        start = quadrature.offsets[positions]
        stop = np.r_[quadrature.offsets, len(quadrature.nodes)][positions + 1]
        for lower, upper in zip(start, stop):
            points = zip(
                quadrature.nodes.x[lower:upper], quadrature.nodes.z[lower:upper]
            )
            assert all(
                parent.buffer(1e-12).covers(shapely.geometry.Point(point))
                for point in points
            )
