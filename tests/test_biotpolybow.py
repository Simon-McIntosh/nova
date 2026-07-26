"""The element class for a polygon cross-section arc segment.

:mod:`tests.test_biotpolygonarc` holds the reduction's own acceptance gate --
four independent references plus ``Bow`` -- and none of it goes through a frame.
This module holds what the CLASS adds on top and nothing else: the section it
builds from the frame's own descriptor, the local-to-global transform, and the
segment registry that lets :class:`nova.biot.solve.Solve` reach it.

So every check here is a comparison against the bare reduction or against
``Bow`` THROUGH the same frame machinery, and the reduction's accuracy is
somebody else's test.  Two of them are worth naming:

* ``Bow`` at zero edge slope, through the class rather than through the bare
  function.  It is the only end-to-end check of the whole path, and it holds to
  ``Bow``'s own fixed-node zeta quadrature rather than to round-off.
* A HEXAGONAL section, which is what the element exists for.  A swept winding of
  non-rectangular section is routed to ``Bow`` today and evaluated as a rectangle
  of its own width and height while being normalised by its true area, so a
  hexagon comes back a third too large.  That factor is asserted, not just the
  agreement, because it is the size of the gap this element closes.

The frame's stored matrices are read rather than
:attr:`nova.frame.coilset.CoilSet.point`'s accessors: the accessors cast to
single precision, which is enough to hide every comparison below.
"""

import numpy as np
import pytest
from scipy.constants import mu_0

from nova.biot.polybow import PolyBow
from nova.biot.polygonarc import polygon_arc_greens
from nova.biot.solve import Solve
from nova.frame.coilset import CoilSet
from nova.geometry.polygen import PolyGen

ROWS = ("A_r", "A_phi", "B_r", "B_phi", "B_z")
FIELD_ATTRS = ["Ax", "Ay", "Bx", "By", "Bz"]

RADIUS, ELEVATION = 3.0, 0.2
WIDTH, THICKNESS = 0.06, 0.04

# Sweeps spanning what the fold has to get right: a short arc, one straddling the
# target azimuths, one that crosses the branch cut, and a half turn.
SWEEPS = {
    "short": (0.3, 1.9),
    "straddle": (-1.0, 2.5),
    "wrapped": (5.6, 7.4),
    "half_turn": (0.0, np.pi),
}

TARGET_R = np.array([3.5, 2.5, 3.1, 3.0, 3.3])
TARGET_PHI = np.array([0.2, 1.0, 2.4, -0.4, 4.5])
TARGET_Z = np.array([0.5, -0.2, 0.25, 0.35, -0.5])


def target_points():
    """Return the target cloud as ``(n, 3)`` cartesian points."""
    return np.stack(
        [TARGET_R * np.cos(TARGET_PHI), TARGET_R * np.sin(TARGET_PHI), TARGET_Z],
        axis=-1,
    )


def winding(section, sweep, segment=None, radius=RADIUS, elevation=ELEVATION):
    """Return a solved coilset carrying one swept winding.

    ``segment`` overrides the element the frame routes to, which is how a
    polygon-section arc is reached today: nothing assigns ``polybow`` on its own
    yet, because a swept winding's section is not always one a corner list can
    carry.  Setting the column directly is the same route
    :mod:`benchmarks.polygon_route_cost` takes to reach ``PolySection``.
    """
    start, end = sweep
    angle = np.array([start, 0.5 * (start + end), end])
    path = np.stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            elevation * np.ones_like(angle),
        ],
        axis=-1,
    )
    coilset = CoilSet(field_attrs=FIELD_ATTRS)
    coilset.winding.insert(
        path,
        section,
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment
    coilset.point.solve(target_points())
    return coilset


def cylindrical_rows(coilset):
    """Return the five rows per ampere in the TARGET's own cylindrical basis.

    The reduction's own convention, so the class can be held to the function it
    wraps: the vector potential carries Urankar eq 3a's ``mu0`` where nova's frame
    convention does not, and the field carries it in both.
    """
    matrix = np.stack(
        [np.asarray(coilset.point.data[attr]).sum(axis=1) for attr in FIELD_ATTRS]
    )
    potential_x, potential_y, field_x, field_y, field_z = matrix
    cosine, sine = np.cos(TARGET_PHI), np.sin(TARGET_PHI)
    return np.stack(
        [
            mu_0 * (potential_x * cosine + potential_y * sine),
            mu_0 * (-potential_x * sine + potential_y * cosine),
            field_x * cosine + field_y * sine,
            -field_x * sine + field_y * cosine,
            field_z,
        ]
    )


def frame_section(coilset, centre=(RADIUS, ELEVATION), column=0):
    """Return the section corners the frame's own descriptor implies.

    Placed at the ARC's radius and height, which is where the element's local
    frame puts it -- not at the swept element's centroid, which is inside the
    chord and is what the frame's ``x`` and ``z`` columns carry.
    """
    name = PolyGen(str(np.asarray(coilset.subframe["section"])[column])).shape
    shape = PolyGen(name)(
        centre[0],
        centre[1],
        float(np.asarray(coilset.subframe["width"])[column]),
        float(np.asarray(coilset.subframe["height"])[column]),
    )
    points = np.asarray(shape.exterior.coords, dtype=np.float64)
    gap = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    return points[gap > 1e-9 * np.max(np.ptp(points, axis=0))]


def worst_overall(got, want):
    """Return the worst deviation against the LARGEST row's scale."""
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def worst_by_row(got, want):
    """Return the worst deviation of each row against its OWN scale."""
    got, want = np.asarray(got), np.asarray(want)
    scale = np.max(np.abs(want), axis=1)[:, None]
    return np.max(np.abs(got - want) / scale, axis=1)


# ---------------------------------------------------------------------------
# The registry, and what the class declares itself to be.


def test_the_segment_registry_reaches_the_element():
    """A frame labelled ``polybow`` builds this class and nothing else."""
    assert Solve.generator["polybow"] is PolyBow
    assert PolyBow.name == "polybow"
    assert PolyBow.axisymmetric is False


def test_the_element_is_a_peer_of_bow_rather_than_a_subclass():
    """The decision, asserted where it can be read.

    ``Bow``'s body is a rectangle's four corners stacked onto every geometry
    array; a polygon section needs none of it.  What the two share is the frame
    plumbing, which is ``Arc``'s, so the two elements are siblings.
    """
    from nova.biot.arc import Arc
    from nova.biot.bow import Bow

    assert issubclass(PolyBow, Arc)
    assert not issubclass(PolyBow, Bow)
    assert issubclass(Bow, Arc)


# ---------------------------------------------------------------------------
# What the class adds to the reduction: a section, and a frame transform.


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_the_class_adds_nothing_to_the_reduction_but_the_transform(sweep):
    """The five rows reach the frame unchanged.

    A local frame whose axis is the arc's own and whose origin is off it, rotated
    to global and read back in the target's cylindrical basis -- the round trip is
    two rotations and its residue is what this measures.
    """
    coilset = winding(
        {"rect": (0, 0, WIDTH, THICKNESS)}, SWEEPS[sweep], segment="polybow"
    )
    start, end = SWEEPS[sweep]
    want = np.stack(
        polygon_arc_greens(
            TARGET_R, TARGET_Z, TARGET_PHI, frame_section(coilset), start, end
        )
    )
    got = cylindrical_rows(coilset)
    assert worst_overall(got, want) <= 1e-11  # measured 2.2e-12
    assert np.max(worst_by_row(got, want)) <= 1e-10  # measured 3.0e-11


def test_a_section_the_frame_cannot_describe_is_refused():
    """A hollow section has an interior boundary a corner list cannot carry.

    ``skin`` and ``box`` also store a hollowness FACTOR where the solid sections
    store a height, so building one from ``(width, height)`` would silently return
    a nearly-solid square rather than the ring the frame means.
    """
    coilset = winding({"sk": (0, 0, WIDTH, WIDTH, 0.2)}, SWEEPS["short"])
    coilset.subframe.loc[:, "segment"] = "polybow"
    with pytest.raises(NotImplementedError, match="width and height"):
        coilset.point.solve(target_points())


# ---------------------------------------------------------------------------
# Bow, through the class rather than through the bare function.


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_a_rectangular_section_reproduces_bow_through_the_class(sweep):
    """The whole path end to end, against Urankar Part IV.

    Bounded from above only, and by ``Bow``'s accuracy rather than by either
    reduction's: ``Bow`` reaches its answer through a fixed-node zeta quadrature
    where this is closed form, and the landed gate already records the closed form
    as the value the quadrature is measured against.
    """
    section = {"rect": (0, 0, WIDTH, THICKNESS)}
    bow = cylindrical_rows(winding(section, SWEEPS[sweep]))
    poly = cylindrical_rows(winding(section, SWEEPS[sweep], segment="polybow"))
    assert worst_overall(poly, bow) <= 2e-08  # measured 4.9e-09
    assert np.max(worst_by_row(poly, bow)) <= 5e-08  # measured 1.1e-08


def test_the_bow_agreement_is_not_a_tie_at_single_precision():
    """The frame's stored matrices carry more than the accessors report.

    ``CoilSet.point``'s accessors come back single precision, at which every
    comparison in this module reads as exact -- so the test above would pass on a
    class that returned ``Bow``'s own numbers.  Reading the stored matrix is what
    makes it evidence, and this is the assertion that the two differ at all.
    """
    section = {"rect": (0, 0, WIDTH, THICKNESS)}
    bow = cylindrical_rows(winding(section, SWEEPS["short"]))
    poly = cylindrical_rows(winding(section, SWEEPS["short"], segment="polybow"))
    assert worst_overall(poly, bow) >= 1e-13


# ---------------------------------------------------------------------------
# The hexagon, which is the gap the element closes.


def test_a_hexagonal_section_is_evaluated_as_a_hexagon():
    """The class reads the frame's section rather than its bounding box."""
    coilset = winding({"hex": (0, 0, WIDTH, THICKNESS)}, SWEEPS["short"], "polybow")
    corners = frame_section(coilset)
    assert len(corners) == 6
    start, end = SWEEPS["short"]
    want = np.stack(
        polygon_arc_greens(TARGET_R, TARGET_Z, TARGET_PHI, corners, start, end)
    )
    got = cylindrical_rows(coilset)
    assert worst_overall(got, want) <= 1e-11  # measured 3.5e-12
    assert np.max(worst_by_row(got, want)) <= 1e-10  # measured 2.0e-11


def test_routing_a_hexagon_to_bow_overstates_it_by_a_third():
    """The size of the gap, asserted rather than described.

    ``Bow`` integrates over the rectangle its width and height bound and divides
    by the frame's area, which is the hexagon's -- and a regular hexagon fills
    three quarters of that box, so every row comes back at four thirds of its
    value.  The ratio is the same on all five because the error is a normalisation
    and not a shape.
    """
    section = {"hex": (0, 0, WIDTH, THICKNESS)}
    bow = cylindrical_rows(winding(section, SWEEPS["short"]))
    poly = cylindrical_rows(winding(section, SWEEPS["short"], segment="polybow"))
    ratio = bow / poly
    assert np.allclose(ratio, 4.0 / 3.0, rtol=2e-03)


# ---------------------------------------------------------------------------
# More than one source in a segment.


def test_two_windings_share_one_segment_and_sum():
    """One generator carries every source of its segment, each with its own
    section and its own sweep, and the frame sums their columns."""
    coilset = CoilSet(field_attrs=FIELD_ATTRS)
    sweeps = [SWEEPS["short"], SWEEPS["wrapped"]]
    sections = [{"rect": (0, 0, WIDTH, THICKNESS)}, {"hex": (0, 0, WIDTH, THICKNESS)}]
    elevations = [ELEVATION, -ELEVATION]
    for section, (start, end), elevation in zip(sections, sweeps, elevations):
        angle = np.array([start, 0.5 * (start + end), end])
        coilset.winding.insert(
            np.stack(
                [
                    RADIUS * np.cos(angle),
                    RADIUS * np.sin(angle),
                    elevation * np.ones_like(angle),
                ],
                axis=-1,
            ),
            section,
            nturn=1,
            Ic=1,
            minimum_arc_nodes=3,
            filament=False,
            ifttt=False,
        )
    coilset.subframe.loc[:, "segment"] = "polybow"
    assert len(coilset.subframe) == 2
    coilset.point.solve(target_points())
    want = sum(
        np.stack(
            polygon_arc_greens(
                TARGET_R,
                TARGET_Z,
                TARGET_PHI,
                frame_section(coilset, (RADIUS, elevations[column]), column),
                *sweeps[column],
            )
        )
        for column in range(2)
    )
    assert worst_overall(cylindrical_rows(coilset), want) <= 1e-11
