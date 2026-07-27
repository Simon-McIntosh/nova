"""The element class for a polygon cross-section arc segment.

:mod:`tests.test_biotpolygonarc` holds the reduction's own acceptance gate --
four independent references plus ``Bow`` -- and none of it goes through a frame.
This module holds what the CLASS adds on top and nothing else: the section it
takes from the frame's ``poly`` column, the local-to-global transform, the
routing that reaches it, and the coupled pair a hollow section arrives as.

So every check here is a comparison against the bare reduction or against
``Bow`` THROUGH the same frame machinery, and the reduction's accuracy is
somebody else's test.  Three of them are worth naming:

* ``Bow`` at zero edge slope, through the class rather than through the bare
  function.  It is the only end-to-end check of the whole path, and it holds to
  ``Bow``'s own fixed-node zeta quadrature rather than to round-off.
* A HEXAGONAL section, which is what the element exists for.  ``Bow`` evaluates a
  rectangle of the section's width and height while normalising by its true area,
  so a hexagon comes back a third too large.  That factor is asserted, not just
  the agreement, because it is the size of the gap this element closes -- and it
  is now measured against a ``Bow`` the caller has to NAME, because a swept
  winding routes here.
* A HOLLOW section, which arrives as a linked pair of solid ones at ``+j`` and
  ``-j`` and is checked against a partition of the annulus into solid cells --
  a direct integral over the same boundaries, decomposed the other way.

The frame's stored matrices are read rather than
:attr:`nova.frame.coilset.CoilSet.point`'s accessors: the accessors cast to
single precision, which is enough to hide every comparison below.
"""

import numpy as np
import pytest
import shapely.geometry
from scipy.constants import mu_0

from nova.biot.polybow import PolyBow, section_corners
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

    A thickened arc routes to ``polybow`` on its own, so ``segment`` is what a
    caller uses to reach ``Bow`` instead -- which is the direction the override now
    runs in, and is how ``Bow`` serves as the independent Urankar Part IV oracle
    below rather than as the default.
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


def test_a_thickened_arc_routes_here_without_the_caller_naming_it():
    """The routing, asserted where it can be read.

    A swept winding reaches the element that can evaluate its section, for every
    section, with no segment named at the call site.  ``Bow`` would evaluate the box
    its width and height bound and normalise by the section's own area, so it is
    right for a rectangle and wrong for everything else -- and cannot express a
    section that is not a rectangle at all.  A filament winding is unaffected: it
    carries no section, so it stays an ``arc``.
    """
    for section in (
        {"rect": (0, 0, WIDTH, THICKNESS)},
        {"hex": (0, 0, WIDTH, THICKNESS)},
        {"disc": (0, 0, WIDTH, WIDTH)},
        {"box": (0, 0, WIDTH, 0.2)},
        {"sk": (0, 0, WIDTH, 0.2)},
    ):
        coilset = winding(section, SWEEPS["short"])
        assert set(np.asarray(coilset.subframe["segment"]).tolist()) == {"polybow"}
    filament = CoilSet()
    start, end = SWEEPS["short"]
    angle = np.array([start, 0.5 * (start + end), end])
    filament.winding.insert(
        np.stack(
            [
                RADIUS * np.cos(angle),
                RADIUS * np.sin(angle),
                ELEVATION * np.ones_like(angle),
            ],
            axis=-1,
        ),
        {"hex": (0, 0, WIDTH, THICKNESS)},
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        ifttt=False,
    )
    assert np.asarray(filament.subframe["segment"]).tolist() == ["arc"]


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


def test_the_section_reaches_the_kernel_at_double_precision():
    """The corners the reduction integrates are the section's own, exactly.

    The frame's ``poly`` column is built from the sweep's own float64 corner loops
    rather than from the vtk mesh they build: VTK's points default to single
    precision, which lands a corner authored at 2.97 on 2.96999979.  That is 8e-09
    relative on a corner and 7.9e-09 on the rows -- four orders above the closed
    form's own 3.5e-12 -- so the column has to be exact for reading it to be worth
    anything.  Measured against the descriptor's corners, which for a named section
    is the same polygon.
    """
    coilset = winding({"hex": (0, 0, WIDTH, THICKNESS)}, SWEEPS["short"])
    corners = section_corners(np.asarray(coilset.subframe["poly"])[0])
    want = frame_section(coilset)
    order = [
        int(np.argmin(np.linalg.norm(want - corner, axis=1))) for corner in corners
    ]
    assert sorted(order) == list(range(len(want)))
    assert np.max(np.abs(corners - want[order])) <= 1e-15  # measured 4.4e-16


def test_a_swept_hexagon_reaches_the_kernel_with_six_corners():
    """A projection splits an edge; the reduction should not pay for it.

    The union of a sweep's projected stations puts a corner part way along the
    section's own edges -- one per station that moved, sixteen over the sixteen-
    station arc here -- each agreeing with the straight line to round-off.  A
    closed-form section reduction costs one evaluation per corner and a corner is
    what an arc's cost tracks, so the run is collapsed where the footprint is
    formed and again where the section is read.
    """
    coilset = winding({"hex": (0, 0, WIDTH, THICKNESS)}, SWEEPS["short"])
    stored = np.asarray(np.asarray(coilset.subframe["poly"])[0].points)
    assert len(section_corners(np.asarray(coilset.subframe["poly"])[0])) == 6
    assert len(stored) == 7  # six corners and the ring's closing repeat


def test_a_free_form_polygon_section_reaches_the_kernel():
    """The capability no descriptor can express, which is why the column is read.

    An irregular pentagon has no ``(width, height)`` pair that reproduces it, so a
    class rebuilding its section from the frame's descriptor could only ever return
    a rectangle of its bounding box.  Read from ``poly`` it arrives as itself.
    """
    loop = np.array(
        [[-0.03, -0.02], [0.03, -0.02], [0.02, 0.0], [0.03, 0.02], [-0.01, 0.015]]
    )
    coilset = winding(shapely.geometry.Polygon(loop), SWEEPS["short"])
    assert np.asarray(coilset.subframe["segment"]).tolist() == ["polybow"]
    corners = section_corners(np.asarray(coilset.subframe["poly"])[0])
    assert np.min(np.linalg.norm(corners - np.array([3.02, 0.2]), axis=1)) <= 1e-04
    start, end = SWEEPS["short"]
    want = np.stack(
        polygon_arc_greens(TARGET_R, TARGET_Z, TARGET_PHI, corners, start, end)
    )
    got = cylindrical_rows(coilset)
    assert worst_overall(got, want) <= 1e-11  # measured 1.6e-12
    assert np.max(worst_by_row(got, want)) <= 1e-10  # measured 1.9e-11
    # The section the frame carries is the swept solid's own footprint, which for a
    # section NOT symmetric about the poloidal plane is a few parts in 1e5 wider
    # than the loop as authored: a discretised path takes a CHORD direction for its
    # end tangents, so the two end stations are tilted by O(h^2) and their
    # projections reach beyond the rest.  Bounded here so the departure is a
    # measured quantity rather than a surprise, and it is the path's, not the
    # section's -- every named section is symmetric and lands on round-off.
    ideal = np.stack(
        polygon_arc_greens(
            TARGET_R,
            TARGET_Z,
            TARGET_PHI,
            loop + np.array([RADIUS, ELEVATION]),
            start,
            end,
        )
    )
    assert worst_overall(got, ideal) <= 1e-04  # measured 4.7e-05


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
    bow = cylindrical_rows(winding(section, SWEEPS[sweep], segment="bow"))
    poly = cylindrical_rows(winding(section, SWEEPS[sweep]))
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
    bow = cylindrical_rows(winding(section, SWEEPS["short"], segment="bow"))
    poly = cylindrical_rows(winding(section, SWEEPS["short"]))
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
    bow = cylindrical_rows(winding(section, SWEEPS["short"], segment="bow"))
    poly = cylindrical_rows(winding(section, SWEEPS["short"]))
    ratio = bow / poly
    assert np.allclose(ratio, 4.0 / 3.0, rtol=2e-03)


@pytest.mark.parametrize(
    "section,ratio",
    [
        ({"disc": (0, 0, WIDTH, WIDTH)}, 4.0 / np.pi),
        ({"ellipse": (0, 0, WIDTH, THICKNESS)}, 4.0 / np.pi),
    ],
)
def test_routing_a_round_section_to_bow_overstates_it_by_four_over_pi(section, ratio):
    """The same normalisation gap on the sections whose corner count is the cost.

    A disc fills ``pi/4`` of its bounding box and an ellipse the same fraction of
    its own, so ``Bow`` returns ``4/pi`` -- 27 % -- too much on every row.  What the
    correct answer costs is sixty-four corners against a hexagon's six, which is the
    trade the round sections force and the reason this ratio is recorded rather than
    described.
    """
    bow = cylindrical_rows(winding(section, SWEEPS["short"], segment="bow"))
    poly = cylindrical_rows(winding(section, SWEEPS["short"]))
    assert np.allclose(bow / poly, ratio, rtol=3e-03)


# ---------------------------------------------------------------------------
# A hollow section, as a coupled pair of solid ones.

HOLLOW = 0.2  # hollowness factor, 1 - r/R


def annulus_cells(coilset):
    """Return a partition of the annulus into solid quadrilateral cells.

    Built between corresponding corners of the two members' own boundaries, so the
    cells tile exactly the region the pair means -- the same integral decomposed
    the other way round, a sum of positive cells instead of a signed superposition.
    """
    outer, core = (
        section_corners(poly) for poly in np.asarray(coilset.subframe["poly"])[:2]
    )
    assert len(outer) == len(core)
    return [
        np.array(
            [
                outer[i],
                outer[(i + 1) % len(outer)],
                core[(i + 1) % len(core)],
                core[i],
            ]
        )
        for i in range(len(outer))
    ]


def cell_integral(cells, sweep):
    """Return the five rows of a partition, at the partition's own density."""
    start, end = sweep
    total = np.zeros((5, len(TARGET_R)))
    area = 0.0
    for cell in cells:
        rolled = np.roll(cell, -1, axis=0)
        cell_area = 0.5 * abs(
            float(np.sum(cell[:, 0] * rolled[:, 1] - rolled[:, 0] * cell[:, 1]))
        )
        total += cell_area * np.stack(
            polygon_arc_greens(TARGET_R, TARGET_Z, TARGET_PHI, cell, start, end)
        )
        area += cell_area
    return total / area, area


@pytest.mark.parametrize("section", ["box", "sk"])
def test_a_hollow_section_is_a_coupled_pair_rather_than_a_refusal(section):
    """The construction, read off the frame's own columns.

    An annulus is the outer boundary at ``+j`` and the interior one at ``-j``, both
    of them solid sections the reduction already evaluates.  The density is the
    annulus's, which is why both members carry it as their ``area`` and the core
    carries the ``-1`` the frame's link machinery already understands -- the
    reference row of a linked group has factor one by definition, so the density
    cannot live in that column.  The member currents then sum to the conductor's::

        I_outer + I_core = j (A_outer - A_core) = I
    """
    coilset = winding({section: (0, 0, WIDTH, HOLLOW)}, SWEEPS["short"])
    assert len(coilset.subframe) == 2
    outer, core = (
        section_corners(poly) for poly in np.asarray(coilset.subframe["poly"])
    )
    area = np.asarray(coilset.subframe["area"], dtype=float)
    factor = np.asarray(coilset.subframe["factor"], dtype=float)
    outer_area = shapely.geometry.Polygon(outer).area
    core_area = shapely.geometry.Polygon(core).area
    assert factor.tolist() == [1.0, -1.0]
    assert np.allclose(area, outer_area - core_area, rtol=1e-14)
    current = 1.0  # the winding's own Ic
    density = current / area[0]
    assert np.isclose(
        factor[0] * density * outer_area + factor[1] * density * core_area,
        current,
        rtol=1e-14,
    )


@pytest.mark.parametrize("section", ["box", "sk"])
def test_the_hollow_pair_reproduces_a_direct_integral_over_the_annulus(section):
    """The pair against the annulus integrated directly, cell by cell.

    A partition into solid quadrilaterals covers exactly the material the pair
    means, evaluated by the same reduction with no cancellation between members --
    so the agreement is a statement about the SUPERPOSITION rather than about the
    kernel.  Exact rather than converged, because the cells tile the boundaries the
    pair itself carries; refining them radially changes nothing.
    """
    coilset = winding({section: (0, 0, WIDTH, HOLLOW)}, SWEEPS["short"])
    want, area = cell_integral(annulus_cells(coilset), SWEEPS["short"])
    assert np.isclose(area, float(np.asarray(coilset.subframe["area"])[0]), rtol=1e-12)
    got = cylindrical_rows(coilset)
    assert worst_overall(got, want) <= 1e-09  # measured 6.1e-12 box, 7.8e-11 skin
    assert np.max(worst_by_row(got, want)) <= 1e-08  # measured 4.0e-10


def test_a_swept_skin_is_a_circular_annulus_and_not_a_square_one():
    """The section of record decides, where a relabelling used to.

    A ``skin`` is a disc with a disc removed; a ``box`` is a square with a square
    removed.  Both boundaries are read off the section itself, so the two are
    different conductors -- where relabelling ``skin`` to ``box`` made a swept skin
    into a square annulus and its area into the square's, a factor of ``4/pi``.
    """
    skin = winding({"sk": (0, 0, WIDTH, HOLLOW)}, SWEEPS["short"])
    box = winding({"box": (0, 0, WIDTH, HOLLOW)}, SWEEPS["short"])
    assert np.asarray(skin.subframe["section"]).tolist() == ["skin", "skin"]
    skin_area = float(np.asarray(skin.subframe["area"])[0])
    box_area = float(np.asarray(box.subframe["area"])[0])
    assert np.isclose(box_area, WIDTH**2 * HOLLOW * (2 - HOLLOW), rtol=1e-12)
    # the 64-gon under-fills its circle, which is the whole of the discrepancy
    exact = np.pi / 4 * WIDTH**2 * HOLLOW * (2 - HOLLOW)
    assert np.isclose(skin_area, exact, rtol=2e-03)
    assert np.isclose(box_area / skin_area, 4 / np.pi, rtol=2e-03)


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
