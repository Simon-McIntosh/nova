"""The prism: a straight segment coupled through its polygon cross-section.

The reduction is held to three independent references, in the order they were
built, and each answers something the others cannot:

* ``Beam``, on the rectangle the two elements share.  Two independent closed
  forms of the same integral -- a corner tensor over an axis-aligned box against
  a contour sum over a general edge -- so the agreement is EXACT rather than
  asymptotic, and round-off is the whole tolerance.  It is also the only
  reference that pins the sign and normalisation conventions the frame expects.
* a converged quadrature of the filament kernel over the section, which is the
  only reference that reaches a section ``Beam`` cannot express.  The rule is a
  SIGNED triangle fan pitched at the target itself, graded radially and split
  angularly at the ray of closest approach -- three choices which between them
  turn the kernel's logarithm into an analytic integrand rather than something to
  be resolved by brute refinement.  It converges to round-off at forty-eight
  nodes, which is asserted before it is used, and :func:`section_average` sets out
  what each of the three is for.
* the large-radius limit of the swept element :class:`nova.biot.polybow.PolyBow`,
  at fixed section and fixed arc LENGTH, which is an independent check on the arc
  evaluation as much as on this one.  Measured rather than asserted as a limit,
  and what it measures on the way is where the ARC's own conditioning stops it.

The frame's stored matrices are read rather than
:attr:`nova.frame.coilset.CoilSet.point`'s accessors: the accessors cast to
single precision, at which every comparison here reads as exact.
"""

import numpy as np
import pytest
import shapely.geometry
from scipy.constants import mu_0

from nova.biot.beam import Beam
from nova.biot.biotframe import Source, Target
from nova.biot.polybeam import PolyBeam, polygon_beam_greens
from nova.biot.polybow import section_area, section_corners
from nova.biot.polygonarc import polygon_arc_greens
from nova.biot.solve import Solve
from nova.frame.coilset import CoilSet

FIELD_ATTRS = ["Ax", "Ay", "Az", "Bx", "By", "Bz"]

WIDTH, HEIGHT = 0.06, 0.04
RADIUS, ELEVATION = 3.0, 0.2
AXIS = np.array([0.35, -0.12])  # the section is placed off the local origin
LIMITS = (-0.35, 0.55)

# Sections spanning what a general edge has to get right: edges along both local
# axes, oblique edges, a corner count above four, and a boundary that turns the
# other way.  Placed absolutely, because the reduction locates the section rather
# than offsetting it from the axis.
SECTIONS = {
    "rectangle": np.array(
        [
            [-WIDTH / 2, -HEIGHT / 2],
            [WIDTH / 2, -HEIGHT / 2],
            [WIDTH / 2, HEIGHT / 2],
            [-WIDTH / 2, HEIGHT / 2],
        ]
    ),
    "hexagon": np.stack(
        [
            WIDTH / 2 * np.cos(np.arange(6) * np.pi / 3),
            HEIGHT / 2 * np.sin(np.arange(6) * np.pi / 3),
        ],
        axis=-1,
    ),
    "trapezoid": np.array(  # one edge on each axis, two oblique
        [
            [-WIDTH / 2, -HEIGHT / 2],
            [WIDTH / 2, -HEIGHT / 2],
            [WIDTH / 4, HEIGHT / 2],
            [-WIDTH / 2, HEIGHT / 4],
        ]
    ),
    "wedge": np.array(  # concave, which the signed fan tiles like any other
        [
            [-WIDTH / 2, -HEIGHT / 2],
            [WIDTH / 2, -HEIGHT / 2],
            [WIDTH / 2, HEIGHT / 2],
            [0.0, 0.0],
            [-WIDTH / 2, HEIGHT / 2],
        ]
    ),
}


def section(name: str) -> np.ndarray:
    """Return a named section placed on the prism's axis."""
    return SECTIONS[name] + AXIS


def targets(name: str) -> np.ndarray:
    """Return a target cloud spanning the distances the bands are drawn in.

    Every entry is deliberate: a far field, a near field, the section's own
    corners and edges reached from outside and from inside, the axis, and the
    plane of each end face.  A target aligned with a corner is where ``Beam``'s
    own ratios go singular, and a target inside the conductor is the reason a
    finite-section element exists at all.
    """
    corners = section(name)
    edge = 0.5 * (corners[0] + corners[1])
    inside = 0.5 * (AXIS + corners[0])
    return np.array(
        [
            [AXIS[0] + 0.8, AXIS[1] + 0.5, 0.1],
            [AXIS[0] + 0.09, AXIS[1] + 0.02, 0.1],
            [AXIS[0] + 0.031, AXIS[1], 0.1],
            [corners[0][0], corners[0][1], 0.1],
            [corners[1][0], corners[1][1], LIMITS[1]],
            [edge[0], edge[1], 0.1],
            [edge[0], edge[1], LIMITS[0]],
            [AXIS[0], AXIS[1], 0.1],
            [AXIS[0], AXIS[1], LIMITS[0]],
            [inside[0], inside[1], 0.2],
            [inside[0], inside[1], LIMITS[1]],
        ]
    )


# ---------------------------------------------------------------------------
# The converged quadrature, and its own convergence.


def filament_rows(offset_x, offset_y, axial_1, axial_2):
    """Return the filament kernel's ``(A_z, B_x, B_y)`` at an in-plane offset.

    :class:`nova.biot.line.Line`'s own rows, per unit current and in nova's
    convention, written on the offset from the target rather than on a pair of
    absolute positions: the quadrature pitches its fan AT the target, so the
    offsets are the small quantities the rule generates and forming them by
    subtracting two coordinates of order the major radius would cost every digit
    the grading is there to win.
    """
    plane = np.hypot(offset_x, offset_y)
    rows = np.zeros((3,) + plane.shape)
    for axial, sign in ((axial_2, 1.0), (axial_1, -1.0)):
        radius = np.sqrt(plane**2 + axial**2)
        rows[0] += sign * np.arcsinh(axial / plane)
        rows[1] += sign * axial / (radius * plane**2) * offset_y
        rows[2] += sign * axial / (radius * plane**2) * -offset_x
    return rows / (4 * np.pi)


def section_average(x, y, z, vertices, z1, z2, order=48, grade=3):
    """Return the section average of the filament kernel, by a signed fan.

    Triangle ``i`` spans the target and edge ``i`` and carries the SIGN of its own
    cross product, which is what lets the fan be pitched at the target wherever
    the target lies: a signed fan sums to the polygon from any apex, inside it, on
    its boundary or outside it altogether.  Putting the apex on the kernel's
    logarithm is then worth doing, because the singularity becomes a coordinate
    line of the map rather than a point in the interior of a panel, and both of
    the map's variables can be shaped around it:

    * RADIALLY, ``t = s**grade`` along the ray from the apex.  Its jacobian
      ``grade * s**(2 grade - 1)`` vanishes fast enough at the apex that
      ``log t`` integrates as an analytic function of ``s``.
    * ANGULARLY, the ray of closest approach.  The offset at fixed radius is the
      affine interpolation between the edge's two ends, whose LENGTH carries a
      logarithm of its own, and for an apex just outside the section that length
      passes close to zero part way along the edge -- a near-singularity interior
      to the angular range, which is what a plain product rule misses by four
      orders.  Splitting the range at the perpendicular foot puts it on an
      endpoint of each piece instead.

    With both, the rule is at round-off by forty-eight nodes for every section
    and every target here, which :func:`test_the_section_quadrature_is_converged`
    asserts before anything is measured against it.
    """
    corners = np.asarray(vertices, dtype=float)
    rolled = np.roll(corners, -1, axis=0)
    area = section_area(corners)
    x, y, z = (np.atleast_1d(np.asarray(value, dtype=float)) for value in (x, y, z))
    node, weight = np.polynomial.legendre.leggauss(order)
    node = 0.5 * (node + 1.0)
    weight = 0.5 * weight
    scaled = node**grade
    jacobian = grade * node ** (grade - 1)
    rows = np.zeros((3, len(x)))
    for index in range(len(x)):
        apex = np.array([x[index], y[index]])
        total = np.zeros(3)
        for start, end in zip(corners - apex, rolled - apex):
            cross = float(start[0] * end[1] - start[1] * end[0])
            if cross == 0.0:
                continue
            edge = end - start
            foot = -float(start @ edge) / float(edge @ edge)
            break_point = [0.0] + ([foot] if 0.0 < foot < 1.0 else []) + [1.0]
            for low, high in zip(break_point[:-1], break_point[1:]):
                ray, along = np.meshgrid(node, low + (high - low) * node, indexing="ij")
                quadrature = np.outer(weight, weight * (high - low))
                offset = (ray**grade)[..., np.newaxis] * (
                    (1 - along)[..., np.newaxis] * start + along[..., np.newaxis] * end
                )
                kernel = filament_rows(
                    offset[..., 0], offset[..., 1], z1 - z[index], z2 - z[index]
                )
                total += cross * np.einsum(
                    "uv,kuv->k",
                    quadrature * (scaled * jacobian)[:, np.newaxis],
                    kernel,
                )
        rows[:, index] = total / area
    return rows


def contour_distance(name: str, target: np.ndarray) -> np.ndarray:
    """Return each target's transverse distance to the section CONTOUR.

    The contour rather than the centroid, because that is what the reduction's
    accuracy tracks: a target a centroid-distance away can sit on an edge of a
    thin section, and a target inside the conductor has no centroid distance worth
    banding by at all.
    """
    boundary = shapely.geometry.LinearRing(section(name))
    return np.array(
        [boundary.distance(shapely.geometry.Point(*point[:2])) for point in target]
    )


def worst_by_row(got, want):
    """Return the worst deviation of each row against its OWN scale."""
    got, want = np.asarray(got), np.asarray(want)
    scale = np.max(np.abs(want), axis=-1)
    scale = np.where(scale > 0, scale, 1.0)[..., np.newaxis]
    return np.max(np.abs(got - want) / scale, axis=-1)


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_the_section_quadrature_is_converged(name):
    """The oracle asserts its own convergence before anything is measured against it.

    Raised until it stops moving, and the last step is what is asserted: at
    forty-eight nodes the rule agrees with the sixty-four-node one to round-off on
    every row and every target, including the ones on the contour where the
    integrand is singular and the one just outside it where the singularity is
    near but not reached.  A rule still moving at its finest setting would put its
    own truncation into every band below.
    """
    target = targets(name)
    coarse, fine = (
        section_average(*target.T, section(name), *LIMITS, order=order)
        for order in (48, 64)
    )
    assert np.max(worst_by_row(coarse, fine)) <= 1e-13  # measured 5.3e-15


@pytest.mark.parametrize("name", sorted(SECTIONS))
def test_the_closed_form_matches_the_converged_quadrature(name):
    """The reduction against the section average, banded by contour distance.

    A prism has no modulus approaching unity and no parameter confluence to
    manage, so there is no near-field band to widen: the envelope is round-off at
    every distance, on the contour and inside the conductor as much as far from
    it.  The bands are drawn anyway, and the near one is the tighter of the two,
    because a band that only ever holds where the answer is easy measures nothing.
    """
    target = targets(name)
    got = np.stack(
        polygon_beam_greens(
            target[:, 0], target[:, 1], target[:, 2], section(name), *LIMITS
        )
    )
    want = section_average(*target.T, section(name), *LIMITS)
    assert np.all(np.isfinite(got))
    distance = contour_distance(name, target)
    scale = max(np.max(np.ptp(section(name), axis=0)), 1e-12)
    near = distance <= 0.5 * scale
    envelope = np.abs(got - want) / np.max(np.abs(want), axis=-1)[:, np.newaxis]
    assert np.max(envelope[:, near]) <= 1e-12  # measured 1.5e-13
    assert np.max(envelope[:, ~near]) <= 1e-11  # measured 4.1e-13


# ---------------------------------------------------------------------------
# Beam, on the rectangle the two elements share.


def straight_winding(cross_section, segment=None, angle=(0.1, 0.4, 0.7)):
    """Return a solved coilset carrying one thickened straight-segment winding.

    ``minimum_arc_nodes`` of zero is what keeps the path straight: an arc fit over
    three points on a circle would otherwise take them as one arc, and the chord
    is what a thickened LINE element exists to carry.  Two segments, so the
    generator is exercised with more than one source and each carries its own
    frame.
    """
    node = np.asarray(angle, dtype=float)
    path = np.stack(
        [
            RADIUS * np.cos(node),
            RADIUS * np.sin(node),
            ELEVATION * np.ones_like(node),
        ],
        axis=-1,
    )
    coilset = CoilSet(field_attrs=FIELD_ATTRS)
    coilset.winding.insert(
        path,
        cross_section,
        nturn=1,
        Ic=1,
        minimum_arc_nodes=0,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment
    coilset.point.solve(FRAME_TARGETS)
    return coilset


FRAME_TARGETS = np.array(
    [
        [3.4, 0.9, 0.5],
        [2.6, 1.1, -0.3],
        [3.0, 1.16, 0.21],
        [3.02, 0.9, 0.2],
        [2.99, 0.7, 0.19],
    ]
)


def frame_rows(coilset):
    """Return the six global rows per ampere, summed over the sources."""
    return np.stack(
        [np.asarray(coilset.point.data[attr]).sum(axis=1) for attr in FIELD_ATTRS]
    )


def element_inputs(cross_section):
    """Return a source and target built from one straight winding."""
    coilset = straight_winding(cross_section)
    frame = coilset.subframe
    source = Source(
        {column: np.asarray(frame[column]) for column in frame.columns},
        index=list(np.asarray(frame.index)),
    )
    target = Target(
        {
            "x": FRAME_TARGETS[:, 0],
            "y": FRAME_TARGETS[:, 1],
            "z": FRAME_TARGETS[:, 2],
        }
    )
    return source, target


def elements(cross_section):
    """Return a ``(Beam, PolyBeam)`` pair for a rectangular section."""
    source, target = element_inputs(cross_section)
    return Beam(source, target), PolyBeam(source, target)


def prism_element(cross_section):
    """Return a polygon-prism element without constructing a rectangle element."""
    source, target = element_inputs(cross_section)
    return PolyBeam(source, target)


def globalise(element, potential, field):
    """Return the global vectors a local row triple rotates to.

    The rotation and the ``mu0`` the field carries are
    :class:`nova.biot.matrix.Matrix`'s, applied here to a row triple the caller
    formed, so a comparison against an element's own ``Avector`` and ``Bvector``
    is a statement about the ROWS and not about the transform.
    """
    return (
        element.loc.rotate(np.stack(potential, axis=-1), "to_global"),
        mu_0 * element.loc.rotate(np.stack(field, axis=-1), "to_global"),
    )


def beam_rectangle(beam):
    """Return the rectangle ``Beam`` integrates, per source, in local corners."""
    corner = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return [
        np.array([[beam.xs[i, 0, column], beam.ys[j, 0, column]] for i, j in corner])
        for column in range(beam.shape[1])
    ]


def test_a_rectangular_section_reproduces_beam_to_round_off():
    """The exact oracle, on all six global components.

    ``Beam`` sums a ``(2, 2, 2)`` corner tensor of an axis-aligned box; this sums
    a contour of general edges.  The two are independent closed forms of the same
    integral, so they agree to round-off and not to a truncation -- and both are
    rotated to global by the same frame, which is what makes the six components a
    statement about the rows.  Every quantity is read at double precision:
    ``CoilSet.point``'s accessors are single, at which a wrong element reads as
    exact.
    """
    beam, prism = elements({"rect": (0, 0, WIDTH, HEIGHT)})
    zero = np.zeros(beam.shape)
    rows = np.zeros((3,) + beam.shape)
    for column, vertices in enumerate(beam_rectangle(beam)):
        rows[:, :, column] = np.stack(
            polygon_beam_greens(
                np.asarray(beam("target", "x"))[:, column],
                np.asarray(beam("target", "y"))[:, column],
                np.asarray(beam("target", "z"))[:, column],
                vertices,
                np.asarray(beam("source", "z1"))[:, column],
                np.asarray(beam("source", "z2"))[:, column],
            )
        )
    # the frame's area column is the swept footprint's, which is the density both
    # elements work at; the reduction normalised by the polygon it integrated
    scale = np.array(
        [
            section_area(vertices) / float(np.asarray(beam.source["area"])[column])
            for column, vertices in enumerate(beam_rectangle(beam))
        ]
    )
    potential, field = globalise(
        beam, (zero, zero, scale * rows[0]), (scale * rows[1], scale * rows[2], zero)
    )
    # measured 8.9e-15 on the potential and 1.1e-14 on the field
    assert np.max(worst_by_row(potential.T, beam.Avector.T)) <= 1e-13
    assert np.max(worst_by_row(field.T, beam.Bvector.T)) <= 1e-13
    assert prism.name == "polybeam"


def test_beam_is_finite_at_a_transverse_section_corner():
    """Bounded products preserve all three local rows at an aligned corner."""
    template, _ = elements({"rect": (0, 0, WIDTH, HEIGHT)})
    vertices = beam_rectangle(template)[0]
    local_target = np.array([vertices[2, 0], vertices[2, 1], 0.1])
    axes = template.coordinate_axes[0, 0]
    origin = template.coordinate_origin[0, 0]
    global_target = np.einsum("i,ji->j", local_target, axes) + origin
    source = Source(
        {
            column: np.asarray(template.source[column])[:1]
            for column in template.source.columns
        },
        index=[template.source.index[0]],
    )
    target = Target(
        {"x": [global_target[0]], "y": [global_target[1]], "z": [global_target[2]]}
    )
    beam = Beam(source, target)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        got = np.array(
            [
                beam._intergrate(beam._Az_hat)[0, 0],
                beam._intergrate(beam._Bx_hat)[0, 0],
                beam._intergrate(beam._By_hat)[0, 0],
            ]
        )
    target_x = np.asarray(beam("target", "x"))[:, 0]
    target_y = np.asarray(beam("target", "y"))[:, 0]
    target_z = np.asarray(beam("target", "z"))[:, 0]
    source_z1 = float(np.asarray(beam("source", "z1"))[0, 0])
    source_z2 = float(np.asarray(beam("source", "z2"))[0, 0])
    expected = section_average(
        target_x, target_y, target_z, vertices, source_z1, source_z2
    )[:, 0]
    expected *= section_area(vertices) / float(np.asarray(source["area"])[0])
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# The two limits: the filament it thickens, and the arc it straightens.


def test_the_far_field_is_the_filament_it_thickens():
    """Far from the conductor the prism is the line on its axis.

    The leading departure is the section's own second moment over the squared
    distance, so it is the RATE that is the statement rather than the agreement:
    a target ten section widths out and one a hundred out differ by two orders.
    """
    corners = section("hexagon")
    envelope = []
    for reach in (10.0, 100.0):
        offset = reach * WIDTH
        target = np.array([[AXIS[0] + offset, AXIS[1] + 0.4 * offset, 0.1]])
        got = np.stack(
            polygon_beam_greens(
                target[:, 0], target[:, 1], target[:, 2], corners, *LIMITS
            )
        )
        want = filament_rows(
            AXIS[0] - target[:, 0],
            AXIS[1] - target[:, 1],
            LIMITS[0] - target[:, 2],
            LIMITS[1] - target[:, 2],
        )
        envelope.append(float(np.max(np.abs(got - want) / np.abs(want))))
    assert envelope[0] <= 2e-03  # measured 1.1e-03
    assert envelope[1] <= 2e-05  # measured 1.1e-05
    assert envelope[0] / envelope[1] >= 50  # second order, measured 100


def test_the_swept_element_straightens_onto_the_prism():
    """``PolyBow`` at fixed section and fixed arc LENGTH, as its radius grows.

    A common frame first: with the local ``x`` axis on the radial direction and
    the local ``z`` on the toroidal, right-handedness puts the local ``y`` on
    MINUS the vertical -- so the section and the target both mirror, and the
    prism's ``B_y`` is minus the arc's ``B_z``.  Getting that wrong is a sign
    error the magnitudes cannot see.

    The departure is first order in the reciprocal radius and the table is the
    evidence.  It stops improving beyond a radius of about ``1e4`` -- not because
    the limit fails but because the ARC's rows are differences of quantities of
    order the squared major radius, so the comparison there measures the arc
    kernel's own conditioning.  The potential loses first, at a radius near
    ``1e3``, being the row whose cancellation is worst.
    """
    length = 0.5
    base = SECTIONS["hexagon"]
    offset = np.array([[0.10, 0.05], [-0.08, 0.02], [0.0, 0.12], [0.012, 0.006]])
    measured = {}
    for radius in (10.0, 1e02, 1e03):
        vertices = base + np.array([radius, 0.0])
        target_r, target_z = radius + offset[:, 0], offset[:, 1]
        arc = (
            np.stack(
                polygon_arc_greens(
                    target_r,
                    target_z,
                    np.zeros_like(target_r),
                    vertices,
                    -0.5 * length / radius,
                    0.5 * length / radius,
                )
            )
            / mu_0
        )
        prism = np.stack(
            polygon_beam_greens(
                target_r,
                -target_z,
                np.zeros_like(target_r),
                vertices * np.array([1.0, -1.0]),
                -0.5 * length,
                0.5 * length,
            )
        )
        measured[radius] = worst_by_row(
            np.stack([arc[1], arc[2], arc[4]]),
            np.stack([prism[0], prism[1], -prism[2]]),
        )
        # a symmetric sweep read at its own mid-azimuth has no radial potential
        # and no toroidal field, exactly, at every radius
        assert np.max(np.abs(arc[0])) == 0.0
        assert np.max(np.abs(arc[3])) == 0.0
    assert np.max(measured[10.0]) <= 1e-02  # measured 7.9e-03
    assert np.max(measured[1e02]) <= 1e-03  # measured 8.0e-04
    assert np.max(measured[1e03]) <= 1e-04  # measured 8.0e-05
    for row in range(3):
        ratio = measured[1e02][row] / measured[1e03][row]
        assert 8.0 <= ratio <= 12.0  # first order in 1/radius, measured 9.9-10.0


# ---------------------------------------------------------------------------
# The registry, and what the class declares itself to be.


def test_the_segment_registry_reaches_the_element():
    """A frame labelled ``polybeam`` builds this class and nothing else."""
    assert Solve.generator["polybeam"] is PolyBeam
    assert PolyBeam.name == "polybeam"
    assert PolyBeam.axisymmetric is False


def test_the_element_is_a_peer_of_beam_rather_than_a_subclass():
    """The decision, asserted where it can be read.

    ``Beam``'s body is a rectangle's corner tensor and four ratios whose
    denominators are its offsets; a polygon section needs none of it.  What the
    two share is the frame plumbing, which is ``Matrix``'s, so they are siblings
    on it.
    """
    from nova.biot.matrix import Matrix

    assert issubclass(PolyBeam, Matrix)
    assert not issubclass(PolyBeam, Beam)
    assert issubclass(Beam, Matrix)


# ---------------------------------------------------------------------------
# What the class adds to the reduction: a section, and a frame transform.


def test_the_section_is_read_out_of_the_frames_own_projection():
    """The column is a poloidal PROJECTION, and the section plane is not it.

    A thickened straight segment reaches the frame as a chord, whose section plane
    is turned from the radial direction by half the chord's own angle -- so the
    footprint the column carries is narrower than the conductor by that cosine,
    which for the chord here is 1.1e-02 and would be the element's error if the
    column were read as the section.  Inverting the frame's own map recovers the
    authored section instead, to the second-order term the projection's curvature
    leaves.
    """
    _, prism = elements({"rect": (0, 0, WIDTH, HEIGHT)})
    projection = prism._projection
    assert np.allclose(projection[:, 0, 1], 0.0, atol=1e-14)
    assert np.allclose(projection[:, 1, 0], 0.0, atol=1e-14)
    assert np.allclose(projection[:, 1, 1], -1.0, atol=1e-14)
    cosine = float(np.cos(0.5 * 0.3))  # half the chord's turn
    assert np.allclose(projection[:, 0, 0], cosine, rtol=1e-12)
    for vertices in prism._section_vertices:
        extent = np.ptp(vertices, axis=0)
        assert np.allclose(extent, [WIDTH, HEIGHT], rtol=2e-04)  # measured 1.1e-04
        assert np.isclose(section_area(vertices), WIDTH * HEIGHT, rtol=1e-05)
    # what reading the column unmapped would have cost, which is what the map buys
    for poly in np.asarray(prism.source["poly"]):
        footprint = np.ptp(section_corners(poly), axis=0)[0]
        assert np.isclose(footprint, WIDTH * cosine, rtol=1e-06)


def test_the_axis_is_the_local_z_and_the_section_rides_on_it():
    """A straight segment holds its transverse position along its whole length.

    The local frame puts the path on the local ``z``, which is what lets one
    section serve both axial limits -- and is the assumption the reduction's
    single corner list encodes.
    """
    _, prism = elements({"rect": (0, 0, WIDTH, HEIGHT)})
    for coord in ("x", "y"):
        start = np.asarray(prism("source", f"{coord}1"))
        end = np.asarray(prism("source", f"{coord}2"))
        assert np.max(np.abs(start - end)) <= 1e-12
        assert np.max(np.abs(start - start[0])) <= 1e-12  # and across the targets


def test_a_rectangular_section_reproduces_beam_through_the_frame():
    """The whole path end to end, at the accuracy the projection round trip has.

    Bounded from above AND from below.  From above because the two elements
    integrate the same rectangle and normalise by the same area column, so all
    that separates them is the section the class recovered from a projection.
    From below because the frame's stored matrices carry more than the accessors
    report: at single precision the two would tie, and a tie is what a class
    returning ``Beam``'s own numbers would also produce.
    """
    beam = frame_rows(straight_winding({"rect": (0, 0, WIDTH, HEIGHT)}, "beam"))
    prism = frame_rows(straight_winding({"rect": (0, 0, WIDTH, HEIGHT)}, "polybeam"))
    assert np.max(worst_by_row(prism, beam)) <= 1e-04  # measured 1.9e-05
    assert np.max(np.abs(prism - beam)) / np.max(np.abs(beam)) >= 1e-07


def test_a_free_form_polygon_section_reaches_the_kernel():
    """The capability no descriptor can express, which is why the column is read.

    An irregular pentagon has no ``(width, height)`` pair that reproduces it, so a
    class rebuilding its section from the frame's descriptor could only ever
    return a rectangle of its bounding box.  Read from ``poly`` and mapped into
    the section plane it arrives as itself, corner for corner.
    """
    loop = np.array(
        [[-0.03, -0.02], [0.03, -0.02], [0.02, 0.0], [0.03, 0.02], [-0.01, 0.015]]
    )
    coilset = straight_winding(shapely.geometry.Polygon(loop))
    prism = prism_element(shapely.geometry.Polygon(loop))
    # the local frame's y axis is MINUS the vertical -- right-handedness demands it,
    # with x on the radial direction and z on the path -- so the section arrives
    # mirrored in its second coordinate.  Asserted on an ASYMMETRIC section, because
    # that is the only kind that can tell: every named section is symmetric about
    # both its own axes and would read as correct either way round.
    want = loop * np.array([1.0, -1.0])
    for vertices in prism._section_vertices:
        assert len(vertices) == len(loop)
        order = [
            int(np.argmin(np.linalg.norm(vertices - corner, axis=1))) for corner in want
        ]
        assert sorted(order) == list(range(len(loop)))
        assert np.max(np.abs(vertices[order] - want)) <= 2e-04  # measured 3.4e-06
    assert np.all(np.isfinite(frame_rows(coilset)))


def test_a_swept_hexagon_reaches_the_kernel_with_six_corners():
    """A projection splits an edge; the reduction should not pay for it.

    The union of a sweep's projected stations puts a corner part way along the
    section's own edges.  A closed-form section reduction costs one evaluation per
    corner, so the run is collapsed before the section is handed over.
    """
    prism = prism_element({"hex": (0, 0, WIDTH, HEIGHT)})
    for vertices in prism._section_vertices:
        assert len(vertices) == 6


def test_beam_rejects_a_hexagonal_section():
    """A rectangle-only kernel cannot be forced onto a six-corner section."""
    with pytest.raises(ValueError, match="axis-aligned rectangle"):
        straight_winding({"hex": (0, 0, WIDTH, HEIGHT)}, "beam")


def test_automatic_straight_section_routing_preserves_geometry():
    """Named rectangles use Beam while general sections use PolyBeam."""
    rectangle = straight_winding({"rect": (0, 0, WIDTH, HEIGHT)})
    hexagon = straight_winding({"hex": (0, 0, WIDTH, HEIGHT)})
    assert np.array_equal(np.unique(rectangle.subframe.segment), ["beam"])
    assert np.array_equal(np.unique(hexagon.subframe.segment), ["polybeam"])


@pytest.mark.parametrize(
    "cross_section,ratio",
    [
        ({"disc": (0, 0, WIDTH, WIDTH)}, 4.0 / np.pi),
        ({"ellipse": (0, 0, WIDTH, HEIGHT)}, 4.0 / np.pi),
    ],
)
def test_beam_rejects_a_round_section(cross_section, ratio):
    """Circular and elliptical sections remain on their polygon route."""
    assert ratio == 4.0 / np.pi
    with pytest.raises(ValueError, match="axis-aligned rectangle"):
        straight_winding(cross_section, "beam")


# ---------------------------------------------------------------------------
# A hollow section, as a coupled pair of solid ones.

HOLLOW = 0.2  # hollowness factor, 1 - r/R


def annulus_cells(outer, core):
    """Return a partition of an annulus into solid quadrilateral cells.

    Built between CORRESPONDING corners of the two boundaries, which takes work
    the pair itself does not need: the two arrive wound opposite ways round and
    starting from different corners, because an interior ring is stored reversed
    with respect to its exterior, and a cell spanning a mismatched pair is a
    self-intersecting quadrilateral whose shoelace area is meaningless.  Wound the
    same way and rolled onto a common start, the cells tile exactly the material
    the pair means.
    """

    def counterclockwise(loop):
        rolled = np.roll(loop, -1, axis=0)
        signed = np.sum(loop[:, 0] * rolled[:, 1] - rolled[:, 0] * loop[:, 1])
        return loop if signed > 0 else loop[::-1].copy()

    outer, core = counterclockwise(outer), counterclockwise(core)
    assert len(outer) == len(core)
    centre = outer.mean(axis=0)
    bearing = [np.arctan2(*(loop - centre).T[::-1]) for loop in (outer, core)]
    turn = np.angle(np.exp(1j * (bearing[1] - bearing[0][0])))
    core = np.roll(core, -int(np.argmin(np.abs(turn))), axis=0)
    return [
        np.array(
            [outer[i], outer[(i + 1) % len(outer)], core[(i + 1) % len(core)], core[i]]
        )
        for i in range(len(outer))
    ]


@pytest.mark.parametrize("name", ["box", "sk"])
def test_a_hollow_section_is_a_coupled_pair_rather_than_a_refusal(name):
    """An annulus, which one corner list cannot carry, by superposition.

    The outer boundary at ``+j`` and the interior one at ``-j``, both of them solid
    sections the reduction already evaluates, both carrying the ANNULUS as their
    area because that is what sets the density::

        I_outer + I_core = j (A_outer - A_core) = I

    Checked against a partition of the annulus into solid quadrilateral cells --
    the same integral decomposed the other way, a sum of positive cells instead of
    a signed pair -- so the agreement is a statement about the superposition and
    not about the kernel.
    """
    coilset = straight_winding({name: (0, 0, WIDTH, HOLLOW)}, "polybeam")
    prism = prism_element({name: (0, 0, WIDTH, HOLLOW)})
    assert len(coilset.subframe) == 4  # two segments, each an outer and a core
    factor = np.asarray(coilset.subframe["factor"], dtype=float)
    assert factor.tolist() == [1.0, 1.0, -1.0, -1.0]
    area = np.asarray(coilset.subframe["area"], dtype=float)
    outer, core = prism._section_vertices[0], prism._section_vertices[2]
    assert np.allclose(area, area[0], rtol=1e-14)  # every member at the annulus
    # the column is the FOOTPRINT's annulus, so the sections the element recovered
    # exceed it by the projection's own determinant, exactly and on both members
    tilt = abs(float(np.linalg.det(prism._projection[0])))
    assert np.isclose(
        (section_area(outer) - section_area(core)) * tilt, area[0], rtol=1e-04
    )
    assert np.all(np.isfinite(frame_rows(coilset)))

    # the pair against the annulus integrated directly, cell by cell: a partition
    # into solid quadrilaterals covers exactly the material the pair means, with no
    # cancellation between members, so the agreement is about the superposition
    cell = annulus_cells(outer, core)
    angle = np.linspace(0, 2 * np.pi, 7)[:-1]
    reach = 0.6 * np.max(np.abs(outer))
    local = np.stack(
        [reach * np.cos(angle), reach * np.sin(angle), np.full_like(angle, 0.1)],
        axis=-1,
    )

    def rows(vertices):
        """Return the section's rows weighted by the area it covers."""
        return section_area(vertices) * np.stack(
            polygon_beam_greens(*local.T, vertices, *LIMITS)
        )

    partition = sum(rows(quad) for quad in cell)
    assert np.isclose(
        sum(section_area(quad) for quad in cell),
        section_area(outer) - section_area(core),
        rtol=1e-12,
    )
    pair = rows(outer) - rows(core)
    # the skin's sixty-four cells are slivers whose sum cancels, which is where the
    # two orders between the two sections go; the box's four cells are not
    assert np.max(worst_by_row(partition, pair)) <= 1e-10  # measured 1.3e-13, 4.3e-11


def test_two_sources_share_one_segment_and_sum():
    """One generator carries every source of its segment, each with its own section.

    The frame sums their columns, and the sum is what the stored matrix holds --
    so a segment carrying two sources is checked against the two evaluated apart.
    """
    coilset = straight_winding({"hex": (0, 0, WIDTH, HEIGHT)}, "polybeam")
    prism = prism_element({"hex": (0, 0, WIDTH, HEIGHT)})
    assert len(coilset.subframe) == 2
    assert prism._rows.shape == (3, len(FRAME_TARGETS), 2)
    assert len(prism._section_vertices) == 2
    # the two chords are similar but not identical, so a generator that carried only
    # the first source would still look plausible on magnitudes alone
    assert np.max(np.abs(prism.Bvector[:, 0] - prism.Bvector[:, 1])) > 1e-08
    # the winding's members are linked, so the stored matrix carries one column and
    # it is the sum -- which is the element's own source axis, summed
    assert np.asarray(coilset.point.data["Bz"]).shape[1] == 1
    for row, attr in enumerate(FIELD_ATTRS[:3]):
        assert np.allclose(
            prism.Avector[..., row].sum(axis=1),
            np.asarray(coilset.point.data[attr]).sum(axis=1),
            rtol=1e-14,
            atol=0.0,
        )
    for row, attr in enumerate(FIELD_ATTRS[3:]):
        assert np.allclose(
            prism.Bvector[..., row].sum(axis=1),
            np.asarray(coilset.point.data[attr]).sum(axis=1),
            rtol=1e-14,
            atol=0.0,
        )
