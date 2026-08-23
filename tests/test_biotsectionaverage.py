"""What the target-section rule integrates, and what its order buys.

The rule in :mod:`nova.biot.sectionaverage` turns a finite-section kernel evaluated
at one target point into the DOUBLE integral an inductance operator wants. Four
things have to hold for that to mean anything: the rule must integrate the area it
claims to, its order must be converged on the hardest integrand it sees (a section's
own flux), the quantity it produces must be the self-inductance -- checked against a
published closed form rather than against itself -- and its ORDER has to be measured
against a reference built from neither of the two evaluations it sits between.

Everything here is the UNIFORM-CURRENT limit -- constant current density over the
whole of a section polygon, no jacket, no insulation, no void, no turn structure
inside it. The oracle below shares that assumption by construction, so what these
tests decide is whether the shipped rule evaluates the uniform-current double
integral correctly, and nothing more. None of them bears on a real winding pack,
which concentrates the same current into discrete sub-conductors and therefore sits
ABOVE this limit.

The brute-force oracle
----------------------
That reference is below: a direct four-dimensional quadrature over source section x
target section of the coaxial ring kernel, assembled from the complete elliptic
integrals and sharing no code with :mod:`nova.biot.polygonanalytic` (the closed form
the shipped rule drives) or with :mod:`nova.biot.sectionaverage` (the rule itself).
It handles the coincident-section case -- the only one with a singularity -- by
pinning the inner fan's degenerate vertex at the TARGET POINT, so the radial
Jacobian of the polar decomposition cancels the kernel's logarithm exactly and what
is left is ``t log t``, resolved to round-off by a mesh graded geometrically towards
the origin. It converges algebraically in the rule order rather than spectrally,
because the outer integrand's second derivatives are only bounded, so it is read at
a Richardson limit over three rungs of its own ladder.

Those limits are RECORDED here, not re-derived. A ladder to the orders the limit
needs is minutes of four-dimensional quadrature per section, and it returns the same
numbers every time -- they are constants of fixed geometry. Re-deriving them on every
run would make this file a benchmark that happens to assert. The ladder itself lives
in :mod:`benchmarks.section_average_oracle`, which prints every rung and is what to
run when the rule changes, when a section changes, or when a constant is doubted.
Each recorded constant is guarded by the fingerprint of the polygon it belongs to, so
a changed section fails rather than quietly comparing against the wrong number.
"""

import numpy as np
import pytest
from scipy.special import ellipe, ellipkm1
from shapely.geometry import MultiPolygon, Point, Polygon

from nova.biot.greens import section_centroid
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.sectionaverage import ORDER, averaged_greens, section_nodes

MU0 = 4.0e-7 * np.pi

SQUARE_GMD = 0.447049
"""The tabulated geometric mean distance of a square of unit side.

Grover's tabulated constant, and the same number
:class:`nova.biot.crosssection.CrossSection` carries as its ``square`` section
factor -- ``2 x 0.447049``, a diameter rather than a radius.
"""


def rectangle(major, height, width, depth):
    """Return a rectangular section's four ``(r, z)`` corners."""
    return np.array(
        [
            [major - width / 2, height - depth / 2],
            [major + width / 2, height - depth / 2],
            [major + width / 2, height + depth / 2],
            [major - width / 2, height + depth / 2],
        ]
    )


def hexagon(major, height, radius):
    """Return a regular hexagon's six ``(r, z)`` corners, one corner outboard."""
    angle = np.arange(6) * np.pi / 3.0
    return np.c_[major + radius * np.cos(angle), height + radius * np.sin(angle)]


def ring_flux(r, z, source_r, source_z):
    """Return the coaxial ring kernel [Wb/A], built from the elliptic integrals.

    The modulus complement is formed from the near separation over the far one, so
    the logarithm a coincident pair drives is carried by ``ellipkm1`` without ever
    subtracting the modulus from one. A coincident pair rounds the modulus itself
    just above one, outside ``ellipe``'s domain, so it is clipped there.
    """
    far = (r + source_r) ** 2 + (z - source_z) ** 2
    complement = ((r - source_r) ** 2 + (z - source_z) ** 2) / far
    modulus = np.sqrt(np.minimum(4.0 * r * source_r / far, 1.0))
    return (
        MU0
        * np.sqrt(r * source_r)
        * (
            (2.0 / modulus - modulus) * ellipkm1(complement)
            - 2.0 / modulus * ellipe(modulus**2)
        )
    )


def unit_interval(order):
    """Return Gauss-Legendre nodes and weights mapped onto ``(0, 1)``."""
    node, weight = np.polynomial.legendre.leggauss(order)
    return 0.5 * (node + 1.0), 0.5 * weight


def graded_interval(order, levels=5, ratio=0.25):
    """Return a rule on ``(0, 1]`` geometrically graded towards the origin.

    The radial direction of a fan pinned at the singular point leaves ``t log t``;
    a geometric mesh resolves that to round-off in a handful of levels where a
    single Gauss panel would converge at first order.
    """
    edge = [0.0] + [ratio**level for level in range(levels, -1, -1)]
    node, weight = unit_interval(order)
    return (
        np.concatenate(
            [low + (high - low) * node for low, high in zip(edge, edge[1:])]
        ),
        np.concatenate([(high - low) * weight for low, high in zip(edge, edge[1:])]),
    )


def signed_fan(vertices, apex, radial, angular):
    """Return ``(points, weights)`` over the polygon, fanned from ``apex``.

    Signed, so either orientation and a concave polygon both work, and the apex may
    sit anywhere -- inside the polygon, on its boundary, or outside it.
    """
    corner = np.asarray(vertices, dtype=float)
    start, end = corner - apex, np.roll(corner, -1, axis=0) - apex
    signed = 0.5 * (start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0])
    radius, radius_weight = radial
    along, along_weight = angular
    edge = (1.0 - along)[None, :, None] * start[:, None, :] + along[
        None, :, None
    ] * end[:, None, :]
    points = apex + radius[None, :, None, None] * edge[:, None, :, :]
    weights = (
        2.0
        * signed[:, None, None]
        * radius[None, :, None]
        * radius_weight[None, :, None]
        * along_weight[None, None, :]
    )
    return points.reshape(-1, 2), weights.reshape(-1)


def brute_force(target, source, order):
    """Return the ring kernel averaged over ``source`` and then over ``target``.

    The inner fan is pinned at the target point itself, which is what makes the
    coincident case ordinary; the outer fan is pinned at the target's own centroid,
    where the integrand is smooth.

    Minutes per section at the orders its limit needs, so nothing here calls it: the
    limits it produces are recorded below as constants and re-derived on demand by
    :mod:`benchmarks.section_average_oracle`, which is what imports this. It stays
    beside the primitives it is built from because a second copy of a reference
    implementation would drift from this one, and not drifting is its whole value.
    """
    radial, angular = graded_interval(order), unit_interval(2 * order)
    node, weight = signed_fan(target, section_centroid(target), radial, angular)
    value = np.empty(len(node))
    for index, point in enumerate(node):
        inner, inner_weight = signed_fan(source, point, radial, angular)
        value[index] = (
            inner_weight @ ring_flux(point[0], point[1], inner[:, 0], inner[:, 1])
        ) / inner_weight.sum()
    return float(weight @ value / weight.sum())


def section_fingerprint(vertices):
    """Return ``(area, centroid r, centroid z)`` -- what a recorded limit belongs to.

    A self-flux limit is a constant of ONE polygon. Recording the number without
    recording what it is a number for is how a stored reference rots: change a
    section definition and every comparison quietly moves to the wrong constant while
    still passing. This triple is asserted beside each recorded limit so that change
    fails instead.
    """
    corner = np.asarray(vertices, dtype=float)
    rolled = np.roll(corner, -1, axis=0)
    cross = corner[:, 0] * rolled[:, 1] - rolled[:, 0] * corner[:, 1]
    return (0.5 * float(cross.sum()), *section_centroid(corner))


SECTIONS = {
    "coil filament": rectangle(3.9431, 7.5641, 0.2398, 0.2460),
    "plasma cell": hexagon(6.2, 0.5, 0.075),
    "slender": rectangle(1.722, 5.313, 0.05, 0.5),
    "wall-clipped": np.array(
        [[6.10, 0.42], [6.24, 0.42], [6.27, 0.50], [6.18, 0.57], [6.10, 0.53]]
    ),
}

BRUTE_FORCE_SELF_FLUX = {
    "coil filament": 1.819748608e-05,
    "plasma cell": 3.767926648e-05,
    "slender": 5.903458509e-06,
    "wall-clipped": 3.630384886e-05,
}
"""Richardson limit of the oracle ladder on each section's own self term [Wb/A].

Orders 10, 12 and 14 of the four-dimensional quadrature described above, extrapolated
over the last three rungs. The ladders and their steps::

    coil filament  1.819753105e-05  1.819750928e-05  1.819749805e-05  ratio 1.94
    plasma cell    3.767928771e-05  3.767927711e-05  3.767927180e-05  ratio 2.00
    slender        5.903321680e-06  5.903422636e-06  5.903449104e-06  ratio 3.81
    wall-clipped   3.630388938e-05  3.630386955e-05  3.630385943e-05  ratio 1.96

The RATIO is the extrapolation's conditioning -- the first step over the second. A
ladder inside its asymptotic regime roughly halves its step, so the tail it implies
is about the size of the last step; a ratio near one divides by almost nothing and
extrapolates a rung that has not begun to converge. Everything recorded here sits
between 1.86 and 3.81. Re-derive with ``python -m benchmarks.section_average_oracle``.
"""

SECTION_FINGERPRINT = {
    "coil filament": (0.0589908000, 3.9431000000, 7.5641000000),
    "plasma cell": (0.0146141787, 6.2000000000, 0.5000000000),
    "slender": (0.0250000000, 1.7220000000, 5.3130000000),
    "wall-clipped": (0.0195500000, 6.1763086104, 0.4845950554),
}
"""``(area, centroid r, centroid z)`` each recorded limit above was taken on."""

ASPECT_MAJOR = 1.722
"""Major radius the aspect sweep holds fixed [m]."""

ASPECT_AREA = 0.719 * 2.075
"""Section area the aspect sweep holds fixed, so only the SHAPE varies [m^2]."""

ASPECT_SELF_FLUX = {
    0.5: 2.701725429e-06,
    1.0: 2.807142095e-06,
    2.0: 2.713280798e-06,
    2.89: 2.589510367e-06,
    10.0: 1.954746276e-06,
}
"""Oracle limit against height over width, at fixed area and major radius [Wb/A].

2.89 is the ITER CS section itself, which is the aspect the coil element meets. Step
ratios 1.92, 1.94, 1.92, 1.86, 3.81 -- all well conditioned. Aspect 5 is swept by the
benchmark and deliberately NOT recorded: its ladder gives 2.3369024e-06, 2.3368939e-06
and 2.3368858e-06, a step ratio of 1.03, so the extrapolation divides by 0.03 and
amplifies the last step thirtyfold. That is a rung outside its asymptotic regime, not
a limit, and nothing here asserts against it.
"""


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_weights_sum_to_the_section_area(name):
    """The rule integrates the polygon, so a constant integrand comes back exactly.

    The shoelace area is formed independently of the fan the rule sums, and the mean
    a caller takes divides by the rule's own weight sum -- so a constant's mean is
    that constant whatever the order, which is what makes a weighted mean of a
    kernel a mean rather than a rescaling of one.
    """
    vertices = SECTIONS[name]
    _, weights = section_nodes(vertices, ORDER)
    corner = np.asarray(vertices)
    rolled = np.roll(corner, -1, axis=0)
    local = corner - corner[0]
    rolled = np.roll(local, -1, axis=0)
    shoelace = 0.5 * np.sum(local[:, 0] * rolled[:, 1] - rolled[:, 0] * local[:, 1])
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(abs(shoelace), rel=1e-12, abs=0)
    assert np.sum(weights * 3.0) / weights.sum() == pytest.approx(3.0, rel=1e-15)


def test_a_concave_section_has_only_positive_area_weights():
    """A re-entrant wall clip is decomposed into material, never a signed fan."""
    vertices = np.array(
        [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)],
        dtype=float,
    )
    points, weights = section_nodes(vertices)
    assert np.all(np.isfinite(points))
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert weights.sum() == pytest.approx(7.0, rel=1e-14)
    np.testing.assert_allclose(
        weights @ points / weights.sum(), [1.3571428571428572, 1.5]
    )


@pytest.mark.parametrize("order", [1, 2, 3, ORDER])
def test_holes_and_disconnected_material_keep_positive_weights(order):
    """Interior voids and separated pieces contribute only their material area."""
    hollow = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    disconnected = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ]
    )
    for section, area in ((hollow, 12.0), (disconnected, 2.0)):
        points, weights = section_nodes(section, order=order)
        assert np.all(weights > 0.0)
        assert weights.sum() == pytest.approx(area, rel=1e-14)
        assert all(section.covers(Point(point)) for point in points)
        centroid = np.array([section.centroid.x, section.centroid.y])
        np.testing.assert_allclose(
            weights @ points / weights.sum(), centroid, rtol=2e-14, atol=2e-14
        )


@pytest.mark.parametrize("translation", [(1.0e8, -1.0e8), (1.0e9, -1.0e9)])
def test_section_area_is_stable_under_large_translation(translation):
    """A small section keeps its area when absolute coordinates are very large."""
    vertices = np.array([(0.0, 0.0), (0.2, 0.0), (0.2, 0.04), (0.0, 0.04)])
    _, weights = section_nodes(vertices + np.asarray(translation))
    assert weights.sum() == pytest.approx(0.008, rel=3e-6)


def test_nonpositive_section_area_is_rejected():
    """No mean can be formed from an empty or collinear target section."""
    with pytest.raises(ValueError, match="positive finite area"):
        section_nodes(np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]))


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_rule_reproduces_the_centroid_of_its_own_section(name):
    """First moments: the node stack's own centroid is the section's.

    A rule whose weighted mean of ``(r, z)`` missed the area centroid would carry a
    dipole error into every average it took, and a dipole error against a ring
    Green's function is first order in the section rather than second.
    """
    vertices = SECTIONS[name]
    points, weights = section_nodes(vertices, ORDER)
    centre = weights @ points / weights.sum()
    assert centre == pytest.approx(section_centroid(vertices), rel=1e-13, abs=0)


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_shipped_order_is_converged_on_the_coincident_term(name):
    """The self term against a doubled rule, which is the hardest case the rule sees.

    The integrand is the section's own flux: smooth on the open section but with only
    bounded second derivatives across it, so the convergence is algebraic and the
    order cannot be argued from polynomial exactness. Doubling the order is the
    convergence check.

    The bound is one, and it is set by the ASPECT RATIO rather than by the corner
    count: a compact section -- a coil filament, a plasma cell, a wall-clipped cell --
    holds a few parts in ten million, and a section ten times taller than it is wide
    holds a few parts in ten thousand, because the collapsed fan resolves the long
    direction with the same node count as the short one. That is the figure a caller
    with a genuinely slender section should raise the order against.
    """
    vertices = SECTIONS[name]
    got = averaged_greens([vertices], vertices, ORDER)
    want = averaged_greens([vertices], vertices, 2 * ORDER)
    # measured: 3.2e-06 psi and 3.1e-06 Bz compact, 2.0e-04 and 3.7e-04 at aspect 10
    assert abs(got[0][0] / want[0][0] - 1.0) < 1e-03
    assert abs(got[2][0] / want[2][0] - 1.0) < 1e-03


def test_the_double_integral_is_reciprocal_where_the_single_integral_is_not():
    """Two neighbouring sections, each averaged over the other, both ways.

    Green's reciprocity makes the DOUBLE integral symmetric exactly -- the same
    kernel integrated over the same pair of areas in the other order. The single
    integral at a point is not, and the gap between the two directions is what the
    target-section average removes; both are measured here so the comparison says
    which one moved.
    """
    left = rectangle(3.9, 7.5, 0.24, 0.24)
    right = rectangle(3.9 + 0.24, 7.5, 0.24, 0.24)
    forward = averaged_greens([right], left)[0][0]
    backward = averaged_greens([left], right)[0][0]
    assert forward == pytest.approx(backward, rel=1e-08, abs=0)  # measured 4.6e-10
    point_forward = polygon_analytic_greens(*section_centroid(right)[:, None], left)[0][
        0
    ]
    point_backward = polygon_analytic_greens(*section_centroid(left)[:, None], right)[
        0
    ][0]
    assert abs(point_forward / point_backward - 1.0) > 1e-04  # measured 1.1e-03


def test_the_oracle_reproduces_the_closed_form_where_both_are_ordinary():
    """The brute force against the closed form, at points inside and outside a section.

    The two share no code: one sums a polar fan of the elliptic-integral ring kernel,
    the other is the analytic reduction of the same area integral. Agreeing to ten
    digits at an exterior point, at an interior point, on the contour and on a corner
    is what makes each of them an independent check on the other, and it is the
    precondition for reading anything into the comparison the two tests below make.
    """
    source = rectangle(1.0, 0.0, 0.2, 0.2)
    radial, angular = graded_interval(10), unit_interval(20)
    for probe in ([1.5, 0.0], [1.05, 0.02], [1.0, 0.0], [1.0, 0.1], [1.1, 0.1]):
        probe = np.array(probe)
        node, weight = signed_fan(source, probe, radial, angular)
        got = (
            weight @ ring_flux(probe[0], probe[1], node[:, 0], node[:, 1])
        ) / weight.sum()
        want = polygon_analytic_greens(probe[:1], probe[1:], source)[0][0]
        assert got == pytest.approx(want, rel=1e-08, abs=0), probe


def test_the_coincident_integrand_is_bounded_on_the_closed_section():
    """What the rule does about the singularity: there is none left for it to do.

    The kernel is log-singular where target and source POINTS coincide, and the
    coincident term integrates over exactly that configuration -- but the source
    integral is taken first and in closed form, so what the target-side rule sees is
    the section's own flux, which is bounded and continuous on the CLOSED section,
    corners included. The rule therefore needs no grading, no exclusion disc and no
    singular subtraction. What it does pay is the second derivative, which is not
    bounded, and that is why the convergence measured above is algebraic rather than
    spectral and why the order is measured rather than argued.
    """
    vertices = SECTIONS["coil filament"]
    probe = np.vstack(
        [
            section_centroid(vertices),
            vertices,
            0.5 * (vertices + np.roll(vertices, -1, axis=0)),
        ]
    )
    value = polygon_analytic_greens(probe[:, 0], probe[:, 1], vertices)[0]
    assert np.all(np.isfinite(value))
    assert np.min(value) > 0.5 * np.max(value)  # measured 0.79 corner over centroid
    assert np.argmax(value) == 0  # the centroid links the most flux


@pytest.mark.parametrize("name", list(SECTIONS))
def test_the_shipped_order_holds_the_self_term_against_the_brute_force(name):
    """The rule's ORDER, measured against a reference built from neither evaluation.

    The doubling check above says the rule has stopped moving; it cannot say what it
    has stopped on. This one can, because the oracle shares no code with either the
    closed form or the fan. The shipped order lands within a few parts in a hundred
    thousand of it on every section the element meets, and doubling the order closes
    that to a couple of parts in a million -- which is the residual of the ORACLE's
    own Richardson limit, not of the rule.

    The limit is read from :data:`BRUTE_FORCE_SELF_FLUX` and the section it was taken
    on is checked first, so this asserts the same figures the ladder produced without
    paying for the ladder.
    """
    vertices = SECTIONS[name]
    assert section_fingerprint(vertices) == pytest.approx(
        SECTION_FINGERPRINT[name], rel=1e-09, abs=1e-12
    ), "the section moved; re-derive its limit with benchmarks.section_average_oracle"
    limit = BRUTE_FORCE_SELF_FLUX[name]
    shipped = averaged_greens([vertices], vertices, ORDER)[0][0]
    doubled = averaged_greens([vertices], vertices, 2 * ORDER)[0][0]
    # measured at the shipped order: 2.7e-06 coil filament, 2.0e-07 plasma cell,
    # 1.7e-06 wall-clipped, 1.9e-04 slender; doubled, 3.9e-07 / 8.9e-08 / 1.7e-07 /
    # 7.5e-06, and the compact three have then reached the oracle's own residual
    assert abs(shipped / limit - 1.0) < 1e-03
    assert abs(doubled / limit - 1.0) < 1e-04


def test_hexagonal_target_rule_convergence_against_the_double_integral_arbiter():
    """The plasma-cell fan closes monotonically onto the independent four-D limit.

    Orders one through five use 4, 16, 36, 64 and 100 target nodes.  Their relative
    self-flux errors are 2.428e-2, 4.262e-5, 3.752e-6, 1.398e-7 and 8.297e-9;
    higher orders reach the recorded arbiter's own roughly 9e-8 residual instead of
    defining a more accurate reference from the rule under test.
    """
    vertices = SECTIONS["plasma cell"]
    reference = BRUTE_FORCE_SELF_FLUX["plasma cell"]
    error = np.array(
        [
            abs(averaged_greens([vertices], vertices, order)[0][0] / reference - 1.0)
            for order in range(1, 6)
        ]
    )
    expected = np.array(
        [
            2.4280703427516537e-2,
            4.2623666611785183e-5,
            3.7520486706466016e-6,
            1.3980087421039400e-7,
            8.2973203863190292e-9,
        ]
    )
    # The production-path vector is banked at full fp64 precision; its 5e-13
    # relative bound is four orders below the smallest recorded error.
    np.testing.assert_allclose(error, expected, rtol=5.0e-13, atol=0.0)
    assert np.all(np.diff(error) < 0.0)


def test_the_rule_loses_order_to_the_section_aspect_ratio():
    """The diagnostic that separates a quadrature error from a reference's model.

    The collapsed fan gives the long direction of a section the same node count as
    the short one, so the rule's own error rises with the ASPECT RATIO. Sweeping it at
    fixed area and fixed major radius is therefore the measurement that says whether a
    deviation belongs to the rule: an error that grows with aspect is the rule's order,
    and one that does not is a property of whatever it is being compared against.

    Measured against the oracle at fixed area: 8e-06 flat from aspect 0.5 to 2, 2.9e-05
    at the ITER CS section's own 2.89, and 6.0e-04 at aspect 10 -- so a disagreement at
    the part-in-a-thousand level with anything else cannot be blamed on this rule for
    any section the element meets. The monotone chain is asserted from aspect 2 up,
    which is the direction the claim needs; below 2 the rule is at its floor and the
    three values there are within a few parts in a million of each other.
    """
    error = {}
    for aspect, limit in ASPECT_SELF_FLUX.items():
        width = np.sqrt(ASPECT_AREA / aspect)
        vertices = rectangle(ASPECT_MAJOR, 0.0, width, aspect * width)
        assert section_fingerprint(vertices) == pytest.approx(
            (ASPECT_AREA, ASPECT_MAJOR, 0.0), rel=1e-09, abs=1e-12
        ), aspect
        error[aspect] = abs(
            averaged_greens([vertices], vertices, ORDER)[0][0] / limit - 1.0
        )
    assert error[1.0] < 3e-05  # measured 7.8e-06
    assert error[10.0] > 10.0 * error[1.0]  # measured 77x
    assert error[10.0] > error[2.89] > error[2.0]  # 6.0e-04, 2.9e-05, 9.6e-06
    assert error[10.0] < 1e-03  # measured 6.0e-04


@pytest.mark.parametrize("aspect", [0.2, 0.05, 0.01, 0.005])
def test_the_self_term_converges_onto_the_published_ring_inductance(aspect):
    """A square-section ring against ``mu_0 R (ln(8 R/GMD) - 2)``, both candidates.

    The asymptotic ring inductance built on a section's geometric mean distance is
    the independent oracle, and it is an ASYMPTOTE: its own error falls as
    ``(w/R)^2``, so the comparison only says something if that error is inside it.
    It is, and both candidate diagonals are held against the same formula over the
    same aspect sweep:

    * the double integral tracks the oracle onto its own error floor -- 3.1e-03 at
      ``w/R = 0.2`` falling to 1.3e-06 at 0.005, which is the ``(w/R)^2`` the
      formula itself carries;
    * the single integral at the section centroid does NOT converge. It is a
      different quantity -- the section's arithmetic mean logarithmic distance in
      place of its geometric mean -- so it holds a fixed additive offset of
      ``mu_0 R`` times a constant, showing up as 10% at ``w/R = 0.2`` and still 4%
      at 0.005, shrinking only as the logarithm it sits beside grows.

    That is the whole case for the diagonal being the double integral, and this test
    fails if it is reverted to a point evaluation. Grover's constant is the geometric
    mean distance of a UNIFORMLY FILLED square, so both sides of this comparison make
    the same current assumption and it is silent about a subdivided conductor.
    """
    major = 1.0
    width = aspect * major
    vertices = rectangle(major, 0.0, width, width)
    oracle = MU0 * major * (np.log(8.0 * major / (SQUARE_GMD * width)) - 2.0)
    double = averaged_greens([vertices], vertices, 2 * ORDER)[0][0]
    point = polygon_analytic_greens(np.array([major]), np.array([0.0]), vertices)[0][0]
    assert abs(double / oracle - 1.0) < 5.0 * aspect**2  # measured 0.08 of the bound
    assert point / oracle - 1.0 > 0.04  # measured +0.104 at 0.2, +0.041 at 0.005
