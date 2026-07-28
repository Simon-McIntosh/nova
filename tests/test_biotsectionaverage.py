"""What the target-section rule integrates, and what its order buys.

The rule in :mod:`nova.biot.sectionaverage` turns a finite-section kernel evaluated
at one target point into the DOUBLE integral an inductance operator wants. Three
things have to hold for that to mean anything: the rule must integrate the area it
claims to, its order must be converged on the hardest integrand it sees (a section's
own flux), and the quantity it produces must be the self-inductance -- which is
checked here against a published closed form rather than against itself.
"""

import numpy as np
import pytest

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


SECTIONS = {
    "coil filament": rectangle(3.9431, 7.5641, 0.2398, 0.2460),
    "plasma cell": hexagon(6.2, 0.5, 0.075),
    "slender": rectangle(1.722, 5.313, 0.05, 0.5),
    "wall-clipped": np.array(
        [[6.10, 0.42], [6.24, 0.42], [6.27, 0.50], [6.18, 0.57], [6.10, 0.53]]
    ),
}


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
    shoelace = 0.5 * np.sum(corner[:, 0] * rolled[:, 1] - rolled[:, 0] * corner[:, 1])
    assert weights.sum() == pytest.approx(shoelace, rel=1e-12, abs=0)
    assert np.sum(weights * 3.0) / weights.sum() == pytest.approx(3.0, rel=1e-15)


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
    fails if it is reverted to a point evaluation.
    """
    major = 1.0
    width = aspect * major
    vertices = rectangle(major, 0.0, width, width)
    oracle = MU0 * major * (np.log(8.0 * major / (SQUARE_GMD * width)) - 2.0)
    double = averaged_greens([vertices], vertices, 2 * ORDER)[0][0]
    point = polygon_analytic_greens(np.array([major]), np.array([0.0]), vertices)[0][0]
    assert abs(double / oracle - 1.0) < 5.0 * aspect**2  # measured 0.08 of the bound
    assert point / oracle - 1.0 > 0.04  # measured +0.104 at 0.2, +0.041 at 0.005
