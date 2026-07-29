"""The acceptance gate for the finite-arc polygon-section evaluation.

The arc is unusually well supplied with independent checks and this module holds
four, each isolating a different failure mode:

1. **Per edge, per limit, at an interior amplitude.**  A direct panelled
   quadrature of the same ``integral_0^alpha da w_l g_l`` the reduction replaces,
   transcribed literally from the paper's own integrands.  It shares no machinery
   with the closed form and it catches anything wrong in the four integrations by
   parts, in their new boundary terms, or in the partial-range moment stack --
   uncontaminated by the fold, the assembly or the section sum.

2. **A quarter-turn amplitude against the full turn's own per-limit values.**  The
   arc's ``4 X_l(pi/2)`` for the three rows the ring forms IS
   :mod:`nova.biot.polygonanalytic`'s ``_Edge.terms``, so two reductions built on
   different moment families -- complete and partial -- must return the same
   number.  Anything that survives this is not in the special functions.

3. **Sweep to a full turn against the landed ring.**  The strongest single check
   the plan has: the fold must collapse the two amplitudes onto the ring, the two
   rows a full turn's parity annihilates must vanish, and the three it keeps must
   reproduce the gated closed form.

4. **A converged direct volume integral over the arc.**  Built from the
   Biot-Savart integral itself -- a triangle-fan rule over the section crossed
   with a graded rule over the sweep -- rather than from anything the reduction
   knows about.  It is the only check the azimuthal field row has, because
   ``Bow`` carries that row through a different formulation and the ring does not
   carry it at all.

On top of those, ``Bow`` at zero edge slope is the fifth: it is the same arc for a
rectangular section through Urankar Part IV, five rows and all, and it agrees to
its own quadrature's accuracy.

The volume integral needs its azimuth GRADED and the grading has to wrap.  A
target sits a fraction ``gap/r`` of a radian from the source ring in azimuth, and
that fraction is a hundredth for a plasma-scale cell; a single Gauss rule over the
sweep misses the peak entirely -- 2e-3 of the answer at 160 nodes, where the
graded rule holds 1e-12 at 24.  The peak is at the target's azimuth PLUS ANY WHOLE
TURN that lands inside the sweep, which is what makes a full sweep a different
problem from a partial one.
"""

import numpy as np
import pytest

from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.polygonarc import _Edge, _Vertex, polygon_arc_greens

MU0 = 4.0e-7 * np.pi
ROWS = ("A_r", "A_phi", "B_r", "B_phi", "B_z")

R0 = 6.2


def hexagon(r0=R0, z0=0.0, radius=0.06):
    """Return the plasma cell section: a regular hexagon."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def rectangle(r0=R0, z0=0.0, width=0.1, height=0.08):
    """Return an axis-aligned rectangle -- every edge has zero or infinite slope."""
    return np.array(
        [
            [r0 - width / 2, z0 - height / 2],
            [r0 + width / 2, z0 - height / 2],
            [r0 + width / 2, z0 + height / 2],
            [r0 - width / 2, z0 + height / 2],
        ]
    )


def trapezium(r0=2.0, z0=0.1):
    """Return a slanted, non-symmetric quadrilateral -- every edge slope differs."""
    return np.array(
        [
            [r0, z0],
            [r0 + 0.25, z0 - 0.05],
            [r0 + 0.3, z0 + 0.2],
            [r0 - 0.05, z0 + 0.12],
        ]
    )


def thin_plate(r0=3.0):
    """Return a high aspect-ratio parallelogram -- the hardest conditioning."""
    return np.array([[r0, 0.0], [r0 + 0.4, 0.06], [r0 + 0.4, 0.075], [r0, 0.015]])


SECTIONS = {
    "hexagon": hexagon(),
    "rectangle": rectangle(),
    "trapezium": trapezium(),
    "thin_plate": thin_plate(),
}

# Sweeps spanning what the fold has to get right: a short arc, one straddling the
# target azimuths, one past a half turn, and the full turn the ring is.
SWEEPS = {
    "short": (0.4, 2.1),
    "straddling": (-0.5, 0.5),
    "long": (0.0, 4.0),
    "turn": (0.4, 0.4 + 2.0 * np.pi),
}


def section_radius(vertices):
    """Return the section's bounding radius about its own centroid."""
    v = np.asarray(vertices, float)
    return float(np.max(np.linalg.norm(v - v.mean(axis=0), axis=1)))


def contour_distance(target_r, target_z, vertices):
    """Return the distance from each target to the section CONTOUR, in radii."""
    v = np.asarray(vertices, float)
    point = np.stack([np.ravel(target_r), np.ravel(target_z)], axis=-1)
    best = np.full(len(point), np.inf)
    for index in range(len(v)):
        start, end = v[index], v[(index + 1) % len(v)]
        span = end - start
        length = float(span @ span)
        along = np.clip(((point - start) @ span) / max(length, 1e-30), 0.0, 1.0)
        nearest = start + along[:, None] * span
        best = np.minimum(best, np.linalg.norm(point - nearest, axis=1))
    return best / section_radius(vertices)


# Centroid distances the gate reads the volume integral at, in section radii.  The
# innermost is one and a half, because ``0.8`` puts a target inside the conductor
# for a hexagon -- whose inradius is 0.87 of its circumradius -- and the reference
# does not converge there.  See the convergence test below.
STANDOFF = (1.5, 3.0, 8.0)


def gate_targets(vertices, directions=12, standoff=STANDOFF):
    """Return targets ringing the section at several contour distances."""
    v = np.asarray(vertices, float)
    centre = v.mean(axis=0)
    radius = section_radius(vertices)
    angle = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    reach = np.array(standoff)[:, None] * radius
    target_r = (centre[0] + reach * np.cos(angle)[None, :]).ravel()
    target_z = (centre[1] + reach * np.sin(angle)[None, :]).ravel()
    return target_r, target_z


def gate_azimuths(count):
    """Return target azimuths spanning inside, outside and on an arc's end plane."""
    return np.resize(np.array([0.4, 1.2, -0.7, 2.6, 0.0, 3.9]), count)


# ---------------------------------------------------------------------------
# ORACLE 1 -- the edge integral at an interior amplitude, transcribed literally.


def alpha_quadrature(target_r, target_z, edge, which, alpha, nodes=40, panels=64):
    """Return ``4 X_l(alpha)`` for the five rows, by panelled quadrature in ``a``.

    A literal transcription of eq 9's antiderivative and eq 11b's field integrand,
    weighted by each row's own ``cos phi``, ``-sin phi``, ``sin phi`` or one, with
    the SAME upper limit the closed form uses.  Deliberately naive: no machinery is
    shared with the reduction under test.

    Panels graded geometrically into BOTH ends, because both are where the
    integrand's own features sit -- ``arsinh beta1`` turns over at ``a = 0`` where
    the target is level with the edge end, and the radical's layer of width ``k'``
    sits at ``a = pi/2``.  Uniform panels are not a reference at either.
    """
    ra, za, rb, zb = edge
    b1 = (rb - ra) / (zb - za)
    a02 = 1.0 + b1 * b1
    a03 = a02 * np.sqrt(a02)
    node, weight = np.polynomial.legendre.leggauss(nodes)
    r = np.atleast_1d(target_r)[:, None]
    z = np.atleast_1d(target_z)[:, None]
    alpha = np.atleast_1d(alpha)[:, None]
    half = np.logspace(-9, np.log10(0.5), panels // 2)
    fraction = np.unique(np.concatenate([[0.0], half, 1.0 - half, [1.0]]))
    lo, hi = fraction[:-1], fraction[1:]
    step = 0.5 * (hi - lo)
    angle = alpha * (step * (node[:, None] + 1.0) + lo).T.ravel()[None, :]
    spread = np.broadcast_to(step * weight[:, None], (nodes, len(step)))
    scale = alpha * spread.T.ravel()
    cos_phi = -np.cos(2.0 * angle)
    sin_phi = np.sin(2.0 * angle)
    r1 = ra - b1 * (za - z)
    u = (zb - z) if which else (za - z)
    offset = (r1 + b1 * u) - r * cos_phi
    plane_offset = r1 - r * cos_phi
    g_squared = u * u + (r * sin_phi) ** 2
    b_squared = plane_offset**2 + a02 * (r * sin_phi) ** 2
    distance = np.sqrt(g_squared + offset**2)
    gamma = u + b1 * offset
    first = np.arcsinh(offset / np.sqrt(g_squared))
    second = np.arcsinh(gamma / np.sqrt(b_squared))
    third = np.arctan((u * offset - b1 * g_squared) / (r * sin_phi * distance))
    potential = (
        gamma * distance / (2.0 * a02)
        + u * r * cos_phi * first
        + (b_squared + 2.0 * a02 * r * cos_phi * plane_offset) / (2.0 * a03) * second
        - 0.5 * r * r * np.sin(2.0 * (np.pi - 2.0 * angle)) * third
    )
    bracket = (
        distance / a02
        + r * cos_phi * first
        - b1 / a03 * (r1 + b1 * b1 * r * cos_phi) * second
    )
    vertical = (
        u * first
        + (b1 * b1 * r1 - (2.0 * a02 - 1.0) * r * cos_phi) / a03 * second
        - r * sin_phi * third
        - b1 / a02 * distance
    )
    return tuple(
        4.0 * np.sum(scale * integrand, axis=1)
        for integrand in (
            -sin_phi * potential,
            cos_phi * potential,
            cos_phi * bracket,
            sin_phi * bracket,
            vertical,
        )
    )


# ---------------------------------------------------------------------------
# ORACLE 4 -- the Biot-Savart volume integral over the arc.


def _section_rule(vertices, nodes):
    """Return a triangle-fan product rule over the section, signed.

    Fanned from corner zero with SIGNED triangle areas, which integrates any simple
    polygon rather than only a convex one, and collapsed onto the square by the
    Duffy map so a plain Gauss rule serves each triangle.
    """
    v = np.asarray(vertices, float)
    node, weight = np.polynomial.legendre.leggauss(nodes)
    unit = 0.5 * (node + 1.0)
    scale = 0.5 * weight
    first, second = np.meshgrid(unit, unit, indexing="ij")
    area = np.outer(scale, scale) * (1.0 - first)
    one, two = first, second * (1.0 - first)
    radius, height, share = [], [], []
    for index in range(1, len(v) - 1):
        base, left, right = v[0], v[index], v[index + 1]
        twice = (left[0] - base[0]) * (right[1] - base[1]) - (left[1] - base[1]) * (
            right[0] - base[0]
        )
        along = base[0] + one * (left[0] - base[0]) + two * (right[0] - base[0])
        across = base[1] + one * (left[1] - base[1]) + two * (right[1] - base[1])
        radius.append(along.ravel())
        height.append(across.ravel())
        share.append((twice * area).ravel())
    return np.concatenate(radius), np.concatenate(height), np.concatenate(share)


def _sweep_rule(start, end, azimuth, width, nodes):
    """Return a rule over the sweep, graded into every nearest approach in it.

    The integrand peaks wherever the source ring passes the target, which is the
    target's azimuth plus ANY whole turn that lands inside the sweep -- so the
    grading is built per turn and clipped, and a full sweep gets it twice.
    """
    bound = {float(start), float(end)}
    turn = 2.0 * np.pi
    low = int(np.floor((start - azimuth) / turn)) - 1
    high = int(np.ceil((end - azimuth) / turn)) + 1
    for count in range(low, high + 1):
        centre = azimuth + count * turn
        if start < centre < end:
            bound.add(float(centre))
        offset = width
        while offset < turn:
            for point in (centre - offset, centre + offset):
                if start < point < end:
                    bound.add(float(point))
            offset *= 2.0
    edge = np.array(sorted(bound))
    node, weight = np.polynomial.legendre.leggauss(nodes)
    lo, hi = edge[:-1, None], edge[1:, None]
    return (
        (0.5 * (hi - lo) * (node[None, :] + 1.0) + lo).ravel(),
        (0.5 * (hi - lo) * weight[None, :]).ravel(),
    )


def volume_quadrature(
    target_r, target_z, target_phi, vertices, start, end, *, nodes=20, sweep=24
):
    """Return the five rows by direct integration of the Biot-Savart integral.

    Nothing here knows about the reduction: the current density is uniform and
    azimuthal, the potential is its own integral over the conductor volume, and the
    field is the curl's integrand taken directly,

        A = mu0/(4 pi) integral J phi'-hat r' dr' dz' dphi'/D
        B = mu0/(4 pi) integral J [phi'-hat x (r - r')] r' dr' dz' dphi'/D^3

    resolved in the TARGET's own cylindrical basis and normalised per ampere.
    """
    radius, height, share = _section_rule(vertices, nodes)
    area = share.sum()
    target_r = np.atleast_1d(np.asarray(target_r, float))
    target_z = np.atleast_1d(np.asarray(target_z, float))
    target_phi = np.atleast_1d(np.asarray(target_phi, float))
    gap = contour_distance(target_r, target_z, vertices) * section_radius(vertices)
    out = np.zeros((5, len(target_r)))
    for index in range(len(target_r)):
        width = max(gap[index] / max(target_r[index], 1e-9), 1e-6)
        angle, weight = _sweep_rule(start, end, target_phi[index], width, sweep)
        separation = angle - target_phi[index]
        cosine = np.cos(separation)[None, :]
        sine = np.sin(separation)[None, :]
        source_r = radius[:, None]
        level = target_z[index] - height[:, None]
        squared = (
            target_r[index] ** 2
            + source_r**2
            - 2.0 * target_r[index] * source_r * cosine
            + level**2
        )
        distance = np.sqrt(squared)
        cube = squared * distance
        element = (share[:, None] * source_r) * weight[None, :]
        out[0, index] = np.sum(element * -sine / distance)
        out[1, index] = np.sum(element * cosine / distance)
        out[2, index] = np.sum(element * cosine * level / cube)
        out[3, index] = np.sum(element * sine * level / cube)
        out[4, index] = np.sum(element * (source_r - target_r[index] * cosine) / cube)
    return out * MU0 / (4.0 * np.pi * area)


# A row whose own largest value is below this fraction of the largest row's has
# CANCELLED over the target set rather than merely being small, and its own scale
# is then the reference's round-off rather than a physical size.  Both rows a full
# turn's parity annihilates do exactly that at a full sweep.
CANCELLED = 1e-6


def worst_overall(got, want):
    """Return the worst deviation against the LARGEST row's scale.

    The measure that stays meaningful when a row cancels: an absolute agreement,
    normalised once for the whole table, so a row that has gone to zero is held to
    the size of the rows beside it rather than to its own residue.
    """
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def worst_by_row(got, want):
    """Return the worst deviation of each row against its OWN scale.

    The measure that says what a small component is worth -- the azimuthal field
    row is four orders below the rows beside it at a plasma-cell aspect, and an
    absolute bound would say nothing about it.  A row that has CANCELLED returns
    zero here and is covered by :func:`worst_overall` instead.
    """
    got, want = np.asarray(got), np.asarray(want)
    scale = np.max(np.abs(want), axis=1)
    live = scale > CANCELLED * np.max(scale)
    held = np.where(live, scale, 1.0)[:, None]
    return np.where(live, np.max(np.abs(got - want) / held, axis=1), 0.0)


# ---------------------------------------------------------------------------
# The references, before anything is held to them.

EDGES = {
    "sloped": (2.0, 0.1, 2.25, 0.05),
    "steep": (2.0, 0.1, 2.05, 0.6),
    "vertical": (2.2, -0.3, 2.2, 0.3),
    "reversed": (2.4, 0.5, 2.0, 0.1),
}
EDGE_TARGET_R = np.array([2.6, 3.0, 1.6, 4.0, 0.9])
EDGE_TARGET_Z = np.array([0.4, -0.45, 0.35, 0.2, -0.7])
AMPLITUDES = (0.35, 0.7, 1.2, 0.5 * np.pi)


def test_the_edge_reference_is_converged_so_it_can_be_used_as_one():
    """Raising the rule past the reference must not move it.

    Without this a candidate can pass by matching the reference's error instead of
    the integral's value.
    """
    edge = np.asarray(EDGES["steep"], float)
    alpha = np.full_like(EDGE_TARGET_R, 0.7)
    coarse = alpha_quadrature(EDGE_TARGET_R, EDGE_TARGET_Z, edge, 0, alpha, 30, 32)
    fine = alpha_quadrature(EDGE_TARGET_R, EDGE_TARGET_Z, edge, 0, alpha, 60, 96)
    assert worst_overall(coarse, fine) <= 1e-14
    assert np.max(worst_by_row(coarse, fine)) <= 1e-13


def test_the_volume_reference_is_converged_where_the_gate_reads_it():
    """Both directions raised well past the rule used, and only beyond the standoff.

    The standoff is not a convenience.  ``gate_targets`` measures from the section's
    CENTROID, so its innermost ring sits partly inside the conductor, where the
    volume integrand is singular and no product rule converges: the reference moves
    by 1.5e-02 there against 4.5e-13 one ring out.  The gate therefore reads the
    reference beyond :data:`STANDOFF` only, and this test is what says where that
    is rather than assuming it.
    """
    vertices = SECTIONS["hexagon"]
    target_r, target_z = gate_targets(vertices, directions=4)
    phi = gate_azimuths(len(target_r))
    coarse = volume_quadrature(target_r, target_z, phi, vertices, 0.4, 2.1)
    fine = volume_quadrature(
        target_r, target_z, phi, vertices, 0.4, 2.1, nodes=32, sweep=40
    )
    assert worst_overall(coarse, fine) <= 1e-12
    assert np.max(worst_by_row(coarse, fine)) <= 1e-9

    inside_r, inside_z = gate_targets(vertices, directions=4, standoff=(0.8,))
    inside_phi = gate_azimuths(len(inside_r))
    near = volume_quadrature(inside_r, inside_z, inside_phi, vertices, 0.4, 2.1)
    near_fine = volume_quadrature(
        inside_r, inside_z, inside_phi, vertices, 0.4, 2.1, nodes=32, sweep=40
    )
    assert worst_overall(near, near_fine) >= 1e-4


def test_a_single_gauss_rule_over_the_sweep_is_not_a_reference():
    """The grading is load-bearing, and this is how much.

    A target a hundredth of a radian from the source ring in azimuth puts a peak of
    that width into the integrand; a single rule over the sweep, even at seven
    times the graded rule's node count, misses it by parts in a thousand.
    """
    vertices = SECTIONS["hexagon"]
    target_r, target_z = gate_targets(vertices, directions=2, standoff=(1.5,))
    phi = gate_azimuths(len(target_r))
    graded = volume_quadrature(target_r, target_z, phi, vertices, 0.4, 2.1)
    flat = np.zeros_like(graded)
    node, weight = np.polynomial.legendre.leggauss(160)
    for index in range(len(target_r)):
        angle = 0.5 * (2.1 - 0.4) * (node + 1.0) + 0.4
        # one panel, no grading: the same integrand under a plain rule
        flat[:, index] = _one_target_flat(
            target_r[index],
            target_z[index],
            phi[index],
            vertices,
            angle,
            0.5 * (2.1 - 0.4) * weight,
        )
    assert worst_overall(flat, graded) >= 1e-4


def _one_target_flat(target_r, target_z, target_phi, vertices, angle, weight):
    """Return the five rows at one target from a given ungraded sweep rule."""
    radius, height, share = _section_rule(vertices, 20)
    separation = angle - target_phi
    cosine = np.cos(separation)[None, :]
    sine = np.sin(separation)[None, :]
    source_r = radius[:, None]
    level = target_z - height[:, None]
    squared = target_r**2 + source_r**2 - 2.0 * target_r * source_r * cosine + level**2
    distance = np.sqrt(squared)
    cube = squared * distance
    element = (share[:, None] * source_r) * weight[None, :]
    rows = (
        np.sum(element * -sine / distance),
        np.sum(element * cosine / distance),
        np.sum(element * cosine * level / cube),
        np.sum(element * sine * level / cube),
        np.sum(element * (source_r - target_r * cosine) / cube),
    )
    return np.array(rows) * MU0 / (4.0 * np.pi * share.sum())


# ---------------------------------------------------------------------------
# ORACLE 1.


@pytest.mark.parametrize("amplitude", AMPLITUDES)
@pytest.mark.parametrize("edge", sorted(EDGES))
def test_the_reduction_reproduces_the_edge_integral_at_an_interior_amplitude(
    edge, amplitude
):
    """Per edge, per limit, per row -- the transcription check.

    Run where the reduction's small quantities are NOT small, so anything wrong in
    the four integrations by parts, in the boundary terms an interior limit
    reinstates, or in the partial-range moment stack shows at full size.
    """
    vertices = np.asarray(EDGES[edge], float)
    alpha = np.full_like(EDGE_TARGET_R, amplitude)
    angle = (alpha, np.sin(alpha), np.cos(alpha))
    part = _Edge(EDGE_TARGET_R, EDGE_TARGET_Z, vertices, 128)
    for which in (0, 1):
        corner = (vertices[2], vertices[3]) if which else (vertices[0], vertices[1])
        vertex = _Vertex(
            EDGE_TARGET_R, EDGE_TARGET_Z, corner, angle, 128, residual=True
        )
        got = [
            term + corner_term
            for term, corner_term in zip(part.terms(vertex), vertex.arsinh_terms())
        ]
        want = alpha_quadrature(EDGE_TARGET_R, EDGE_TARGET_Z, vertices, which, alpha)
        assert worst_overall(got, want) <= 1e-13, f"limit {which}"
        assert np.max(worst_by_row(got, want)) <= 1e-11, f"limit {which}"


# ---------------------------------------------------------------------------
# ORACLE 2.


@pytest.mark.parametrize("edge", sorted(EDGES))
def test_a_quarter_turn_amplitude_reproduces_the_full_turns_own_values(edge):
    """Two reductions on different moment families, on the same three rows.

    ``4 X_l(pi/2)`` is exactly what :mod:`nova.biot.polygonanalytic` returns per
    limit for the flux and both field components, and the arc reaches it through
    the PARTIAL-range families -- a tridiagonal solve where the ring runs a
    downward recursion, an amplitude-carrying descent where the ring runs Bartky's.
    Nothing but the geometry is shared, so agreement here places every remaining
    error in the fold or the assembly.
    """
    from nova.biot.polygonanalytic import _Edge as RingEdge, _Vertex as RingVertex

    vertices = np.asarray(EDGES[edge], float)
    ones = np.ones_like(EDGE_TARGET_R)
    angle = (0.5 * np.pi * ones, ones, 0.0 * ones)
    part = _Edge(EDGE_TARGET_R, EDGE_TARGET_Z, vertices, 128)
    for which in (0, 1):
        corner = (vertices[2], vertices[3]) if which else (vertices[0], vertices[1])
        vertex = _Vertex(
            EDGE_TARGET_R, EDGE_TARGET_Z, corner, angle, 128, residual=True
        )
        ring = RingVertex(
            EDGE_TARGET_R, EDGE_TARGET_Z, corner[0], corner[1], 128, residual=True
        )
        got = [
            term + corner_term
            for term, corner_term in zip(part.terms(vertex), vertex.arsinh_terms())
        ]
        # the ring forms only the three rows whose fold is odd, in its own order
        want = [
            term + corner_term
            for term, corner_term in zip(
                RingEdge.terms(part, ring), ring.arsinh_terms()
            )
        ]
        rows = np.stack([got[1], got[2], got[4]])
        assert worst_overall(rows, np.stack(want)) <= 1e-14, f"limit {which}"
        assert np.max(worst_by_row(rows, np.stack(want))) <= 1e-13, f"limit {which}"


# ---------------------------------------------------------------------------
# ORACLE 3.


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_sweep_to_a_full_turn_reproduces_the_ring(section):
    """The fold's own arithmetic must close the arc onto the landed full turn.

    Three of the five rows carry the ring's value and two must vanish, and the two
    that vanish do so by the SAME parity argument that leaves an axisymmetric ring
    with no toroidal field -- so this is the assembly's statement of the physics as
    well as a regression against the gated closed form.

    The residual is not round-off and is not meant to be.  The two ends' folded
    amplitudes are the same angle reached from separations a whole turn apart, so
    their antiderivatives cancel to their own last bits -- and an antiderivative is
    of order the squared major radius where the flux is not.
    """
    vertices = SECTIONS[section]
    target_r, target_z = gate_targets(vertices, directions=6)
    phi = gate_azimuths(len(target_r))
    start = 0.4
    rows = polygon_arc_greens(
        target_r, target_z, phi, vertices, start, start + 2.0 * np.pi
    )
    psi, radial, vertical = polygon_analytic_greens(target_r, target_z, vertices)
    ring = np.stack(
        [
            np.zeros_like(psi),
            psi / (2.0 * np.pi * target_r),
            radial,
            np.zeros_like(psi),
            vertical,
        ]
    )
    kept = np.stack([rows[1], rows[2], rows[4]])
    assert np.max(worst_by_row(kept, ring[[1, 2, 4]])) <= 1e-10
    # the two rows the parity annihilates, against the scale of the rows it keeps
    scale = np.max(np.abs(ring[[1, 2, 4]]), axis=1).max()
    assert np.max(np.abs(rows[0])) <= 1e-12 * scale
    assert np.max(np.abs(rows[3])) <= 1e-12 * scale


# ---------------------------------------------------------------------------
# ORACLE 4 -- the gate proper.

# Worst deviation from the volume integral, per section, over every sweep and every
# target beyond the standoff.  Measured 2026-07-26 and asserted in BOTH directions,
# with two numbers per section because they answer different questions.
#
# OVERALL is the absolute agreement normalised by the largest row, and it is the
# tight one: a few parts in 1e12, which is the assembly's own differencing floor.
# Each per-limit antiderivative is of order the squared major radius where the
# answer is of order the section's own scale, so an edge's two limits are
# differenced against each other before anything else is added to them and what
# survives is round-off on the LARGER quantity -- 7e-19 absolute on a rectangle,
# against rows reaching 1e-6.
#
# BY ROW measures each row against its own size, which is the only measure that
# says anything about a row that has become small.  Two of them do: the azimuthal
# field row is four orders below the rows beside it at a plasma-cell aspect and
# holds 3e-11 of itself, which is the one statement anything makes about the row
# with no other oracle.  The looser entries below are set by the RADIAL field row
# over the sweep symmetric about the target azimuths, where it too falls four
# orders and the absolute floor above shows through it -- so the number records how
# small a row can get before that floor is visible, not an accuracy the reduction
# lacks.
SECTION_ENVELOPE = {
    "hexagon": (5e-12, 1e-09),  # measured 2.9e-12 / 6.7e-10
    "rectangle": (2e-12, 5e-09),  # measured 5.4e-13 / 3.0e-09
    "trapezium": (1e-12, 5e-12),  # measured 1.7e-13 / 2.0e-12
    "thin_plate": (1e-11, 5e-11),  # measured 4.8e-12 / 2.1e-11
}


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_closed_form_matches_the_volume_integral(section, sweep):
    """All five rows against an integral that shares nothing with the reduction.

    The azimuthal field row has no other check at a partial sweep: the ring does
    not carry it at all and ``Bow`` carries it through a different formulation, so
    this is what holds it.

    At a full sweep two of the five rows cancel to nothing and the reference cannot
    resolve them relative to themselves -- its own residue is what is left.  Those
    are held by the OVERALL measure here and by
    :func:`test_the_sweep_to_a_full_turn_reproduces_the_ring` against zero.
    """
    vertices = SECTIONS[section]
    start, end = SWEEPS[sweep]
    target_r, target_z = gate_targets(vertices, directions=4)
    phi = gate_azimuths(len(target_r))
    got = np.stack(polygon_arc_greens(target_r, target_z, phi, vertices, start, end))
    want = volume_quadrature(target_r, target_z, phi, vertices, start, end)
    overall, by_row = SECTION_ENVELOPE[section]
    assert worst_overall(got, want) <= overall
    assert np.max(worst_by_row(got, want)) <= by_row


@pytest.mark.parametrize("section", sorted(SECTIONS))
def test_the_recorded_envelopes_are_not_loose(section):
    """Asserted from below as well, over every sweep.

    An envelope only bounded from above stops being evidence the moment the
    reduction improves: it records what the evaluation achieved, so it has to fail
    if the evaluation gets better, and be re-measured when it does.
    """
    vertices = SECTIONS[section]
    target_r, target_z = gate_targets(vertices, directions=4)
    phi = gate_azimuths(len(target_r))
    overall, by_row = SECTION_ENVELOPE[section]
    reached = []
    for start, end in SWEEPS.values():
        got = np.stack(
            polygon_arc_greens(target_r, target_z, phi, vertices, start, end)
        )
        want = volume_quadrature(target_r, target_z, phi, vertices, start, end)
        reached.append((worst_overall(got, want), np.max(worst_by_row(got, want))))
    assert max(one for one, _ in reached) >= 0.02 * overall
    assert max(one for _, one in reached) >= 0.02 * by_row


# ---------------------------------------------------------------------------
# The structural reductions.


def test_a_horizontal_edge_contributes_nothing():
    """Paper eq 7a. The reduction divides by the edge's dz to form its slope.

    Splitting a horizontal edge in two adds an edge whose contribution must be
    exactly nothing, so the two evaluations agree to round-off rather than to the
    gate tolerance -- both sides are the same closed form.  It also moves the
    ``arsinh beta1`` corner term, which cancels around an unbroken chain and does
    not around a broken one, so the two sections reach the same answer by
    different bookkeeping.
    """
    vertices = rectangle(r0=3.0, width=1.0, height=0.8)
    low, high = vertices[0, 0], vertices[1, 0]
    bottom, top = vertices[0, 1], vertices[2, 1]
    split = np.array(
        [
            [low, bottom],
            [0.5 * (low + high), bottom],
            [high, bottom],
            [high, top],
            [low, top],
        ]
    )
    target_r, target_z = gate_targets(vertices, directions=6)
    phi = gate_azimuths(len(target_r))
    plain = np.stack(polygon_arc_greens(target_r, target_z, phi, vertices, 0.4, 2.1))
    divided = np.stack(polygon_arc_greens(target_r, target_z, phi, split, 0.4, 2.1))
    assert np.all(np.isfinite(divided))
    assert worst_overall(divided, plain) <= 1e-13
    assert np.max(worst_by_row(divided, plain)) <= 1e-10


def test_a_rectangular_section_reproduces_bow():
    """Zero edge slope, all five rows, against Urankar Part IV.

    ``Bow`` is the same finite arc for a rectangular cross-section through a
    different chain of the same paper series, and it reaches its own answer through
    a fixed-node zeta quadrature rather than in closed form -- so agreement here is
    to ITS accuracy, not to round-off, and it is the only check of the two potential
    rows that is not a quadrature of the definition.

    nova's point solver reports the vector potential without the ``mu0`` of eq 3a,
    so the two potential rows carry that factor between them; the field rows do not.
    Its output precision is also not fixed across processes, which is what sets the
    tolerance below rather than either reduction's accuracy.
    """
    from nova.frame.coilset import CoilSet

    radius, height, width, thick = 3.0, 0.2, 0.06, 0.04
    start, end = 0.3, 1.9
    theta = np.array([start, 0.5 * (start + end), end])
    path = np.stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            height * np.ones_like(theta),
        ],
        axis=-1,
    )
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay"])
    coilset.winding.insert(
        path,
        {"rect": (0, 0, width, thick)},
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    assert coilset.subframe.segment.tolist() == ["bow"]
    target_r = np.array([3.5, 2.5, 3.1, 3.0])
    target_phi = np.array([0.2, 1.0, 2.4, -0.4])
    target_z = np.array([0.5, -0.2, 0.25, 0.35])
    coilset.point.solve(
        np.stack(
            [
                target_r * np.cos(target_phi),
                target_r * np.sin(target_phi),
                target_z,
            ],
            axis=-1,
        )
    )
    cosine, sine = np.cos(target_phi), np.sin(target_phi)
    potential_x = MU0 * np.asarray(coilset.point.ax)
    potential_y = MU0 * np.asarray(coilset.point.ay)
    field_x = np.asarray(coilset.point.bx)
    field_y = np.asarray(coilset.point.by)
    bow = np.stack(
        [
            potential_x * cosine + potential_y * sine,
            -potential_x * sine + potential_y * cosine,
            field_x * cosine + field_y * sine,
            -field_x * sine + field_y * cosine,
            np.asarray(coilset.point.bz),
        ]
    )
    vertices = rectangle(r0=radius, z0=height, width=width, height=thick)
    got = np.stack(
        polygon_arc_greens(target_r, target_z, target_phi, vertices, start, end)
    )
    # Bounded from above only, and the bound is neither evaluation's accuracy.  The
    # point solver's own output PRECISION is not fixed: it comes back float32 in one
    # process and float64 in another, and the deviation moves with it -- 2.2e-07 in
    # the first case, which is a few ulp of float32, and 2.0e-10 in the second.  So
    # the bound has to hold in the looser of the two, and a lower bound would be a
    # claim about nova's storage rather than about either reduction.
    assert np.max(worst_by_row(got, bow)) <= 1e-6
    assert worst_overall(got, bow) <= 1e-6


def test_the_arc_is_finite_where_a_target_is_level_with_a_corner():
    """``u = 0`` drives both of the ring denominator's roots onto the range ends.

    A grid across a section hits it by alignment, and the arc adds a second way in:
    the amplitude itself can reach either end of the range, which is a target in the
    plane of one of the arc's own ends.  Both are evaluated rather than approached,
    and the neighbourhood is continuous through them.
    """
    vertices = SECTIONS["trapezium"]
    levels = np.asarray(vertices)[:, 1]
    target_r = np.repeat(np.array([1.7, 2.4]), len(levels))
    target_z = np.tile(levels, 2)
    # an azimuth exactly on the arc's start plane sends one amplitude to a quarter
    # turn, where the folded cosine is exactly zero
    phi = np.full_like(target_r, 0.4)
    rows = np.stack(polygon_arc_greens(target_r, target_z, phi, vertices, 0.4, 2.1))
    assert np.all(np.isfinite(rows))
    nudged = np.stack(
        polygon_arc_greens(target_r, target_z + 1e-9, phi + 1e-9, vertices, 0.4, 2.1)
    )
    assert worst_overall(nudged, rows) <= 1e-6
