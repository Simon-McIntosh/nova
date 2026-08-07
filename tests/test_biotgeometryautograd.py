"""Geometry autograd contract for the coupling kernel family.

Every traced kernel here takes its SOURCE geometry -- section vertices, a
filament position, a rectangle descriptor, an arc's azimuths -- as trace
inputs, so ``jax.jacfwd``/``jacrev`` return exact geometry Jacobians rather
than finite differences.  That property is what a coil-geometry inverse solve
consumes, and it is pinned per kernel class in three parts:

1. VALUE PARITY -- the traced form reproduces the shipped numpy reference.
   Where the two share every operation the pin is at round-off; where the
   trace substitutes a branch-free rule for a host-side routing (the zeta
   quadrature) the pin is the rules' measured mutual accuracy.
2. JACFWD vs CENTRAL DIFFERENCES -- forward-mode derivatives of the traced
   kernel agree with central differences of the same arithmetic executed by
   numpy, at the finite difference's own floor.  That floor is the kernel's
   round-off over the step (``~1e-13/h``), so the step is chosen per kernel
   and the pin sits just above the measured agreement.  Pointwise relative
   error is meaningless where a derivative component passes through zero, so
   the closed-form pins are stated against each component's own peak, exactly
   as the tiled-backend parity is.
3. JACREV vs JACFWD -- reverse mode (what a large least-squares wants) agrees
   with forward mode.

The section pack runs INSIDE the trace (:func:`~nova.biot.polygon.
traced_pack_section`) with its zero-weight topology mask computed OUTSIDE
(:func:`~nova.biot.polygon.horizontal_edges`): edge weights are a discrete
property of which integrand the kernel evaluates, so the mask is static and a
perturbation that would change it is a re-pack, not a derivative.

The three closed-form SECTION traces -- the full-turn polygon on both sections,
and the finite arc -- carry the ``slow`` marker.  They keep all three parts; what
puts them in the full lane is that their graphs are set by the reductions' moment
recurrences, so an eager trace is op-count bound and neither a smaller grid nor
fewer targets moves it.  Every other kernel class keeps a fast-lane test.
"""

import numpy as np
import pytest

from nova.biot.arcbandedcoupling import (
    arc_filament_greens,
    traced_arc_filament_greens,
)
from nova.biot.greens import (
    cylinder_greens,
    greens_bz_br,
    greens_psi,
    traced_cylinder_greens,
    traced_filament_greens,
)
from nova.biot.polygon import (
    _phi_rule,
    horizontal_edges,
    pack_section,
    polygon_greens,
    traced_pack_section,
)
from nova.biot.polygonanalytic import packed_analytic_greens, polygon_analytic_greens
from nova.biot.polygonarc import packed_arc_greens, polygon_arc_greens
from nova.biot.tiledassembly import _traced_psi_gradient
from nova.biot.zeta import traced_zeta, zeta

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

# A pentagon with no horizontal edges (every edge carries weight one, so the
# pack is smooth in every vertex) and a trapezoid whose one horizontal edge
# exercises the static zero-weight mask and the broken-chain corner terms.
PENTAGON = np.array(
    [
        [1.42, -1.16],
        [1.55, -1.19],
        [1.60, -1.07],
        [1.50, -1.015],
        [1.405, -1.06],
    ]
)
TRAPEZOID = np.array([[1.42, -1.16], [1.58, -1.16], [1.55, -1.02], [1.44, -1.04]])
TRIANGLE = np.array([[1.42, -1.16], [1.58, -1.13], [1.47, -1.02]])

TARGET_R = np.array([1.8449, 0.1803, 1.5913])
TARGET_Z = np.array([0.3070, 0.5000, -0.7904])
TARGET_PHI = np.array([0.15, 1.2, 2.5])

ARC_SPAN = (-0.3, 1.9)

# Residual-quadrature nodes for the closed-form sections: accuracy is not what
# these tests measure, so the count is held low to keep the traces small.
NODES = 64

# The finite arc takes fewer still.  It is the only kernel here whose trace is
# large enough for the node count to show at all, and even there it reaches a
# fifth of the primal and nothing of the shape -- the arc's graph is set by its
# moment recurrences, not by its quadrature, so the runtime is op-count bound.
ARC_NODES = 24


def rigid(vertices, delta):
    """Return the section rigidly shifted by ``delta = (dR, dZ)``."""
    return vertices + delta[None, :]


def relative_gap(traced, reference):
    """Return the pointwise relative disagreement of two stacked results."""
    traced = np.asarray(traced)
    reference = np.asarray(reference)
    return np.max(np.abs(traced - reference) / (np.abs(reference) + 1e-300))


def peak_scaled_gap(traced, reference):
    """Return the disagreement against each component's own peak.

    The smallest entries of a Jacobian pass through zero, where a pointwise
    ratio measures the finite difference's cancellation instead of the
    derivative's accuracy.
    """
    traced = np.asarray(traced)
    reference = np.asarray(reference)
    peak = np.max(np.abs(reference), axis=-1, keepdims=True)
    return np.max(np.abs(traced - reference) / peak)


def central_difference(function, argument, axis, step):
    """Return d ``function`` /d ``argument[axis]`` by central differences."""
    move = np.asarray(argument, dtype=np.float64).copy()
    move[axis] += step
    plus = function(move)
    move[axis] -= 2.0 * step
    minus = function(move)
    return (np.asarray(plus) - np.asarray(minus)) / (2.0 * step)


# --- the traced pack -----------------------------------------------------


def test_the_traced_pack_reproduces_pack_section_exactly():
    """Same edges, same weights, same norm -- the pack is one arithmetic."""
    for section in (PENTAGON, TRAPEZOID):
        mask = horizontal_edges(section)
        edge, weight, norm = pack_section(section)
        traced = traced_pack_section(jnp, jnp.asarray(section), mask)
        assert np.array_equal(np.asarray(traced[0]), edge)
        assert np.array_equal(np.asarray(traced[1]), weight)
        assert abs(float(traced[2]) - norm) <= 1e-15 * abs(norm)


def test_the_pack_topology_mask_marks_only_horizontal_edges():
    """The trapezoid's one flat edge is dropped; the pentagon keeps all five."""
    assert not horizontal_edges(PENTAGON).any()
    assert horizontal_edges(TRAPEZOID).tolist() == [True, False, False, False]


# --- polygon quadrature kernel ---------------------------------------------


def quadrature_traced(vertices):
    """(psi, B_R, B_Z) at the targets, fully traced from the vertices."""
    edge, weight, norm = traced_pack_section(jnp, vertices, horizontal_edges(PENTAGON))
    phi, wts = _phi_rule(16, 48)
    nodes = tuple(
        jnp.asarray(array)
        for array in (np.cos(phi), np.sin(phi), np.sin(2.0 * phi), wts * np.cos(phi))
    )
    r = jnp.asarray(TARGET_R)[:, None]
    z = jnp.asarray(TARGET_Z)[:, None]
    psi, dpsi_dr, dpsi_dz = _traced_psi_gradient(
        jnp, r, z, edge[..., None], weight[:, None], *nodes, norm
    )
    two_pi_r = 2.0 * jnp.pi * r[:, 0]
    return jnp.stack([psi, -dpsi_dz / two_pi_r, dpsi_dr / two_pi_r])


def quadrature_reference(vertices):
    return np.stack(polygon_greens(TARGET_R, TARGET_Z, vertices))


def test_polygon_quadrature_geometry_autograd():
    """Vertices -> pack -> quadrature kernel: values, jacfwd vs FD, jacrev."""
    assert (
        relative_gap(
            quadrature_traced(jnp.asarray(PENTAGON)), quadrature_reference(PENTAGON)
        )
        < 1e-10
    )

    def shifted(delta):
        return quadrature_traced(jnp.asarray(PENTAGON) + delta[None, :])

    forward = jax.jacfwd(shifted)(jnp.zeros(2))
    step = 1e-6
    for axis in (0, 1):
        difference = central_difference(
            lambda d: quadrature_reference(rigid(PENTAGON, d)),
            np.zeros(2),
            axis,
            step,
        )
        assert peak_scaled_gap(np.asarray(forward[..., axis]), difference) < 1e-6

    # one vertex coordinate, where the kernel has vertices
    vertex_forward = jax.jacfwd(quadrature_traced)(jnp.asarray(PENTAGON))
    move = np.zeros_like(PENTAGON)
    move[2, 1] = step
    difference = (
        quadrature_reference(PENTAGON + move) - quadrature_reference(PENTAGON - move)
    ) / (2.0 * step)
    assert peak_scaled_gap(np.asarray(vertex_forward[..., 2, 1]), difference) < 1e-6

    reverse = jax.jacrev(shifted)(jnp.zeros(2))
    assert relative_gap(reverse, forward) < 1e-10


# --- closed-form full-turn polygon kernel -----------------------------------


def analytic_packed(xp, vertices, mask):
    edge, weight, norm = traced_pack_section(xp, vertices, mask)
    return xp.stack(
        packed_analytic_greens(
            xp,
            xp.asarray(TARGET_R),
            xp.asarray(TARGET_Z),
            edge[..., None],
            weight[:, None],
            norm,
            nodes=NODES,
        )
    )


@pytest.mark.slow
@pytest.mark.parametrize("section", (PENTAGON, TRAPEZOID), ids=("smooth", "masked"))
def test_closed_form_polygon_geometry_autograd(section):
    """Values against the host driver; jacfwd vs FD of the same packed path.

    The closed-form section reductions are the expensive traces in the file --
    this pair and the finite arc below -- so all three sit in the full lane
    rather than the fast one.  The kernels they pin have the same reach either
    way; what the fast lane keeps is one representative of every OTHER kernel
    class, each of them seconds rather than a minute.
    """
    mask = horizontal_edges(section)
    traced = analytic_packed(jnp, jnp.asarray(section), mask)
    host = np.stack(polygon_analytic_greens(TARGET_R, TARGET_Z, section, nodes=NODES))
    assert relative_gap(traced, host) < 1e-9

    def shifted(delta):
        return analytic_packed(jnp, jnp.asarray(section) + delta[None, :], mask)

    forward = jax.jacfwd(shifted)(jnp.zeros(2))
    assert np.all(np.isfinite(np.asarray(forward)))
    step = 1e-4
    for axis in (0, 1):
        difference = central_difference(
            lambda d: analytic_packed(np, rigid(section, d), mask),
            np.zeros(2),
            axis,
            step,
        )
        assert peak_scaled_gap(np.asarray(forward[..., axis]), difference) < 1e-5

    reverse = jax.jacrev(shifted)(jnp.zeros(2))
    assert relative_gap(reverse, forward) < 1e-8


# --- closed-form finite-arc kernel -------------------------------------------


def arc_packed(xp, vertices, mask, start, end, nodes=NODES):
    edge, weight, norm = traced_pack_section(xp, vertices, mask)
    return xp.stack(
        packed_arc_greens(
            xp,
            xp.asarray(TARGET_R),
            xp.asarray(TARGET_Z),
            xp.asarray(TARGET_PHI),
            edge[..., None],
            weight[:, None],
            norm,
            start,
            end,
            nodes=nodes,
        )
    )


# The four things the arc is differentiated in at once: the section's two rigid
# components and the arc's two azimuths.  Held as one parameter vector so the
# whole Jacobian comes off ONE forward pass -- the four columns are the same four
# partials taken one group at a time, and the arc's trace is expensive enough
# that the number of passes is the runtime.
ARC_PARAMETERS = np.array([0.0, 0.0, *ARC_SPAN])
ARC_STEPS = (1e-4, 1e-4, 1e-6, 1e-6)


@pytest.mark.slow
def test_finite_arc_geometry_autograd():
    """The arc's five rows differentiate in the vertices AND its two azimuths.

    The heaviest kernel in the file by a factor of five, and it carries the
    ``slow`` marker for the reason its own runtime is a measurement rather than
    an accident: the arc's graph is fixed by its moment recurrences, so under an
    eager trace the cost is op-count bound and neither the target count nor the
    residual quadrature moves it much.  The reverse pass alone is a minute with a
    SINGLE cotangent, so the two economies below are the whole difference between
    this and a test nobody runs -- and neither of them is coverage:

    * one forward pass over all four parameters rather than one per group;
    * a reverse pass contracted onto a fixed covector rather than the whole
      five-by-three Jacobian.  A VJP that reproduces the JVP contracted on a
      covector with no zero entries reproduces it on every row, and the cost of
      reverse mode is in its transpose rather than in the cotangent count.
    """
    mask = horizontal_edges(TRIANGLE)

    def rows(xp, parameters):
        """The five rows at every target, from the four-parameter geometry."""
        vertices = xp.asarray(TRIANGLE) + parameters[:2][None, :]
        return arc_packed(
            xp, vertices, mask, parameters[2:3], parameters[3:4], nodes=ARC_NODES
        )

    traced = rows(jnp, jnp.asarray(ARC_PARAMETERS))
    host = np.stack(
        polygon_arc_greens(
            TARGET_R, TARGET_Z, TARGET_PHI, TRIANGLE, *ARC_SPAN, nodes=ARC_NODES
        )
    )
    assert relative_gap(traced, host) < 1e-9

    forward = jax.jacfwd(lambda p: rows(jnp, p))(jnp.asarray(ARC_PARAMETERS))
    assert np.all(np.isfinite(np.asarray(forward)))
    for axis, step in enumerate(ARC_STEPS):
        difference = central_difference(
            lambda p: rows(np, p), ARC_PARAMETERS, axis, step
        )
        assert peak_scaled_gap(np.asarray(forward[..., axis]), difference) < 1e-5

    # a covector with no zero entries and no symmetry, so no row of the Jacobian
    # can drop out of the contraction
    covector = np.linspace(0.37, 1.73, forward.shape[0] * forward.shape[1]).reshape(
        forward.shape[:2]
    )
    reverse = jax.grad(lambda p: jnp.sum(rows(jnp, p) * jnp.asarray(covector)))(
        jnp.asarray(ARC_PARAMETERS)
    )
    contracted = np.einsum("ij,ijk->k", covector, np.asarray(forward))
    assert peak_scaled_gap(np.asarray(reverse), contracted) < 1e-8


# --- ring filament kernel -----------------------------------------------------


def filament_host(source):
    psi = greens_psi(TARGET_R, TARGET_Z, source[0], source[1])
    bz, br = greens_bz_br(TARGET_R, TARGET_Z, source[0], source[1])
    return np.stack([psi, br, bz])


def filament_traced(source):
    return jnp.stack(
        traced_filament_greens(
            jnp, jnp.asarray(TARGET_R), jnp.asarray(TARGET_Z), source[0], source[1]
        )
    )


def test_ring_filament_geometry_autograd():
    """The loop position differentiates; the elliptic pair costs a few ulp."""
    source = np.array([1.52, -1.08])
    assert (
        relative_gap(filament_traced(jnp.asarray(source)), filament_host(source))
        < 1e-12
    )

    forward = jax.jacfwd(filament_traced)(jnp.asarray(source))
    for axis in (0, 1):
        difference = central_difference(filament_host, source, axis, 1e-6)
        assert relative_gap(np.asarray(forward[..., axis]), difference) < 1e-6

    reverse = jax.jacrev(filament_traced)(jnp.asarray(source))
    assert relative_gap(reverse, forward) < 1e-11


def test_ring_filament_axis_targets_keep_finite_tangents():
    """An on-axis target returns the loop limits with a clean geometry tangent."""
    jacobian = jax.jacfwd(
        lambda source: jnp.stack(
            traced_filament_greens(
                jnp, jnp.zeros(1), jnp.full(1, 0.3), source[0], source[1]
            )
        )
    )(jnp.asarray([1.52, -1.08]))
    assert np.all(np.isfinite(np.asarray(jacobian)))


# --- rectangular-section kernel -------------------------------------------------


def cylinder_traced(descriptor):
    return jnp.stack(
        traced_cylinder_greens(
            jnp, jnp.asarray(TARGET_R), jnp.asarray(TARGET_Z), *descriptor
        )
    )


def test_rectangle_section_geometry_autograd():
    """Centroid and extents differentiate; parity is the two zeta rules' own."""
    descriptor = np.array([1.52, -1.08, 0.12, 0.09])
    host = np.stack(cylinder_greens(TARGET_R, TARGET_Z, *descriptor))
    assert relative_gap(cylinder_traced(jnp.asarray(descriptor)), host) < 1e-11

    forward = jax.jacfwd(cylinder_traced)(jnp.asarray(descriptor))
    step = 1e-5
    for axis in range(4):
        difference = central_difference(
            lambda d: np.stack(traced_cylinder_greens(np, TARGET_R, TARGET_Z, *d)),
            descriptor,
            axis,
            step,
        )
        assert peak_scaled_gap(np.asarray(forward[..., axis]), difference) < 3e-6

    reverse = jax.jacrev(cylinder_traced)(jnp.asarray(descriptor))
    assert relative_gap(reverse, forward) < 1e-7


# --- arc filament kernel ---------------------------------------------------------


def arc_filament_traced(geometry):
    return jnp.stack(
        traced_arc_filament_greens(
            jnp,
            jnp.asarray(TARGET_R),
            jnp.asarray(TARGET_Z),
            jnp.asarray(TARGET_PHI),
            geometry[0],
            geometry[1],
            geometry[2],
            geometry[3],
        )
    )


def arc_filament_host(geometry):
    return np.stack(
        arc_filament_greens(
            TARGET_R,
            TARGET_Z,
            TARGET_PHI,
            geometry[0],
            geometry[1],
            geometry[2],
            geometry[3],
        )
    )


def test_arc_filament_geometry_autograd():
    """Position and both azimuths differentiate off the same fixed-node rule."""
    geometry = np.array([1.52, -1.08, *ARC_SPAN])
    assert (
        relative_gap(
            arc_filament_traced(jnp.asarray(geometry)), arc_filament_host(geometry)
        )
        < 1e-13
    )

    forward = jax.jacfwd(arc_filament_traced)(jnp.asarray(geometry))
    for axis in range(4):
        difference = central_difference(arc_filament_host, geometry, axis, 1e-6)
        assert relative_gap(np.asarray(forward[..., axis]), difference) < 1e-6

    reverse = jax.jacrev(arc_filament_traced)(jnp.asarray(geometry))
    assert relative_gap(reverse, forward) < 1e-12


# --- the branch-free zeta rule ----------------------------------------------------


def test_traced_zeta_matches_the_routed_host_rule_in_both_regimes():
    """One tanh-sinh rule reproduces the host's per-element rule selection."""
    generator = np.random.default_rng(7)
    rs = generator.uniform(0.5, 2.0, 64)
    r = generator.uniform(0.5, 2.0, 64)
    gamma = np.concatenate(
        [generator.uniform(-0.05, 0.05, 32), generator.uniform(0.3, 1.0, 32)]
    )
    alpha = np.full(64, np.pi / 2.0)
    host = zeta(rs, r, gamma, alpha)
    traced = np.asarray(traced_zeta(jnp, rs, r, gamma, alpha))
    assert relative_gap(traced, host) < 1e-13
    # a zero-length interval integrates to zero without a divide
    assert float(traced_zeta(jnp, 1.0, 1.0, 0.0, 0.0)) == 0.0
