"""Cost floor of the closed-form polygon evaluation, from its primitives alone.

This times only the pieces that cannot be avoided -- the special-function calls and
the two residual ``arsinh`` quadratures, per polygon edge, per target-source pair --
and ignores the coefficient algebra between them. The result is a LOWER bound on
the closed form's cost and therefore an UPPER bound on the speedup available.

What the reduction needs is counted the way the evaluation is ORGANISED, which is by
corner rather than by edge limit. Per CORNER of a full-turn ring: the complete
integrals of the first and second kind, the harmonic moment stack built from them and
its radical-weighted fold, and a third-kind integral for each of the ring
denominator's two pole factors. Per EDGE LIMIT: the plane denominator's two pole
factors, contracted against the corner's own stack, and one residual ``arsinh``
integral. A closed section has one corner per edge and shares each between two of
them, so the corner work is half of what an edge-limit organisation would do.
Everything else is rational algebra in quantities already computed.

Only ONE residual integral per edge limit, though the reduction leaves two. The
first is a function of the corner alone in all three components, so a closed chain of
edges carries it twice with opposite signs and it cancels exactly; it survives only
at the two ends of a dropped horizontal edge, and for a plasma cell -- a hexagon with
no horizontal edge -- it is never formed at all.

The residual quadratures carry their grading, because that is not optional: each
integrand has a boundary layer at each end of the range whose width falls with the
target's offset from the edge's end level, and a plain rule at any practical order
steps over it. The grading costs a ``sinh`` per node per pair, which is why it
appears in the floor rather than beside it.

``measured_cost`` times the assembled evaluation on the same shape of problem, for
the flux alone and for the flux with both field components, so the gap to the floor
is the coefficient algebra and the pole bookkeeping the floor deliberately ignores.

Measured, one SLURM compute core, 4096 pairs against a hexagonal cell, median of
three in a fresh process, at the 128 residual nodes the acceptance gate needs:

    third-kind integrals        12.7 us/pair
    moment stacks and all       26.8 us/pair
    graded g_p quadrature       73.3 us/pair
    floor                      100.1 us/pair
    assembled psi              189.8 us/pair       1.90 x floor
    assembled psi + field      189.6 us/pair       1.89 x floor

Three things to read from that. The field is FREE: eq 11b's rows share every
reduction the flux uses and differ only in polynomial weights, so both components
cost nothing measurable on top of the flux -- which is the case for transcribing
them rather than differentiating the reduced flux.

The CORNER organisation is worth 1.9x on the assembled evaluation, and how much
depends on the section's shape. Both arrangements timed in one job on one idle core,
fresh process per variant, median of three:

    section       edge-limit    by corner
    hexagon        333.9         176.6      1.89 x
    thin plate     220.7         119.3      1.85 x
    rectangle      109.4         113.2      0.97 x

Through this module's own harness, where the floor model has already run in the
process, the same pair is 334.9 to 189.8 -- 1.77x, and that is the figure the table
above belongs to.

The rectangle gains nothing, and that is the honest shape of the win rather than a
defect: two of its four edges are horizontal and dropped, so each of its corners
belongs to exactly ONE live edge and there is nothing to share, and its chain is
broken at every corner, so the first residual survives everywhere. What is left for
it is the extra object and the release bookkeeping, a few per cent either way across
runs. A hexagonal plasma cell is the opposite case -- every corner shared and the
chain closed -- and it halves both counts. Peak allocation moves the other way, 10.6
to 12.1 kB/pair, because up to three corners are live at once where one edge limit
used to be; the residual quadrature's (pairs x nodes) blocks dominate that figure
either way.

The residual quadrature is still the whole story, at 73 of the 100 floor. Half of
that is the node count, which doubled to 128 because the most slender section's
near-contour targets do not converge at 64 -- a limit that predates the harmonic
reduction and is its own investigation. The other half is what removing the boundary
logarithm costs per node -- a model, its root and a logarithm in place of one
``arsinh`` -- and it buys five orders at a corner and one to two everywhere else.

Next after that is the moment stack's LENGTH. It is carried forty orders past the
harmonics the numerators reach, because the pole families' downward recursion is
closed there; but the families are skipped outright where no root is far enough out to
need one, which is the common case, and then two thirds of the stack is built for
nothing. Deciding the length from the denominators' shifts -- which need no moments --
would cut it, at the cost of changing the recursion's own starting order and so every
accuracy entry the gate records.

    python benchmarks/analytic_cost_floor.py [nodes]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.special

from nova.biot.elliptic import (
    POLE_HEADROOM,
    cn_pole_moment,
    complete_pi,
    harmonic_moments,
    harmonic_pole_moments,
    harmonic_root_moments,
    sn_pole_moment,
)
from nova.biot.polygonanalytic import polygon_analytic_flux, polygon_analytic_greens

PAIRS = 4096
EDGES = 6  # a hexagonal plasma cell contributes all six
CORNERS = EDGES  # a closed section has one corner per edge, shared by two of them
LIMITS = 2  # every edge is evaluated at both of its ends
ORDERS = 9


def special_function_cost(pairs: int, edges: int) -> float:
    """Return seconds for the unavoidable moment work of one build.

    The modulus is the CORNER's, so one harmonic stack and one radical-weighted
    fold per corner rather than per edge limit, and with them the ring
    denominator's own split -- a seed and a family for each of its two pole
    factors.  The plane denominator is the edge's and its split is per limit, so
    its two seeds and families are counted that way.  The stack is carried past
    the harmonics the numerators reach because the pole families' system is closed
    there, which is what sets its length.
    """
    rng = np.random.default_rng(0)
    parameter = rng.uniform(0.05, 0.95, pairs)
    complement = 1.0 - parameter
    shift = rng.uniform(1e-4, 3.0, pairs)

    def split(moments):
        for seed, mirrored in (
            (cn_pole_moment(shift, parameter, parameter_complement=complement), False),
            (sn_pole_moment(shift, parameter, parameter_complement=complement), True),
        ):
            harmonic_pole_moments(shift, seed, moments, ORDERS, mirrored=mirrored)

    start = time.perf_counter()
    for _ in range(edges):  # once per corner
        moments = harmonic_moments(
            parameter, ORDERS + POLE_HEADROOM + 1, complement=complement
        )
        harmonic_root_moments(moments, parameter)
        split(moments)
    for _ in range(edges * LIMITS):  # the plane split, against the corner's stack
        split(moments)
    return time.perf_counter() - start


def third_kind_cost(pairs: int, edges: int) -> float:
    """Return seconds for the third-kind integrals alone, the irreducible core.

    The first and second kind and the ring denominator's two characteristics are
    the corner's; the plane denominator's two are the edge's, at each limit.
    """
    rng = np.random.default_rng(3)
    parameter = rng.uniform(0.05, 0.95, pairs)
    characteristic = rng.uniform(-1.0, 0.9, pairs)
    start = time.perf_counter()
    for _ in range(edges):  # once per corner
        scipy.special.ellipk(parameter)
        scipy.special.ellipe(parameter)
        for _ in range(2):  # the ring denominator's two pole factors
            complete_pi(characteristic, parameter)
    for _ in range(edges * LIMITS):
        for _ in range(2):  # the plane denominator's two, per limit
            complete_pi(characteristic, parameter)
    return time.perf_counter() - start


def residual_quadrature_cost(pairs: int, edges: int, nodes: int) -> float:
    """Return seconds for the graded arsinh integral, once per edge limit.

    The regularised integrand replaces one ``arsinh`` with the model, its root and
    two logarithms, which is what removing the boundary logarithm costs per node.

    ONE integral per edge limit, not two.  The reduction leaves two, but the first
    is a function of the corner alone in all three components, so a closed chain of
    edges carries it twice with opposite signs and it cancels: it is formed only at
    the two ends of a dropped horizontal edge, and for a section with no horizontal
    edge -- the plasma cell -- it is never formed at all.
    """
    rng = np.random.default_rng(1)
    radius = rng.uniform(5.0, 7.0, pairs)[:, None]
    offset = rng.uniform(0.1, 2.0, pairs)[:, None]
    width = rng.uniform(1e-4, 1.0, pairs)[:, None]
    node, weight = np.polynomial.legendre.leggauss(nodes // 2)
    start = time.perf_counter()
    for _ in range(edges * LIMITS):
        for _ in range(2):  # the two graded halves of the range
            span = np.arcsinh(0.25 * np.pi / width)
            stretch = 0.5 * span * (node + 1.0)[None, :]
            angle = width * np.sinh(stretch)
            jacobian = 0.5 * span * width * np.cosh(stretch)
            near = np.sin(angle) ** 2
            numerator = radius - offset * near
            denominator = np.sqrt(offset**2 + radius**2 * near * (1.0 - near))
            model = np.hypot(offset, radius * angle)
            regular = np.log(numerator + np.hypot(numerator, denominator)) + np.log(
                model / denominator
            )
            (jacobian * regular) @ weight
    return time.perf_counter() - start


def measured_cost(pairs: int, edges: int, nodes: int, repeats: int = 3) -> float:
    """Return the median seconds for flux alone and for flux with both field rows."""
    rng = np.random.default_rng(2)
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, edges, endpoint=False)
    vertices = np.column_stack([6.2 + 0.06 * np.cos(angle), 0.06 * np.sin(angle)])
    target_r = 6.2 + rng.uniform(-1.5, 1.5, pairs)
    target_z = rng.uniform(-1.5, 1.5, pairs)
    out = []
    for evaluate in (polygon_analytic_flux, polygon_analytic_greens):
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            evaluate(target_r, target_z, vertices, nodes=nodes)
            elapsed.append(time.perf_counter() - start)
        out.append(float(np.median(elapsed)))
    return out


if __name__ == "__main__":
    nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    third = third_kind_cost(PAIRS, EDGES)
    special = special_function_cost(PAIRS, EDGES)
    residual = residual_quadrature_cost(PAIRS, EDGES, nodes)
    total = special + residual
    print(f"pairs {PAIRS}, edges per pair {EDGES}, limits {LIMITS}, g_p nodes {nodes}")
    print(f"  third-kind integrals   {1e6 * third / PAIRS:9.2f} us/pair")
    print(f"  moment stacks and all  {1e6 * special / PAIRS:9.2f} us/pair")
    print(f"  graded g_p quadrature  {1e6 * residual / PAIRS:9.2f} us/pair")
    print(f"  floor                  {1e6 * total / PAIRS:9.2f} us/pair")
    flux, field = measured_cost(PAIRS, EDGES, nodes)
    print(f"  assembled psi          {1e6 * flux / PAIRS:9.2f} us/pair")
    print(f"  assembled psi + field  {1e6 * field / PAIRS:9.2f} us/pair")
    print(f"  psi over floor         {flux / total:9.2f} x")
    print(f"  psi + field over floor {field / total:9.2f} x")
