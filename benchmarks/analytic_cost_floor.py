"""Cost floor of the closed-form polygon evaluation, from its primitives alone.

This times only the pieces that cannot be avoided -- the special-function calls and
the two residual ``arsinh`` quadratures, per polygon edge, per target-source pair --
and ignores the coefficient algebra between them. The result is a LOWER bound on
the closed form's cost and therefore an UPPER bound on the speedup available.

Per edge of a full-turn ring the finished reduction needs, once per edge limit: the
complete integrals of the first and second kind; the plain and complement moment
stacks built from them; one complete integral of the third kind for each of the four
pole factors -- two denominators, each split between a root past either end of the
range -- and the two residual ``arsinh`` integrals the paper leaves numerical.
Everything else is rational algebra in quantities already computed.

The residual quadratures carry their grading, because that is not optional: each
integrand has a boundary layer at each end of the range whose width falls with the
target's offset from the edge's end level, and a plain rule at any practical order
steps over it. The grading costs a ``sinh`` and a ``cosh`` per node per pair, which
is why it appears in the floor rather than beside it.

``measured_cost`` times the assembled evaluation on the same shape of problem, for
the flux alone and for the flux with both field components, so the gap to the floor
is the coefficient algebra and the pole bookkeeping the floor deliberately ignores.

Measured, one SLURM compute core, 4096 pairs against a hexagonal cell, median of
three in a fresh process:

    third-kind integrals        17.2 us/pair
    moment stacks and all       52.1 us/pair
    graded g_p quadrature       41.3 us/pair
    floor                       93.4 us/pair
    assembled psi              134.0 us/pair       1.43 x floor
    assembled psi + field      134.4 us/pair       1.44 x floor

Two things to read from that. The field is FREE: eq 11b's rows share every
reduction the flux uses and differ only in polynomial weights, so both components
cost 0.4 us/pair on top of the flux -- which is the case for transcribing them
rather than differentiating the reduced flux. And the assembly now sits within half
again of its own primitives, against fifteen times when every pole family was
rebuilt per term; what is left to win is in the primitives, and two thirds of those
are the two moment stacks, whose length is set by the headroom a downward recursion
needs rather than by the nine orders the numerators reach.

    python benchmarks/analytic_cost_floor.py [nodes]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.special

from nova.biot.elliptic import (
    cn_pole_moments,
    complete_pi,
    sn_pole_moments,
    stable_cn_moments,
    stable_sn_moments,
)
from nova.biot.polygonanalytic import polygon_analytic_flux, polygon_analytic_greens

PAIRS = 4096
EDGES = 6  # a hexagonal plasma cell contributes all six
LIMITS = 2  # every edge is evaluated at both of its ends
ORDERS = 9


def special_function_cost(pairs: int, edges: int) -> float:
    """Return seconds for the unavoidable special-function calls of one build."""
    rng = np.random.default_rng(0)
    parameter = rng.uniform(0.05, 0.95, pairs)
    complement = 1.0 - parameter
    shift = rng.uniform(1e-4, 3.0, pairs)
    start = time.perf_counter()
    for _ in range(edges * LIMITS):
        plain = stable_sn_moments(parameter, ORDERS + 60)
        complete = stable_cn_moments(parameter, ORDERS + 60, complement=complement)
        for _ in range(2):
            cn_pole_moments(
                shift,
                parameter,
                ORDERS,
                moments=complete,
                parameter_complement=complement,
            )
            sn_pole_moments(
                shift,
                parameter,
                ORDERS,
                moments=plain,
                parameter_complement=complement,
            )
    return time.perf_counter() - start


def third_kind_cost(pairs: int, edges: int) -> float:
    """Return seconds for the third-kind integrals alone, the irreducible core."""
    rng = np.random.default_rng(3)
    parameter = rng.uniform(0.05, 0.95, pairs)
    characteristic = rng.uniform(-1.0, 0.9, pairs)
    start = time.perf_counter()
    for _ in range(edges * LIMITS):
        scipy.special.ellipk(parameter)
        scipy.special.ellipe(parameter)
        for _ in range(4):
            complete_pi(characteristic, parameter)
    return time.perf_counter() - start


def residual_quadrature_cost(pairs: int, edges: int, nodes: int) -> float:
    """Return seconds for the two graded arsinh integrals, per edge limit."""
    rng = np.random.default_rng(1)
    radius = rng.uniform(5.0, 7.0, pairs)[:, None]
    offset = rng.uniform(0.1, 2.0, pairs)[:, None]
    width = rng.uniform(1e-4, 1.0, pairs)[:, None]
    node, weight = np.polynomial.legendre.leggauss(nodes // 2)
    start = time.perf_counter()
    for _ in range(edges * LIMITS):
        for _ in range(2):  # the two residual integrals
            for _ in range(2):  # the two graded halves of the range
                span = np.arcsinh(0.25 * np.pi / width)
                stretch = 0.5 * span * (node + 1.0)[None, :]
                angle = width * np.sinh(stretch)
                jacobian = 0.5 * span * width * np.cosh(stretch)
                near = np.sin(angle) ** 2
                argument = (radius - offset * near) / np.sqrt(
                    offset**2 + radius**2 * near * (1.0 - near)
                )
                (jacobian * np.arcsinh(argument)) @ weight
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
    nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
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
