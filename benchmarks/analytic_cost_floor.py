"""Cost floor of a closed-form polygon evaluation, from its primitives alone.

Transcribing Urankar Part V's closed form is a large job, so it is worth knowing
what it can possibly buy before doing it. This times only the pieces that cannot
be avoided -- the special-function calls and the two residual ``arsinh``
quadratures, per polygon edge, per target-source pair -- and ignores the
coefficient algebra between them. The result is a LOWER bound on the closed
form's cost and therefore an UPPER bound on the speedup available.

Per edge of a full-turn ring the closed form needs the complete integrals of the
first and second kind, four complete integrals of the third kind (characteristics
``n_1, n_2, m_1, m_2``), the three moment families built from them, and the two
``arsinh`` integrals ``g_p`` that the paper leaves numerical. Everything else is
rational algebra in quantities already computed.

    python benchmarks/analytic_cost_floor.py [nodes]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.special

from nova.biot.elliptic import complete_pi, pole_moment, sn_cn_moments, sn_moments

PAIRS = 4096
EDGES = 6  # a hexagonal plasma cell contributes all six


def special_function_cost(pairs: int, edges: int) -> float:
    """Return seconds for the unavoidable special-function calls of one build."""
    rng = np.random.default_rng(0)
    parameter = rng.uniform(0.05, 0.95, pairs)
    characteristic = rng.uniform(-1.0, 0.9, pairs)
    start = time.perf_counter()
    for _ in range(edges):
        scipy.special.ellipk(parameter)
        scipy.special.ellipe(parameter)
        for _ in range(4):
            complete_pi(characteristic, parameter)
            pole_moment(np.abs(characteristic) * 0.9, parameter)
        sn_moments(parameter, 5)
        sn_cn_moments(parameter, 4)
    return time.perf_counter() - start


def residual_quadrature_cost(pairs: int, edges: int, nodes: int) -> float:
    """Return seconds for the two arsinh integrals the paper leaves numerical."""
    rng = np.random.default_rng(1)
    radius = rng.uniform(5.0, 7.0, pairs)[:, None]
    offset = rng.uniform(0.1, 2.0, pairs)[:, None]
    node, weight = np.polynomial.legendre.leggauss(nodes)
    alpha = 0.25 * np.pi * (node + 1.0)
    start = time.perf_counter()
    for _ in range(edges):
        for _ in range(2):  # g_1 and g_2
            phi = np.pi - 2.0 * alpha
            argument = (radius - offset * np.cos(phi)) / np.sqrt(
                offset**2 + radius**2 * np.sin(phi) ** 2
            )
            np.arcsinh(argument) @ (0.25 * np.pi * weight)
    return time.perf_counter() - start


if __name__ == "__main__":
    nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    special = special_function_cost(PAIRS, EDGES)
    residual = residual_quadrature_cost(PAIRS, EDGES, nodes)
    total = special + residual
    print(f"pairs {PAIRS}, edges per pair {EDGES}, g_p nodes {nodes}")
    print(f"  special functions      {1e6 * special / PAIRS:9.2f} us/pair")
    print(f"  residual g_p quadrature{1e6 * residual / PAIRS:9.2f} us/pair")
    print(f"  floor                  {1e6 * total / PAIRS:9.2f} us/pair")
