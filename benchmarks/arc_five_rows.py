"""Evidence for the finite arc's five rows: the collapse, the gate and the constant.

Three panels, one per claim the assembly rests on.

**Left -- the sweep closes onto the ring.**  Every row plotted against the arc's
sweep, with the landed full-turn values marked.  Two of the five go to zero at a
whole turn and three land on the ring, and the two that vanish do so by the parity
argument that leaves an axisymmetric conductor with no toroidal field -- the same
statement as ``cn K = 0`` annihilating the potential's radial row.  The fold is
what carries it, and the panel is the fold's own arithmetic seen from outside.

**Middle -- against an integral that shares nothing with the reduction.**  Each
row's worst deviation from a direct Biot-Savart volume integral, over four
sections and four sweeps, measured two ways: absolutely, normalised by the largest
row, and relative to each row's OWN size.  The two disagree by three decades on
exactly the rows that have become small, which is what the panel is for.

**Right -- the by-parts constant an odd row's second residual needs.**  A row
weighted by ``sin phi`` has a pure sine weight and its antiderivative's constant
is a choice.  Taken to vanish at the LOWER limit -- which is what the first
residual needs, and what the paper's own eq 19a does -- the upper boundary term is
that antiderivative's whole-range value times ``arsinh beta2`` at the amplitude,
and the second of those runs to a logarithm as the target reaches the edge's
extended line.  The near-pole seed runs there with it and the two cancel.  The
panel measures the size of that pair against the row it belongs to: it grows
without bound as the offset falls, so the cancellation is a real one, and taking
the constant to vanish at the UPPER limit removes both terms instead of balancing
them.

Run as ``python benchmarks/arc_five_rows.py [output.png]``.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.polygonarc import _Edge, _Vertex, polygon_arc_greens
from nova.biot.rangefunction import (
    across_the_range,
    product,
    rising_integral,
    scaled,
    total,
)

ROWS = ("$A_r$", "$A_\\phi$", "$B_r$", "$B_\\phi$", "$B_z$")
COLOUR = ("C0", "C1", "C2", "C3", "C4")
R0 = 6.2
CELL = 0.06


def hexagon(r0=R0, z0=0.0, radius=CELL):
    """Return the plasma cell section: a regular hexagon."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def collapse(axes):
    """Plot every row against the sweep, with the ring's own values marked."""
    vertices = hexagon()
    target_r = np.array([R0 + 3.0 * CELL])
    target_z = np.array([2.0 * CELL])
    phi = np.array([0.7])
    sweep = np.linspace(0.05, 2.0 * np.pi, 60)
    rows = np.array(
        [
            polygon_arc_greens(target_r, target_z, phi, vertices, 0.0, end)
            for end in sweep
        ]
    )[:, :, 0]
    psi, radial, vertical = polygon_analytic_greens(target_r, target_z, vertices)
    ring = (
        0.0,
        float(psi[0] / (2.0 * np.pi * target_r[0])),
        float(radial[0]),
        0.0,
        float(vertical[0]),
    )
    for index, (name, colour) in enumerate(zip(ROWS, COLOUR)):
        scale = max(abs(ring[index]), np.max(np.abs(rows[:, index])))
        axes.plot(sweep, rows[:, index] / scale, colour, label=name)
        axes.plot(
            2.0 * np.pi, ring[index] / scale, colour, marker="o", ms=7, mfc="none"
        )
    axes.axvline(2.0 * np.pi, color="0.7", lw=0.8, zorder=0)
    axes.set_xlabel("arc sweep [rad]")
    axes.set_ylabel("row, scaled by its own reach")
    axes.set_title("the sweep closes onto the ring\n(rings: the landed full turn)")
    axes.legend(fontsize=8, ncol=2, loc="lower left")


def gate(axes, table):
    """Plot each row's worst deviation from the volume integral, both measures."""
    position = np.arange(len(ROWS))
    axes.semilogy(position - 0.12, table["overall"], "o", color="C0", label="absolute")
    axes.semilogy(position + 0.12, table["by_row"], "s", color="C3", label="own scale")
    for index in range(len(ROWS)):
        axes.plot(
            [position[index] - 0.12, position[index] + 0.12],
            [table["overall"][index], table["by_row"][index]],
            color="0.7",
            lw=0.8,
            zorder=0,
        )
    axes.set_xticks(position)
    axes.set_xticklabels(ROWS)
    axes.set_ylabel("worst deviation from the volume integral")
    axes.set_title(
        "four sections, four sweeps\n(the gap is where a row has become small)"
    )
    axes.legend(fontsize=8)
    axes.grid(axis="y", which="both", color="0.9", lw=0.6)


def _odd_second_arsinh_parts(offset):
    """Return the odd row's row value and the pair the other constant would form.

    ``offset`` is the target's distance from the edge's extended line, which is what
    both the ``arsinh beta2`` divergence and the near-pole shift are set by.  The
    amplitude is a quarter turn, which is where the near end of the range is
    reached -- a target in the plane of one of the arc's ends.
    """
    edge = np.array([R0 - 0.5 * CELL, -0.5 * CELL, R0 + 0.5 * CELL, 0.6 * CELL])
    ra, za, rb, zb = edge
    slope = (rb - ra) / (zb - za)
    # place the target ON the edge's extended line, less the offset: r1 - r = offset
    target_z = np.full_like(offset, 0.35 * CELL)
    target_r = ra + slope * (target_z - za) - offset
    ones = np.ones_like(target_r)
    angle = (0.5 * np.pi * ones, ones, 0.0 * ones)
    vertex = _Vertex(target_r, target_z, (ra, za), angle, 128, residual=True)
    part = _Edge(target_r, target_z, edge, 128)
    row = part.terms(vertex)[0] + vertex.arsinh_terms()[0]
    # the antiderivative the OTHER constant leaves, and the transcendental it would
    # multiply at the amplitude: eq 9's arsinh beta2 coefficient, weighted by
    # -sin phi, integrated in the range variable from zero
    squared_slope = 1.0 + slope * slope
    coefficient = scaled(
        total(
            part.plane_squared,
            scaled(
                product(vertex.cosine, part.plane_radius),
                2.0 * squared_slope * vertex.radius,
            ),
        ),
        -0.5 / (squared_slope * np.sqrt(squared_slope)),
    )
    rising = rising_integral(across_the_range(coefficient))
    reached = part._second_arsinh(vertex, vertex.sine_squared, vertex.cosine_squared)
    return np.abs(row), np.abs(4.0 * vertex.value(rising) * reached)


def constant(axes):
    """Plot the size of the pair the other by-parts constant would have to cancel."""
    offset = np.logspace(-14, -1, 40) * CELL
    row, pair = _odd_second_arsinh_parts(offset)
    axes.loglog(offset / CELL, pair / row, "C3", label="term the other constant forms")
    axes.loglog(
        offset / CELL, np.ones_like(offset), "C0", label="the row it belongs to"
    )
    axes.set_xlabel("target offset from the edge's line, in section radii")
    axes.set_ylabel("size relative to the row")
    axes.set_title(
        "the odd rows' by-parts constant\n(vanishing at the lower limit forms both)"
    )
    axes.legend(fontsize=8, loc="upper right")
    axes.grid(which="both", color="0.9", lw=0.6)


def measure_gate():
    """Return the gate table, computed rather than quoted."""
    import tests.test_biotpolygonarc as harness

    overall = np.zeros(len(ROWS))
    by_row = np.zeros(len(ROWS))
    for vertices in harness.SECTIONS.values():
        target_r, target_z = harness.gate_targets(vertices, directions=4)
        phi = harness.gate_azimuths(len(target_r))
        for start, end in harness.SWEEPS.values():
            got = np.stack(
                polygon_arc_greens(target_r, target_z, phi, vertices, start, end)
            )
            want = harness.volume_quadrature(
                target_r, target_z, phi, vertices, start, end
            )
            scale = np.max(np.abs(want))
            overall = np.maximum(overall, np.max(np.abs(got - want), axis=1) / scale)
            by_row = np.maximum(by_row, harness.worst_by_row(got, want))
    return {"overall": overall, "by_row": np.maximum(by_row, overall)}


def main(path: str) -> None:
    """Build the three panels and write the figure."""
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    collapse(axes[0])
    gate(axes[1], measure_gate())
    constant(axes[2])
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    print(path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "arc_five_rows.png")
