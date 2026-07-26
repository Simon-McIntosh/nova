"""Fresh-process per-pair cost of the finite-arc polygon-section closed form.

What the arc costs against the two evaluations it sits between: the full-turn
closed form (:mod:`nova.biot.polygonanalytic`), which is the same reduction with
every integral complete, and ``Bow``, which is the same finite arc for a
RECTANGULAR section through a fixed-node zeta quadrature.

Three things make the arc dearer than the ring and all three are structural
rather than incidental:

* the moment families are a tridiagonal solve carried a hundred and thirty orders
  past the last one wanted, where the ring runs a downward recursion over forty;
* there are TWO of them, because the rows weighted by ``sin phi`` contract against
  their own family;
* a corner is evaluated at BOTH of the arc's amplitudes plus, for the three rows
  whose fold is odd, at a quarter turn -- and that last one is the ring's own
  evaluation, so it is the ring's cost paid once more per corner.

Against that, the arc returns FIVE rows where the ring returns three, so the
per-row comparison is not the per-call one.

The variants come in two families and they are not interchangeable.  ``arc-`` and
``ring-`` call a reduction directly, which is the sharpest comparison between two
evaluations.  ``bow-`` and ``polybow-`` go through the FRAME -- a coilset, a
source assembly, the local-to-global transform and the matrix storage -- which is
the comparison a caller actually pays, and the only one in which the two element
classes are measured on the same terms.

One variant per process: repeats inside a single interpreter warm the allocator
and understate the first-build cost operator assembly actually pays.  Run as
``python benchmarks/arc_section_cost.py <variant>``; ``list`` prints the variants.
Output is one JSON record on stdout.

Run it on a compute node.  On a shared login node the same call has been observed
to vary five-fold, which is larger than every difference this is meant to resolve.
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np

R0, Z0 = 6.2, 0.0
CELL_RADIUS = 0.06  # hexagon circumradius of a ~560-cell ITER plasma mesh
RECT = (0.1, 0.08)  # a representative rectangular winding-pack section
N_TARGET = 512
REPEATS = 3
SWEEP = (0.4, 2.1)


def hexagon(r0=R0, z0=Z0, radius=CELL_RADIUS):
    """Return regular hexagon vertices, counter-clockwise."""
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    return np.column_stack([r0 + radius * np.cos(angle), z0 + radius * np.sin(angle)])


def rectangle(r0=R0, z0=Z0, width=RECT[0], height=RECT[1]):
    """Return rectangle vertices, counter-clockwise."""
    return np.array(
        [
            [r0 - width / 2, z0 - height / 2],
            [r0 + width / 2, z0 - height / 2],
            [r0 + width / 2, z0 + height / 2],
            [r0 - width / 2, z0 + height / 2],
        ]
    )


def trapezium(r0=R0, z0=Z0):
    """Return a slanted quadrilateral -- every edge slope differs."""
    return np.array(
        [
            [r0 - 0.05, z0 - 0.04],
            [r0 + 0.06, z0 - 0.05],
            [r0 + 0.07, z0 + 0.05],
            [r0 - 0.04, z0 + 0.03],
        ]
    )


def thin_plate(r0=R0, z0=Z0):
    """Return a high aspect-ratio parallelogram -- two horizontal edges dropped."""
    return np.array(
        [
            [r0 - 0.2, z0],
            [r0 + 0.2, z0 + 0.03],
            [r0 + 0.2, z0 + 0.0375],
            [r0 - 0.2, z0 + 0.0075],
        ]
    )


SECTIONS = {
    "hexagon": hexagon,
    "rectangle": rectangle,
    "trapezium": trapezium,
    "thin-plate": thin_plate,
}


def targets(count=N_TARGET, reach=8.0 * CELL_RADIUS):
    """Return a target cloud ringing the section, with azimuths across the sweep.

    A spiral rather than a grid, so no target is degenerate with a corner and none
    repeats another's contour distance; the azimuths span the sweep and both sides
    of it, which is what makes the amplitude fold do its work rather than land on
    one branch.
    """
    turn = np.linspace(0.0, 6.0 * np.pi, count)
    radius = R0 + reach * (0.3 + 0.7 * turn / turn[-1]) * np.cos(turn)
    height = Z0 + reach * (0.3 + 0.7 * turn / turn[-1]) * np.sin(turn)
    azimuth = np.linspace(SWEEP[0] - 1.0, SWEEP[1] + 1.0, count)
    return radius, height, azimuth


def _arc(name: str) -> tuple:
    """Return the finite-arc closed form's per-pair cost on one section."""
    from nova.biot.polygonarc import polygon_arc_greens

    vertices = SECTIONS[name]()
    radius, height, azimuth = targets()

    def once():
        polygon_arc_greens(radius, height, azimuth, vertices, *SWEEP)

    return _timed(once, len(radius)), len(radius)


def _ring(name: str) -> tuple:
    """Return the full-turn closed form's per-pair cost on the same section."""
    from nova.biot.polygonanalytic import polygon_analytic_greens

    vertices = SECTIONS[name]()
    radius, height, _ = targets()

    def once():
        polygon_analytic_greens(radius, height, vertices)

    return _timed(once, len(radius)), len(radius)


# Frame sections, as a winding's own descriptor writes them: a width and a height
# about the arc, which is what the element classes build their corners from.  The
# rectangle is the same section the bare variants use, so ``arc-rectangle`` against
# ``polybow-rectangle`` measures what the FRAME costs on top of the reduction.  The
# hexagon is the same size but not the same orientation -- the generator puts a
# corner on the r axis where ``hexagon()`` puts one on z, which leaves four live
# edges rather than six -- so read that pair as bow against polybow, not as framed
# against bare.
FRAME_SECTIONS = {
    "rectangle": {"rect": (0, 0, *RECT)},
    "hexagon": {"hex": (0, 0, np.sqrt(3.0) * CELL_RADIUS, 2.0 * CELL_RADIUS)},
}


def _frame(section: str, segment: str | None) -> tuple:
    """Return an element class's per-pair cost through the frame, same cloud.

    ``segment`` ``None`` leaves the frame's own routing alone, which sends a swept
    winding to ``Bow`` whatever its section is -- the rectangular-section arc
    through Urankar Part IV and the fixed-node zeta quadrature, and what a
    non-rectangular section is approximated by today.  Naming a segment overrides
    that.  Going through the frame rather than calling either reduction directly
    is what makes the two comparable: the local-to-global transform, the source
    assembly and the matrix storage are the same for both and are not free.
    """
    from nova.frame.coilset import CoilSet

    radius, height, azimuth = targets()
    start, end = SWEEP
    theta = np.array([start, 0.5 * (start + end), end])
    path = np.stack(
        [R0 * np.cos(theta), R0 * np.sin(theta), Z0 * np.ones_like(theta)], axis=-1
    )
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay"])
    coilset.winding.insert(
        path,
        FRAME_SECTIONS[section],
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    if segment is not None:
        coilset.subframe.loc[:, "segment"] = segment
    cloud = np.stack(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), height], axis=-1
    )

    def once():
        coilset.point.solve(cloud)

    return _timed(once, len(radius)), len(radius)


def _timed(call, pairs: int) -> float:
    """Return the median of ``REPEATS`` timings, in microseconds per pair.

    The median rather than the minimum: the first call in a process carries the
    import-time and allocator warmup that operator assembly pays too, and the
    minimum would hide exactly that.
    """
    elapsed = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return 1e6 * statistics.median(elapsed) / pairs


VARIANTS = {
    **{f"arc-{name}": (lambda name=name: _arc(name)) for name in SECTIONS},
    **{f"ring-{name}": (lambda name=name: _ring(name)) for name in SECTIONS},
    **{
        f"{element}-{section}": (
            lambda section=section, segment=segment: _frame(section, segment)
        )
        for element, segment in (("bow", None), ("polybow", "polybow"))
        for section in FRAME_SECTIONS
    },
}


def main(name: str) -> None:
    """Run one variant and print its record."""
    if name == "list":
        print("\n".join(sorted(VARIANTS)))
        return
    if name not in VARIANTS:
        raise SystemExit(f"unknown variant {name!r}; try 'list'")
    cost, pairs = VARIANTS[name]()
    print(
        json.dumps(
            {
                "variant": name,
                "microseconds_per_pair": round(cost, 4),
                "pairs": pairs,
                "repeats": REPEATS,
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "list")
