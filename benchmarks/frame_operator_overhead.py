"""Decompose fixed FrameSpace overhead from per-pair Biot arithmetic.

The archived polygon-prism benchmark divided one framed solve by only 512 pairs
and called the difference "assembly".  This benchmark changes the target count
while holding one source fixed.  A fixed FrameSpace/xarray cost therefore appears
as a constant intercept, while the direct kernel appears as the slope.

Both a straight polygon prism and a swept polygon arc are measured.  Geometry and
the target cloud are built before timing, matching the archived benchmark: this
is operator orchestration, not polygon or path construction.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

RADIUS = 6.2
SECTION_RADIUS = 0.06
START, END = 0.4, 2.1
LIMITS = (-0.4, 0.6)


def timed(call, repeats=3):
    """Return the median wall time after one untimed warm call."""
    call()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed)


def hexagon(r0=RADIUS, z0=0.0):
    """Return a representative polygon section."""
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return np.column_stack(
        [
            r0 + SECTION_RADIUS * np.cos(angle),
            z0 + SECTION_RADIUS * np.sin(angle),
        ]
    )


def rectangle(r0=RADIUS, z0=0.0):
    """Return the rectangular section used by the finite-arc frame."""
    width, height = 0.1, 0.08
    return np.array(
        [
            [r0 - width / 2.0, z0 - height / 2.0],
            [r0 + width / 2.0, z0 - height / 2.0],
            [r0 + width / 2.0, z0 + height / 2.0],
            [r0 - width / 2.0, z0 + height / 2.0],
        ]
    )


def target_cylinders(count):
    """Return a deterministic cylindrical target cloud."""
    turn = np.linspace(0.0, 6.0 * np.pi, count)
    grow = 0.3 + 0.7 * np.arange(count) / max(count - 1, 1)
    radius = RADIUS + 8.0 * SECTION_RADIUS * grow * np.cos(turn)
    height = 8.0 * SECTION_RADIUS * grow * np.sin(turn)
    azimuth = np.linspace(START - 0.3, END + 0.3, count)
    cloud = np.stack(
        [radius * np.cos(azimuth), radius * np.sin(azimuth), height], axis=-1
    )
    return radius, height, azimuth, cloud


def straight_coilset():
    """Return one explicit polygon-prism source."""
    from nova.frame.coilset import CoilSet

    angle = np.array([0.1, 0.9])
    path = np.stack(
        [
            RADIUS * np.cos(angle),
            RADIUS * np.sin(angle),
            np.zeros_like(angle),
        ],
        axis=-1,
    )
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay"])
    coilset.winding.insert(
        path,
        {"hex": (0, 0, np.sqrt(3.0) * SECTION_RADIUS, 2.0 * SECTION_RADIUS)},
        nturn=1,
        Ic=1,
        minimum_arc_nodes=0,
        filament=False,
        ifttt=False,
    )
    coilset.subframe.loc[:, "segment"] = "polybeam"
    return coilset


def arc_coilset():
    """Return one explicit polygon-section arc source."""
    from nova.frame.coilset import CoilSet

    theta = np.array([START, 0.5 * (START + END), END])
    path = np.stack(
        [
            RADIUS * np.cos(theta),
            RADIUS * np.sin(theta),
            np.zeros_like(theta),
        ],
        axis=-1,
    )
    coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Ax", "Ay"])
    coilset.winding.insert(
        path,
        {"rect": (0, 0, 0.1, 0.08)},
        nturn=1,
        Ic=1,
        minimum_arc_nodes=3,
        filament=False,
        ifttt=False,
    )
    coilset.subframe.loc[:, "segment"] = "polybow"
    return coilset


def measure(geometry, counts):
    """Return direct and framed wall time across target counts."""
    if geometry == "straight":
        from nova.biot.polybeam import polygon_beam_greens

        coilset = straight_coilset()
        vertices = hexagon()

        def direct(radius, height, azimuth):
            along = np.linspace(LIMITS[0] - 0.3, LIMITS[1] + 0.3, len(radius))
            polygon_beam_greens(radius, height, along, vertices, *LIMITS)

    else:
        from nova.biot.polygonarc import polygon_arc_greens

        coilset = arc_coilset()
        vertices = rectangle()

        def direct(radius, height, azimuth):
            polygon_arc_greens(radius, height, azimuth, vertices, START, END)

    rows = []
    for count in counts:
        radius, height, azimuth, cloud = target_cylinders(count)
        direct_seconds = timed(lambda: direct(radius, height, azimuth))
        frame_seconds = timed(lambda: coilset.point.solve(cloud))
        rows.append(
            {
                "targets": count,
                "pairs": count,
                "direct_seconds": direct_seconds,
                "frame_seconds": frame_seconds,
                "direct_microseconds_per_pair": 1.0e6 * direct_seconds / count,
                "frame_microseconds_per_pair": 1.0e6 * frame_seconds / count,
                "ratio": frame_seconds / direct_seconds,
                "overhead_seconds": frame_seconds - direct_seconds,
            }
        )
    coefficient = np.polyfit(
        [row["pairs"] for row in rows],
        [row["frame_seconds"] for row in rows],
        1,
    )
    return {
        "geometry": geometry,
        "rows": rows,
        "fit_seconds_per_pair": float(coefficient[0]),
        "fit_fixed_seconds": float(coefficient[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", choices=("straight", "arc"))
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 64, 512, 4096])
    args = parser.parse_args()
    print(json.dumps(measure(args.geometry, args.counts), indent=2))


if __name__ == "__main__":
    main()
