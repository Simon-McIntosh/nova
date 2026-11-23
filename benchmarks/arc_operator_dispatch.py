"""Measure finite-arc banding and packed accelerator dispatch.

Examples:

    python benchmarks/arc_operator_dispatch.py placement
    python benchmarks/arc_operator_dispatch.py wall --targets 4096
    python benchmarks/arc_operator_dispatch.py gpu --devices 1 --tile 8

The placement study compares the area-centroid moment filament, the RMS-radius
moment filament, and the bare RMS-radius filament on rays within the sweep and
beyond both end planes.  The wall study uses a three-dimensional target cloud and
reports the exact fraction selected by the finite swept-volume coordinate.  The
GPU study reports the first dispatch, the warm dispatch, and device count
separately; persistent compilation caching is controlled by the caller's
environment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from nova.biot.arcbandedcoupling import (
    ARC_FAR_LIMIT,
    arc_band,
    arc_far_limit,
    arc_moment_filament,
    banded_arc_greens,
)
from nova.biot.greens import section_centroid
from nova.biot.polygon import pad_batch
from nova.biot.polygonarc import polygon_arc_greens
from nova.biot.tiledassembly import TilePlan, tile_evaluator

RADIUS = 6.2
SECTION_RADIUS = 0.06
START, END = 0.4, 2.1


def hexagon(r0=RADIUS, z0=0.0):
    """Return one representative symmetric section."""
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    return np.column_stack(
        [
            r0 + SECTION_RADIUS * np.cos(angle),
            z0 + SECTION_RADIUS * np.sin(angle),
        ]
    )


def trapezium(r0=RADIUS, z0=0.0):
    """Return a skewed section whose third moments survive."""
    return np.array(
        [
            [r0 - 0.05, z0 - 0.04],
            [r0 + 0.07, z0 - 0.05],
            [r0 + 0.055, z0 + 0.045],
            [r0 - 0.035, z0 + 0.03],
        ]
    )


def thin_plate(r0=RADIUS, z0=0.0):
    """Return the elongated parallelogram from the arc acceptance geometry."""
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
    "trapezium": trapezium,
    "thin-plate": thin_plate,
}


def target_routes(vertices, levels):
    """Return nonsymmetric target rays within the sweep and past both ends."""
    levels = np.asarray(levels)
    centre = section_centroid(vertices)
    offset = vertices - centre
    radius = float(np.max(np.hypot(offset[:, 0], offset[:, 1])))
    return {
        "poloidal": (
            centre[0] + levels * radius,
            np.full_like(levels, centre[1] + 0.35 * radius),
            np.full_like(levels, 0.5 * (START + END) + 0.17),
        ),
        "before-start": (
            np.full_like(levels, centre[0] + 0.27 * radius),
            np.full_like(levels, centre[1] + 0.31 * radius),
            START - levels * radius / centre[0],
        ),
        "after-end": (
            np.full_like(levels, centre[0] - 0.23 * radius),
            np.full_like(levels, centre[1] - 0.29 * radius),
            END + levels * radius / centre[0],
        ),
    }


def relative_envelope(got, exact):
    """Return each pair's worst deviation, scaled by each row's peak."""
    scale = np.max(np.abs(exact), axis=1)[:, None]
    return np.max(np.abs(got - exact) / scale, axis=0)


def placement_study() -> dict:
    """Return the three filament placements' accuracy across all target routes."""
    levels = np.geomspace(4.0, 80.0, 20)
    records = {}
    limits = {}
    worst = {label: 0.0 for label in ("centroid-moment", "rms-moment", "rms-bare")}
    for section, factory in SECTIONS.items():
        vertices = factory()
        far_limit = arc_far_limit(vertices)
        limits[section] = far_limit
        section_record = {}
        for route, (target_r, target_z, target_phi) in target_routes(
            vertices, levels
        ).items():
            exact = np.stack(
                polygon_arc_greens(target_r, target_z, target_phi, vertices, START, END)
            )
            route_record = {}
            for label, placement, corrected in (
                ("centroid-moment", "centroid", True),
                ("rms-moment", "rms", True),
                ("rms-bare", "rms", False),
            ):
                got = np.stack(
                    arc_moment_filament(
                        target_r,
                        target_z,
                        target_phi,
                        vertices,
                        START,
                        END,
                        placement=placement,
                        corrected=corrected,
                    )
                )
                envelope = relative_envelope(got, exact)
                route_record[label] = envelope.tolist()
                worst[label] = max(
                    worst[label], float(np.max(envelope[levels >= far_limit]))
                )
            section_record[route] = route_record
        records[section] = section_record
    return {
        "levels": levels.tolist(),
        "base_far_limit": ARC_FAR_LIMIT,
        "far_limits": limits,
        "sections": records,
        "worst_beyond_seam": worst,
    }


def cloud(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic three-dimensional cloud spanning near and far pairs."""
    index = np.arange(count)
    fraction = (index + 0.5) / count
    turn = 18.0 * np.pi * fraction
    reach = SECTION_RADIUS * (0.2 + 42.0 * fraction)
    target_r = RADIUS + reach * np.cos(turn)
    target_z = reach * np.sin(turn)
    target_phi = START - 0.7 + (END - START + 1.4) * np.remainder(fraction * 17.0, 1.0)
    return target_r, target_z, target_phi


def timed(call, repeats: int = 3) -> float:
    """Return the median wall time of ``call``."""
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        elapsed.append(time.perf_counter() - start)
    return statistics.median(elapsed)


def wall_study(count: int) -> dict:
    """Return exact and banded host wall time at a realistic column width."""
    vertices = hexagon()
    target_r, target_z, target_phi = cloud(count)
    assignment = arc_band(target_r, target_z, target_phi, vertices, START, END)
    exact_seconds = timed(
        lambda: polygon_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    banded_seconds = timed(
        lambda: banded_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    exact = np.stack(
        polygon_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    banded = np.stack(
        banded_arc_greens(target_r, target_z, target_phi, vertices, START, END)
    )
    return {
        "targets": count,
        "exact_pairs": int(np.count_nonzero(assignment == 0)),
        "filament_pairs": int(np.count_nonzero(assignment == 1)),
        "exact_fraction": float(np.mean(assignment == 0)),
        "exact_seconds": exact_seconds,
        "banded_seconds": banded_seconds,
        "speedup": exact_seconds / banded_seconds,
        "worst_relative": float(np.max(relative_envelope(banded, exact))),
    }


def gpu_study(devices: int, tile: int, block: int) -> dict:
    """Return cold and warm packed finite-arc tile dispatch on visible GPUs."""
    source_count = tile
    target_count = tile
    sections = [
        hexagon(RADIUS + 0.004 * source, 0.003 * (source % 3))
        for source in range(source_count)
    ]
    edge, weight, norm = pad_batch(sections)
    target_r, target_z, target_phi = cloud(target_count)
    start = START + 0.002 * np.arange(source_count)
    end = END + 0.002 * np.arange(source_count)
    plan = TilePlan(
        target_tile=target_count,
        source_tile=source_count,
        block=block,
        n_panels=1,
        n_nodes=1,
    )
    evaluate = tile_evaluator(
        plan,
        batched=True,
        kernel="closed",
        geometry="arc",
        devices=devices,
    )
    geometry = (target_r, target_z, target_phi, edge, weight, norm, start, end)
    first_start = time.perf_counter()
    first = np.stack(evaluate(*geometry))
    first_seconds = time.perf_counter() - first_start
    warm_seconds = timed(lambda: evaluate(*geometry))
    expected = np.column_stack(
        [
            polygon_arc_greens(
                target_r[pair // source_count],
                target_z[pair // source_count],
                target_phi[pair // source_count],
                sections[pair % source_count],
                start[pair % source_count],
                end[pair % source_count],
            )
            for pair in range(target_count * source_count)
        ]
    ).reshape(5, target_count, source_count)
    record = {
        "devices": devices,
        "targets": target_count,
        "sources": source_count,
        "pairs": target_count * source_count,
        "block": block,
        "first_seconds": first_seconds,
        "warm_seconds": warm_seconds,
        "warm_microseconds_per_pair": 1.0e6
        * warm_seconds
        / (target_count * source_count),
        "compile_count": evaluate.compile_count,
        "worst_relative": float(
            np.max(relative_envelope(first.reshape(5, -1), expected.reshape(5, -1)))
        ),
    }
    if target_count * source_count <= 4:
        record["computed"] = first.tolist()
        record["expected"] = expected.tolist()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=("placement", "wall", "gpu"))
    parser.add_argument("--targets", type=int, default=4096)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--block", type=int, default=1)
    args = parser.parse_args()
    if args.study == "placement":
        result = placement_study()
    elif args.study == "wall":
        result = wall_study(args.targets)
    else:
        result = gpu_study(args.devices, args.tile, args.block)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
