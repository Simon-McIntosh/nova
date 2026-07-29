"""Measure the tiled polygon build: memory against tile size, time against cores.

    python benchmarks/tiled_assembly.py tile <output.json> <targets>x<sources>
    python benchmarks/tiled_assembly.py budget <output.json>
    python benchmarks/tiled_assembly.py cores <output.json> <n1,n2,...>
    python benchmarks/tiled_assembly.py project <output.json>

``tile`` measures the resident-memory high-water mark of ONE tile evaluation, in
a fresh process, against a baseline taken after the imports -- the per-worker
peak the tiling exists to bound. It has to be one point per process: peak RSS is
a high-water mark, so a second tile in the same interpreter reports the first
tile's peak. ``budget`` sweeps the byte budget over a whole build. ``cores``
holds the budget fixed and sweeps the pool size. ``project`` extrapolates a
full plasma-mesh build from a measured per-pair rate.

Every mode writes JSON so a driver can collect several runs. Run on a compute
node; a shared login node cannot resolve any of this.
"""

from __future__ import annotations

import json
import pathlib
import resource
import shutil
import sys
import tempfile
import time

import numpy as np

from nova.biot.tiledassembly import TilePlan, assemble, plan_tiles

CELL_RADIUS = 0.06
R0 = 6.2


def hex_mesh(count: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return a hexagonally-tiled plasma-like cell set and its cell centres.

    Cells sit on the standard hexagonal lattice -- neighbouring centres at
    ``sqrt(3)`` circumradii -- inside a circular plasma-sized boundary, so the
    near-neighbour geometry of the real mesh is reproduced rather than assumed.
    """
    pitch = np.sqrt(3.0) * CELL_RADIUS
    span = int(np.ceil(np.sqrt(count)))
    centres = []
    for row in range(-span, span + 1):
        for column in range(-span, span + 1):
            r = R0 + pitch * (column + 0.5 * (row % 2))
            z = 1.5 * CELL_RADIUS * row
            centres.append((r, z))
    centres = np.array(centres)
    order = np.argsort(np.hypot(centres[:, 0] - R0, centres[:, 1]))
    centres = centres[order][:count]
    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    sections = [
        np.column_stack(
            [r + CELL_RADIUS * np.cos(angle), z + CELL_RADIUS * np.sin(angle)]
        )
        for r, z in centres
    ]
    return sections, centres[:, 0].copy(), centres[:, 1].copy()


def peak_bytes() -> int:
    """Return the peak resident set of this process and its children."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return 1024 * max(usage, children)


def run_build(sections, target_r, target_z, plan, workers) -> dict:
    """Return timing and footprint for one build into a throwaway store."""
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tiled-assembly-"))
    try:
        before = peak_bytes()
        start = time.perf_counter()
        assemble(
            directory / "coupling.zarr",
            target_r,
            target_z,
            sections,
            plan=plan,
            workers=workers,
        )
        seconds = time.perf_counter() - start
        pairs = target_r.size * len(sections)
        return {
            "workers": workers,
            "seconds": seconds,
            "pairs": pairs,
            "us_per_pair": 1e6 * seconds / pairs,
            "target_tile": plan.target_tile,
            "source_tile": plan.source_tile,
            "block": plan.block,
            "planned_peak_bytes": plan.peak_bytes,
            "tile_count": plan.tile_count(target_r.size, len(sections)),
            "measured_peak_bytes": peak_bytes(),
            "measured_peak_bytes_before": before,
        }
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def measure_tile(n_target: int, n_source: int) -> dict:
    """Return the resident high-water mark of one tile evaluation of this shape."""
    from nova.biot.polygon import pad_batch

    from nova.biot.tiledassembly import tile_coupling

    sections, target_r, target_z = hex_mesh(n_source)
    edge, weight, norm = pad_batch(sections)
    baseline = peak_bytes()
    start = time.perf_counter()
    tile_coupling(target_r[:n_target], target_z[:n_target], edge, weight, norm)
    seconds = time.perf_counter() - start
    return {
        "n_target": n_target,
        "n_source": n_source,
        "pairs": n_target * n_source,
        "seconds": seconds,
        "us_per_pair": 1e6 * seconds / (n_target * n_source),
        "baseline_bytes": baseline,
        "peak_bytes": peak_bytes(),
        "tile_bytes": peak_bytes() - baseline,
    }


def sweep_budget(cells=180) -> list[dict]:
    """Return one record per byte budget, at a single worker."""
    sections, target_r, target_z = hex_mesh(cells)
    out = []
    for budget in (1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20):
        plan = plan_tiles(target_r.size, len(sections), budget_bytes=budget)
        record = run_build(sections, target_r, target_z, plan, 1)
        record["budget_bytes"] = budget
        out.append(record)
    return out


def sweep_cores(counts: list[int], cells=320, tile=40) -> list[dict]:
    """Return one record per pool size, at a tile small enough to distribute.

    The tile is set explicitly rather than from a budget: a budget generous
    enough to be realistic puts a mesh this size in ONE tile, and a pool with
    one task measures nothing about scaling.
    """
    sections, target_r, target_z = hex_mesh(cells)
    plan = TilePlan(
        target_tile=tile, source_tile=tile, block=16, n_panels=16, n_nodes=48
    )
    return [run_build(sections, target_r, target_z, plan, n) for n in counts]


def project(cells=(560, 1000, 2000), rate_cells=140) -> list[dict]:
    """Return projected full-mesh build times from one measured per-pair rate."""
    sections, target_r, target_z = hex_mesh(rate_cells)
    plan = plan_tiles(target_r.size, len(sections), budget_bytes=16 << 20)
    measured = run_build(sections, target_r, target_z, plan, 1)
    rate = measured["us_per_pair"]
    out = []
    for count in cells:
        pairs = count * count
        out.append(
            {
                "cells": count,
                "pairs": pairs,
                "us_per_pair": rate,
                "serial_hours": rate * 1e-6 * pairs / 3600.0,
                "sixteen_core_minutes": rate * 1e-6 * pairs / 60.0 / 16.0,
            }
        )
    return [measured] + out


if __name__ == "__main__":
    mode, destination = sys.argv[1], pathlib.Path(sys.argv[2])
    match mode:
        case "tile":
            rows, columns = (int(part) for part in sys.argv[3].split("x"))
            records = [measure_tile(rows, columns)]
        case "budget":
            records = sweep_budget()
        case "cores":
            records = sweep_cores([int(n) for n in sys.argv[3].split(",")])
        case "project":
            records = project()
        case _:
            raise SystemExit(f"unknown mode {mode}")
    destination.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))
