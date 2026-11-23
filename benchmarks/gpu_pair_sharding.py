"""Measure whole-operator GPU sharding over independent pair blocks.

The polygon-ring tile is the established accelerator workload.  This benchmark
keeps its tile and operator shapes fixed while changing only the number of
visible devices used by :func:`nova.biot.tiledassembly.assemble`.  Geometry is
replicated; pair blocks and their outputs are divided evenly across devices.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import time

from benchmarks.tiled_assembly import hex_mesh
from nova.biot.tiledassembly import TilePlan, assemble, tile_evaluator


def measure(kernel: str, devices: int, cells: int, tile: int, block: int) -> dict:
    """Return first and warm whole-build wall time."""
    sections, target_r, target_z = hex_mesh(cells)
    plan = TilePlan(tile, tile, block, 16, 48)
    evaluator = tile_evaluator(plan, batched=True, kernel=kernel, devices=devices)
    directory = pathlib.Path(tempfile.mkdtemp(prefix="gpu-pair-sharding-"))
    elapsed = []
    try:
        for index in range(2):
            start = time.perf_counter()
            assemble(
                directory / f"operator-{index}.zarr",
                target_r,
                target_z,
                sections,
                plan=plan,
                backend="jax",
                batched=True,
                kernel=kernel,
                evaluator=evaluator,
                devices=devices,
            )
            elapsed.append(time.perf_counter() - start)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    pairs = target_r.size * len(sections)
    return {
        "kernel": kernel,
        "devices": devices,
        "cells": cells,
        "pairs": pairs,
        "tile": tile,
        "block": block,
        "tiles": plan.tile_count(target_r.size, len(sections)),
        "first_seconds": elapsed[0],
        "warm_seconds": elapsed[1],
        "warm_microseconds_per_pair": 1.0e6 * elapsed[1] / pairs,
        "compile_count": evaluator.compile_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", choices=("quadrature", "closed"), default="quadrature"
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--cells", type=int, default=320)
    parser.add_argument("--tile", type=int, default=40)
    parser.add_argument("--block", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            measure(args.kernel, args.devices, args.cells, args.tile, args.block),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
