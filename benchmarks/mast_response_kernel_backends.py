"""Time the MAST response kernel on the host path against the device path.

The frozen response carrier is built by ``nova.imas.mast_vacuum_response``
through :func:`nova.biot.polygon.polygon_greens`, which is NumPy with a Python
block loop over targets and cannot accept a traced array. Beside it the
repository ships :mod:`nova.biot.tiledassembly`, whose ``tile_evaluator``
traces the same fixed-node ring quadrature so one compile serves a whole build
and the pairs fill a device. Nothing calls it from the response build.

This measures the pair on the carrier's OWN problem: every stored winding-pack
section of the shot against the carrier's own resolved targets, at both pinned
grids. Compile and kernel wall are separated, because a build pays compile once
and kernel per tile, and a ratio that folds them together answers neither
question.

Agreement is measured, not assumed. A device path that returns a different
operator is not a faster build, so the two blocks are compared elementwise and
the largest absolute and relative differences are reported beside the timings.
The comparison is on the per-section flux rows rather than on assembled circuit
columns: the circuit assembly is a weighted sum both paths would share, so
summing first would only dilute a kernel disagreement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/mast-centroid-capability/kernel-backends/receipt.json"
)
SHOT = 22086
#: The two pinned carrier grids, named by the row count of their response.
GRID_ROWS = (1126, 4262)
#: Quadrature shape.
PANELS = 16
NODES = 48
#: A BATCHED tile is sized from a measurement, not from a byte budget. The
#: tiled-assembly docstring is explicit that TilePlan.peak_bytes models one
#: block's working set while mapping the blocks makes the whole tile's
#: quadrature live at once, and reports the measured high-water mark at the
#: 16x48 rule: 131 MB at 400 pairs, 864 MB at 1600, 1.4 GB at 6400. 64 x 64 is
#: 4096 pairs, so roughly a gigabyte, which leaves ample headroom on a 16 GB
#: card. Asking the byte planner for a batched tile instead returns the whole
#: pair space and the device refuses the 236 GiB that implies.
TILE_TARGET = 64
TILE_SOURCE = 64


def _sections(shot: int) -> list[np.ndarray]:
    """Return every stored winding-pack section of the shot as a polygon."""
    import zarr

    from nova.catalog.mast_geometry import shaped_section_vertices
    from nova.imas.mast_solve_inputs import SHOT_STORE

    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    field = {
        name: np.asarray(group[f"fcoil_{name}"], dtype=np.float64).reshape(-1)
        for name in ("r", "z", "width", "height", "ang1", "ang2")
    }
    return [
        shaped_section_vertices(
            field["r"][index],
            field["z"][index],
            field["width"][index],
            field["height"][index],
            field["ang1"][index],
            field["ang2"][index],
        )
        for index in range(field["r"].size)
    ]


def _targets(rows: int) -> np.ndarray:
    """Return the resolved targets of the pinned carrier with this row count."""
    from benchmarks.mast_response_carrier_warm import CARRIER_GRIDS

    for grid in CARRIER_GRIDS.values():
        if grid.pinned and grid.response_shape[0] == rows:
            with np.load(grid.path(), allow_pickle=False) as archive:
                return np.asarray(archive["resolved_targets"], dtype=np.float64)
    raise ValueError(f"no pinned grid carries {rows} response rows")


def _host_block(targets: np.ndarray, sections: list[np.ndarray]) -> tuple[Any, float]:
    """Return the per-section flux rows the shipped build path produces."""
    from nova.biot.polygon import polygon_greens

    target_r = np.ascontiguousarray(targets[:, 0])
    target_z = np.ascontiguousarray(targets[:, 1])
    block = np.empty((targets.shape[0], len(sections)), dtype=np.float64)
    started = perf_counter()
    for index, vertices in enumerate(sections):
        block[:, index] = polygon_greens(target_r, target_z, vertices)[0]
    return block, perf_counter() - started


def _device_block(
    targets: np.ndarray, sections: list[np.ndarray], tile_target: int, tile_source: int
) -> tuple[Any, dict[str, float | int]]:
    """Return the same rows through the traced tile kernel, timed in stages.

    The tile shape is a declared input rather than a byte budget, because this
    is the BATCHED kernel: mapping the quadrature blocks makes the whole tile's
    temporaries live at once, which the byte planner does not model. Every tile
    is padded to the plan shape, so one compile serves the build, and the
    compile count is recorded so a per-tile retrace shows rather than hides.
    """
    import jax

    from nova.biot.polygon import pad_batch
    from nova.biot.tiledassembly import TilePlan, tile_evaluator
    from nova.jax.config import Precision

    edge, weight, norm = pad_batch(sections)
    n_target = int(targets.shape[0])
    n_source = len(sections)
    plan = TilePlan(
        target_tile=min(tile_target, n_target),
        source_tile=min(tile_source, n_source),
        block=min(tile_target, n_target) * min(tile_source, n_source),
        n_panels=PANELS,
        n_nodes=NODES,
    )
    evaluator = tile_evaluator(
        plan,
        batched=True,
        kernel="quadrature",
        precision=Precision.DOUBLE,
        edge_count=int(edge.shape[0]),
    )
    target_r = np.ascontiguousarray(targets[:, 0])
    target_z = np.ascontiguousarray(targets[:, 1])
    block = np.empty((n_target, n_source), dtype=np.float64)

    def geometry_of(target_slice: slice, source_slice: slice):
        return (
            target_r[target_slice],
            target_z[target_slice],
            edge[..., source_slice],
            weight[:, source_slice],
            norm[source_slice],
        )

    tiles = list(plan.tiles(n_target, n_source))
    first = geometry_of(*tiles[0])
    started = perf_counter()
    prepared = evaluator.prepare(*first, synchronize=True)
    transfer_seconds = perf_counter() - started
    started = perf_counter()
    executable = evaluator.compile(prepared)
    compile_seconds = perf_counter() - started
    started = perf_counter()
    jax.block_until_ready(evaluator.launch(prepared, executable))
    first_launch_seconds = perf_counter() - started

    kernel_seconds = 0.0
    for target_slice, source_slice in tiles:
        tile_prepared = evaluator.prepare(
            *geometry_of(target_slice, source_slice), synchronize=True
        )
        started = perf_counter()
        rows = evaluator.launch(tile_prepared, executable)
        jax.block_until_ready(rows)
        kernel_seconds += perf_counter() - started
        values = evaluator.materialize(
            rows,
            target_slice.stop - target_slice.start,
            source_slice.stop - source_slice.start,
        )
        block[target_slice, source_slice] = values[0]
    return block, {
        "transfer_seconds": transfer_seconds,
        "compile_seconds": compile_seconds,
        "first_launch_seconds": first_launch_seconds,
        "kernel_seconds": kernel_seconds,
        "compile_count": int(evaluator.compile_count),
        "edge_count": int(edge.shape[0]),
        "tile_count": len(tiles),
        "tile_target": int(plan.target_tile),
        "tile_source": int(plan.source_tile),
        "batched": True,
        "tile_sizing": (
            "declared, not from TilePlan.peak_bytes: a batched tile does not "
            "respect that model"
        ),
    }


def _agreement(host: np.ndarray, device: np.ndarray) -> dict[str, float]:
    """Return how far the two operators are from being the same operator."""
    difference = np.abs(host - device)
    scale = np.maximum(np.abs(host), np.abs(device))
    relative = np.where(scale > 0.0, difference / np.maximum(scale, 1e-300), 0.0)
    return {
        "maximum_absolute_difference": float(difference.max()),
        "maximum_relative_difference": float(relative.max()),
        "host_magnitude_maximum": float(np.abs(host).max()),
        "finite_both": bool(np.all(np.isfinite(host)) and np.all(np.isfinite(device))),
    }


def measure(
    output: Path,
    grids: tuple[int, ...],
    host: bool,
    tile_target: int,
    tile_source: int,
) -> None:
    """Time both backends on each pinned grid and record their agreement."""
    import jax

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("extended precision did not take before any array was built")
    sections = _sections(SHOT)
    device = jax.devices()[0]
    rows: list[dict[str, Any]] = []
    for count in grids:
        targets = _targets(count)
        record: dict[str, Any] = {
            "response_rows": int(count),
            "target_count": int(targets.shape[0]),
            "section_count": len(sections),
            "pair_count": int(targets.shape[0] * len(sections)),
        }
        device_block, timings = _device_block(
            targets, sections, tile_target, tile_source
        )
        record["device"] = timings
        if host:
            host_block, host_seconds = _host_block(targets, sections)
            record["host_seconds"] = host_seconds
            record["agreement"] = _agreement(host_block, device_block)
            record["kernel_speedup"] = host_seconds / timings["kernel_seconds"]
            record["build_speedup_including_compile"] = host_seconds / (
                timings["compile_seconds"] + timings["kernel_seconds"]
            )
        rows.append(record)
        print(
            f"GRID rows={count} pairs={record['pair_count']} "
            f"host_s={record.get('host_seconds')} "
            f"compile_s={timings['compile_seconds']:.3f} "
            f"kernel_s={timings['kernel_seconds']:.4f} "
            f"speedup={record.get('kernel_speedup')}",
            flush=True,
        )
    document = {
        "artifact": "MAST response kernel on the host path against the device path",
        "shot": SHOT,
        "runtime": {
            "host": platform.node(),
            "platform": device.platform,
            "device_kind": device.device_kind,
            "jax": jax.__version__,
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "measurement_contract": {
            "host_path": (
                "nova.biot.polygon.polygon_greens, one call per section, which "
                "is what nova.imas.mast_vacuum_response builds the carrier with"
            ),
            "device_path": (
                "nova.biot.tiledassembly.tile_evaluator, batched quadrature "
                "kernel, one tile covering the whole pair space"
            ),
            "compared": (
                "per-section flux rows, not assembled circuit columns: the "
                "circuit sum is shared by both paths and would dilute a "
                "kernel disagreement"
            ),
            "kernel_seconds": "second launch of the same executable",
        },
        "grids": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"RECEIPT={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, nargs="*", default=list(GRID_ROWS))
    parser.add_argument("--no-host", action="store_true")
    parser.add_argument("--tile-target", type=int, default=TILE_TARGET)
    parser.add_argument("--tile-source", type=int, default=TILE_SOURCE)
    arguments = parser.parse_args()
    measure(
        arguments.output,
        tuple(arguments.rows),
        not arguments.no_host,
        arguments.tile_target,
        arguments.tile_source,
    )


if __name__ == "__main__":
    main()
