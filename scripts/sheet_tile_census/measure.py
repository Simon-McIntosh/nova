"""Measure the production sheet tile and a warm full WEST interaction build."""

import argparse
import json
import os
from pathlib import Path
import re
import resource
import shlex
import socket
import subprocess
import sys
from time import perf_counter

import imas
import jax
import numpy as np
import shapely

from nova.biot import tiledassembly as tile
from nova.jax.config import Precision, configure_dtypes
from scripts.h200_test_lane import cache_guard
from scripts.prototypes.iron_magnetisation.magnetisation import Cells, mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=("miss", "hit"), required=True)
    parser.add_argument("--tile-only", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision} tree: {Path.cwd().resolve()} command: "
        + shlex.join([sys.executable, *sys.argv]),
        flush=True,
    )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    device = jax.devices()[0]
    assert "H200" in device.device_kind, device
    root = Path(os.environ["NOVA_COMPILATION_CACHE_ROOT"]).resolve()
    assert str(root) != "/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm"
    configuration = tile.configure_production_compilation_cache()
    directory = configuration.resolved_directory
    assert directory.is_relative_to(root), (directory, root)
    ledger = cache_guard.install()
    receipt = dict(
        size_m=args.size,
        host=socket.gethostname(),
        device=device.device_kind,
        job=os.environ["SLURM_JOB_ID"],
        cache_root=str(root),
        cache_directory=str(directory),
        pid=os.getpid(),
        revision=revision,
        module=tile.__file__,
        cwd=str(Path.cwd().resolve()),
    )

    def save(phase):
        receipt["phase"] = phase
        receipt["peak_host_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print("RECEIPT " + json.dumps(receipt), flush=True)

    save("starting")
    try:
        machine = (
            "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/WEST/machine/dd4-1-0"
        )
        with imas.DBEntry(
            f"imas:hdf5?path={machine}", "r", dd_version="4.1.0"
        ) as entry:
            iron = entry.get("iron_core", autoconvert=False)
        polygons = []
        for segment in iron.segment:
            region = shapely.Polygon(
                np.column_stack(
                    (segment.geometry.outline.r, segment.geometry.outline.z)
                )
            )
            polygons.extend(mesh(region.buffer(0), args.size).polygons)
        cells = Cells(tuple(polygons))
        count = len(polygons)
        assert count == 1696 if args.size == 0.08 else count > 1696
        targets = cells.centroids
        edges = tile.cell_edges(polygons)
        plan = tile.TilePlan(128, 128, 256, 16, 48)
        evaluator = tile._warm_sheet_evaluator(
            plan, False, 1, Precision.DOUBLE, configuration
        )
        geometry = (
            targets[:128, 0],
            targets[:128, 1],
            edges.start_r[:128],
            edges.start_z[:128],
            edges.end_r[:128],
            edges.end_z[:128],
        )
        prepared = evaluator.prepare(*geometry, synchronize=True)
        receipt.update(
            cells=count, edges=len(edges.start_r), tile_shape=[128, 128], block=256
        )
        save("inputs-ready")
        started = perf_counter()
        lowered = evaluator._kernel.lower(*prepared[2])
        receipt["lower_seconds"] = perf_counter() - started
        text = lowered.as_text()
        receipt["stablehlo_instructions"] = len(
            re.findall(r"\bstablehlo\.[a-z_]+\b", text)
        )
        receipt["stablehlo_bytes"] = len(text.encode())
        assert receipt["stablehlo_instructions"] > 0
        started = perf_counter()
        executable = lowered.compile()
        receipt["backend_compile_seconds"] = perf_counter() - started
        receipt["compile_wall_seconds"] = (
            receipt["lower_seconds"] + receipt["backend_compile_seconds"]
        )
        receipt["executable_bytes"] = len(executable.runtime_executable().serialize())
        receipt["compile_peak_host_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        receipt["cache_rows"] = ledger.rows()
        rows = [row for row in ledger.rows() if "jit__lambda" in row["cache_key"]]
        assert len(rows) == 1, rows
        row = rows[0]
        receipt["tile_cache_key"] = row["cache_key"]
        receipt["tile_cache_result"] = "hit" if row["hits"] else "miss"
        assert receipt["tile_cache_result"] == args.expect, row
        assert not row.get("unpersisted_misses"), row
        assert any(directory.glob("*-cache")), directory
        save("compiled")
        jax.block_until_ready(evaluator.launch(prepared, executable))
        started = perf_counter()
        for _ in range(3):
            jax.block_until_ready(evaluator.launch(prepared, executable))
        receipt["warm_tile_seconds"] = (perf_counter() - started) / 3
        if not args.tile_only:
            evaluator(*geometry)
            started = perf_counter()
            matrix = tile.cell_interaction(
                polygons, targets[:, 0], targets[:, 1], plan=plan
            )
            receipt["warm_build_seconds"] = perf_counter() - started
            receipt["matrix_shape"] = list(matrix.shape)
            receipt["matrix_finite"] = bool(np.isfinite(matrix).all())
            receipt["matrix_max_abs"] = float(np.max(np.abs(matrix)))
            assert matrix.shape == (2 * count, 2 * count)
            assert receipt["matrix_finite"] and receipt["matrix_max_abs"] > 0
        receipt["bounds"] = dict(
            compile=receipt["compile_wall_seconds"] <= 60,
            executable=receipt["executable_bytes"] <= 50_000_000,
            host=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            <= 64_000_000_000,
        )
        save("complete")
        return 0 if all(receipt["bounds"].values()) else 2
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        save("failed")
        raise


if __name__ == "__main__":
    sys.exit(main())
