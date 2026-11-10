"""Compare every material qualification target-edge pair with precise rings."""

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import types
import warnings

import numpy as np
import shapely

from nova.jax.config import configure_dtypes

from nova.biot import sheet
from nova.biot.tiledassembly import cell_edges
from scripts.edge_sheet.reference import graded_precise_ring_reference

configure_dtypes()


def meshes():
    owner = types.ModuleType("committed_mesh_owner")
    sys.modules[owner.__name__] = owner
    source = subprocess.check_output(
        ["git", "show", "25eabd5de:nova/biot/magnetisation.py"], text=True
    )
    exec(compile(source, "<committed material mesh owner>", "exec"), owner.__dict__)
    sphere = (
        shapely.Point(0, 0)
        .buffer(0.5, 256)
        .intersection(shapely.box(0, -0.5, 0.5, 0.5))
    )
    shell = (
        shapely.Point(0, 0)
        .buffer(1, 512)
        .difference(shapely.Point(0, 0).buffer(0.9, 512))
        .intersection(shapely.box(0, -1, 1, 1))
    )
    return [
        ("sphere", owner._mesh(sphere, 0.1), np.array([[1.0, 0.0], [0.0, 1.0]])),
        (
            "shell",
            owner._mesh(shell, 0.04),
            np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0]]),
        ),
    ]


def reference_chunk(rows):
    result = []
    for row in rows:
        r, z, a, h, b, k = row
        value = graded_precise_ring_reference((a, h), (b, k), (r, z))
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            scalar = sheet.sheet_field(r, z, (a, h), (b, k))
        result.append((*scalar, *value[1:]))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "module:",
        sheet.__file__,
        "cwd:",
        Path.cwd(),
        "revision:",
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        flush=True,
    )
    import jax
    import jax.numpy as jnp

    assert jax.config.jax_enable_x64
    evaluate = jax.jit(lambda *p: jnp.stack(sheet.packed_sheet_field(jnp, *p), axis=-1))
    scales = jax.jit(
        lambda *p: jnp.stack(sheet.packed_sheet_end_scale(jnp, *p), axis=-1)
    )
    failures = 0
    for name, mesh, probes in meshes():
        # The material owner uses this exact dilation before edge identity rounding.
        edges = cell_edges(tuple(p * 1024 for p in mesh.polygons))
        sources = (
            np.column_stack((edges.start_r, edges.start_z, edges.end_r, edges.end_z))
            / 1024
        )
        targets = np.concatenate((mesh.centroids, probes))
        rows = np.column_stack(
            (
                np.repeat(targets, len(sources), axis=0),
                np.tile(sources, (len(targets), 1)),
            )
        )
        print(
            name,
            "cells:",
            len(mesh.centroids),
            "edges:",
            len(sources),
            "targets:",
            len(targets),
            "pairs:",
            len(rows),
            flush=True,
        )
        packed = []
        end_scales = []
        dilated = []
        for start in range(0, len(rows), 256):
            batch = rows[np.minimum(np.arange(start, start + 256), len(rows) - 1)]
            n = min(256, len(rows) - start)
            packed.append(np.asarray(evaluate(*map(jnp.asarray, batch.T)))[:n])
            dilated.append(
                np.asarray(evaluate(*map(jnp.asarray, (batch * 1024).T)))[:n]
            )
            end_scales.append(np.asarray(scales(*map(jnp.asarray, batch.T)))[:n])
        packed = np.concatenate(packed)
        dilated = np.concatenate(dilated)
        end_scales = np.concatenate(end_scales)
        np.savez(
            args.output / f"{name}-operands.npz",
            rows=rows,
            packed=packed,
            dilated=dilated,
            end_scales=end_scales,
        )
        chunks = [rows[i : i + 32] for i in range(0, len(rows), 32)]
        result = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for index, values in enumerate(pool.map(reference_chunk, chunks)):
                result.append(values)
                if index % 100 == 0:
                    print(
                        name,
                        "reference pairs:",
                        min((index + 1) * 32, len(rows)),
                        flush=True,
                    )
        result = np.concatenate(result)
        scalar = result[:, :2]
        reference = result[:, 2:]
        np.savez(
            args.output / f"{name}-reference.npz",
            rows=rows,
            scalar=scalar,
            reference=reference,
        )
        # A difference of end primitives has roundoff bounded by gamma_n times
        # their absolute sum. n=128 covers the elementary operations and sums.
        floor = 128 * np.finfo(float).eps * end_scales
        tolerance = 1e-10 * np.abs(reference) + floor
        summary = {
            "mesh": name,
            "pairs": len(rows),
            "absolute_floor": "128 eps times absolute endpoint sum",
            "max_floor": float(floor.max()),
        }
        for route, values in [
            ("scalar", scalar),
            ("packed", packed),
            ("dilated", dilated),
        ]:
            bad = (np.abs(values - reference) > tolerance) | ~np.isfinite(values)
            count = int(bad.any(axis=1).sum())
            failures += count
            ratio = np.abs(values - reference) / np.maximum(
                tolerance, np.finfo(float).tiny
            )
            worst = np.unravel_index(np.argmax(ratio), ratio.shape)
            summary[route] = {
                "nonfinite": int((~np.isfinite(values).all(axis=1)).sum()),
                "failed_pairs": count,
                "worst_budget_ratio": float(ratio[worst]),
                "worst_pair": rows[worst[0]].tolist(),
                "got": values[worst[0]].tolist(),
                "reference": reference[worst[0]].tolist(),
            }
        print(json.dumps(summary), flush=True)
        (args.output / f"{name}-summary.json").write_text(json.dumps(summary, indent=2))
    assert failures == 0, f"{failures} inaccurate pairs"


if __name__ == "__main__":
    main()
