"""Publish compute-separated recovery matrices and bank their build receipt."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import socket
from time import perf_counter

import jax
import numpy as np

from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import _horizontal_reflection, _section_centroid
from nova.biot.tiledassembly import TilePlan, tile_evaluator
from nova.database.filepath import compute_provenance
from nova.database.zarrstore import ZarrStore
from nova.jax.config import Precision, configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "cache-provenance-receipt.json"
TILE = 32
PAIR_BLOCK = TILE * TILE


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _identity(resolution: str, provenance: str) -> dict[str, object]:
    return fixture.cache_identity(
        fixture.analytic_case(),
        requested_cells=fixture.FIXTURE_REQUESTS[resolution],
        wall_nodes=fixture.WALL_POINT_COUNT,
    ) | {"compute_provenance": provenance}


def _store(resolution: str, machine, provenance: str) -> tuple[str, str]:
    identity = _identity(resolution, provenance)
    store = ZarrStore(
        filename=f"{fixture.CACHE_FILENAME}_{abs(fixture.FIXTURE_REQUESTS[resolution])}",
        dirname=".nova",
    )
    store.group = store.hash_attrs(identity)
    store.data = fixture._dataset(machine, identity, store.group)
    store.store_overwrite()
    reader = ZarrStore(
        filename=store.filename, dirname=store.dirname, group=store.group
    )
    reader.load()
    fixture._from_dataset(reader.data, identity, store.group)
    return str(store.filepath), store.group


def publish_cpu(resolution: str, result: Path) -> None:
    """Republish the banked scalar-NumPy sibling under explicit provenance."""
    started = perf_counter()
    machine = fixture.cached_machine(
        fixture.analytic_case(),
        fixture.FIXTURE_REQUESTS[resolution],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    provenance = compute_provenance("numpy")
    store, group = _store(resolution, machine, provenance)
    _write_json(
        result,
        {
            "compute_provenance": provenance,
            "group": group,
            "hostname": socket.gethostname(),
            "matrix_build_seconds": machine.cache["build_seconds"],
            "resolution": resolution,
            "semantic_identity": _identity(resolution, provenance),
            "source_cache_hit": machine.cache["hit"],
            "store": store,
            "wall_seconds": perf_counter() - started,
        },
    )


def _geometry(machine):
    sections = list(machine.cell_polygons)
    edge, weight, norm = pad_batch(sections)
    section_centre = np.column_stack([_section_centroid(item) for item in sections])
    expansion_centre = np.asarray(
        machine.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    ).T
    reflection_axis = np.full(len(sections), np.nan, dtype=np.float64)
    reflection_partner = np.repeat(
        np.arange(edge.shape[0], dtype=np.int32)[:, None], len(sections), axis=1
    )
    for column, vertices in enumerate(sections):
        reflection = _horizontal_reflection(vertices)
        if reflection is None:
            continue
        axis, vertex_partner = reflection
        reflection_axis[column] = axis
        for index in range(len(vertices)):
            reflection_partner[index, column] = vertex_partner[
                (index + 1) % len(vertices)
            ]
    return (
        edge,
        weight,
        norm,
        section_centre,
        expansion_centre,
        reflection_axis,
        reflection_partner,
    )


def _family(
    evaluator, executable, targets: np.ndarray, geometry
) -> tuple[np.ndarray, float]:
    output = np.empty((3, len(targets), len(geometry[2])), dtype=np.float64)
    kernel_seconds = 0.0
    for row_start in range(0, len(targets), TILE):
        row_stop = min(row_start + TILE, len(targets))
        for column_start in range(0, len(geometry[2]), TILE):
            column_stop = min(column_start + TILE, len(geometry[2]))
            columns = slice(column_start, column_stop)
            arguments = (
                targets[row_start:row_stop, 0],
                targets[row_start:row_stop, 1],
                geometry[0][:, :, columns],
                geometry[1][:, columns],
                geometry[2][columns],
                geometry[3][:, columns],
                geometry[4][:, columns],
                geometry[5][columns],
                geometry[6][:, columns],
            )
            prepared = evaluator.prepare(*arguments, synchronize=True)
            launched = perf_counter()
            rows = evaluator.launch(prepared, executable)
            jax.block_until_ready(rows)
            kernel_seconds += perf_counter() - launched
            values = evaluator.materialize(
                rows, row_stop - row_start, column_stop - column_start
            )
            output[:, row_start:row_stop, columns] = np.asarray(values[:3])
    return output, kernel_seconds


def build_h200(resolution: str, result: Path) -> None:
    """Cold-build all recovery matrices with the production H200 graph."""
    configure_dtypes()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected gpu backend, got {jax.default_backend()!r}")
    carrier_started = perf_counter()
    carrier = fixture.cached_machine(
        fixture.analytic_case(),
        fixture.FIXTURE_REQUESTS[resolution],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    carrier_seconds = perf_counter() - carrier_started
    geometry = _geometry(carrier)
    evaluator = tile_evaluator(
        TilePlan(TILE, TILE, PAIR_BLOCK, 16, 48),
        batched=True,
        kernel="moments",
        precision=Precision.DOUBLE,
        edge_count=int(geometry[0].shape[0]),
    )
    first = (
        carrier.node[:TILE, 0],
        carrier.node[:TILE, 1],
        geometry[0][:, :, :TILE],
        geometry[1][:, :TILE],
        geometry[2][:TILE],
        geometry[3][:, :TILE],
        geometry[4][:, :TILE],
        geometry[5][:TILE],
        geometry[6][:, :TILE],
    )
    cold_started = perf_counter()
    prepared = evaluator.prepare(*first, synchronize=True)
    compile_started = perf_counter()
    executable = evaluator.compile(prepared)
    compile_seconds = perf_counter() - compile_started
    grid, grid_seconds = _family(evaluator, executable, carrier.node, geometry)
    wall, wall_seconds = _family(evaluator, executable, carrier.wall_node, geometry)
    sample, sample_seconds = _family(
        evaluator, executable, carrier.sample_coordinates, geometry
    )
    matrix_seconds = perf_counter() - cold_started
    built = replace(
        carrier,
        plasma_to_grid=grid[0],
        plasma_to_grid_r=grid[1],
        plasma_to_grid_z=grid[2],
        plasma_to_wall=wall[0],
        plasma_to_wall_r=wall[1],
        plasma_to_wall_z=wall[2],
        plasma_to_sample=sample[0],
        plasma_to_sample_r=sample[1],
        plasma_to_sample_z=sample[2],
        cache={},
    )
    provenance = compute_provenance("jax")
    store_started = perf_counter()
    store, group = _store(resolution, built, provenance)
    store_seconds = perf_counter() - store_started
    pair_count = len(geometry[2]) * (
        len(carrier.node) + len(carrier.wall_node) + len(carrier.sample_coordinates)
    )
    _write_json(
        result,
        {
            "carrier_load_seconds": carrier_seconds,
            "compile_seconds": compile_seconds,
            "compute_provenance": provenance,
            "device_kind": jax.devices()[0].device_kind,
            "group": group,
            "hostname": socket.gethostname(),
            "kernel_seconds": grid_seconds + wall_seconds + sample_seconds,
            "matrix_build_seconds": matrix_seconds,
            "pair_count": pair_count,
            "resolution": resolution,
            "semantic_identity": _identity(resolution, provenance),
            "store": store,
            "store_seconds": store_seconds,
        },
    )


def merge(cpu_results: list[Path], gpu_results: list[Path]) -> None:
    cpu = [json.loads(path.read_text(encoding="utf-8")) for path in cpu_results]
    gpu = [json.loads(path.read_text(encoding="utf-8")) for path in gpu_results]
    entries = {}
    for resolution in fixture.FIXTURE_REQUESTS:
        host = next(item for item in cpu if item["resolution"] == resolution)
        device = next(item for item in gpu if item["resolution"] == resolution)
        entries[resolution] = {
            "cpu_sibling": host,
            "h200_cold_build": device,
            "groups_are_separate": host["group"] != device["group"],
            "shared_store": host["store"] == device["store"],
        }
    _write_json(
        RECEIPT,
        {
            "fixtures": entries,
            "semantic_key_field": "compute_provenance",
            "verdict": {
                "cpu_never_loads_device_artifact": all(
                    item["groups_are_separate"] for item in entries.values()
                ),
                "device_never_loads_cpu_artifact": all(
                    item["groups_are_separate"] for item in entries.values()
                ),
                "h200_and_cpu_siblings_share_store": all(
                    item["shared_store"] for item in entries.values()
                ),
                "mechanism": (
                    "the semantic identity hashes the code-generator platform, "
                    "device family and jaxlib version, so unlike provenance is a "
                    "cache miss while sibling groups coexist in one zarr store"
                ),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("publish-cpu", "build-h200"):
        command = commands.add_parser(name)
        command.add_argument(
            "--resolution", choices=fixture.FIXTURE_REQUESTS, required=True
        )
        command.add_argument("--result", type=Path, required=True)
    merging = commands.add_parser("merge")
    merging.add_argument("--cpu-result", type=Path, action="append", required=True)
    merging.add_argument("--gpu-result", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.command == "publish-cpu":
        publish_cpu(args.resolution, args.result)
    elif args.command == "build-h200":
        build_h200(args.resolution, args.result)
    else:
        merge(args.cpu_result, args.gpu_result)


if __name__ == "__main__":
    main()
