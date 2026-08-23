"""Build and receipt the default plasma carrier's exact GPU interactions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
from time import perf_counter

import jax
import jaxlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import LineString
import zarr

from nova.biot.plasmagrid import PlasmaGrid
from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import _horizontal_reflection, _section_centroid
from nova.biot.tiledassembly import MOMENT_COMPONENTS, TilePlan, tile_evaluator
from nova.database.filepath import compute_provenance
from nova.database.zarrstore import ZarrStore
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.frame.coilset import CoilSet
from nova.jax.config import Precision, configure_dtypes
from scripts.analytic_oracle_fixtures.measure import analytic_case, limiter_contour


HERE = Path(__file__).resolve().parents[1]
RECEIPT = (
    HERE / "docs/figures/forward-operator-refinement/global-dplasma-gpu-rebuild.json"
)
FIGURE = RECEIPT.with_suffix(".png")
KERNEL_MERGE = "80979f40"
SCHEMA = "global-default-exact-moment-interactions"
STORE_FILENAME = "global_default_exact_moment_interactions"
TILE = 32
PAIR_BLOCK = TILE * TILE
COMPARATORS = {
    "H200 cold nine-block": 26.576,
    "coarse provenance fixture": 28.029,
    "fine provenance fixture": 34.638,
}


@dataclass(frozen=True)
class Carrier:
    """Fixed plasma-section and target geometry for one default machine."""

    node: np.ndarray
    polygons: tuple[np.ndarray, ...]
    wall: np.ndarray
    sample: np.ndarray
    expansion_centres: np.ndarray


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        packed = np.ascontiguousarray(values)
        digest.update(packed.dtype.str.encode("ascii"))
        digest.update(np.asarray(packed.shape, dtype="<i8").tobytes())
        digest.update(packed.tobytes())
    return digest.hexdigest()


def _clean_vertices(vertices: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(np.abs(vertices))), float(np.ptp(vertices)), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale
    kept = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - kept[-1]) > tolerance:
            kept.append(vertex)
    if len(kept) > 1 and np.linalg.norm(kept[-1] - kept[0]) <= tolerance:
        kept.pop()
    return np.asarray(kept, dtype=np.float64)


def _carrier() -> Carrier:
    wall = limiter_contour(analytic_case(), points=121)
    coilset = CoilSet(tplasma="hex")
    coilset.firstwall.insert(wall, turn="hex")
    plasma = np.asarray(coilset.subframe.loc[:, "plasma"], dtype=bool)
    material = np.asarray(coilset.subframe.loc[:, "poly"], dtype=object)[plasma]
    centres = np.c_[
        np.asarray(coilset.subframe.loc[plasma, "x"], dtype=np.float64),
        np.asarray(coilset.subframe.loc[plasma, "z"], dtype=np.float64),
    ]
    polygons = tuple(
        _clean_vertices(np.asarray(item.poly.exterior.coords)[:-1, :2])
        for item in material
    )
    areas = np.asarray([item.poly.area for item in material], dtype=np.float64)
    triangulation = Delaunay(centres)
    boundary = LineString(wall)
    boundary_cells = np.asarray(
        [
            position
            for position, item in enumerate(material)
            if item.poly.intersects(boundary)
        ]
    )
    stencil, _ = PlasmaGrid.loop_neighbour_vertices(
        centres, triangulation.vertex_neighbor_vertices, boundary_cells
    )
    mesh = StencilMesh(centres, stencil, areas)
    sections = np.asarray(coilset.aloc["plasma", "section"], dtype=object).astype(str)
    complete = np.flatnonzero(sections == "hexagon")
    dimensions = np.c_[
        np.asarray(coilset.aloc["plasma", "dl"], dtype=float)[complete],
        np.asarray(coilset.aloc["plasma", "dt"], dtype=float)[complete],
    ]
    width, height = dimensions[0]
    radius = min(width / 2.0, height / np.sqrt(3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    offsets = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    sampling = centres[:, None, :] + offsets[None, :, :]
    moments = MomentGeometry.from_cells(mesh, polygons, sampling_vertices=sampling)
    return Carrier(
        node=centres,
        polygons=polygons,
        wall=wall,
        sample=np.asarray(moments.sample_node_coordinates, dtype=np.float64),
        expansion_centres=np.asarray(moments.atomic_mesh.centroids, dtype=np.float64),
    )


def _geometry(carrier: Carrier):
    edge, weight, norm = pad_batch(list(carrier.polygons))
    section_centre = np.column_stack(
        [_section_centroid(item) for item in carrier.polygons]
    )
    expansion_centre = carrier.expansion_centres.T
    reflection_axis = np.full(len(carrier.polygons), np.nan, dtype=np.float64)
    reflection_partner = np.repeat(
        np.arange(edge.shape[0], dtype=np.int32)[:, None],
        len(carrier.polygons),
        axis=1,
    )
    for column, vertices in enumerate(carrier.polygons):
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


def _evaluate_family(evaluator, executable, targets, geometry):
    output = np.empty(
        (len(MOMENT_COMPONENTS), len(targets), len(geometry[2])),
        dtype=np.float64,
    )
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
            output[:, row_start:row_stop, columns] = np.asarray(values)
    return output, kernel_seconds


def _identity(carrier: Carrier, provenance: str) -> dict[str, object]:
    polygon_coordinates = np.concatenate(carrier.polygons)
    polygon_lengths = np.asarray([len(item) for item in carrier.polygons], dtype="<i8")
    return {
        "schema": SCHEMA,
        "analytic_reference": "moderate-rotation-conventional",
        "dplasma": -2100,
        "achieved_cells": len(carrier.node),
        "wall_nodes": len(carrier.wall),
        "sample_nodes": len(carrier.sample),
        "geometry_digest": _digest(
            carrier.node,
            carrier.wall,
            carrier.sample,
            polygon_lengths,
            polygon_coordinates,
        ),
        "kernel": "closed-form moments",
        "kernel_merge": KERNEL_MERGE,
        "precision": "float64",
        "routes": "jax-packed-analytic-moments",
        "components": list(MOMENT_COMPONENTS),
        "tile": [TILE, TILE],
        "compute_provenance": provenance,
    }


def _publish(group, matrices: dict[str, np.ndarray], identity):
    store = ZarrStore(filename=STORE_FILENAME, dirname=".nova")
    root = zarr.open_group(str(store.filepath), mode="a")
    if group in root:
        raise FileExistsError(
            f"cold publication group {group!r} already exists in {store.filepath}"
        )
    published = root.create_group(group)
    published.attrs.update(identity | {"publication_complete": False})
    for family, values in matrices.items():
        target_group = published.create_group(family)
        for index, component in enumerate(MOMENT_COMPONENTS):
            shape = values[index].shape
            target_group.create_array(
                component,
                data=values[index],
                chunks=(min(512, shape[0]), min(512, shape[1])),
            )
    published.attrs["publication_complete"] = True
    return str(store.filepath)


def _plot(receipt: dict[str, object]) -> None:
    timings = COMPARATORS | {
        "bumped default\nlive rebuild": receipt["timing"]["matrix_build_seconds"]
    }
    labels = list(timings)
    values = list(timings.values())
    colours = ["#8c8c8c"] * len(COMPARATORS) + ["#006d77"]
    figure, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    bars = axis.bar(labels, values, color=colours)
    axis.axhline(120.0, color="#b22222", linestyle="--", linewidth=1.5)
    axis.text(-0.45, 121.5, "120 s bar", color="#8b1a1a", fontsize=9)
    axis.set_ylabel("cold matrix-build wall time (s)")
    axis.set_title(
        f"Exact H200 interaction rebuild at {receipt['mesh']['achieved_cells']} cells"
    )
    axis.set_ylim(0.0, 132.0)
    axis.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2.0,
            f"{value:.3f} s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def run() -> None:
    configure_dtypes()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected gpu backend, got {jax.default_backend()!r}")
    carrier_started = perf_counter()
    carrier = _carrier()
    carrier_seconds = perf_counter() - carrier_started
    if len(carrier.node) < 2141:
        raise AssertionError(f"default carrier has only {len(carrier.node)} cells")
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
    matrices = {}
    kernel_seconds = 0.0
    targets = {
        "plasma_to_grid": carrier.node,
        "plasma_to_wall": carrier.wall,
        "plasma_to_sample": carrier.sample,
    }
    for family, coordinates in targets.items():
        matrices[family], elapsed = _evaluate_family(
            evaluator, executable, coordinates, geometry
        )
        kernel_seconds += elapsed
    matrix_seconds = perf_counter() - cold_started
    provenance = compute_provenance("jax")
    identity = _identity(carrier, provenance)
    identity_store = ZarrStore(filename=STORE_FILENAME, dirname=".nova")
    group = identity_store.hash_attrs(identity)
    publish_started = perf_counter()
    store_path = _publish(group, matrices, identity)
    publish_seconds = perf_counter() - publish_started

    old_frame = CoilSet(dplasma=-500)
    current_frame = CoilSet()
    old_group = identity_store.hash_attrs(old_frame.coilset_attrs)
    current_group = identity_store.hash_attrs(current_frame.coilset_attrs)
    pair_count = len(carrier.node) * sum(len(item) for item in targets.values())
    receipt = {
        "verdict": {
            "reference_native_interior_met": len(carrier.node) >= 2141,
            "under_120_seconds": matrix_seconds < 120.0,
            "exact_kernel_everywhere": True,
            "all_moment_companions_built": all(
                values.shape[0] == len(MOMENT_COMPONENTS)
                for values in matrices.values()
            ),
            "cold_semantic_group_published": True,
        },
        "mesh": {
            "requested_cells": -2100,
            "achieved_cells": len(carrier.node),
            "reference_native_interior_equivalent": 2141,
            "wall_nodes": len(carrier.wall),
            "sample_nodes": len(carrier.sample),
        },
        "operator": {
            "target_families": {key: len(value) for key, value in targets.items()},
            "base_block_count": 9,
            "moment_companion_block_count": 18,
            "stored_matrix_count": 27,
            "pair_count": pair_count,
            "components": list(MOMENT_COMPONENTS),
            "kernel": "jax packed closed-form analytic moments",
            "precision": "float64",
            "kernel_path_merge": KERNEL_MERGE,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "device_identity": jax.devices()[0].device_kind,
            "platform": jax.default_backend(),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "compute_provenance": provenance,
            "compilation_cache_setting": os.environ.get(
                "NOVA_COMPILATION_CACHE", "default"
            ),
        },
        "timing": {
            "carrier_geometry_seconds": carrier_seconds,
            "compile_seconds": compile_seconds,
            "kernel_seconds": kernel_seconds,
            "matrix_build_seconds": matrix_seconds,
            "publication_seconds": publish_seconds,
            "comparators_seconds": COMPARATORS,
            "bar_seconds": 120.0,
        },
        "cache": {
            "store": store_path,
            "compute_provenance_group": group,
            "semantic_identity": identity,
            "old_default_frame_group": old_group,
            "current_default_frame_group": current_group,
            "implicit_default_identity_changed": old_group != current_group,
            "invalidation_scope": [
                "implicit CoilSet frame identity",
                "plasma cell carrier",
                "plasma-to-grid interactions and moment companions",
                "plasma-to-wall interactions and moment companions",
                "plasma-to-sample interactions and moment companions",
            ],
            "unaffected_scope": [
                "explicit dplasma instances",
                "semantic identities for other discretisations",
                "CPU and other compute-provenance sibling groups",
            ],
        },
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
