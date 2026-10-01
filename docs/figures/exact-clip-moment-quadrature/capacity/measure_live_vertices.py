"""Measure the live vertex count the two-arc capacity fixture presents.

Runs the head fixture from tests/test_equilibrium_separatrix_clip.py on the
login node (CPU) and records, for every call into the fixed-shape packer
``_pack_traced_vertices``, the packed array's full shape and the number of live
entries on its vertex axis.  A second arm raises the derived capacity for one
diagnostic run so the control's admitted count can be read beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp

from nova.equilibrium import separatrix_clip as sc
from nova.equilibrium.separatrix_clip import (
    AtomicCellMesh,
    traced_polygon_vertex_capacity,
)
from nova.jax.config import configure_dtypes

configure_dtypes()
print("jax_enable_x64 =", jax.config.jax_enable_x64, file=sys.stderr)

records: list[dict] = []
_original_pack = sc._pack_traced_vertices


def recording_pack(vertices, valid, capacity):
    packed, count = _original_pack(vertices, valid, capacity)
    live = np.asarray(jax.device_get(jnp.sum(valid, axis=1)))
    records.append(
        {
            "source": "separatrix_clip._pack_traced_vertices",
            "vertices_name": "_pack_traced_vertices.vertices",
            "vertices_shape": list(vertices.shape),
            "valid_shape": list(valid.shape),
            "capacity": int(capacity),
            "live_vertex_count": [int(v) for v in np.atleast_1d(live)],
        }
    )
    return packed, count


def build_fixture():
    cell = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mesh = AtomicCellMesh.from_cells([cell], centroids=np.asarray([[0.5, 0.5]]))
    return mesh


def saddle_level(points):
    radial, vertical = points[..., 0], points[..., 1]
    return (radial - 0.5) * (vertical - 0.5)


def run_head():
    sc._pack_traced_vertices = recording_pack
    try:
        mesh = build_fixture()
        capacity = traced_polygon_vertex_capacity(int(mesh.support_capacity))
        signed = saddle_level(jnp.asarray(mesh.node_coordinates))
        support = mesh.traced_clip(signed, curve_evaluator=saddle_level)
        return {
            "support_capacity_mesh": int(mesh.support_capacity),
            "derived_capacity": int(capacity),
            "refused_cells": int(support.refused_cells()),
            "vertex_capacity": int(np.asarray(support.vertex_capacity)),
            "admitted_vertex_count": [
                int(v) for v in np.atleast_1d(np.asarray(support.vertex_count))
            ],
            "area": [float(a) for a in np.atleast_1d(np.asarray(support.area))],
        }
    finally:
        sc._pack_traced_vertices = _original_pack


def run_raised():
    """Raise the derived capacity for one diagnostic run; read the admitted count."""
    mesh = build_fixture()
    mesh_capacity = int(mesh.support_capacity)
    raised = 4096

    original_derived = sc.traced_polygon_vertex_capacity
    sc.traced_polygon_vertex_capacity = lambda straight: raised
    sc._pack_traced_vertices = recording_pack
    try:
        signed = saddle_level(jnp.asarray(mesh.node_coordinates))
        support = mesh.traced_clip(signed, curve_evaluator=saddle_level)
        return {
            "support_capacity_mesh": mesh_capacity,
            "raised_capacity": raised,
            "refused_cells": int(support.refused_cells()),
            "vertex_capacity": int(np.asarray(support.vertex_capacity)),
            "admitted_vertex_count": [
                int(v) for v in np.atleast_1d(np.asarray(support.vertex_count))
            ],
            "area": [float(a) for a in np.atleast_1d(np.asarray(support.area))],
        }
    finally:
        sc.traced_polygon_vertex_capacity = original_derived
        sc._pack_traced_vertices = _original_pack


head = run_head()
head_records = list(records)
records.clear()
raised = run_raised()
raised_records = list(records)

report = {
    "head": head,
    "head_pack_calls": head_records,
    "raised": raised,
    "raised_pack_calls": raised_records,
}

out = Path(__file__).resolve().parent / "live-vertex-fixture.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2))
