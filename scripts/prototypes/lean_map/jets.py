"""Store all source jets and gather bounded rows around active support."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import fixed_points
from scripts.prototypes.read_route.jets import build_jet_matrix


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JetMap:
    rows: jax.Array
    exterior: jax.Array
    nodes: jax.Array
    sample_index: jax.Array
    mesh: fixed_points.BellMesh

    def gather(self, moments):
        return self.exterior + jnp.einsum("ncmq,cm->nq", self.rows, moments)

    def field(self, moments):
        return fixed_points.field_from_jets(
            self.gather(moments), self.sample_index, self.mesh
        )


def build_jets(coupling, geometry, total_field, reference_moments):
    """Store the six value and derivative responses at every unique node."""
    columns = tuple(reference_moments[:, index] for index in range(3))
    rows, exterior, nodes, sample_index, seconds = build_jet_matrix(
        coupling, geometry, total_field, columns, batch=4, persist=False
    )
    mesh = fixed_points.build_mesh(
        np.asarray(nodes), np.asarray(sample_index), geometry
    )
    jet_map = JetMap(rows, exterior, nodes, sample_index, mesh)
    return jet_map, {
        "nodes": len(nodes),
        "source_cells": rows.shape[1],
        "matrix_bytes": rows.size * rows.dtype.itemsize,
        "assembly_seconds": seconds,
    }


def gather_window(jet_map, geometry, membership, roots, *, margin_pitches=4.0):
    """Freeze a machine-specific boundary and root window with spare row slots."""
    centre = np.asarray(geometry.centre)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    points = np.asarray(roots)
    distance = np.min(np.linalg.norm(centre[:, None] - points[None], axis=-1), axis=1)
    fraction = np.asarray(membership)
    boundary_band = (fraction > 0) & (
        (fraction < 1 - 1e-14) | (np.asarray(geometry.vertex_count) != 6)
    )
    live_edges = (
        np.arange(geometry.neighbour.shape[1])[None]
        < np.asarray(geometry.vertex_count)[:, None]
    )
    wall_band = np.any((np.asarray(geometry.neighbour) < 0) & live_edges, axis=1)
    window_cells = boundary_band | wall_band | (distance <= margin_pitches * pitch)
    selected = np.unique(np.asarray(jet_map.sample_index)[window_cells])
    needed = len(selected)
    capacity = int(np.ceil(1.25 * needed)) + 16
    lookup = np.full(len(jet_map.nodes), -1, dtype=np.int32)
    lookup[selected] = np.arange(needed)
    rows = np.zeros((capacity,) + jet_map.rows.shape[1:], dtype=np.float64)
    exterior = np.zeros((capacity, 6), dtype=np.float64)
    rows[:needed] = np.asarray(jet_map.rows[selected])
    exterior[:needed] = np.asarray(jet_map.exterior[selected])
    cache = fixed_points.RowCache(
        jet_map.mesh,
        jet_map.nodes,
        jnp.asarray(lookup),
        jnp.asarray(rows),
        jnp.asarray(exterior),
    )
    return cache, {
        "boundary_cells": int(np.sum(boundary_band)),
        "wall_cells": int(np.sum(wall_band)),
        "root_margin_pitches": margin_pitches,
        "required_rows": needed,
        "capacity": capacity,
        "margin_rows": capacity - needed,
        "gather_bytes": rows.nbytes,
    }
