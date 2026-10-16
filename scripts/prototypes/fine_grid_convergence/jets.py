"""Persist source jets while constructing the fixed-shape moment carrier."""

import numpy as np

from nova.equilibrium import fixed_points
from scripts.prototypes.lean_map.jets import JetMap
from scripts.prototypes.read_route.jets import build_jet_matrix


def build_jets(coupling, geometry, total_field, reference_moments):
    """Replace lean_map.jets.build_jets with a persistent source matrix."""
    columns = tuple(reference_moments[:, index] for index in range(3))
    rows, exterior, nodes, sample_index, seconds = build_jet_matrix(
        coupling, geometry, total_field, columns, batch=4, persist=True
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
