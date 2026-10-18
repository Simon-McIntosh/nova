"""Print the sample-point stencil against each cell centre."""

from __future__ import annotations

import numpy as np
from scripts.prototypes.efit_field_swap.crosscheck import (
    lattice,
    load_pairs,
    read_trace_record,
    wall_units,
)

from nova.equilibrium import topology

rows = load_pairs()
record = read_trace_record(rows[0][1])
cells, offsets = lattice(record.wall_ring, 16)
geometry = topology.TopologyGeometry.from_cells(cells, offsets, wall_units(record))
centre = np.asarray(geometry.centre)
samples = np.asarray(geometry.sample_points)
delta = samples - centre[:, None, :]
reach = np.abs(delta).max(axis=(1, 2))
print("cells", len(cells), "max reach", reach.max(), "median reach", np.median(reach))
worst = int(np.argmax(reach))
print("worst cell", worst, "centre", centre[worst], "reach", reach[worst])
print("worst delta rows", np.round(delta[worst], 4))
print("authored cell", np.round(cells[worst], 4))
print("named offsets row", np.round(offsets[worst], 4))
print("cell 5 centre", centre[5], "reach", reach[5])