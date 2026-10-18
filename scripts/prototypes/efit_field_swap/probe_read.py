"""Print nova's census counts and decision for one record."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scripts.prototypes.efit_field_swap.crosscheck import (
    lattice,
    load_pairs,
    read_trace_record,
    wall_units,
)
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.efit_field_swap.orientation import read_convention
from scripts.prototypes.fast_map_production.kernel import read as nova_read
from scripts.prototypes.read_robustness.census import census

from nova.equilibrium import topology
from nova.equilibrium.topology import TopologyPolicy

rows = load_pairs()
for pair_index in (0, 1):
    record = read_trace_record(rows[pair_index][1])
    field = TraceSplineField.from_record(record)
    cells, offsets = lattice(record.wall_ring, 16)
    geometry = topology.TopologyGeometry.from_cells(cells, offsets, wall_units(record))
    field = field.sampled_for(geometry)
    convention = read_convention(polarity=record.polarity)
    policy = TopologyPolicy()
    initial = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    valid = np.asarray(initial.x_point_valid)
    stack = [np.asarray(initial.axis)[None]]
    if valid.any():
        stack.append(np.asarray(initial.x_points)[valid])
    previous = jnp.asarray(np.vstack(stack))
    result = census(field, geometry, convention, policy, previous)
    print(
        rows[pair_index][0],
        "candidates",
        int(result.candidates),
        "axes",
        int(result.axes),
        "saddles",
        int(result.saddles),
        "initial_axis",
        np.round(np.asarray(initial.axis), 3),
    )
    if int(result.candidates) > 0:
        reading = nova_read(field, result.roots, geometry, convention, policy)
        print(
            "  nova axis",
            np.round(np.asarray(reading.axis), 3),
            "boundary_flux",
            round(float(np.asarray(reading.boundary_flux)), 5),
            "class",
            int(np.asarray(reading.boundary_class)),
            "| efit axis",
            np.round([record.axis_r, record.axis_z], 3),
            "fed_back",
            round(record.fed_back_psi, 5),
            record.verdict,
        )
