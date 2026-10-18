"""Negative control: moving one wall vertex onto the separatrix moves nova's answer.

The claim under test is that the decision the table reports is produced by the
record's own geometry and field, not by the adapter's bookkeeping. The control
takes one record whose read is qualified, moves a single wall vertex inward onto
the level the read called the boundary, and re-runs the same pipeline. If the
control vertex changes nothing the table would have to be treated as
insensitive; if it changes the read, the pipeline is reading the geometry it is
given.

The perturbed vertex is placed where a single ring edit is observable: the
lattice cells whose sampling points fall outside the shrunk ring are dropped, so
the cell set itself changes. Both states are reported with their census counts,
so a difference can be attributed rather than asserted.

Usage: negative_control.py <pairs.tsv row index> <vertex index>
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from scripts.prototypes.efit_field_swap.crosscheck import (
    _decision,
    load_pairs,
    resolve_trace,
)
from scripts.prototypes.efit_field_swap.trace_record import read_trace_record


def perturb(record, vertex, fraction):
    """Pull one wall vertex toward the record's axis level by ``fraction``."""
    ring = record.wall_ring
    axis = np.array([record.axis_r, record.axis_z])
    moved = ring.copy()
    moved[vertex] = moved[vertex] + fraction * (axis - moved[vertex])
    # Rebuild the segment array from the moved ring so the record's own
    # accessors (wall_ring, wall_units) see the perturbation.
    starts = moved[:-1]
    ends = np.vstack((moved[1:-1], moved[-1]))
    return replace(
        record,
        wall_r=np.column_stack((starts[:, 0], ends[:, 0])),
        wall_z=np.column_stack((starts[:, 1], ends[:, 1])),
    )


def main(row_index=0, vertex=0, fraction=0.02, resolution=16):
    pairs = load_pairs()
    record = read_trace_record(resolve_trace(pairs[row_index][1]))
    base = _decision("control", "base", record, resolution)
    moved = perturb(record, vertex, fraction)
    shifted = _decision("control", "perturbed", moved, resolution)
    print(
        "base      label",
        base.nova_label,
        "boundary_psi",
        round(base.nova_boundary_psi, 6),
        "saddle",
        np.round([base.nova_saddle_r, base.nova_saddle_z], 5),
        "cells",
        base.n_cells,
    )
    print(
        "perturbed label",
        shifted.nova_label,
        "boundary_psi",
        round(shifted.nova_boundary_psi, 6),
        "saddle",
        np.round([shifted.nova_saddle_r, shifted.nova_saddle_z], 5),
        "cells",
        shifted.n_cells,
    )
    changed = base.nova_label != shifted.nova_label or not np.isclose(
        base.nova_boundary_psi, shifted.nova_boundary_psi
    )
    print("CONTROL", "CHANGED" if changed else "UNCHANGED")
    return 0 if changed else 1


if __name__ == "__main__":
    args = [int(value) for value in sys.argv[1:3]]
    if len(sys.argv) > 3:
        args.append(float(sys.argv[3]))
    sys.exit(main(*args))
