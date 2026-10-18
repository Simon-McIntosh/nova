"""Negative control: perturbing the input the decision is about moves the answer.

The read's input is the field and the cell geometry. A control that perturbs
only the wall cannot falsify the table, because the record's field is its own
spline and does not depend on the wall at all; the wall reaches the decision
only by changing which cells the lattice contains, and dropping one cell in 132
does not move an admitted saddle whose position is set by the field. That was
measured: 2 and 25 per cent wall-vertex moves left the label, the boundary level
and the admitted saddle byte-identical.

So the control perturbs the field. The perturbation is a smooth shape change
over the record's own grid, psi -> psi * (1 + amplitude * shape), where shape is
a normalised sinusoid in r and z that vanishes nowhere. A shape change is chosen
over a constant offset or a global scale because the read is invariant to both:
an offset moves the axis and the boundary level by the same amount and a scale
cancels, so neither can move a span. The mode is selected on the command line so
the wall result stays reproducible.

Usage: negative_control.py field <row index> <amplitude>
       negative_control.py wall  <row index> <vertex> <fraction>
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

SPAN_TOLERANCE = 1e-3
AXIS_TOLERANCE = 1e-3


def perturb_field(record, amplitude):
    """Multiply the grid psi by a smooth, nowhere-vanishing shape."""
    rgrid = np.asarray(record.rgrid, dtype=np.float64)
    zgrid = np.asarray(record.zgrid, dtype=np.float64)
    r = (rgrid - rgrid.mean()) / max(np.ptp(rgrid), 1e-30)
    z = (zgrid - zgrid.mean()) / max(np.ptp(zgrid), 1e-30)
    shape = 0.5 * (np.sin(2.0 * np.pi * r)[:, None] * np.cos(2.0 * np.pi * z)[None, :])
    psi = np.asarray(record.psi, dtype=np.float64)
    return replace(record, psi=psi * (1.0 + amplitude * shape))


def perturb_wall(record, vertex, fraction):
    """Pull one wall vertex toward the record's axis by ``fraction``."""
    ring = record.wall_ring
    axis = np.array([record.axis_r, record.axis_z])
    moved = ring.copy()
    moved[vertex] = moved[vertex] + fraction * (axis - moved[vertex])
    starts = moved[:-1]
    ends = np.vstack((moved[1:-1], moved[-1]))
    return replace(
        record,
        wall_r=np.column_stack((starts[:, 0], ends[:, 0])),
        wall_z=np.column_stack((starts[:, 1], ends[:, 1])),
    )


def report(base, shifted):
    """Print both decisions and their deltas; return whether the answer moved."""
    print(
        "base      label=%-9s boundary_psi=%.6f axis=(%.5f,%.5f) span=%.6f cells=%d"
        % (
            base.nova_label,
            base.nova_boundary_psi,
            base.nova_axis_r,
            base.nova_axis_z,
            base.nova_span,
            base.n_cells,
        )
    )
    print(
        "perturbed label=%-9s boundary_psi=%.6f axis=(%.5f,%.5f) span=%.6f cells=%d"
        % (
            shifted.nova_label,
            shifted.nova_boundary_psi,
            shifted.nova_axis_r,
            shifted.nova_axis_z,
            shifted.nova_span,
            shifted.n_cells,
        )
    )
    label_moved = base.nova_label != shifted.nova_label
    span_delta = abs(base.nova_span - shifted.nova_span)
    span_moved = span_delta > SPAN_TOLERANCE * max(abs(base.nova_span), 1e-30)
    axis_moved = (
        np.hypot(
            base.nova_axis_r - shifted.nova_axis_r,
            base.nova_axis_z - shifted.nova_axis_z,
        )
        > AXIS_TOLERANCE
    )
    print(
        "deltas    label=%s span=%.3e axis=%.3e"
        % (
            label_moved,
            span_delta,
            np.hypot(
                base.nova_axis_r - shifted.nova_axis_r,
                base.nova_axis_z - shifted.nova_axis_z,
            ),
        )
    )
    moved = label_moved or span_moved or axis_moved
    print("CONTROL", "CHANGED" if moved else "UNCHANGED")
    return moved


def main_field(row_index, amplitude, resolution=16):
    pairs = load_pairs()
    record = read_trace_record(resolve_trace(pairs[row_index][1]))
    base = _decision("control", "base", record, resolution)
    shifted = _decision(
        "control", "perturbed", perturb_field(record, amplitude), resolution
    )
    return 0 if report(base, shifted) else 1


def main_wall(row_index, vertex, fraction, resolution=16):
    pairs = load_pairs()
    record = read_trace_record(resolve_trace(pairs[row_index][1]))
    base = _decision("control", "base", record, resolution)
    shifted = _decision(
        "control", "perturbed", perturb_wall(record, vertex, fraction), resolution
    )
    return 0 if report(base, shifted) else 1


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "wall":
        sys.exit(main_wall(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])))
    sys.exit(main_field(int(sys.argv[2]), float(sys.argv[3])))
