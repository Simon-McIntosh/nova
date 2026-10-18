"""Score nova's boundary read on TCV 77549 LIUQE slices against LIUQE itself.

The input is a LIUQE reconstruction stored as an IMAS ``equilibrium`` IDS: a
rectilinear psi grid per time slice and a boundary polygon on a separate TCV
``wall`` IDS. The merged field-swap adapter supplies the pieces: the grid spline
field (``efit_field_*swap.spline_field``) and the per-loop wall geometry and the
nova read call (``efit_field_swap.crosscheck``). nova's read is run unchanged on
LIUQE's own psi field with the per-loop TCV wall; LIUQE's stored axis, X-point,
flux levels and label are the reference.

Two facts about the shipped LIUQE archives drive the slice choice and are
reported beside every number:

* ``boundary.type`` is unset (-999999999) on every slice, so LIUQE's own
  limited-or-diverted label is read from whether it stored a boundary X-point
  (``boundary/x_point``), which is absent through the limited phase and present
  through the diverted one.
* ``profiles_2d/psi`` is piecewise-constant in time: the 1703-slice store
  carries only 17 distinct psi grids, each frozen over one contiguous block.
  A per-slice read inside a block would score one field repeatedly, so the
  snapshot takes one representative slice per distinct grid block.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imas
import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from nova.equilibrium.wall_mask import wall_units_from_ids
from nova.equilibrium.topology import TopologyPolicy
from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap import crosscheck
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField

configure_dtypes()
assert jax.config.jax_enable_x64 is True

RESOLUTIONS = (16, 24, 32)
SLICES = (0, 14, 34, 46, 118, 206, 248, 291, 456, 542, 796, 1257)
REFUSED_CASES = {
    "jt60sa-5slice-s5-133-134:A",
    "jt60sa-5slice-s5-133-134:B",
    "jt60sa-5slice-s1-15-16:B",
    "jt60sa-5slice-s2-135-136:A",
    "jt60sa-5slice-s2-135-136:B",
}


@dataclass(frozen=True)
class LatticeState:
    """The duck-typed record the field-swap adapter consumes.

    ``spline_field.from_record`` reads ``rgrid``/``zgrid``/``psi`` and the wall
    geometry helpers read ``wall_rings`` and the stored-axis attribute, so one
    object carries a LIUQE slice in the shape the adapter already expects.
    """

    rgrid: np.ndarray
    zgrid: np.ndarray
    psi: np.ndarray
    wall_rings: tuple[np.ndarray, ...]
    axis_r: float
    axis_z: float


def build_state(rgrid, zgrid, psi, wall_rings, axis_r, axis_z):
    return LatticeState(
        rgrid=np.asarray(rgrid, dtype=np.float64),
        zgrid=np.asarray(zgrid, dtype=np.float64),
        psi=np.asarray(psi, dtype=np.float64),
        wall_rings=tuple(np.asarray(r, dtype=np.float64) for r in wall_rings),
        axis_r=float(axis_r),
        axis_z=float(axis_z),
    )


def read_wall(path):
    """Every limiter unit outline of a wall IDS as one closed RZ ring each."""
    db = imas.DBEntry(f"imas:hdf5?path={path}", "r", dd_version="4.1.0")
    wall = db.get("wall")
    rings = [unit.vertices for unit in wall_units_from_ids(wall) if unit.r.size >= 3]
    db.close()
    return tuple(rings)


def _closed(vertices):
    keep = np.ones(len(vertices), dtype=bool)
    keep[1:] = np.any(vertices[1:] != vertices[:-1], axis=1)
    ring = vertices[keep]
    if not np.array_equal(ring[-1], ring[0]):
        ring = np.vstack((ring, ring[0]))
    return ring


def slice_view(eq, index):
    """LIUQE slice ``index`` as the field, wall ring selection happens in the driver."""
    ts = eq.time_slice[index]
    p2 = ts.profiles_2d[0]
    rgrid = np.asarray(p2.grid.dim1, dtype=np.float64)
    zgrid = np.asarray(p2.grid.dim2, dtype=np.float64)
    psi = np.asarray(p2.psi, dtype=np.float64)
    gq = ts.global_quantities
    axis = (float(gq.magnetic_axis.r), float(gq.magnetic_axis.z))
    xpoints = [
        (float(p.r), float(p.z))
        for p in ts.boundary.x_point
        if abs(float(p.r)) < 1e3 and abs(float(p.z)) < 1e3
    ]
    outline_r = np.asarray(ts.boundary.outline.r, dtype=np.float64)
    outline_z = np.asarray(ts.boundary.outline.z, dtype=np.float64)
    outline = (
        np.column_stack((outline_r, outline_z))
        if outline_r.size >= 3
        else np.empty((0, 2))
    )
    return dict(
        time=float(np.asarray(eq.time)[index]),
        rgrid=rgrid,
        zgrid=zgrid,
        psi=psi,
        axis=axis,
        xpoints=xpoints,
        outline=outline,
        psi_axis=float(gq.psi_axis),
        psi_boundary=float(gq.psi_boundary),
    )


def _value(field, point):
    return float(np.asarray(field.value(jnp.asarray(point))))


def _grid_level(field, points):
    values = np.array([_value(field, p) for p in points])
    return float(np.mean(values))


def axis_boundary_span(axis, outline):
    """Shortest distance from the axis to LIUQE's piecewise-linear boundary."""
    points = np.asarray(outline, dtype=np.float64)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("LIUQE boundary outline is missing or nonfinite")
    start = points
    edge = np.roll(points, -1, axis=0) - start
    squared = np.einsum("ij,ij->i", edge, edge)
    usable = squared > 0
    if not np.any(usable):
        raise ValueError("LIUQE boundary outline has no segments")
    start, edge, squared = start[usable], edge[usable], squared[usable]
    position = np.asarray(axis, dtype=np.float64)
    fraction = np.clip(np.einsum("ij,ij->i", position - start, edge) / squared, 0, 1)
    nearest = start + fraction[:, None] * edge
    span = float(np.min(np.linalg.norm(position - nearest, axis=1)))
    if not np.isfinite(span) or span <= 0:
        raise ValueError("LIUQE axis-to-boundary span is not positive")
    return span


def measure_slice(view, wall_rings, resolutions=RESOLUTIONS):
    """nova's read on one LIUQE slice, against LIUQE's own stored values."""
    state = build_state(
        view["rgrid"],
        view["zgrid"],
        view["psi"],
        wall_rings,
        view["axis"][0],
        view["axis"][1],
    )
    field = TraceSplineField.from_record(state)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    boundary_ring, obstacles = crosscheck.domain_rings(state)
    outcome = crosscheck.read_state(
        field, boundary_ring, obstacles, convention, policy, resolutions
    )
    coverage = outcome.coverage
    row = dict(
        resolution=outcome.resolution,
        fitted=outcome.capacity_fit,
        nova_candidates=outcome.candidates,
        nova_saddles=outcome.saddles,
        nova_axis=[float(outcome.axis[0]), float(outcome.axis[1])],
        nova_axis_psi=outcome.axis_psi,
        nova_boundary=[float(outcome.boundary[0]), float(outcome.boundary[1])],
        nova_boundary_psi=outcome.boundary_flux,
        nova_label=outcome.label,
        nova_span=outcome.span,
        attempts=outcome.attempts,
    )
    for name in (
        "outer_winding",
        "obstacle_winding",
        "total",
        "index_sum",
        "min_gradient",
        "floor",
        "reliable",
        "certified",
        "cells_cleared",
        "cells_uncertified",
        "cells_occupied",
        "subdivision_depth",
    ):
        row[f"coverage_{name}"] = None if coverage is None else getattr(coverage, name)
    return row


def liuqe_reference(view, field):
    """LIUQE's own levels, both as stored and as read from its own grid."""
    ref = dict(
        liuqe_time=view["time"],
        liuqe_axis=list(view["axis"]),
        liuqe_axis_psi_stored=view["psi_axis"],
        liuqe_boundary_psi_stored=view["psi_boundary"],
        liuqe_xpoints=[list(p) for p in view["xpoints"]],
        liuqe_label="diverted" if view["xpoints"] else "limited",
        liuqe_axis_boundary_span=axis_boundary_span(view["axis"], view["outline"]),
    )
    ref["liuqe_axis_psi_grid"] = _value(field, view["axis"])
    if view["xpoints"]:
        ref["liuqe_boundary_psi_grid"] = _grid_level(field, view["xpoints"])
    elif len(view["outline"]):
        ref["liuqe_boundary_psi_grid"] = _grid_level(field, view["outline"])
    else:
        ref["liuqe_boundary_psi_grid"] = float("nan")
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liuqe", required=True)
    parser.add_argument("--wall", required=True)
    parser.add_argument("--slices", required=True, help="comma-separated indices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--field-swap-out")
    parser.add_argument("--resolutions", default="16,24,32")
    args = parser.parse_args()
    print("nova.__file__", __import__("nova").__file__, flush=True)

    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    resolutions = tuple(int(r) for r in args.resolutions.split(","))
    wall_rings = read_wall(args.wall)
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    if int(eq.ids_properties.homogeneous_time) != 1:
        raise ValueError("LIUQE equilibrium must use its IDS-level time base")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for index in slices:
            view = slice_view(eq, index)
            state = build_state(
                view["rgrid"],
                view["zgrid"],
                view["psi"],
                wall_rings,
                view["axis"][0],
                view["axis"][1],
            )
            field = TraceSplineField.from_record(state)
            row = dict(index=index)
            row.update(liuqe_reference(view, field))
            try:
                row.update(measure_slice(view, wall_rings, resolutions))
            except Exception as error:  # noqa: BLE001 - preserve the failed slice
                row.update(
                    resolution=None,
                    nova_label="unread",
                    attempts=f"{type(error).__name__}: {error}",
                )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print("done", index, row["liuqe_label"], row["nova_label"], flush=True)
    db.close()
    print("ROWS", len(slices), flush=True)
    if args.field_swap_out:
        field_out = Path(args.field_swap_out)
        field_out.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        with field_out.open("w") as handle:
            for case, source, path in crosscheck.cases():
                if case not in REFUSED_CASES:
                    continue
                seen.add(case)
                row = asdict(crosscheck._row(case, source, path, resolutions))
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print("field-swap", case, row["nova_label"], flush=True)
        if seen != REFUSED_CASES:
            missing = sorted(REFUSED_CASES - seen)
            raise ValueError(f"missing field-swap cases: {missing}")
        print("FIELD_SWAP_ROWS", len(seen), flush=True)


if __name__ == "__main__":
    main()
