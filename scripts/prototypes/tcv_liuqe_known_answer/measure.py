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
from dataclasses import dataclass
from pathlib import Path

import imas
import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import topology
from nova.equilibrium.topology import TopologyPolicy
from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap import crosscheck
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.fast_map_production.kernel import read as nova_read
from scripts.prototypes.read_robustness.census import census

configure_dtypes()

RESOLUTIONS = (24, 32, 48)


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
    rings = []
    for description in wall.description_2d:
        for unit in description.limiter.unit:
            r = np.asarray(unit.outline.r, dtype=np.float64)
            z = np.asarray(unit.outline.z, dtype=np.float64)
            if r.size >= 3:
                rings.append(np.column_stack((r, z)))
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
    sigma = convention.sigma
    boundary_ring, obstacles = crosscheck.domain_rings(state)
    attempts = []
    for resolution in resolutions:
        cells, samples = crosscheck.lattice(boundary_ring, resolution, obstacles)
        geometry, fitted = crosscheck.fitted_geometry(
            cells, samples, (boundary_ring, *obstacles)
        )
        if geometry is None:
            attempts.append(f"lattice {resolution}: contour mesh admits no vertex")
            continue
        initial = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
        axis_valid = np.asarray(initial.x_point_valid)
        points = np.asarray(initial.x_points)
        stack = [np.asarray(initial.axis)[None]]
        if axis_valid.any():
            stack.append(points[axis_valid])
        previous = jnp.asarray(np.vstack(stack))
        roots = census(field, geometry, convention, policy, previous)
        attempts.append(f"lattice {resolution}: candidates {int(roots.candidates)}")
        if int(roots.candidates) == 0 or int(roots.saddles) == 0:
            attempts[-1] += f", admitted saddles {int(roots.saddles)}"
            continue
        reading = nova_read(field, roots.roots, geometry, convention, policy)
        boundary = np.asarray(reading.boundary, dtype=np.float64)
        axis = np.asarray(reading.axis, dtype=np.float64)
        label = int(np.asarray(reading.boundary_class))
        return dict(
            resolution=resolution,
            fitted=fitted,
            nova_axis=[float(axis[0]), float(axis[1])],
            nova_axis_psi=sigma * _value(field, axis),
            nova_boundary=[float(boundary[0]), float(boundary[1])],
            nova_boundary_psi=float(np.asarray(reading.boundary_flux)),
            nova_label="diverted" if label else "limited",
            nova_span=abs(
                float(np.asarray(reading.boundary_flux)) - sigma * _value(field, axis)
            ),
            attempts="; ".join(attempts),
        )
    return dict(
        resolution=None,
        fitted=False,
        nova_axis=[float("nan")] * 2,
        nova_axis_psi=float("nan"),
        nova_boundary=[float("nan")] * 2,
        nova_boundary_psi=float("nan"),
        nova_label="unqualified",
        nova_span=float("nan"),
        attempts="; ".join(attempts) + "; no lattice admitted a saddle",
    )


def liuqe_reference(view, field):
    """LIUQE's own levels, both as stored and as read from its own grid."""
    ref = dict(
        liuqe_time=view["time"],
        liuqe_axis=list(view["axis"]),
        liuqe_axis_psi_stored=view["psi_axis"],
        liuqe_boundary_psi_stored=view["psi_boundary"],
        liuqe_xpoints=[list(p) for p in view["xpoints"]],
        liuqe_label="diverted" if view["xpoints"] else "limited",
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
    parser.add_argument("--resolutions", default="24,32,48")
    args = parser.parse_args()
    print("nova.__file__", __import__("nova").__file__, flush=True)

    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    resolutions = tuple(int(r) for r in args.resolutions.split(","))
    wall_rings = read_wall(args.wall)
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    rows = []
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
        except Exception as error:  # noqa: BLE001 - one slice must not stop the sweep
            row.update(
                resolution=None,
                fitted=False,
                nova_axis=[float("nan")] * 2,
                nova_axis_psi=float("nan"),
                nova_boundary=[float("nan")] * 2,
                nova_boundary_psi=float("nan"),
                nova_label="unread",
                nova_span=float("nan"),
                attempts=f"{type(error).__name__}: {error}",
            )
        rows.append(row)
        print("done", index, row["liuqe_label"], row["nova_label"], flush=True)
    db.close()
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=2))
    print("ROWS", len(rows))


if __name__ == "__main__":
    main()
