"""Cross-check nova's boundary decision against imas-efit's on each other's fields.

Each imas-efit boundary trace record carries a rectilinear psi grid and the wall
polygon it was traced against. The adapter rebuilds the record's bicubic spline
and nova's read is run on it unchanged, on a square-lattice cell geometry built
inside the record's own wall polygon. The record also carries imas-efit's own
decision for the same state, so the two codes can be compared case by case
rather than converged.

Approximations this driver states plainly, because they qualify every number it
emits:

* The field is the record's bicubic spline. nova's Biot-Savart operator is not
  used and no plasma current is booked; a comparison therefore measures the two
  codes' *decisions* on one field representation, not two physics models.
* The cell geometry is a square lattice of the record's own resolution inside
  the record's wall polygon, not nova's hex or Delaunay carrier mesh, so the
  Bell fit and the cell-boundary census differ in discretisation from a
  production nova run.
* nova's roots come from its own census on that lattice; imas-efit's saddle is
  read from the record's stored status line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box

from nova.equilibrium import topology
from nova.jax.config import configure_dtypes
from nova.equilibrium.topology import TopologyPolicy
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.efit_field_swap.trace_record import (
    TraceRecord,
    read_trace_record,
)
from scripts.prototypes.fast_map_production.kernel import read as nova_read
from scripts.prototypes.read_robustness.census import census

configure_dtypes()

PAIRS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission/pairs.tsv"
)
CROSSING = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/imas-efit/"
    "continuous-boundary-level/cbl-saddle-admission/crossing"
)


@dataclass
class Decision:
    """One case's nova-versus-imas-efit decision columns."""

    case: str
    record: str
    source: str
    n_cells: int
    efit_verdict: int
    efit_label: str
    efit_axis_psi: float
    efit_axis_r: float
    efit_axis_z: float
    efit_saddle_refined: float
    efit_wall_level: float
    efit_fed_back_psi: float
    nova_axis_psi: float
    nova_axis_r: float
    nova_axis_z: float
    nova_boundary_psi: float
    nova_saddle_psi: float
    nova_saddle_r: float
    nova_saddle_z: float
    nova_wall_level: float
    nova_label: str
    nova_span: float
    efit_span: float
    note: str = ""


def _unqualified(case, source, record, resolution, roots, initial):
    """A case the lattice cannot qualify, reported as such rather than as a decision.

    nova's census proposed no cellwise seed, or proposed seeds and admitted no
    saddle. Reporting the read's own fallback here would put a number in the
    nova column that is the axis level wearing the boundary's name, which is
    exactly the misreading this table exists to prevent.
    """
    return Decision(
        case=case,
        record=record.slice,
        source=source,
        n_cells=0,
        efit_verdict=int(record.verdict),
        efit_label="diverted" if record.diverted else "limited",
        efit_axis_psi=record.axis_psi,
        efit_axis_r=record.axis_r,
        efit_axis_z=record.axis_z,
        efit_saddle_refined=record.saddle_refined,
        efit_wall_level=record.wall_level,
        efit_fed_back_psi=record.fed_back_psi,
        nova_axis_psi=float("nan"),
        nova_axis_r=float("nan"),
        nova_axis_z=float("nan"),
        nova_boundary_psi=float("nan"),
        nova_saddle_psi=float("nan"),
        nova_saddle_r=float("nan"),
        nova_saddle_z=float("nan"),
        nova_wall_level=float("nan"),
        nova_label="unqualified",
        nova_span=float("nan"),
        efit_span=abs(record.fed_back_psi - record.axis_psi),
        note=(
            f"lattice {resolution}: candidates {int(roots.candidates)}, "
            "admitted saddles 0; nova's read's axis fallback is not reported"
        ),
    )


def lattice(wall_ring: np.ndarray, resolution: int):
    """A square lattice of axis-aligned cells whose centres lie in the wall."""
    polygon = ShapelyPolygon(wall_ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    rmin, zmin = wall_ring.min(axis=0)
    rmax, zmax = wall_ring.max(axis=0)
    width = (rmax - rmin) / resolution
    height = (zmax - zmin) / resolution
    cells = []
    offsets = []
    for i in range(resolution):
        for j in range(resolution):
            r0 = rmin + i * width
            z0 = zmin + j * height
            cell = box(r0, z0, r0 + width, z0 + height)
            if not polygon.contains(cell.centroid):
                continue
            r1, z1 = r0 + width, z0 + height
            cells.append(
                np.asarray(((r0, z0), (r1, z0), (r1, z1), (r0, z1)), dtype=np.float64)
            )
            step_r, step_z = 0.5 * width, 0.5 * height
            mid_r, mid_z = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
            # The geometry's sampling vertices are absolute RZ points, one set
            # per cell, and it prepends the cell centre itself.
            around = [
                (mid_r - step_r, mid_z - step_z),
                (mid_r + step_r, mid_z - step_z),
                (mid_r + step_r, mid_z + step_z),
                (mid_r - step_r, mid_z + step_z),
                (mid_r, mid_z - step_z),
                (mid_r, mid_z + step_z),
                (mid_r - step_r, mid_z),
                (mid_r + step_r, mid_z),
            ]
            offsets.append(np.asarray(around, dtype=np.float64))
    if not cells:
        raise ValueError("lattice found no cell inside the wall polygon")
    return cells, np.asarray(offsets)


def wall_units(record: TraceRecord):
    """The record's wall as one closed unit."""
    ring = record.wall_ring
    if not np.allclose(ring[0], ring[-1]):
        ring = np.vstack((ring, ring[0]))
    return (ring,)


def _decision(case, source, record, resolution):
    field = TraceSplineField.from_record(record)
    cells, samples = lattice(record.wall_ring, resolution)
    geometry = topology.TopologyGeometry.from_cells(
        cells, samples, wall_units(record)
    )
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    initial = jax.jit(lambda: topology.read(field, geometry, convention, policy))()
    axis_valid = np.asarray(initial.x_point_valid)
    points = np.asarray(initial.x_points)
    stack = [np.asarray(initial.axis)[None]]
    if axis_valid.any():
        stack.append(points[axis_valid])
    previous = jnp.asarray(np.vstack(stack))
    roots = census(field, geometry, convention, policy, previous)
    qualified = int(roots.candidates) > 0 and int(roots.saddles) > 0
    if not qualified:
        return _unqualified(case, source, record, resolution, roots, initial)
    reading = nova_read(field, roots.roots, geometry, convention, policy)
    sigma = convention.sigma

    boundary = np.asarray(reading.boundary)
    boundary_psi = float(sigma * field.evaluate(jnp.asarray(boundary)).value)
    axis_psi = float(sigma * field.evaluate(jnp.asarray(reading.axis)).value)
    wall_values = np.asarray(
        jax.vmap(lambda p: field.evaluate(p).value)(geometry.wall_points)
    )
    wall_level = float(np.max(sigma * wall_values))
    label = np.asarray(reading.boundary_class)
    return Decision(
        case=case,
        record=record.slice,
        source=source,
        n_cells=len(cells),
        efit_verdict=int(record.verdict),
        efit_label="diverted" if record.diverted else "limited",
        efit_axis_psi=record.axis_psi,
        efit_axis_r=record.axis_r,
        efit_axis_z=record.axis_z,
        efit_saddle_refined=record.saddle_refined,
        efit_wall_level=record.wall_level,
        efit_fed_back_psi=record.fed_back_psi,
        nova_axis_psi=axis_psi,
        nova_axis_r=float(np.asarray(reading.axis)[0]),
        nova_axis_z=float(np.asarray(reading.axis)[1]),
        nova_boundary_psi=float(np.asarray(reading.boundary_flux)),
        nova_saddle_psi=boundary_psi,
        nova_saddle_r=float(boundary[0]),
        nova_saddle_z=float(boundary[1]),
        nova_wall_level=wall_level,
        nova_label="diverted" if int(label) else "limited",
        nova_span=abs(float(np.asarray(reading.boundary_flux)) - axis_psi),
        efit_span=abs(record.fed_back_psi - record.axis_psi),
    )


def load_pairs():
    rows = []
    for line in PAIRS.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[0].lower() in ("pair", "pair_id"):
            continue
        rows.append(parts)
    return rows


def main(resolution=16, out=None):
    results = []
    for parts in load_pairs():
        pair, a_path, b_path = parts[0], parts[1], parts[2]
        for side, path in (("A", a_path), ("B", b_path)):
            record = read_trace_record(path)
            try:
                case = _decision(
                    f"{pair}:{side}", Path(path).name, record, resolution
                )
            except Exception as error:  # noqa: BLE001 - one record must not stop the sweep
                case = Decision(
                    case=f"{pair}:{side}",
                    record=record.slice,
                    source=Path(path).name,
                    n_cells=0,
                    efit_verdict=record.verdict,
                    efit_label="diverted" if record.diverted else "limited",
                    efit_axis_psi=record.axis_psi,
                    efit_axis_r=record.axis_r,
                    efit_axis_z=record.axis_z,
                    efit_saddle_refined=record.saddle_refined,
                    efit_wall_level=record.wall_level,
                    efit_fed_back_psi=record.fed_back_psi,
                    nova_axis_psi=float("nan"),
                    nova_axis_r=float("nan"),
                    nova_axis_z=float("nan"),
                    nova_boundary_psi=float("nan"),
                    nova_saddle_psi=float("nan"),
                    nova_saddle_r=float("nan"),
                    nova_saddle_z=float("nan"),
                    nova_wall_level=float("nan"),
                    nova_label="unread",
                    nova_span=float("nan"),
                    efit_span=abs(record.fed_back_psi - record.axis_psi),
                    note=f"{type(error).__name__}: {error}",
                )
            results.append(asdict(case))
            print("done", pair, side, case.nova_label, flush=True)
    for name in ("synthetic_crossing_a", "synthetic_crossing_b"):
        path = CROSSING / f"{name}.txt"
        if path.is_file():
            record = read_trace_record(path)
            try:
                case = _decision(name, path.name, record, resolution)
            except Exception as error:  # noqa: BLE001
                print("skip", name, type(error).__name__ if False else type(error).__name__, flush=True)
                continue
            results.append(asdict(case))
            print("done", name, case.nova_label, flush=True)
    payload = {"resolution": resolution, "rows": results}
    text = json.dumps(payload, indent=2)
    if out:
        Path(out).write_text(text + "\n")
    print("ROWS", len(results))
    return payload


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(out=target)