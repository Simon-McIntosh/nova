"""Cross-check nova's boundary decision against imas-efit's on each other's fields.

Each imas-efit boundary trace record carries a rectilinear psi grid and the wall
polygon it was traced against. The adapter rebuilds the record's bicubic spline
and nova's read is run on it unchanged, on a square-lattice cell geometry built
inside the record's own wall polygon. The record also carries imas-efit's own
decision for the same state, so the case-by-case table records ``<none>`` with its
reason rather than a guessed value les the two codes can be compared case by case
rather than converged.

Every record in the manifest gets a row. A record whose read fails is recorded
with the exception text, and a record whose lattice admits no saddle is recorded
with the census counts that disqualified it, so a gap is never silent. Two
retries keep the gap small without inventing a decision:

* the cell geometry is rebuilt at a coarser resolution first and, when the census
  admits no saddle, at successively finer ones, because a saddle the lattice
  cannot resolve is a limit of the substituted geometry rather than a property of
  the field;
* the contour mesh is rebuilt with a fitted capacity when nova's own capacity
  truncates it to nothing, which is what happens on every wide wall in this
  manifest. The fit is applied only where the native mesh has no valid vertex at
  all, so no case that already read is changed by it, and the row says which
  capacity served it.

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
TRACE_LIBRARY = Path(
    "/home/ITER/mcintos/Code/imas-efit/tests/unit/fortran/data/boundary_level"
)

RESOLUTIONS = (16, 24, 32)
CAPACITY_GROWTH = 8


def resolve_trace(path):
    """Follow a pairs row whose path names a since-removed worktree.

    The manifest was written from a worker worktree, so its later rows carry
    that worktree's absolute path and no longer resolve. The same records are
    tracked in imas-efit's own test data directory, so the row is resolved by
    record name there rather than dropped.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    fallback = TRACE_LIBRARY / candidate.name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"{path} and {fallback} both missing")


@dataclass
class Decision:
    """One case's nova-versus-imas-efit decision columns."""

    case: str
    record: str
    source: str
    n_cells: int
    resolution: int
    capacity_fit: bool
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
    field_grad_at_efit_axis: float
    field_grad_at_nova_axis: float
    note: str = ""


def _efit_columns(case, source, record):
    """The columns that come from the record's own stored decision."""
    return dict(
        case=case,
        record=record.slice if record is not None else "",
        source=source,
        efit_verdict=int(record.verdict) if record is not None else -1,
        efit_label=(
            ("diverted" if record.diverted else "unread")
            if record is None
            else ("diverted" if record.diverted else "limited")
        ),
        efit_axis_psi=record.axis_psi if record is not None else float("nan"),
        efit_axis_r=record.axis_r if record is not None else float("nan"),
        efit_axis_z=record.axis_z if record is not None else float("nan"),
        efit_saddle_refined=(
            record.saddle_refined if record is not None else float("nan")
        ),
        efit_wall_level=record.wall_level if record is not None else float("nan"),
        efit_fed_back_psi=(record.fed_back_psi if record is not None else float("nan")),
        efit_span=(
            abs(record.fed_back_psi - record.axis_psi)
            if record is not None
            else float("nan")
        ),
    )


def _undecided(case, source, record, label, note, field_grad):
    """A case with no nova decision, reported as such rather than as a decision.

    nova's census proposed no cellwise seed, or proposed seeds and admitted no
    saddle, or the geometry could not be built. Reporting the read's own
    fallback here would put a number in the nova column that is the axis level
    wearing the boundary's name, which is exactly the misreading this table
    exists to prevent.
    """
    nan = float("nan")
    return Decision(
        **_efit_columns(case, source, record),
        n_cells=0,
        resolution=0,
        capacity_fit=False,
        nova_axis_psi=nan,
        nova_axis_r=nan,
        nova_axis_z=nan,
        nova_boundary_psi=nan,
        nova_saddle_psi=nan,
        nova_saddle_r=nan,
        nova_saddle_z=nan,
        nova_wall_level=nan,
        nova_label=label,
        nova_span=nan,
        field_grad_at_efit_axis=field_grad,
        field_grad_at_nova_axis=nan,
        note=note,
    )


def _fitted_capacity(holder):
    """Wrap the contour-mesh builder so a truncating capacity is refitted.

    ``TopologyGeometry.from_cells`` sizes the contour mesh as
    ``len(tree_centres) + len(points) + 1``, which is too small for a wall whose
    segments generate more vertices than that. The builder then returns a mesh
    with *zero* valid vertices instead of raising, and the failure surfaces two
    calls later as an empty index. The wrapper retries at a larger capacity only
    in that case, so a geometry that already meshed is returned untouched.
    """

    def build(*args, **kwargs):
        original = holder["original"]
        mesh = original(*args, **kwargs)
        if np.any(np.asarray(mesh.vertex_valid)):
            return mesh
        holder["fitted"] = True
        grown = dict(kwargs)
        for key in ("vertex_capacity", "edge_capacity", "triangle_capacity"):
            if key in grown:
                grown[key] = CAPACITY_GROWTH * grown[key]
        return original(*args, **grown)

    return build


def fitted_geometry(cells, sampling_vertices, wall_units):
    """Build the cell geometry, refitting a truncating contour-mesh capacity.

    Returns the geometry and whether the fit was needed, or ``(None, True)`` when
    even the fitted capacity admits no vertex.
    """
    from nova.equilibrium import contour_tree_mesh

    holder = {"original": contour_tree_mesh.build_contour_mesh, "fitted": False}
    contour_tree_mesh.build_contour_mesh = _fitted_capacity(holder)
    try:
        try:
            geometry = topology.TopologyGeometry.from_cells(
                cells, sampling_vertices, wall_units
            )
        except IndexError, ValueError:
            return None, holder["fitted"]
    finally:
        contour_tree_mesh.build_contour_mesh = holder["original"]
    return geometry, holder["fitted"]


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
        ring = np.vstack((ring, ring[-1]))
    return (ring,)


def _field_gradient_at_efit_axis(field, record):
    """|grad psi| at the axis imas-efit recorded in this same record.

    This is the adapter's own falsification test and it runs on every row. If
    the rebuilt spline is imas-efit's field, its gradient vanishes at the axis
    imas-efit measured, to numerical precision, whatever the grid ordering,
    units or frame. A row whose value is not near zero says the adapter is
    wrong and nothing below it in the row can be read as a statement about
    nova's decision rule.
    """
    axis = jnp.asarray([record.axis_r, record.axis_z])
    return float(np.linalg.norm(np.asarray(jax.jacfwd(field.value)(axis))))


def _field_gradient_at_nova_axis(field, reading):
    """|grad psi| at the axis nova's own read placed on this field.

    The paired test of the one above. If nova's axis is a true stationary point
    of the same field, the axis disagreement between the two codes is a
    difference in how they select an axis, which is a statement about nova's
    rule. If it is not stationary, nova's axis here is an artefact of the
    square-lattice cell geometry this adapter substitutes, and no claim about
    nova's axis rule can be made from this table.
    """
    axis = jnp.asarray(reading.axis)
    return float(np.linalg.norm(np.asarray(jax.jacfwd(field.value)(axis))))


def _decision(case, source, record, resolutions=RESOLUTIONS):
    """nova's decision on the record's spline, at the coarsest lattice that reads."""
    field = TraceSplineField.from_record(record)
    field_grad = _field_gradient_at_efit_axis(field, record)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    attempts = []
    for resolution in resolutions:
        cells, samples = lattice(record.wall_ring, resolution)
        geometry, fitted = fitted_geometry(cells, samples, wall_units(record))
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
        sigma = convention.sigma
        boundary = np.asarray(reading.boundary)
        boundary_psi = float(sigma * field.evaluate(jnp.asarray(boundary)).value)
        axis_psi = float(sigma * field.evaluate(jnp.asarray(reading.axis)).value)
        wall_values = np.asarray(
            jax.vmap(lambda p: field.evaluate(p).value)(geometry.wall_points)
        )
        wall_level = float(np.max(sigma * wall_values))
        label = np.asarray(reading.boundary_class)
        note = " | ".join(attempts[:-1]) if len(attempts) > 1 else ""
        if fitted:
            note = (note + " | " if note else "") + (
                f"contour mesh refitted at {CAPACITY_GROWTH}x capacity: nova's own "
                "capacity admits no vertex for this wall"
            )
        return Decision(
            **_efit_columns(case, source, record),
            n_cells=len(cells),
            resolution=resolution,
            capacity_fit=fitted,
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
            field_grad_at_efit_axis=field_grad,
            field_grad_at_nova_axis=_field_gradient_at_nova_axis(field, reading),
            note=note,
        )
    return _undecided(
        case,
        source,
        record,
        "unqualified",
        "; ".join(attempts) + "; no lattice admitted a saddle",
        field_grad,
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


def _row(case, source, path, resolutions=RESOLUTIONS):
    """One row for one record, whatever the outcome."""
    try:
        record = read_trace_record(resolve_trace(path))
    except Exception as error:  # noqa: BLE001 - an unreadable record is a row
        return _undecided(
            case,
            source,
            None,
            "unread",
            f"{type(error).__name__}: {error}",
            float("nan"),
        )
    try:
        return _decision(case, source, record, resolutions)
    except Exception as error:  # noqa: BLE001 - one record must not stop the sweep
        return _undecided(
            case,
            source,
            record,
            "unread",
            f"{type(error).__name__}: {error}",
            float("nan"),
        )


def cases():
    """Every input the manifest names, as (case, record name, path).

    The crossing records are inputs too. A record whose read fails belongs in
    the table as a recorded failure rather than as a row the sweep never
    reached, so this enumeration does not depend on whether a read succeeds.
    """
    for parts in load_pairs():
        for side, path in (("A", parts[1]), ("B", parts[2])):
            yield f"{parts[0]}:{side}", Path(path).name, path
    for path in sorted(CROSSING.glob("*.txt")):
        yield path.stem, path.name, str(path)


def main(resolution=RESOLUTIONS, out=None):
    resolutions = (
        tuple(resolution) if isinstance(resolution, (tuple, list)) else (resolution,)
    )
    results = []
    for case, source, path in cases():
        row = _row(case, source, path, resolutions)
        results.append(asdict(row))
        print("done", row.case, row.nova_label, flush=True)
        if out:
            Path(out).write_text(
                json.dumps(
                    {"resolutions": list(resolutions), "rows": results}, indent=2
                )
                + "\n"
            )
    payload = {"resolutions": list(resolutions), "rows": results}
    if out:
        Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print("ROWS", len(results))
    return payload


if __name__ == "__main__":
    import sys

    main(out=sys.argv[1] if len(sys.argv) > 1 else None)
