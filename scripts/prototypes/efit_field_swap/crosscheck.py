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
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon, box

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

# A winding number computed from the unwrapped phase of grad psi lands on an
# integer unless the loop passes through a critical point. The tolerance only
# absorbs that rounding; the floor below is what catches a loop through a zero.
WINDING_TOLERANCE = 0.2
# |grad psi| below this fraction of the loop's own psi extent per sample
# spacing is indistinguishable from zero at the arithmetic's precision, so a
# critical point sitting on the loop makes the winding meaningless rather than
# an integer to be trusted.
GRADIENT_FLOOR_RELATIVE = float(np.sqrt(np.finfo(np.float64).eps))


class Coverage(NamedTuple):
    """The boundary-index certificate for one lattice's census.

    While the census only proposes seeds from the cells it can condition, an
    empty admitted-saddle set is a claim about the field that the census cannot
    itself support. The certificate is read off the field instead: the degree of
    grad psi on the domain boundary equals the index sum of the critical points
    inside it, extrema counting +1 and saddles -1. Agreement means the admitted
    set accounts for every critical point in the domain; disagreement means the
    census missed one (or invented one), whatever it reported.
    """

    outer_winding: float
    obstacle_winding: float
    total: float
    index_sum: float
    min_gradient: float
    floor: float
    reliable: bool
    certified: bool
    cells_cleared: int = 0
    cells_uncertified: int = 0
    cells_occupied: int = 0
    subdivision_depth: int = 0


CELL_SUBDIVISION_DEPTH = 6
CELL_WORK_LIMIT = 20000


def _gradient_controls(field):
    """Physical gradient components as Bernstein control nets per spline patch."""
    spline = field.spline
    radial = np.asarray(spline.radial)
    vertical = np.asarray(spline.vertical)
    coefficient = np.asarray(spline.coefficients)
    dr = np.diff(radial)[None, :, None, None]
    dz = np.diff(vertical)[:, None, None, None]
    grad_r = 3.0 * np.diff(coefficient, axis=-1) / dr
    grad_z = 3.0 * np.diff(coefficient, axis=-2) / dz
    return radial, vertical, grad_r, grad_z


def _split_bernstein(control, fraction, axis):
    """The two exact de Casteljau restrictions of a Bernstein control net."""
    levels = [np.asarray(control, dtype=float)]
    for _ in range(control.shape[axis] - 1):
        first = np.take(levels[-1], range(levels[-1].shape[axis] - 1), axis=axis)
        second = np.take(levels[-1], range(1, levels[-1].shape[axis]), axis=axis)
        levels.append((1.0 - fraction) * first + fraction * second)
    left = np.stack([np.take(level, 0, axis=axis) for level in levels], axis=axis)
    right = np.stack(
        [np.take(level, -1, axis=axis) for level in levels[::-1]], axis=axis
    )
    return left, right


def _restrict_bernstein(control, low, high, axis):
    """Restrict one Bernstein axis to a closed subinterval of its patch."""
    if low > 0.0:
        _, control = _split_bernstein(control, low, axis)
    if high < 1.0:
        control, _ = _split_bernstein(control, (high - low) / (1.0 - low), axis)
    return control


def _strict_sign(control, depth):
    """Outward-rounded coefficient hull excludes zero on the entire patch."""
    scale = float(np.max(np.abs(control)))
    gamma = (64 + 32 * depth) * np.finfo(float).eps
    return bool(np.min(control) > gamma * scale or np.max(control) < -gamma * scale)


def _patch_clear(controls, radial, vertical, ir, iz, box, depth):
    r0, r1, z0, z1 = box
    pr0, pr1 = radial[ir : ir + 2]
    pz0, pz1 = vertical[iz : iz + 2]
    u0 = max(0.0, (r0 - pr0) / (pr1 - pr0))
    u1 = min(1.0, (r1 - pr0) / (pr1 - pr0))
    v0 = max(0.0, (z0 - pz0) / (pz1 - pz0))
    v1 = min(1.0, (z1 - pz0) / (pz1 - pz0))
    for control in controls:
        patch = _restrict_bernstein(control[iz, ir], u0, u1, axis=1)
        patch = _restrict_bernstein(patch, v0, v1, axis=0)
        if _strict_sign(patch, depth):
            return True
    return False


def certify_empty_cells(field, cells, admitted_points, *, depth=CELL_SUBDIVISION_DEPTH):
    """Exclude grad-psi zeros in every lattice cell without an admitted root.

    Each box is intersected with every spline patch it meets. De Casteljau
    restriction gives Bernstein coefficients for each gradient component on
    the intersection. A patch is clear when either component's outward-rounded
    coefficient hull excludes zero. An unresolved box splits into four until
    the depth or total-work cap; any remainder stays uncertified.
    """
    radial, vertical, grad_r, grad_z = _gradient_controls(field)
    points = np.asarray(admitted_points, dtype=float).reshape(-1, 2)
    occupied = 0
    empty_cells = 0
    unresolved_cells = set()
    pending = []
    for cell_number, cell in enumerate(cells):
        low = np.min(cell, axis=0)
        high = np.max(cell, axis=0)
        rounding = 256 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(cell))))
        if np.any(
            np.all((points >= low - rounding) & (points <= high + rounding), axis=1)
        ):
            occupied += 1
        else:
            empty_cells += 1
            pending.append((low[0], high[0], low[1], high[1], 0, cell_number))
    work = 0
    while pending:
        if work >= CELL_WORK_LIMIT:
            unresolved_cells.update(item[-1] for item in pending)
            break
        r0, r1, z0, z1, level, cell_number = pending.pop()
        work += 1
        if r0 < radial[0] or r1 > radial[-1] or z0 < vertical[0] or z1 > vertical[-1]:
            unresolved_cells.add(cell_number)
            continue
        ir0 = max(0, np.searchsorted(radial, r0, side="right") - 1)
        ir1 = min(len(radial) - 2, np.searchsorted(radial, r1, side="left") - 1)
        iz0 = max(0, np.searchsorted(vertical, z0, side="right") - 1)
        iz1 = min(len(vertical) - 2, np.searchsorted(vertical, z1, side="left") - 1)
        clear = all(
            _patch_clear(
                (grad_r, grad_z), radial, vertical, ir, iz, (r0, r1, z0, z1), level
            )
            for iz in range(iz0, iz1 + 1)
            for ir in range(ir0, ir1 + 1)
        )
        if clear:
            pass
        elif level == depth:
            unresolved_cells.add(cell_number)
        else:
            rm, zm = (r0 + r1) / 2, (z0 + z1) / 2
            pending.extend(
                (a, b, c, d, level + 1, cell_number)
                for a, b in ((r0, rm), (rm, r1))
                for c, d in ((z0, zm), (zm, z1))
            )
    return empty_cells - len(unresolved_cells), len(unresolved_cells), occupied


def _counterclockwise(ring):
    """The ring closed and traversed counterclockwise, never the stored order.

    The winding of a vector field along a closed curve changes sign with the
    curve's direction, while a critical point's index does not. Every loop is
    therefore taken in one known orientation, so the domain's index sum is the
    outer loop's winding less each obstacle's -- both measured the same way --
    rather than a sum of numbers whose signs depend on the source file.
    """
    points = np.asarray(ring, dtype=np.float64)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    area = 0.5 * float(
        np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1])
    )
    return points[::-1] if area < 0.0 else points


def loop_winding(field, ring):
    """Winding number of grad psi along a closed loop, and min |grad psi| on it."""
    points = np.asarray(ring, dtype=np.float64)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    gradient = np.asarray(
        jax.vmap(lambda point: field.evaluate(point).gradient)(jnp.asarray(points))
    )
    magnitude = np.linalg.norm(gradient, axis=1)
    phase = np.unwrap(np.arctan2(gradient[:, 1], gradient[:, 0]))
    return float(phase[-1] - phase[0]) / (2.0 * np.pi), float(magnitude.min())


def gradient_floor(field, ring):
    """Rounding-referenced floor for |grad psi| on one loop.

    A spline gradient carries rounding error of order the field's own variation
    across one sample spacing, so that figure -- not any absolute scale -- is
    what makes a loop's value indistinguishable from zero.
    """
    points = np.asarray(ring, dtype=np.float64)
    closed = (
        points
        if np.array_equal(points[0], points[-1])
        else np.vstack((points, points[0]))
    )
    spacing = float(np.min(np.linalg.norm(np.diff(closed, axis=0), axis=1)))
    values = np.asarray([float(field.value(point)) for point in points])
    return GRADIENT_FLOOR_RELATIVE * float(np.ptp(values)) / spacing


def admitted_index_sum(roots):
    """Index sum of the admitted critical points: extrema +1, saddles -1.

    Counted from the sign of the Hessian determinant at each admitted root, so
    an extremum is +1 whatever sign the field's own maxima carry in the working
    convention. The census' own extremum tally is not used: it counts only
    negative-definite roots, so it reads zero on a field whose axis is a
    minimum and would report a closed domain as uncertified.
    """
    hessian = np.asarray(roots.jet.hessian)
    valid = np.asarray(roots.valid).copy()
    determinant = hessian[:, 0, 0] * hessian[:, 1, 1] - hessian[:, 0, 1] ** 2
    return int(np.sum(np.where(determinant[valid] > 0.0, 1, -1)))


def coverage_certificate(field, outer_ring, obstacle_rings, index_sum):
    """Certify that the census admitted every critical point inside the domain.

    The count is the outer loop's winding less the sum of the obstacle loops',
    each taken counterclockwise -- never a per-loop equality,
    which would hold trivially for a loop enclosing nothing. A loop whose
    minimum |grad psi| falls to the rounding floor passes through a critical
    point and carries no integer to compare, so the slice is left uncertified
    rather than trusted on a number the arithmetic cannot support.
    """
    minimum = float("inf")
    floor = 0.0
    windings = []
    for ring in (outer_ring, *obstacle_rings):
        value, least = loop_winding(field, ring)
        windings.append(value)
        minimum = min(minimum, least)
        floor = max(floor, gradient_floor(field, ring))
    outer = windings[0]
    obstacle = float(sum(windings[1:]))
    total = outer - obstacle
    reliable = minimum > floor
    certified = reliable and abs(total - float(index_sum)) < WINDING_TOLERANCE
    return Coverage(
        outer_winding=outer,
        obstacle_winding=obstacle,
        total=total,
        index_sum=float(index_sum),
        min_gradient=minimum,
        floor=floor,
        reliable=reliable,
        certified=certified,
    )


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
    wall_loop_count: int
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
    coverage_outer_winding: float
    coverage_obstacle_winding: float
    coverage_winding: float
    coverage_index_sum: float
    coverage_min_gradient: float
    coverage_floor: float
    coverage_reliable: bool
    coverage_certified: bool
    coverage_cells_cleared: int
    coverage_cells_uncertified: int
    coverage_cells_occupied: int
    note: str = ""


def _efit_columns(case, source, record):
    """The columns that come from the record's own stored decision."""
    return dict(
        case=case,
        record=record.slice if record is not None else "",
        source=source,
        wall_loop_count=(
            len(record.wall_index)
            if record is not None and record.wall_r.size
            else (1 if record is not None and record.polygon_r.size >= 3 else 0)
        ),
        efit_verdict=int(record.verdict) if record is not None else -1,
        efit_label=(
            "unread"
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


def _undecided(case, source, record, label, note, field_grad, coverage=None):
    """A case with no nova decision, reported as such rather than as a decision.

    nova's census proposed no cellwise seed, or proposed seeds and admitted no
    saddle, or the boundary-index certificate could not close, or the geometry
    could not be built. Reporting the read's own fallback here would put a
    number in the nova column that is the axis level wearing the boundary's
    name, which is exactly the misreading this table exists to prevent. The
    certificate, where one was computed, is reported beside the refusal so the
    reason is readable rather than asserted.
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
        coverage_outer_winding=(nan if coverage is None else coverage.outer_winding),
        coverage_obstacle_winding=(
            nan if coverage is None else coverage.obstacle_winding
        ),
        coverage_winding=nan if coverage is None else coverage.total,
        coverage_index_sum=nan if coverage is None else coverage.index_sum,
        coverage_min_gradient=nan if coverage is None else coverage.min_gradient,
        coverage_floor=nan if coverage is None else coverage.floor,
        coverage_reliable=False if coverage is None else coverage.reliable,
        coverage_certified=False if coverage is None else coverage.certified,
        coverage_cells_cleared=0 if coverage is None else coverage.cells_cleared,
        coverage_cells_uncertified=0
        if coverage is None
        else coverage.cells_uncertified,
        coverage_cells_occupied=0 if coverage is None else coverage.cells_occupied,
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


def lattice(wall_ring: np.ndarray, resolution: int, obstacles=()):
    """A square lattice with centres inside the vessel and outside obstacles."""
    polygon = ShapelyPolygon(wall_ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    holes = []
    for ring in obstacles:
        hole = ShapelyPolygon(ring)
        holes.append(hole if hole.is_valid else hole.buffer(0))
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
            if not polygon.contains(cell.centroid) or any(
                hole.covers(cell.centroid) for hole in holes
            ):
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


def domain_rings(record: TraceRecord):
    """Select the one indexed loop containing the stored axis as vessel."""
    rings = record.wall_rings
    if len(rings) == 1:
        return rings[0], ()
    axis = Point(record.axis_r, record.axis_z)
    polygons = []
    for ring in rings:
        polygon = ShapelyPolygon(ring)
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    owners = [index for index, polygon in enumerate(polygons) if polygon.contains(axis)]
    if len(owners) != 1:
        raise ValueError(f"expected one axis-containing wall loop, found {len(owners)}")
    owner = owners[0]
    outer = polygons[owner]
    obstacles = tuple(
        ring
        for index, ring in enumerate(rings)
        if index != owner and outer.covers(polygons[index]) and polygons[index].area > 0
    )
    return rings[owner], obstacles


def wall_units(record: TraceRecord):
    """Keep the vessel and its interior obstacles as separate wall units."""
    boundary, obstacles = domain_rings(record)
    return (boundary, *obstacles)


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


@dataclass
class ReadOutcome:
    """nova's read on one field, with the certificate that qualifies it."""

    resolution: int | None
    cells: int
    capacity_fit: bool
    candidates: int
    saddles: int
    label: str
    axis: np.ndarray
    axis_psi: float
    boundary: np.ndarray
    boundary_psi: float
    boundary_flux: float
    span: float
    wall_level: float
    coverage: Coverage | None
    attempts: str


def _empty_outcome(label, attempts, coverage=None):
    nan = float("nan")
    return ReadOutcome(
        resolution=None,
        cells=0,
        capacity_fit=False,
        candidates=0,
        saddles=0,
        label=label,
        axis=np.asarray([nan, nan]),
        axis_psi=nan,
        boundary=np.asarray([nan, nan]),
        boundary_psi=nan,
        boundary_flux=nan,
        span=nan,
        wall_level=nan,
        coverage=coverage,
        attempts=attempts,
    )


def read_state(
    field, boundary_ring, obstacles, convention, policy, resolutions=RESOLUTIONS
):
    """nova's read on one field, at the coarsest lattice that reads.

    A lattice whose census admits no saddle is a limited state only where the
    boundary-index certificate shows the census accounted for every critical
    point in the domain. Where it does not, the lattice is refused rather than
    decided: with the census' proposal coverage incomplete, an empty saddle set
    is as consistent with a missed saddle as with a genuinely limited plasma,
    and deciding limited on it would turn a coverage miss into a confident
    wrong answer. A lattice with no admitted stationary root at all carries
    nothing to read and is skipped outright.
    """
    outer = _counterclockwise(boundary_ring)
    walls = tuple(_counterclockwise(ring) for ring in obstacles)
    attempts = []
    uncertified = False
    coverage = None
    for resolution in resolutions:
        cells, samples = lattice(boundary_ring, resolution, obstacles)
        geometry, fitted = fitted_geometry(cells, samples, (boundary_ring, *obstacles))
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
        candidates = int(roots.candidates)
        admitted = int(roots.saddles)
        index_sum = admitted_index_sum(roots.roots)
        try:
            coverage = coverage_certificate(field, outer, walls, index_sum)
        except Exception as error:  # noqa: BLE001 - refusal, never a false decision
            attempts.append(
                f"lattice {resolution}: boundary winding failed "
                f"{type(error).__name__}: {error}; coverage uncertified"
            )
            uncertified = True
            continue
        root_valid = np.asarray(roots.roots.valid).copy()
        axis_admitted = bool(root_valid.size and root_valid[0])
        admitted_points = np.asarray(roots.roots.position)[root_valid]
        try:
            cleared, unresolved, occupied = certify_empty_cells(
                field, cells, admitted_points
            )
            enclosure_error = ""
        except Exception as error:  # noqa: BLE001 - an unknown read is caller-visible
            cleared, unresolved, occupied = 0, len(cells), 0
            enclosure_error = f"; enclosure failed {type(error).__name__}: {error}"
        coverage = coverage._replace(
            cells_cleared=cleared,
            cells_uncertified=unresolved,
            cells_occupied=occupied,
            subdivision_depth=CELL_SUBDIVISION_DEPTH,
            certified=coverage.certified and unresolved == 0 and axis_admitted,
        )
        attempts.append(
            f"lattice {resolution}: candidates {candidates}, admitted saddles "
            f"{admitted}, boundary winding {coverage.total:+.0f} (outer "
            f"{coverage.outer_winding:+.0f} less obstacles "
            f"{coverage.obstacle_winding:+.0f}), admitted index sum {index_sum:+d}, "
            f"min |grad psi| {coverage.min_gradient:.2e} against floor "
            f"{coverage.floor:.2e}, cells clear {cleared}, occupied "
            f"{occupied}, uncertified {unresolved}, axis admitted {axis_admitted} "
            f"at depth {CELL_SUBDIVISION_DEPTH} "
            f"and work cap {CELL_WORK_LIMIT}{enclosure_error}"
        )
        if candidates == 0 and admitted == 0 and not bool(np.any(axis_valid)):
            attempts[-1] += "; no stationary root to read"
            continue
        if not coverage.certified:
            attempts[-1] += (
                "; coverage uncertified: boundary index or cell-gradient enclosure "
                "does not exclude an unadmitted critical point"
                if coverage.reliable
                else "; coverage uncertified: a critical point on the loop leaves "
                "the boundary winding undefined"
            )
            uncertified = True
            continue
        try:
            reading = nova_read(field, roots.roots, geometry, convention, policy)
        except Exception as error:  # noqa: BLE001 - one lattice must not stop the sweep
            attempts[-1] += f"; read failed {type(error).__name__}: {error}"
            continue
        sigma = convention.sigma
        boundary = np.asarray(reading.boundary, dtype=np.float64)
        axis = np.asarray(reading.axis, dtype=np.float64)
        boundary_psi = float(sigma * field.evaluate(jnp.asarray(boundary)).value)
        axis_psi = float(sigma * field.evaluate(jnp.asarray(axis)).value)
        wall_values = np.asarray(
            jax.vmap(lambda point: field.evaluate(point).value)(geometry.wall_points)
        )
        wall_level = float(np.max(sigma * wall_values))
        return ReadOutcome(
            resolution=resolution,
            cells=len(cells),
            capacity_fit=fitted,
            candidates=candidates,
            saddles=admitted,
            label="diverted" if int(np.asarray(reading.boundary_class)) else "limited",
            axis=axis,
            axis_psi=axis_psi,
            boundary=boundary,
            boundary_psi=boundary_psi,
            boundary_flux=float(np.asarray(reading.boundary_flux)),
            span=abs(float(np.asarray(reading.boundary_flux)) - axis_psi),
            wall_level=wall_level,
            coverage=coverage,
            attempts="; ".join(attempts),
        )
    return _empty_outcome(
        "coverage uncertified" if uncertified else "unqualified",
        "; ".join(attempts)
        + (
            "; no lattice certified the census' coverage"
            if uncertified
            else "; no lattice admitted a saddle"
        ),
        coverage,
    )


def _decision(case, source, record, resolutions=RESOLUTIONS):
    """nova's decision on the record's spline, at the coarsest lattice that reads."""
    field = TraceSplineField.from_record(record)
    field_grad = _field_gradient_at_efit_axis(field, record)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    boundary_ring, obstacles = domain_rings(record)
    outcome = read_state(
        field, boundary_ring, obstacles, convention, policy, resolutions
    )
    if outcome.resolution is None:
        return _undecided(
            case,
            source,
            record,
            outcome.label,
            outcome.attempts,
            field_grad,
            outcome.coverage,
        )
    coverage = outcome.coverage
    note = outcome.attempts
    if outcome.capacity_fit:
        note = (note + " | " if note else "") + (
            f"contour mesh refitted at {CAPACITY_GROWTH}x capacity: nova's own "
            "capacity admits no vertex for this grid"
        )
    return Decision(
        **_efit_columns(case, source, record),
        n_cells=outcome.cells,
        resolution=outcome.resolution,
        capacity_fit=outcome.capacity_fit,
        nova_axis_psi=outcome.axis_psi,
        nova_axis_r=float(outcome.axis[0]),
        nova_axis_z=float(outcome.axis[1]),
        nova_boundary_psi=outcome.boundary_flux,
        nova_saddle_psi=outcome.boundary_psi,
        nova_saddle_r=float(outcome.boundary[0]),
        nova_saddle_z=float(outcome.boundary[1]),
        nova_wall_level=outcome.wall_level,
        nova_label=outcome.label,
        nova_span=outcome.span,
        coverage_outer_winding=coverage.outer_winding,
        coverage_obstacle_winding=coverage.obstacle_winding,
        coverage_winding=coverage.total,
        coverage_index_sum=coverage.index_sum,
        coverage_min_gradient=coverage.min_gradient,
        coverage_floor=coverage.floor,
        coverage_reliable=coverage.reliable,
        coverage_certified=coverage.certified,
        coverage_cells_cleared=coverage.cells_cleared,
        coverage_cells_uncertified=coverage.cells_uncertified,
        coverage_cells_occupied=coverage.cells_occupied,
        field_grad_at_efit_axis=field_grad,
        field_grad_at_nova_axis=_field_gradient_at_nova_axis(field, outcome.axis),
        note=note,
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
