"""Score nova's boundary read on TCV 77549 LIUQE slices against LIUQE itself.

The input is a LIUQE reconstruction stored as an IMAS ``equilibrium`` IDS: a
rectilinear psi grid per time slice and a boundary polygon on a separate TCV
``wall`` IDS. The merged field-swap adapter supplies the pieces: the grid spline
field (``efit_field_*swap.spline_field``) and the per-loop wall geometry and the
nova read call (``efit_field_swap.crosscheck``). nova's read is confirmed
unchanged on LIUQE's own psi field with the per-loop TCV wall; LIUQE's stored
axis, X-point, flux levels and label are the reference.

Two facts about the shipped LIUQE stores drive the slice choice and are reported
beside every number:

* ``boundary.type`` is unset (-999999999) on every slice. The reference label
  follows the boundary outline's contact with the wall or a stored X-point;
  an X-point outside a wall-limited outline does not make it diverted.
* ``profiles_2d/psi`` is piecewise-constant in time: the 1703-slice store
  carries only 17 distinct psi grids, each frozen over one contiguous block, and
  the block starts are exactly ``SLICES``. A per-slice read inside a block would
  score one field repeatedly, so the snapshot takes one representative slice per
  distinct grid block.

The default reference is imas-efit's corrected full-pulse source
(``SOURCE_LIUQE_STORE``, DD 3.41.0), whose ``profiles_2d/psi`` is negated so the
file is internally COCOS-11: the stored grid and the stored scalars agree, and
the orientation guard admits every slice with no compensating sign. The curated
DD 4.1.0 window (``WINDOW_LIUQE_STORE``) is COCOS-17 and is used only to confirm
the correction against the full pulse across the DD conversion.
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

from nova.equilibrium.wall_mask import wall_units_from_ids
from nova.equilibrium.topology import TopologyPolicy
from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap import crosscheck
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.efit_field_swap.orientation import (
    ReferenceSignError,
    UndecidableSignError,
    check_axis_boundary_sign,
    grid_field,
    grid_value,
    ids_boundary_points,
    ids_cocos,
    ids_declaration,
    read_convention,
)

configure_dtypes()
assert jax.config.jax_enable_x64 is True

SOURCE_LIUQE_STORE = Path(
    "/home/ITER/mcintos/public/efit/validation/TCV/77549/liuqe-corrected"
)
"""imas-efit's corrected TCV 77549 LIUQE full pulse, DD 3.41.0, internally COCOS-11."""

WINDOW_LIUQE_STORE = Path(
    "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/TCV/77549/validation/liuqe"
)
"""The curated 50-slice window of the same pulse, DD 4.1.0, COCOS-17 throughout."""

WINDOW_SECONDS = (0.7930, 0.8390)
"""The time span the curated window covers, in the source's own time base."""

RESOLUTIONS = (16, 24, 32)
SLICES = (
    0,
    6,
    14,
    24,
    34,
    46,
    118,
    206,
    248,
    291,
    333,
    376,
    456,
    542,
    611,
    796,
    1257,
)
"""One representative slice per distinct psi-grid block of the 1703-slice store.

``profiles_2d/psi`` is piecewise-constant in time, so the store carries only
these seventeen distinct grids; the start index of each block is listed so every
distinct grid is guarded and scored exactly once."""

LIMITED_SLICE = 34
"""The limited-phase slice the coverage test reads; also the fallback selector
for the curated window where no sampled slice falls inside it."""

REFUSED_CASES = {
    "jt60sa-5slice-s5-133-134:A",
    "jt60sa-5slice-s5-133-134:B",
    "jt60sa-5slice-s1-15-16:B",
    "jt60sa-5slice-s2-135-136:A",
    "jt60sa-5slice-s2-135-136:B",
}

WALL_CONTACT_TOLERANCE = 0.002
XPOINT_CONTACT_TOLERANCE = 0.02
"""Metre tolerances for sampled wall and X-point contacts on an outline.

The corrected pulse's wall contacts are at most 0.00024 m for limited
outlines, whereas its diverted outlines stand at least 0.009 m off the wall.
The closest X-point to a diverted outline is within 0.011 m, while the nearest
wall-limited outline with a stored X-point remains more than 0.044 m away.
"""


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


def getattr_optional(structure, name):
    """A structure child that a DD version may not carry, or an empty list.

    ``ids_structure`` raises ``AttributeError`` for a name the loaded Data
    Dictionary has no node for, so a version-specific structure such as
    ``boundary/x_point`` (DD 3) is absent rather than empty in DD 4.
    """
    try:
        return getattr(structure, name)
    except AttributeError:
        return []


def slice_view(eq, index):
    """LIUQE slice ``index`` as the field, wall ring selection happens in the driver."""
    ts = eq.time_slice[index]
    p2 = ts.profiles_2d[0]
    rgrid = np.asarray(p2.grid.dim1, dtype=np.float64)
    zgrid = np.asarray(p2.grid.dim2, dtype=np.float64)
    psi = np.asarray(p2.psi, dtype=np.float64)
    gq = ts.global_quantities
    axis = (float(gq.magnetic_axis.r), float(gq.magnetic_axis.z))
    stored_xpoints = getattr_optional(ts.boundary, "x_point")
    xpoints = [
        (float(p.r), float(p.z))
        for p in stored_xpoints
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
        index=index,
        dd_version=str(eq.ids_properties.version_put.data_dictionary),
        plasma_current=float(gq.ip),
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


def _point_to_segments(points, outline):
    """Minimum Euclidean distance from points to a closed RZ polyline."""
    ring = _closed(np.asarray(outline, dtype=np.float64))
    start = ring[:-1]
    edge = ring[1:] - start
    squared = np.einsum("ij,ij->i", edge, edge)
    usable = squared > 0
    if not np.any(usable):
        raise ValueError("boundary polyline has no segments")
    start, edge, squared = start[usable], edge[usable], squared[usable]
    delta = np.asarray(points, dtype=np.float64)[:, None, :] - start[None, :, :]
    fraction = np.clip(np.einsum("pij,ij->pi", delta, edge) / squared, 0, 1)
    foot = start[None, :, :] + fraction[:, :, None] * edge[None, :, :]
    return float(np.min(np.linalg.norm(np.asarray(points)[:, None, :] - foot, axis=2)))


def liuqe_boundary_geometry(view, wall_rings):
    """Label the stored outline by wall and X-point proximity in metres."""
    outline = np.asarray(view["outline"], dtype=np.float64)
    if len(outline) < 3 or not wall_rings:
        raise ValueError("LIUQE boundary and wall geometry are required")
    wall_distance = min(_point_to_segments(outline, ring) for ring in wall_rings)
    xpoint_distance = min(
        (_point_to_segments(np.asarray([point]), outline) for point in view["xpoints"]),
        default=float("nan"),
    )
    if xpoint_distance <= XPOINT_CONTACT_TOLERANCE:
        label = "diverted"
    elif wall_distance <= WALL_CONTACT_TOLERANCE:
        label = "limited"
    else:
        raise ValueError(
            f"LIUQE outline has no certified wall or X-point contact: "
            f"wall {wall_distance:.5f} m, X-point {xpoint_distance:.5f} m"
        )
    return label, wall_distance, xpoint_distance


def _boundary_view(view, wall_rings):
    label, _, _ = liuqe_boundary_geometry(view, wall_rings)
    return {**view, "xpoints": []} if label == "limited" else view


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
    convention = read_convention(
        dd_version=view["dd_version"], plasma_current=view["plasma_current"]
    )
    check_axis_boundary_sign(
        convention,
        view["rgrid"],
        view["zgrid"],
        view["psi"],
        view["axis"],
        ids_boundary_points(_boundary_view(view, wall_rings)),
        scalar_axis_psi=view["psi_axis"],
        scalar_boundary_psi=view["psi_boundary"],
        slice_index=view["index"],
        declared=ids_declaration(view["dd_version"], view["plasma_current"]),
        field=field,
    )
    policy = TopologyPolicy()
    boundary_ring, obstacles = crosscheck.domain_rings(state)
    outcome = crosscheck.read_state(
        field, boundary_ring, obstacles, convention, policy, resolutions
    )
    coverage = outcome.coverage
    row = dict(
        resolution=outcome.resolution,
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
    label, wall_distance, xpoint_distance = liuqe_boundary_geometry(
        view, view["wall_rings"]
    )
    ref = dict(
        liuqe_time=view["time"],
        liuqe_axis=list(view["axis"]),
        liuqe_axis_psi_stored=view["psi_axis"],
        liuqe_boundary_psi_stored=view["psi_boundary"],
        liuqe_xpoints=[list(p) for p in view["xpoints"]],
        liuqe_label=label,
        liuqe_wall_distance=wall_distance,
        liuqe_xpoint_distance=xpoint_distance,
        liuqe_axis_boundary_span=axis_boundary_span(view["axis"], view["outline"]),
    )
    ref["liuqe_axis_psi_grid"] = _value(field, view["axis"])
    if label == "diverted":
        ref["liuqe_boundary_psi_grid"] = _grid_level(field, view["xpoints"])
    elif len(view["outline"]):
        ref["liuqe_boundary_psi_grid"] = _grid_level(field, view["outline"])
    else:
        ref["liuqe_boundary_psi_grid"] = float("nan")
    return ref


def guard_verdict(view):
    """The orientation guard's verdict for one slice, with no compensation.

    Returns a row naming the declaration, the measured grid and scalar spans,
    and the outcome: ``agrees`` when both spans carry the sign the declared
    COCOS requires for the slice's plasma current, else ``contradicts`` or
    ``undecidable`` carrying the refusal's own message. Nothing here negates,
    offsets or otherwise compensates the stored flux.
    """
    convention = read_convention(
        dd_version=view["dd_version"], plasma_current=view["plasma_current"]
    )
    boundary_view = (
        _boundary_view(view, view["wall_rings"]) if "wall_rings" in view else view
    )
    point = ids_boundary_points(boundary_view)
    field = grid_field(view["rgrid"], view["zgrid"], view["psi"])
    axis_value = grid_value(
        view["rgrid"], view["zgrid"], view["psi"], view["axis"], field=field
    )
    boundary_value = (
        None
        if point is None
        else float(
            np.mean(
                [
                    grid_value(
                        view["rgrid"], view["zgrid"], view["psi"], p, field=field
                    )
                    for p in point
                ]
            )
        )
    )
    row = dict(
        index=view["index"],
        dd_version=view["dd_version"],
        cocos=ids_cocos(view["dd_version"]),
        plasma_current=view["plasma_current"],
        declared_sigma=int(np.sign(float(convention.sigma))),
        boundary_kind="xpoint" if boundary_view["xpoints"] else "outline",
        scalar_span=view["psi_axis"] - view["psi_boundary"],
        grid_axis_psi=axis_value,
        grid_boundary_psi=boundary_value,
        grid_span=None if boundary_value is None else axis_value - boundary_value,
        outcome="",
        refused=False,
        reason="",
    )
    try:
        row["measured_signs"] = check_axis_boundary_sign(
            convention,
            view["rgrid"],
            view["zgrid"],
            view["psi"],
            view["axis"],
            point,
            scalar_axis_psi=view["psi_axis"],
            scalar_boundary_psi=view["psi_boundary"],
            slice_index=view["index"],
            declared=ids_declaration(view["dd_version"], view["plasma_current"]),
            field=field,
        )
    except UndecidableSignError as error:
        row["outcome"], row["refused"], row["reason"] = "undecidable", True, str(error)
    except ReferenceSignError as error:
        row["outcome"], row["refused"], row["reason"] = "contradicts", True, str(error)
    else:
        row["outcome"] = "agrees"
    return row


def grid_block_starts(eq):
    """Indices at which ``profiles_2d/psi`` changes: one per distinct grid block."""
    starts = []
    previous = None
    for index in range(len(eq.time_slice)):
        psi = np.asarray(eq.time_slice[index].profiles_2d[0].psi, dtype=np.float64)
        if previous is None or not np.array_equal(psi, previous):
            starts.append(index)
            previous = psi
    return starts


def grid_motion(eq, slices):
    """Per-slice grid motion against slice 0, beside the scalars' own motion.

    ``grid_motion`` is ``max |psi(t) - psi(t0)|`` over the slice's own grid
    array, so it is zero exactly where the grid is a byte-copy of slice 0's and
    grows with the width of the change where it is not. The scalar motions are
    the same measure on ``psi_axis`` and ``psi_boundary``, which a frozen-grid
    slice still moves.
    """
    times = np.asarray(eq.time, dtype=np.float64)
    base = np.asarray(eq.time_slice[0].profiles_2d[0].psi, dtype=np.float64)
    base_axis = float(eq.time_slice[0].global_quantities.psi_axis)
    base_boundary = float(eq.time_slice[0].global_quantities.psi_boundary)
    rows = []
    for index in slices:
        psi = np.asarray(eq.time_slice[index].profiles_2d[0].psi, dtype=np.float64)
        gq = eq.time_slice[index].global_quantities
        rows.append(
            dict(
                index=index,
                time=float(times[index]),
                grid_motion=float(np.max(np.abs(psi - base))),
                axis_motion=abs(float(gq.psi_axis) - base_axis),
                boundary_motion=abs(float(gq.psi_boundary) - base_boundary),
                psi_axis=float(gq.psi_axis),
                psi_boundary=float(gq.psi_boundary),
            )
        )
    return rows


def window_selection(source_times, window_times, slices):
    """Curated window slices matched to the source's sampled slices.

    A ``SLICES`` entry whose time falls inside ``WINDOW_SECONDS`` selects the
    curated window slice nearest to it. Where no entry falls inside the window
    the selection falls back to the single window time nearest
    ``LIMITED_SLICE``, and each row is marked ``matched: False`` so a fallback
    is never read as a time match.
    """
    picks = []
    for index in slices:
        time = float(source_times[index])
        if WINDOW_SECONDS[0] <= time <= WINDOW_SECONDS[1]:
            picks.append((index, int(np.argmin(np.abs(window_times - time))), True))
    if not picks:
        time = float(source_times[LIMITED_SLICE])
        picks.append((None, int(np.argmin(np.abs(window_times - time))), False))
    return picks


def window_agreement(source_view, window_view, *, source_index, window_index, matched):
    """Whether the curated window agrees with the corrected source at one time.

    The DD 3.41.0 to 4.1.0 conversion negates every flux field, so the source's
    COCOS-11 values are negated to their COCOS-17 equivalents before the
    comparison. The span (axis minus boundary) is invariant under an additive
    gauge constant and is compared as a ratio; the grid, whose coordinates the
    two files share, is compared as a max absolute difference of the converted
    source grid against the window grid.
    """
    source_span = source_view["psi_axis"] - source_view["psi_boundary"]
    converted_span = -source_span
    window_span = window_view["psi_axis"] - window_view["psi_boundary"]
    converted_grid = -np.asarray(source_view["psi"], dtype=np.float64)
    window_grid = np.asarray(window_view["psi"], dtype=np.float64)
    return dict(
        source_index=source_index,
        window_index=window_index,
        matched=matched,
        source_time=source_view["time"],
        window_time=window_view["time"],
        source_psi_axis=source_view["psi_axis"],
        source_psi_boundary=source_view["psi_boundary"],
        converted_psi_axis=-source_view["psi_axis"],
        converted_psi_boundary=-source_view["psi_boundary"],
        window_psi_axis=window_view["psi_axis"],
        window_psi_boundary=window_view["psi_boundary"],
        converted_span=converted_span,
        window_span=window_span,
        span_ratio=window_span / converted_span if converted_span else None,
        grid_max_abs_diff=float(np.max(np.abs(converted_grid - window_grid))),
        window_guard=guard_verdict(window_view),
        source_guard=guard_verdict(source_view),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liuqe", default=str(SOURCE_LIUQE_STORE))
    parser.add_argument("--wall", required=True)
    parser.add_argument("--slices", required=True, help="comma-separated indices")
    parser.add_argument("--out", required=True)
    parser.add_argument("--guard-out")
    parser.add_argument("--grid-motion-out")
    parser.add_argument("--window", default="")
    parser.add_argument("--window-out", default="")
    parser.add_argument("--field-swap-out")
    parser.add_argument("--resolutions", default="16,24,32")
    args = parser.parse_args()
    print("nova.__file__", __import__("nova").__file__, flush=True)

    slices = [int(s) for s in args.slices.split(",") if s.strip()]
    resolutions = tuple(int(r) for r in args.resolutions.split(","))
    wall_rings = read_wall(args.wall)
    db = imas.DBEntry(f"imas:hdf5?path={args.liuqe}", "r", dd_version="3.41.0")
    eq = db.get("equilibrium")
    if eq.ids_properties.homogeneous_time != 1:
        raise ValueError("equilibrium time base is not homogeneous")
    guard_rows = [
        guard_verdict({**slice_view(eq, index), "wall_rings": wall_rings})
        for index in slices
    ]
    for row in guard_rows:
        print(
            "GUARD",
            row["index"],
            f"t={float(np.asarray(eq.time)[row['index']]):.6f}",
            f"cocos={row['cocos']}",
            f"outcome={row['outcome']}",
            f"grid_span={row['grid_span']!r}",
            f"scalar_span={row['scalar_span']:.6e}",
            row["reason"],
            flush=True,
        )
    if args.guard_out:
        guard_out = Path(args.guard_out)
        guard_out.parent.mkdir(parents=True, exist_ok=True)
        guard_out.write_text(json.dumps({"rows": guard_rows}, indent=2) + "\n")
    print(
        "GUARD_ROWS",
        len(guard_rows),
        "AGREES",
        sum(row["outcome"] == "agrees" for row in guard_rows),
        "REFUSED",
        sum(row["refused"] for row in guard_rows),
        flush=True,
    )
    motion = grid_motion(eq, slices)
    for row in motion:
        print(
            "GRID_MOTION",
            row["index"],
            f"t={row['time']:.6f}",
            f"grid={row['grid_motion']:.6e}",
            f"axis={row['axis_motion']:.6e}",
            f"boundary={row['boundary_motion']:.6e}",
            flush=True,
        )
    if args.grid_motion_out:
        motion_out = Path(args.grid_motion_out)
        motion_out.parent.mkdir(parents=True, exist_ok=True)
        motion_out.write_text(json.dumps({"rows": motion}, indent=2) + "\n")
    starts = grid_block_starts(eq)
    print("GRID_BLOCKS", len(starts), ",".join(str(s) for s in starts), flush=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for index in slices:
            view = slice_view(eq, index)
            view["wall_rings"] = wall_rings
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
    if args.window:
        window_pass(eq, args.window, slices, args.window_out)


def _ratio_text(value):
    """A ratio formatted for a log line, or ``None`` where it is undefined."""
    return "None" if value is None else format(value, ".6f")


def window_pass(eq, window_path, slices, out_path=""):
    """Guard the curated window and compare it with the corrected source.

    Every selected window slice is put through the guard under its own written
    DD version (4.1.0, COCOS-17), and the source slice nearest it in time is the
    comparison partner, converted to COCOS-17 by negating its flux.
    """
    source_times = np.asarray(eq.time, dtype=np.float64)
    db = imas.DBEntry(f"imas:hdf5?path={window_path}", "r", dd_version="4.1.0")
    try:
        window = db.get("equilibrium")
        window_times = np.asarray(window.time, dtype=np.float64)
        rows = []
        for source_index, window_index, matched in window_selection(
            source_times, window_times, slices
        ):
            nearest = (
                source_index
                if source_index is not None
                else int(np.argmin(np.abs(source_times - window_times[window_index])))
            )
            row = window_agreement(
                slice_view(eq, nearest),
                slice_view(window, window_index),
                source_index=nearest,
                window_index=window_index,
                matched=matched,
            )
            rows.append(row)
            print(
                "WINDOW",
                row["window_index"],
                f"t={row['window_time']:.6f}",
                f"matched={row['matched']}",
                f"window_guard={row['window_guard']['outcome']}",
                f"source_guard={row['source_guard']['outcome']}",
                f"converted_span={row['converted_span']:.6e}",
                f"window_span={row['window_span']:.6e}",
                f"span_ratio={_ratio_text(row['span_ratio'])}",
                f"grid_max_abs_diff={row['grid_max_abs_diff']:.6e}",
                flush=True,
            )
    finally:
        db.close()
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print("WINDOW_ROWS", len(rows), flush=True)
    return rows


if __name__ == "__main__":
    main()
