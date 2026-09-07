"""Measure every sampled limiter anchor on bounded MAST label frames.

The production labeller is replayed without changing its selection.  For each
requested frame, every sampled wall node is treated as the winner of its
centred three-node quadratic bracket.  The resulting fitted wall flux is then
traced from the public read's magnetic axis with the same 64-angle first-crossing
surface tracer used by the persisted internal-geometry record.

The receipt distinguishes the public writer read from a diagnostic private read
made with the wall shadow derived at the same terminal state.  This is an
observation only: neither read is substituted into the solve or the writer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from nova.equilibrium.connectivity_boundary import _points_inside_polygon
from nova.equilibrium.flux_surface_geometry import (
    SurfaceGeometryError,
    _refine_axis,
    _trace_surfaces,
)
from nova.geometry import select
from nova.media import poloidal
from scripts.labeller_batch import shard


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/playable-forward-solve/limited-anchor"
SHOT = 27079
FRAME_ROWS = tuple(range(11, 18))
ANGLE_COUNT = 64


@dataclass(frozen=True)
class _CapturedRead:
    """Terminal state and the two topology reads inspected by the benchmark."""

    state: np.ndarray
    public_topology: Any
    private_topology: Any
    private_wall_mask: np.ndarray


def _strict(value: Any) -> Any:
    """Convert arrays and non-finite scalars into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one strict, human-readable evidence artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_revision() -> str:
    """Return the exact checkout revision measured by this process."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _node_index(wall: np.ndarray, point: np.ndarray) -> int:
    """Return the first wall node nearest a fitted wall position."""
    return int(np.argmin(np.sum((wall - point) ** 2, axis=1)))


def _topology_read(topology: Any, wall: np.ndarray) -> dict[str, Any]:
    """Serialize the boundary and fitted wall anchor of one topology read."""
    anchor = np.asarray(topology.wall_point, dtype=np.float64)
    node = _node_index(wall, anchor)
    return {
        "selected_anchor_node": node,
        "selected_anchor_node_position_m": wall[node],
        "fitted_anchor_position_m": anchor,
        "wall_flux_wb": float(np.asarray(topology.wall_point_flux)),
        "boundary_position_m": np.asarray(topology.boundary, dtype=np.float64),
        "boundary_flux_wb": float(np.asarray(topology.boundary_flux)),
        "axis_position_m": np.asarray(topology.axis, dtype=np.float64),
        "axis_flux_wb": float(np.asarray(topology.axis_flux)),
        "x_point_position_m": np.asarray(topology.x_point, dtype=np.float64),
        "x_point_flux_wb": float(np.asarray(topology.x_point_flux)),
    }


def _capture_writer_reads(
    output: Path,
) -> tuple[Any, dict[str, Any], list[_CapturedRead]]:
    """Replay the bounded writer and retain its terminal topology inputs."""
    replay = output / "writer-replay"
    replay.mkdir(parents=True, exist_ok=False)
    prepared = shard.prepare_labeller()
    prepared = shard._prepared_with_polarity(prepared, shard._shot_polarity(SHOT))
    captured: list[_CapturedRead] = []
    original = shard._forward_receipt

    def record(prepared_labeller, result, **kwargs):
        receipt = original(prepared_labeller, result, **kwargs)
        operator = prepared_labeller.profile.operator
        requested = kwargs["requested_class"]
        state = result.state
        physical = state[: operator.physical_node_number]
        shadow = operator.residual_shadow_mask(state, requested)
        private_wall_mask = np.asarray(
            operator._previous_wall_shadow(shadow), dtype=bool
        )
        _masks, private, _connected, admitted = operator._fixed_design_read(
            physical,
            requested,
            private_wall_node_mask=private_wall_mask,
        )
        if not bool(np.asarray(admitted)):
            raise RuntimeError("diagnostic private read did not admit its axis")
        captured.append(
            _CapturedRead(
                state=np.asarray(state, dtype=np.float64),
                public_topology=receipt.terminal_state.topology,
                private_topology=private,
                private_wall_mask=private_wall_mask,
            )
        )
        return receipt

    shard._forward_receipt = record
    try:
        _programs, manifest = shard.label_shot(
            prepared,
            SHOT,
            replay,
            programs=shard.LabellerPrograms(),
            include_raster=False,
            condition_on_guard_failure=False,
            setup_wall_seconds=prepared.setup_wall_seconds,
            max_slices=16,
        )
    finally:
        shard._forward_receipt = original

    written = [row for row in manifest["slices"] if row.get("written")]
    if len(written) != len(captured):
        raise RuntimeError(
            f"captured {len(captured)} topology reads for {len(written)} written rows"
        )
    return prepared, manifest, captured


def _candidate_bracket(
    wall: np.ndarray,
    wall_flux: np.ndarray,
    node: int,
    polarity: int,
) -> dict[str, Any]:
    """Fit the centred wall bracket associated with one sampled node."""
    count = wall.shape[0]
    indices = np.asarray([(node - 1) % count, node, (node + 1) % count])
    cluster = wall[indices]
    flux_cluster = wall_flux[indices]
    length = select.length_2d(cluster[:, 0], cluster[:, 1])
    if length[-1] == 0.0 or np.unique(cluster, axis=0).shape[0] < 3:
        return {
            "node": node,
            "node_position_m": wall[node],
            "sampled_flux_wb": float(wall_flux[node]),
            "trace_error": "the centred bracket contains duplicate coordinates",
        }
    coefficients = select.host_quadratic_wall(length, flux_cluster)
    coordinate = float(select.wall_length(coefficients))
    anchor_r, anchor_z = select.wall_coordinate(
        coordinate,
        cluster[:, 0],
        cluster[:, 1],
        length,
    )
    fitted_flux = (
        coefficients[0] * coordinate**2 + coefficients[1] * coordinate + coefficients[2]
    )
    tangent = bool(
        0.0 <= coordinate <= float(length[-1])
        and int(polarity) * float(coefficients[0]) < 0.0
    )
    return {
        "node": node,
        "node_position_m": wall[node],
        "sampled_flux_wb": float(wall_flux[node]),
        "fitted_anchor_position_m": [float(anchor_r), float(anchor_z)],
        "fitted_anchor_flux_wb": float(fitted_flux),
        "quadratic_curvature": float(coefficients[0]),
        "stationary_coordinate_m": coordinate,
        "bracket_length_m": float(length[-1]),
        "tangent_at_wall_node": tangent,
    }


def _tracer(
    prepared: Any,
    state: np.ndarray,
    axis: np.ndarray,
) -> tuple[RectBivariateSpline, tuple[float, float], float]:
    """Build the public geometry reader's interpolant and refined axis."""
    lattice = prepared.profile.lattice
    radius = np.asarray(lattice.radius, dtype=np.float64)
    height = np.asarray(lattice.height, dtype=np.float64)
    count = radius.size * height.size
    values = np.asarray(state[:count], dtype=np.float64).reshape(
        radius.size, height.size
    )
    interpolant = RectBivariateSpline(radius, height, values, kx=3, ky=3, s=0)
    centre = _refine_axis(
        interpolant,
        (float(axis[0]), float(axis[1])),
        radius,
        height,
    )
    return interpolant, centre, float(interpolant.ev(*centre))


def _trace_boundary(
    prepared: Any,
    interpolant: RectBivariateSpline,
    centre: tuple[float, float],
    axis_flux: float,
    boundary_flux: float,
) -> np.ndarray:
    """Trace one 64-vertex first-crossing contour at an absolute flux."""
    span = float(boundary_flux) - float(axis_flux)
    if span == 0.0 or not np.isfinite(span):
        raise SurfaceGeometryError("the candidate has no finite axis-boundary span")
    lattice = prepared.profile.lattice
    traced = _trace_surfaces(
        interpolant,
        centre,
        axis_flux,
        span,
        np.asarray([1.0]),
        np.asarray(lattice.radius, dtype=np.float64),
        np.asarray(lattice.height, dtype=np.float64),
        ANGLE_COUNT,
    )
    contour = np.column_stack((traced.radius[:, 0], traced.height[:, 0]))
    if not np.all(np.isfinite(contour)):
        raise SurfaceGeometryError("the traced candidate contour is not finite")
    return contour


def _polygon_area(points: np.ndarray) -> float:
    """Return the absolute shoelace area of one ordered closed contour."""
    radius = points[:, 0]
    height = points[:, 1]
    return float(
        0.5
        * abs(np.dot(radius, np.roll(height, -1)) - np.dot(height, np.roll(radius, -1)))
    )


def _contour_metrics(
    contour: np.ndarray,
    wall: np.ndarray,
    reconstruction_area: float,
) -> dict[str, Any]:
    """Measure wall containment and area against the reconstruction."""
    inside = np.asarray(
        _points_inside_polygon(contour[:, 0], contour[:, 1], wall[:, 0], wall[:, 1]),
        dtype=bool,
    )
    area = _polygon_area(contour)
    return {
        "inside_vertex_count": int(np.count_nonzero(inside)),
        "boundary_vertex_count": int(contour.shape[0]),
        "containment_fraction": float(np.mean(inside)),
        "enclosed_area_m2": area,
        "enclosed_area_ratio": area / float(reconstruction_area),
    }


def _reconstruction_boundary(group: Any, row: int) -> np.ndarray:
    """Return the finite reconstruction boundary vertices for one row."""
    radius = np.asarray(group["lcfs_r"][row], dtype=np.float64)
    height = np.asarray(group["lcfs_z"][row], dtype=np.float64)
    finite = np.isfinite(radius) & np.isfinite(height)
    boundary = np.column_stack((radius[finite], height[finite]))
    if boundary.shape[0] < 3:
        raise RuntimeError(f"row {row} has no finite reconstruction boundary")
    return boundary


def _measure_frame(
    prepared: Any,
    group: Any,
    manifest_row: dict[str, Any],
    captured: _CapturedRead,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure all wall-node candidates and both reads for one frame."""
    row = int(manifest_row["row"])
    operator = prepared.profile.operator
    wall = np.asarray(operator.wall.coordinate, dtype=np.float64)
    physical = captured.state[: operator.physical_node_number]
    _grid_flux, wall_flux = operator.topology.split_flux_map(physical)
    wall_flux = np.asarray(wall_flux, dtype=np.float64)
    public = _topology_read(captured.public_topology, wall)
    private = _topology_read(captured.private_topology, wall)
    interpolant, centre, axis_flux = _tracer(
        prepared, captured.state, public["axis_position_m"]
    )
    reconstruction_area = float(group["plasma_area"][row])
    if not np.isfinite(reconstruction_area) or reconstruction_area <= 0.0:
        raise RuntimeError(f"row {row} has no positive reconstruction area")

    contours: dict[int, np.ndarray] = {}
    candidates = []
    for node in range(wall.shape[0]):
        candidate = _candidate_bracket(wall, wall_flux, node, int(operator.polarity))
        if "trace_error" not in candidate:
            try:
                contour = _trace_boundary(
                    prepared,
                    interpolant,
                    centre,
                    axis_flux,
                    candidate["fitted_anchor_flux_wb"],
                )
                contours[node] = contour
                candidate.update(_contour_metrics(contour, wall, reconstruction_area))
            except (SurfaceGeometryError, ValueError, np.linalg.LinAlgError) as error:
                candidate["trace_error"] = f"{type(error).__name__}: {error}"
        candidates.append(candidate)

    read_contours: dict[str, np.ndarray] = {}
    for name, reading in (("public", public), ("private", private)):
        try:
            contour = _trace_boundary(
                prepared,
                interpolant,
                centre,
                axis_flux,
                reading["boundary_flux_wb"],
            )
            read_contours[name] = contour
            reading.update(_contour_metrics(contour, wall, reconstruction_area))
        except SurfaceGeometryError as error:
            reading["contour_error"] = f"{type(error).__name__}: {error}"

    contained = [
        candidate
        for candidate in candidates
        if candidate.get("containment_fraction") == 1.0
        and candidate.get("enclosed_area_ratio") is not None
    ]
    if not contained:
        raise RuntimeError(f"row {row} has no fully contained candidate contour")
    nearest = min(
        contained,
        key=lambda candidate: abs(candidate["enclosed_area_ratio"] - 1.0),
    )
    for candidate in candidates:
        candidate["selected_by_public_read"] = (
            candidate["node"] == public["selected_anchor_node"]
        )
        candidate["selected_by_private_read"] = (
            candidate["node"] == private["selected_anchor_node"]
        )
        candidate["nearest_fully_contained_ratio_one"] = (
            candidate["node"] == nearest["node"]
        )

    reconstruction_boundary = _reconstruction_boundary(group, row)
    return (
        {
            "manifest_row": row,
            "time_s": float(manifest_row["time"]),
            "converged": bool(manifest_row["converged"]),
            "qualified": bool(manifest_row["qualified"]),
            "reconstruction_area_m2": reconstruction_area,
            "public_read": public,
            "private_active_partition_read": private,
            "private_wall_masked_node_count": int(
                np.count_nonzero(captured.private_wall_mask)
            ),
            "nearest_fully_contained_candidate_node": int(nearest["node"]),
            "nearest_fully_contained_candidate_ratio": float(
                nearest["enclosed_area_ratio"]
            ),
            "candidates": candidates,
        },
        {
            "wall": wall,
            "field": np.asarray(captured.state, dtype=np.float64)[
                : prepared.profile.lattice.node_count
            ].reshape(
                len(prepared.profile.lattice.radius),
                len(prepared.profile.lattice.height),
            ),
            "nearest_boundary": contours[int(nearest["node"])],
            "reconstruction_boundary": reconstruction_boundary,
            **{f"{name}_boundary": contour for name, contour in read_contours.items()},
        },
    )


def _draw_frame(
    prepared: Any,
    frame: dict[str, Any],
    geometry: dict[str, np.ndarray],
    output: Path,
) -> None:
    """Draw one self-contained wall-anchor diagnosis panel."""
    figure, axes = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
    wall = geometry["wall"]
    public = frame["public_read"]
    candidates = frame["candidates"]
    lattice = prepared.profile.lattice

    levels = np.linspace(public["axis_flux_wb"], public["boundary_flux_wb"], 12)[1:-1]
    poloidal.draw_flux_contours(
        axes,
        lattice.radius,
        lattice.height,
        geometry["field"].T,
        levels,
        color="#b8c2cc",
        linewidth=0.45,
    )
    poloidal.draw_wall(axes, wall[:, 0], wall[:, 1], color="#202020")
    poloidal.draw_boundary(
        axes,
        geometry["reconstruction_boundary"][:, 0],
        geometry["reconstruction_boundary"][:, 1],
        color="#222222",
        linestyle="--",
        linewidth=1.4,
        label="reconstruction",
    )
    for name, color, linestyle, label in (
        ("public", "#cc3344", "solid", "public writer read"),
        ("private", "#e69500", ":", "private shadow read"),
    ):
        contour = geometry.get(f"{name}_boundary")
        if contour is None:
            reading = frame[
                "public_read" if name == "public" else "private_active_partition_read"
            ]
            poloidal.draw_flux_contours(
                axes,
                lattice.radius,
                lattice.height,
                geometry["field"].T,
                [reading["boundary_flux_wb"]],
                color=color,
                linewidth=2.0,
            )
        else:
            poloidal.draw_boundary(
                axes,
                contour[:, 0],
                contour[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )
    poloidal.draw_boundary(
        axes,
        geometry["nearest_boundary"][:, 0],
        geometry["nearest_boundary"][:, 1],
        color="#008b72",
        linestyle="-.",
        linewidth=1.7,
        label="nearest contained ratio",
    )
    poloidal.draw_nulls(
        axes,
        public["axis_position_m"],
        np.atleast_2d(public["x_point_position_m"]),
        contain=wall,
    )

    anchor_points = np.asarray(
        [candidate["node_position_m"] for candidate in candidates], dtype=float
    )
    ratios = np.asarray(
        [candidate.get("enclosed_area_ratio", np.nan) for candidate in candidates],
        dtype=float,
    )
    axes.scatter(
        anchor_points[:, 0],
        anchor_points[:, 1],
        c=np.abs(ratios - 1.0),
        cmap="viridis_r",
        s=26,
        edgecolors="white",
        linewidths=0.35,
        zorder=8,
        label="sampled wall candidates",
    )
    selected_node = int(public["selected_anchor_node"])
    nearest_node = int(frame["nearest_fully_contained_candidate_node"])
    axes.scatter(
        *wall[selected_node],
        marker="s",
        s=90,
        color="#cc3344",
        edgecolors="black",
        linewidths=0.7,
        zorder=9,
    )
    axes.scatter(
        *wall[nearest_node],
        marker="D",
        s=75,
        color="#008b72",
        edgecolors="black",
        linewidths=0.7,
        zorder=9,
    )
    axes.set_aspect("equal", adjustable="box")
    margin = 0.08
    axes.set_xlim(
        float(np.min(wall[:, 0]) - margin), float(np.max(wall[:, 0]) + margin)
    )
    axes.set_ylim(
        float(np.min(wall[:, 1]) - margin), float(np.max(wall[:, 1]) + margin)
    )
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_title(
        f"MAST {SHOT}, row {frame['manifest_row']}, "
        f"t = {1e3 * frame['time_s']:.0f} ms\n"
        f"public node {selected_node}: containment "
        f"{public.get('inside_vertex_count', 'unclosed')}/"
        f"{public.get('boundary_vertex_count', 'unclosed')}, "
        f"area ratio "
        f"{public.get('enclosed_area_ratio', float('nan')):.4f}; "
        f"nearest contained node {nearest_node}: "
        f"{frame['nearest_fully_contained_candidate_ratio']:.4f}"
    )
    axes.legend(loc="lower center", fontsize=8, frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def measure(output: Path) -> dict[str, Any]:
    """Run the bounded replay and emit the complete diagnosis receipt."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared, manifest, captured = _capture_writer_reads(output)
    group = shard.zarr.open_group(str(shard.SHOT_STORE / f"{SHOT}.zarr"), mode="r")[
        "efm"
    ]
    written = [row for row in manifest["slices"] if row.get("written")]
    aligned = {
        int(row["row"]): (row, capture)
        for row, capture in zip(written, captured, strict=True)
    }
    missing = sorted(set(FRAME_ROWS) - set(aligned))
    if missing:
        raise RuntimeError(f"bounded replay did not write requested rows {missing}")

    frames = []
    for row in FRAME_ROWS:
        manifest_row, capture = aligned[row]
        frame, geometry = _measure_frame(prepared, group, manifest_row, capture)
        panel = output / f"row-{row:02d}-{1e3 * frame['time_s']:.0f}ms.png"
        _draw_frame(prepared, frame, geometry, panel)
        frame["panel"] = str(panel.relative_to(ROOT))
        frames.append(frame)
        public = frame["public_read"]
        containment = (
            f"{public['inside_vertex_count']}/{public['boundary_vertex_count']}"
            if "inside_vertex_count" in public
            else "unclosed"
        )
        ratio = public.get("enclosed_area_ratio")
        ratio_text = "unclosed" if ratio is None else f"{ratio:.10f}"
        print(
            f"row={row} time_ms={1e3 * frame['time_s']:.0f} "
            f"public_anchor_node={public['selected_anchor_node']} "
            f"public_anchor_rz={public['selected_anchor_node_position_m']} "
            f"containment={containment} "
            f"area_ratio={ratio_text} "
            f"nearest_contained_node="
            f"{frame['nearest_fully_contained_candidate_node']} "
            f"nearest_ratio="
            f"{frame['nearest_fully_contained_candidate_ratio']:.10f}",
            flush=True,
        )

    payload = {
        "schema": "limited-wall-anchor-containment-diagnosis",
        "nova_revision": _source_revision(),
        "shot": SHOT,
        "frame_rows": list(FRAME_ROWS),
        "candidate_definition": (
            "every sampled wall node with the stationary point and flux of its "
            "centred three-node quadratic bracket"
        ),
        "tangency_definition": (
            "stationary point lies within its two wall segments and the signed "
            "quadratic curvature is a maximum; no observed-value tolerance"
        ),
        "contour_definition": (
            "64-angle first outward crossing from the public read's refined "
            "magnetic axis, using the persisted internal-geometry tracer"
        ),
        "containment_definition": (
            "production wall-polygon test including its fixed arithmetic boundary "
            "rule; no structure-name exclusion"
        ),
        "writer_replay": {
            "written_frame_count": int(manifest["written_slice_count"]),
            "converged_frame_count": int(manifest["converged_slice_count"]),
            "manifest": str(
                (output / "writer-replay" / f"{SHOT}.manifest.json").relative_to(ROOT)
            ),
        },
        "frames": frames,
    }
    _write_json(output / "limited-anchor-containment.json", payload)
    return payload


def main() -> None:
    """Parse the output path and run the bounded diagnosis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = measure(arguments.output)
    print(
        json.dumps(
            {
                "nova_revision": receipt["nova_revision"],
                "shot": receipt["shot"],
                "frame_count": len(receipt["frames"]),
                "written_frame_count": receipt["writer_replay"]["written_frame_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
