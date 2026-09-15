"""Render the limited-row residual-shadow comparison from persisted states.

The driver reads three paired Solovev terminal-state receipts and renders their
shadowed and unmasked flux beside the analytic field.  It does not construct a
forward operator or invoke a solve.  Shared absolute contour levels come only
from each row's persisted analytic field, and the weak-row panel marks the raw
private carriers that the former residual shadow excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes

ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/limited-shadow"
DEFAULT_FIGURE = FIGURE_ROOT / "lfs-contours-before-after.png"
DEFAULT_RECEIPT = FIGURE_ROOT / "lfs-contours-before-after-receipt.json"
DEFAULT_ORACLE_FIGURE = FIGURE_ROOT / "oracle-overlay-error.png"
DEFAULT_ORACLE_RECEIPT = FIGURE_ROOT / "oracle-overlay-error-receipt.json"
DEFAULT_ORACLE_REPORT = (
    Path.home()
    / ".config/reckon/crew/reports/nova/s19-handoff/oracle-overlay"
    / "report.md"
)
SOLVE_RECEIPT = FIGURE_ROOT / "solve-receipt.json"
SHADOW_RECEIPT = FIGURE_ROOT / "receipt.json"
SOURCE_PART_ROOT = (
    ROOT / "docs/figures/cut-cell-current-attribution/wholecell-rows/parts/rows"
)
UNMASKED_PART_ROOT = FIGURE_ROOT / "solve-parts/chord"
SOLVE_PART_ROOT = FIGURE_ROOT / "solve-parts"
WEAK_REPRODUCTION_PART = (
    Path.home()
    / ".config/reckon/crew/reports/nova/s19-handoff/limited-shadow"
    / "reproduction/parts/weak-rotation-reactor-static-cells-2500.json"
)
SOURCE_REVISION = "3082be61cb6dd4a4d9cae1889993717c4e63e58c"
ROWS = (
    ("weak-rotation-reactor-static", "weak rotation"),
    ("moderate-rotation-conventional-static", "moderate rotation"),
    ("strong-rotation-compact-static", "strong rotation"),
)
ORACLE_REQUIRED_ROWS = (
    ("weak-rotation-reactor-static", -1000, "weak 1000 whole-cell"),
    ("moderate-rotation-conventional-static", -1000, "moderate 1000 whole-cell"),
    ("strong-rotation-compact-static", -1000, "strong 1000 whole-cell"),
    ("diverted-single-null", -500, "single-null 500 control"),
)
SIGNED_DIFFERENCE_LEVELS = np.asarray(
    [-1e-1, -1e-2, -1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    dtype=np.float64,
)
QUOTED_ROUNDOFF_FLOOR = 1e-14


@dataclass(frozen=True)
class TerminalState:
    """One persisted terminal flux and its analytic oracle."""

    path: Path
    source_mode: str
    display_name: str
    payload: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _part_path(root: Path, case: str) -> Path:
    return root / f"{case}-production-route-cells-1000.json"


def _metric_rows() -> dict[str, dict[str, Any]]:
    receipt = _read_json(SOLVE_RECEIPT)
    rows = {
        row["case"]: row
        for row in receipt["rows"]
        if row["case"] in {case for case, _label in ROWS}
        and row["requested_cells"] == -1000
    }
    expected = {case for case, _label in ROWS}
    if set(rows) != expected:
        raise ValueError(
            f"solve receipt rows {sorted(rows)} do not match {sorted(expected)}"
        )
    return rows


def _weak_shadow_carriers() -> list[dict[str, Any]]:
    receipt = _read_json(SHADOW_RECEIPT)
    rows = [row for row in receipt["rows"] if row["requested_cells"] == -1000]
    if len(rows) != 1:
        raise ValueError(f"expected one weak 1000 shadow census, found {len(rows)}")
    carriers = rows[0]["terminal"]["raw_private_carriers"]
    expected = int(rows[0]["terminal"]["raw_private_count"])
    if len(carriers) != expected or expected == 0:
        raise ValueError(
            f"weak 1000 carrier census has {len(carriers)} rows for count {expected}"
        )
    return carriers


def _terminal_flux_persisted(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    render_data = payload.get("render_data")
    return bool(
        isinstance(render_data, dict)
        and isinstance(render_data.get("terminal_flux_wb"), list)
        and render_data["terminal_flux_wb"]
    )


def _terminal_amplitude(payload: dict[str, Any]) -> float:
    samples = payload["solver"]["lambda_amplitude_history"]["samples"]
    terminal = [sample for sample in samples if sample.get("state") == "terminal"]
    sample = terminal[-1] if terminal else samples[-1]
    amplitude = float(sample["amplitude"])
    if not np.isfinite(amplitude):
        raise ValueError(f"non-finite terminal amplitude in {payload['case']}")
    return amplitude


def _nested_scalar(payload: dict[str, Any], names: tuple[str, ...]) -> float | None:
    stack: list[dict[str, Any]] = [payload]
    while stack:
        item = stack.pop()
        for name in names:
            value = item.get(name)
            if isinstance(value, int | float) and np.isfinite(value) and value > 0:
                return float(value)
        stack.extend(value for value in item.values() if isinstance(value, dict))
    return None


def _roundoff_floor(payload: dict[str, Any]) -> tuple[float, float, str]:
    measured = _nested_scalar(
        payload,
        (
            "measured_roundoff_floor_fraction_of_span",
            "roundoff_floor_fraction_of_span",
            "map_roundoff_floor_fraction_of_span",
        ),
    )
    if measured is not None:
        condition = measured / np.finfo(np.float64).eps
        return measured, condition, "receipt measured floor"
    condition = _nested_scalar(
        payload,
        ("map_condition_number", "map_condition", "condition_of_map"),
    )
    if condition is not None:
        return (
            np.finfo(np.float64).eps * condition,
            condition,
            "receipt map condition times binary64 epsilon",
        )
    quoted_condition = QUOTED_ROUNDOFF_FLOOR / np.finfo(np.float64).eps
    return QUOTED_ROUNDOFF_FLOOR, quoted_condition, "quoted default floor"


def _discover_oracle_states() -> tuple[list[TerminalState], list[dict[str, str]]]:
    paths = sorted(SOLVE_PART_ROOT.glob("*/*.json"))
    if not paths:
        raise ValueError(f"no solve parts found under {SOLVE_PART_ROOT}")
    by_identity: dict[tuple[str, int, str], TerminalState] = {}
    skipped: list[dict[str, str]] = []
    for path in paths:
        source_mode = path.parent.name
        is_required_mode = source_mode == "chord"
        is_exact_mode = "exact" in source_mode.lower()
        if not (is_required_mode or is_exact_mode):
            continue
        if not _terminal_flux_persisted(path):
            skipped.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "reason": "no persisted terminal_flux_wb",
                }
            )
            continue
        payload = _read_json(path)
        case = str(payload["case"])
        requested_cells = int(payload["requested_cells"])
        if is_required_mode:
            required = {
                (required_case, required_cells): display
                for required_case, required_cells, display in ORACLE_REQUIRED_ROWS
            }
            display_name = required.get((case, requested_cells))
            if display_name is None:
                continue
        else:
            display_name = (
                f"{case.replace('-', ' ')} {abs(requested_cells)} "
                f"{source_mode.replace('_', ' ').replace('-', ' ')}"
            )
        identity = (case, requested_cells, source_mode)
        if identity in by_identity:
            raise ValueError(f"duplicate persisted terminal state {identity}")
        by_identity[identity] = TerminalState(
            path=path,
            source_mode=source_mode,
            display_name=display_name,
            payload=payload,
        )

    required_states: list[TerminalState] = []
    for case, requested_cells, _display in ORACLE_REQUIRED_ROWS:
        identity = (case, requested_cells, "chord")
        if identity not in by_identity:
            raise ValueError(f"required persisted terminal state missing: {identity}")
        required_states.append(by_identity.pop(identity))
    exact_states = sorted(
        by_identity.values(),
        key=lambda state: (
            state.payload["case"],
            abs(int(state.payload["requested_cells"])),
            state.source_mode,
        ),
    )
    return required_states + exact_states, skipped


def _raster_field(
    coordinates: np.ndarray,
    values: np.ndarray,
    wall_units: tuple[np.ndarray, ...],
    *,
    samples: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(coordinates, dtype=np.float64)
    field = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(field)
    points, field = points[finite], field[finite]
    if len(points) < 3:
        raise ValueError("persisted field has too few finite samples")
    limits = np.vstack((points, *wall_units))
    radial = np.linspace(
        float(np.min(limits[:, 0])), float(np.max(limits[:, 0])), samples
    )
    height = np.linspace(
        float(np.min(limits[:, 1])), float(np.max(limits[:, 1])), samples
    )
    radius_grid, height_grid = np.meshgrid(radial, height)
    raster = LinearNDInterpolator(points, field, fill_value=np.nan)(
        radius_grid, height_grid
    )
    if not np.any(np.isfinite(raster)):
        raise ValueError("persisted field interpolation produced no finite raster")
    return radial, height, np.asarray(raster, dtype=np.float64)


def _outboard_wall_radius(
    axis_rz: np.ndarray, wall_units: tuple[np.ndarray, ...]
) -> float:
    axis_r, axis_z = (float(value) for value in axis_rz)
    crossings: list[float] = []
    for raw_unit in wall_units:
        unit = np.asarray(raw_unit, dtype=np.float64)
        if len(unit) < 2:
            continue
        if not np.allclose(unit[0], unit[-1]):
            unit = np.vstack((unit, unit[0]))
        for start, end in zip(unit[:-1], unit[1:], strict=True):
            r0, z0 = start
            r1, z1 = end
            if np.isclose(z0, z1):
                if np.isclose(axis_z, z0):
                    crossings.extend(float(r) for r in (r0, r1) if r > axis_r)
                continue
            fraction = (axis_z - z0) / (z1 - z0)
            if -1e-12 <= fraction <= 1.0 + 1e-12:
                radius = float(r0 + fraction * (r1 - r0))
                if radius > axis_r:
                    crossings.append(radius)
    if not crossings:
        raise ValueError(f"no outboard wall crossing from analytic axis {axis_rz}")
    return min(crossings)


def _oracle_metrics(state: TerminalState) -> dict[str, Any]:
    payload = state.payload
    data = payload["render_data"]
    coordinates = np.asarray(data["coordinates_rz_m"], dtype=np.float64)
    solved = np.asarray(data["terminal_flux_wb"], dtype=np.float64)
    analytic = np.asarray(data["analytic_flux_wb"], dtype=np.float64)
    if len(coordinates) != len(solved) or len(solved) != len(analytic):
        raise ValueError(f"persisted field shape mismatch in {state.path}")
    difference = solved - analytic
    finite = np.isfinite(difference)
    if not np.any(finite):
        raise ValueError(f"no finite solved-minus-analytic samples in {state.path}")
    analytic_topology = data["analytic_topology"]
    terminal_topology = data["terminal_topology"]
    span = abs(float(analytic_topology["flux_span_wb"]))
    if not np.isfinite(span) or span <= 0:
        raise ValueError(f"invalid analytic flux span {span} in {state.path}")
    normalized = difference / span
    roundoff_floor, map_condition, floor_source = _roundoff_floor(payload)
    max_over_span = float(np.max(np.abs(normalized[finite])))
    rms_over_span = float(np.sqrt(np.mean(np.square(normalized[finite]))))
    axis_error_m = float(
        np.linalg.norm(
            np.asarray(terminal_topology["axis_rz_m"], dtype=np.float64)
            - np.asarray(analytic_topology["axis_rz_m"], dtype=np.float64)
        )
    )
    pitch = float(payload["characteristic_pitch_m"])
    signed_boundary_error = float(terminal_topology["boundary_flux_wb"]) - float(
        analytic_topology["boundary_flux_wb"]
    )
    oracle_matched = max_over_span <= 100.0 * roundoff_floor
    return {
        "case": payload["case"],
        "requested_cells": int(payload["requested_cells"]),
        "realised_cells": int(payload["realised_cells"]),
        "source_mode": state.source_mode,
        "display_name": state.display_name,
        "source_part": str(state.path.relative_to(ROOT)),
        "source_part_sha256": _sha256(state.path),
        "analytic_flux_span_wb": span,
        "max_absolute_difference_over_span": max_over_span,
        "rms_difference_over_span": rms_over_span,
        "axis_error_in_pitch": axis_error_m / pitch,
        "boundary_level_error_wb": abs(signed_boundary_error),
        "signed_boundary_level_error_wb": signed_boundary_error,
        "boundary_level_error_over_span": abs(signed_boundary_error) / span,
        "plasma_current_amplitude": _terminal_amplitude(payload),
        "converged": bool(payload["solver"]["converged"]),
        "terminal_residual": float(payload["solver"]["terminal_fixed_point_residual"]),
        "roundoff_floor_over_span": roundoff_floor,
        "map_condition": map_condition,
        "roundoff_floor_source": floor_source,
        "within_one_hundred_roundoff_floors": oracle_matched,
        "verdict": "oracle-matched" if oracle_matched else "not-oracle-matched",
        "_coordinates": coordinates,
        "_solved": solved,
        "_analytic": analytic,
        "_difference_over_span": normalized,
        "_wall_units": tuple(
            np.asarray(unit, dtype=np.float64) for unit in data["wall_units_rz_m"]
        ),
        "_analytic_topology": analytic_topology,
        "_terminal_topology": terminal_topology,
        "_pitch_m": pitch,
    }


def _topology_flux(topology: dict[str, Any], name: str) -> float | None:
    value = topology.get(name)
    return float(value) if value is not None and np.isfinite(value) else None


def _draw_null_sets(
    axis,
    *,
    state_topology: dict[str, Any],
    analytic_topology: dict[str, Any],
    state_color: str,
    wall_units: tuple[np.ndarray, ...],
) -> None:
    poloidal.draw_nulls(
        axis,
        magnetic_axis=analytic_topology["axis_rz_m"],
        x_points=analytic_topology["x_point_rz_m"],
        style=DEFAULT_INK.variant(
            axis_marker="^",
            axis_markersize=5.5,
            axis_color="#3366cc",
            xpoint_marker="X",
            xpoint_color="#3366cc",
        ),
        contain=wall_units,
    )
    poloidal.draw_nulls(
        axis,
        magnetic_axis=state_topology["axis_rz_m"],
        x_points=state_topology["x_point_rz_m"],
        style=DEFAULT_INK.variant(
            axis_marker="^",
            axis_markersize=5.5,
            axis_color=state_color,
            xpoint_marker="X",
            xpoint_color=state_color,
        ),
        contain=wall_units,
    )


def _draw_field(
    axis,
    *,
    coordinates: np.ndarray,
    values: np.ndarray,
    levels: np.ndarray,
    wall_units: tuple[np.ndarray, ...],
    color: str,
) -> None:
    radial, height, field = _raster_field(coordinates, values, wall_units)
    poloidal.draw_flux_contours(
        axis,
        radial,
        height,
        field,
        levels,
        color=color,
        linewidth=0.55,
    )
    poloidal.draw_wall(axis, units=wall_units, linewidth=0.8)
    poloidal_axes(axis)


def _validate_pair(
    source: dict[str, Any], unmasked: dict[str, Any], case: str
) -> dict[str, np.ndarray | tuple[np.ndarray, ...] | dict[str, Any]]:
    source_data = source["render_data"]
    unmasked_data = unmasked["render_data"]
    if source["case"] != case or unmasked["case"] != case:
        raise ValueError(f"case identity mismatch for {case}")
    if source["requested_cells"] != -1000 or unmasked["requested_cells"] != -1000:
        raise ValueError(f"cell-count identity mismatch for {case}")

    coordinates = np.asarray(source_data["coordinates_rz_m"], dtype=np.float64)
    other_coordinates = np.asarray(unmasked_data["coordinates_rz_m"], dtype=np.float64)
    np.testing.assert_array_equal(coordinates, other_coordinates)
    analytic = np.asarray(source_data["analytic_flux_wb"], dtype=np.float64)
    other_analytic = np.asarray(unmasked_data["analytic_flux_wb"], dtype=np.float64)
    np.testing.assert_array_equal(analytic, other_analytic)
    shadowed = np.asarray(source_data["terminal_flux_wb"], dtype=np.float64)
    corrected = np.asarray(unmasked_data["terminal_flux_wb"], dtype=np.float64)
    if not (
        len(coordinates) == len(analytic) == len(shadowed) == len(corrected)
        and np.all(np.isfinite(analytic))
        and np.all(np.isfinite(shadowed))
        and np.all(np.isfinite(corrected))
    ):
        raise ValueError(f"persisted field shape or finiteness failed for {case}")
    wall_units = tuple(
        np.asarray(unit, dtype=np.float64) for unit in source_data["wall_units_rz_m"]
    )
    return {
        "coordinates": coordinates,
        "analytic": analytic,
        "shadowed": shadowed,
        "corrected": corrected,
        "wall_units": wall_units,
        "source_topology": source_data["terminal_topology"],
        "corrected_topology": unmasked_data["terminal_topology"],
        "analytic_topology": source_data["analytic_topology"],
    }


def _render(figure_path: Path) -> dict[str, Any]:
    metrics = _metric_rows()
    weak_carriers = _weak_shadow_carriers()
    weak_points = np.asarray(
        [[row["radius_m"], row["height_m"]] for row in weak_carriers],
        dtype=np.float64,
    )
    figure, axes = plt.subplots(len(ROWS), 3, figsize=(10.8, 12.2))
    figure.subplots_adjust(
        left=0.02,
        right=0.99,
        bottom=0.06,
        top=0.88,
        wspace=0.10,
        hspace=0.18,
    )
    receipt_rows: list[dict[str, Any]] = []
    for row_index, (case, label) in enumerate(ROWS):
        source_path = _part_path(SOURCE_PART_ROOT, case)
        unmasked_path = _part_path(UNMASKED_PART_ROOT, case)
        source = _read_json(source_path)
        unmasked = _read_json(unmasked_path)
        pair = _validate_pair(source, unmasked, case)
        analytic_topology = pair["analytic_topology"]
        assert isinstance(analytic_topology, dict)
        levels = poloidal.contour_levels(
            np.asarray(pair["analytic"]),
            count=12,
            boundary=_topology_flux(analytic_topology, "boundary_flux_wb"),
        )
        metric = metrics[case]
        state_columns = (
            (
                "shadowed",
                "shadowed terminal",
                "#cc7722",
                pair["source_topology"],
                metric["before"],
            ),
            (
                "corrected",
                "unmasked terminal",
                "#16877c",
                pair["corrected_topology"],
                metric,
            ),
        )
        for column, (key, title, color, topology, row_metric) in enumerate(
            state_columns
        ):
            axis = axes[row_index, column]
            _draw_field(
                axis,
                coordinates=np.asarray(pair["coordinates"]),
                values=np.asarray(pair[key]),
                levels=levels,
                wall_units=pair["wall_units"],
                color=color,
            )
            assert isinstance(topology, dict)
            _draw_null_sets(
                axis,
                state_topology=topology,
                analytic_topology=analytic_topology,
                state_color=color,
                wall_units=pair["wall_units"],
            )
            if case == ROWS[0][0] and key == "shadowed":
                axis.scatter(
                    weak_points[:, 0],
                    weak_points[:, 1],
                    s=18,
                    facecolors="none",
                    edgecolors="#b13f8c",
                    linewidths=0.8,
                    zorder=DEFAULT_INK.zorder_markers,
                )
            axis.set_title(
                f"{label} · {title}\n"
                f"residual {row_metric['terminal_residual']:.2e} · "
                f"converged {'yes' if row_metric['converged'] else 'no'}",
                fontsize=8.5,
            )

        analytic_axis = axes[row_index, 2]
        _draw_field(
            analytic_axis,
            coordinates=np.asarray(pair["coordinates"]),
            values=np.asarray(pair["analytic"]),
            levels=levels,
            wall_units=pair["wall_units"],
            color="#3366cc",
        )
        poloidal.draw_nulls(
            analytic_axis,
            magnetic_axis=analytic_topology["axis_rz_m"],
            x_points=analytic_topology["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^",
                axis_markersize=5.5,
                axis_color="#3366cc",
                xpoint_marker="X",
                xpoint_color="#3366cc",
            ),
            contain=pair["wall_units"],
        )
        analytic_axis.set_title(
            f"{label} · analytic reference\nresidual n/a · converged reference",
            fontsize=8.5,
        )
        receipt_rows.append(
            {
                "case": case,
                "requested_cells": -1000,
                "realised_cells": source["realised_cells"],
                "source_part": str(source_path.relative_to(ROOT)),
                "unmasked_part": str(unmasked_path.relative_to(ROOT)),
                "source_part_sha256": _sha256(source_path),
                "unmasked_part_sha256": _sha256(unmasked_path),
                "analytic_level_wb": levels.tolist(),
                "shadowed": {
                    "terminal_residual": metric["before"]["terminal_residual"],
                    "axis_error_in_pitch": metric["before"]["axis_error_in_pitch"],
                    "boundary_flux_error_from_analytic_zero_wb": metric["before"][
                        "boundary_flux_error_from_analytic_zero_wb"
                    ],
                    "converged": metric["before"]["converged"],
                },
                "unmasked": {
                    "terminal_residual": metric["terminal_residual"],
                    "axis_error_in_pitch": metric["axis_error_in_pitch"],
                    "boundary_flux_error_from_analytic_zero_wb": metric[
                        "boundary_flux_error_from_analytic_zero_wb"
                    ],
                    "converged": metric["converged"],
                },
                "marked_shadow_carrier_count": (
                    len(weak_carriers) if case == ROWS[0][0] else 0
                ),
            }
        )

    figure.legend(
        handles=(
            Line2D([], [], color="#cc7722", label="shadowed terminal contours"),
            Line2D([], [], color="#16877c", label="unmasked terminal contours"),
            Line2D([], [], color="#3366cc", label="analytic contours / nulls"),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="#b13f8c",
                linestyle="none",
                label="shadow-excluded carrier",
            ),
            Line2D(
                [],
                [],
                marker="^",
                color="#333333",
                linestyle="none",
                label="magnetic axis",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Limited Solovev terminal flux: residual-shadow removal",
        fontsize=13,
        y=0.992,
    )
    figure.text(
        0.5,
        0.003,
        "Each row uses one analytic Wb level array in all three columns. "
        f"Weak 1000 marks {len(weak_carriers)} persisted shadow-excluded carriers. "
        "The weak 2500 reproduction persisted only its 54-carrier census, not its "
        "terminal flux array, so that row is not drawn.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=190, facecolor=DEFAULT_INK.figure_facecolor)
    plt.close(figure)
    return {"rows": receipt_rows, "weak_1000_marked_carriers": len(weak_carriers)}


def _draw_oracle_overlay(axis, metric: dict[str, Any]) -> np.ndarray:
    coordinates = metric["_coordinates"]
    wall_units = metric["_wall_units"]
    analytic_topology = metric["_analytic_topology"]
    radial, height, analytic = _raster_field(
        coordinates, metric["_analytic"], wall_units
    )
    _other_radial, _other_height, solved = _raster_field(
        coordinates, metric["_solved"], wall_units
    )
    levels = poloidal.contour_levels(
        metric["_analytic"],
        count=12,
        boundary=_topology_flux(analytic_topology, "boundary_flux_wb"),
    )
    poloidal.draw_flux_contours(
        axis,
        radial,
        height,
        analytic,
        levels,
        color="#3366cc",
        linewidth=0.62,
    )
    poloidal.draw_flux_contours(
        axis,
        radial,
        height,
        solved,
        levels,
        color="#cc7722",
        linewidth=0.58,
    )
    poloidal.draw_wall(axis, units=wall_units, linewidth=0.8)
    _draw_null_sets(
        axis,
        state_topology=metric["_terminal_topology"],
        analytic_topology=analytic_topology,
        state_color="#cc7722",
        wall_units=wall_units,
    )
    poloidal_axes(axis)
    axis.set_title(
        f"{metric['display_name']} · solved over analytic\n"
        f"12 shared levels · residual {metric['terminal_residual']:.2e} · "
        f"converged {'yes' if metric['converged'] else 'no'}",
        fontsize=8.2,
    )
    return np.asarray(levels, dtype=np.float64)


def _draw_difference_panel(axis, metric: dict[str, Any]) -> list[float]:
    coordinates = metric["_coordinates"]
    wall_units = metric["_wall_units"]
    radial, height, difference = _raster_field(
        coordinates, metric["_difference_over_span"], wall_units
    )
    finite = difference[np.isfinite(difference)]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    visible = SIGNED_DIFFERENCE_LEVELS[
        (SIGNED_DIFFERENCE_LEVELS >= lower) & (SIGNED_DIFFERENCE_LEVELS <= upper)
    ]
    negative = visible[visible < 0]
    positive = visible[visible > 0]
    if len(negative):
        poloidal.draw_flux_contours(
            axis,
            radial,
            height,
            difference,
            negative,
            color="#cc7722",
            linewidth=0.62,
        )
    if len(positive):
        poloidal.draw_flux_contours(
            axis,
            radial,
            height,
            difference,
            positive,
            color="#b13f8c",
            linewidth=0.62,
        )
    floor = float(metric["roundoff_floor_over_span"])
    floor_levels = [level for level in (-floor, floor) if lower <= level <= upper]
    if floor_levels:
        axis.contour(
            radial,
            height,
            difference,
            levels=floor_levels,
            colors="#555555",
            linewidths=0.7,
            linestyles="dotted",
            zorder=DEFAULT_INK.zorder_flux + 0.1,
        )
    poloidal.draw_wall(axis, units=wall_units, linewidth=0.8)
    _draw_null_sets(
        axis,
        state_topology=metric["_terminal_topology"],
        analytic_topology=metric["_analytic_topology"],
        state_color="#cc7722",
        wall_units=wall_units,
    )
    poloidal_axes(axis)
    axis.plot(
        [0.05, 0.22],
        [0.055, 0.055],
        transform=axis.transAxes,
        color="#555555",
        linewidth=0.9,
        linestyle=":",
        clip_on=False,
    )
    axis.text(
        0.24,
        0.055,
        f"roundoff floor {floor:.0e}",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=6.7,
        color="#444444",
    )
    axis.set_title(
        "signed (solved − analytic) ψ / analytic span\n"
        f"max |Δψ|/span {metric['max_absolute_difference_over_span']:.3e} · "
        f"{metric['verdict']}",
        fontsize=8.2,
    )
    return [float(level) for level in visible]


def _draw_outboard_chord(axis, metric: dict[str, Any]) -> dict[str, Any]:
    analytic_axis = np.asarray(
        metric["_analytic_topology"]["axis_rz_m"], dtype=np.float64
    )
    wall_radius = _outboard_wall_radius(analytic_axis, metric["_wall_units"])
    radius = np.linspace(float(analytic_axis[0]), wall_radius, 241)
    height = np.full_like(radius, float(analytic_axis[1]))
    interpolator = LinearNDInterpolator(
        metric["_coordinates"],
        metric["_difference_over_span"],
        fill_value=np.nan,
    )
    difference = np.asarray(interpolator(radius, height), dtype=np.float64)
    distance_in_pitch = (radius - float(analytic_axis[0])) / metric["_pitch_m"]
    finite = np.isfinite(difference)
    if np.count_nonzero(finite) < 3:
        raise ValueError(
            f"outboard chord interpolation failed for {metric['display_name']}"
        )
    absolute = np.abs(difference[finite])
    positive = absolute > 0
    if not np.any(positive):
        raise ValueError(
            f"outboard chord positive control is empty for {metric['display_name']}"
        )
    axis.plot(
        distance_in_pitch[finite][positive],
        absolute[positive],
        color="#16877c",
        linewidth=1.1,
    )
    floor = float(metric["roundoff_floor_over_span"])
    axis.axhline(floor, color="#555555", linewidth=0.8, linestyle=":")
    axis.axvline(1.0, color="#7d6b91", linewidth=0.8, linestyle="--")
    axis.text(
        1.0,
        0.98,
        "one pitch",
        transform=axis.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=6.7,
        color="#6a587e",
    )
    axis.set_yscale("log")
    upper = max(float(np.max(absolute)) * 3.0, floor * 100.0)
    axis.set_ylim(floor / 10.0, upper)
    axis.set_xlim(0.0, float(np.max(distance_in_pitch[finite])))
    axis.set_xlabel("outboard distance from analytic axis (pitch)", fontsize=7.2)
    axis.set_ylabel("|Δψ| / analytic span", fontsize=7.2)
    axis.tick_params(axis="both", labelsize=6.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.set_title(
        f"outboard midplane · axis to wall\n"
        f"wall {distance_in_pitch[-1]:.2f} pitch · floor {floor:.0e}",
        fontsize=8.2,
    )
    return {
        "sample_count": int(np.count_nonzero(finite)),
        "outboard_wall_radius_m": wall_radius,
        "axis_to_wall_in_pitch": float(distance_in_pitch[-1]),
        "max_absolute_difference_over_span": float(np.max(absolute)),
    }


def _public_oracle_row(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metric.items() if not key.startswith("_")}


def _render_oracle(figure_path: Path) -> dict[str, Any]:
    states, skipped = _discover_oracle_states()
    metrics = [_oracle_metrics(state) for state in states]
    figure, axes = plt.subplots(
        len(metrics),
        3,
        figsize=(11.7, max(12.8, 3.25 * len(metrics))),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.045,
        top=0.935,
        wspace=0.17,
        hspace=0.30,
    )
    receipt_rows: list[dict[str, Any]] = []
    for row_index, metric in enumerate(metrics):
        analytic_levels = _draw_oracle_overlay(axes[row_index, 0], metric)
        difference_levels = _draw_difference_panel(axes[row_index, 1], metric)
        chord = _draw_outboard_chord(axes[row_index, 2], metric)
        public = _public_oracle_row(metric)
        public["analytic_contour_levels_wb"] = analytic_levels.tolist()
        public["configured_signed_difference_levels"] = (
            SIGNED_DIFFERENCE_LEVELS.tolist()
        )
        public["visible_signed_difference_levels"] = difference_levels
        public["outboard_midplane_chord"] = chord
        receipt_rows.append(public)

    figure.legend(
        handles=(
            Line2D([], [], color="#3366cc", label="analytic ψ / nulls"),
            Line2D([], [], color="#cc7722", label="solved ψ / nulls; negative Δψ"),
            Line2D([], [], color="#b13f8c", label="positive Δψ"),
            Line2D([], [], color="#555555", linestyle=":", label="roundoff floor"),
            Line2D([], [], color="#7d6b91", linestyle="--", label="one pitch"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=5,
        frameon=False,
        fontsize=7.5,
    )
    figure.suptitle(
        "Limited Solovev terminal states against the analytic oracle",
        fontsize=13,
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "A converged row is a self-consistent fixed point, not an oracle match. "
        "Oracle-matched means max |solved − analytic| is within 100 roundoff "
        "floors; the quoted floor is 1e−14 of analytic flux span unless a part "
        "records a measured floor.",
        ha="center",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=190, facecolor=DEFAULT_INK.figure_facecolor)
    plt.close(figure)
    return {
        "rows": receipt_rows,
        "skipped_exact_parts": skipped,
        "persisted_terminal_state_count": len(receipt_rows),
        "exact_clip_terminal_state_count": sum(
            row["source_mode"] != "chord" for row in receipt_rows
        ),
    }


def _write_receipt(path: Path, figure_path: Path, measurement: dict[str, Any]) -> None:
    weak_2500_persisted = _terminal_flux_persisted(WEAK_REPRODUCTION_PART)
    if weak_2500_persisted:
        raise RuntimeError(
            "weak 2500 reproduction now carries terminal flux; add its paired row"
        )
    payload = {
        "schema": "nova.limited-shadow-terminal-comparison",
        "source_revision": _source_revision(),
        "shadow_mechanism_revision": SOURCE_REVISION,
        "completed": True,
        "render_only": True,
        "solve_calls": 0,
        "figure": {
            "filesystem_path": str(figure_path.relative_to(ROOT)),
            "project_absolute_src": f"/nova/{figure_path.relative_to(ROOT / 'docs')}",
            "sha256": _sha256(figure_path),
        },
        "rows": measurement["rows"],
        "carrier_marker_positive_control": {
            "weak_1000_marked_count": measurement["weak_1000_marked_carriers"],
            "source": str(SHADOW_RECEIPT.relative_to(ROOT)),
        },
        "omitted_weak_2500": {
            "omitted": True,
            "reason": (
                "the reproduction part carries the 54-carrier census but no "
                "terminal_flux_wb array"
            ),
            "reproduction_part": str(WEAK_REPRODUCTION_PART),
            "terminal_flux_persisted": False,
            "census_carrier_count": 54,
            "solve_forbidden": True,
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_oracle_receipt(
    path: Path, figure_path: Path, measurement: dict[str, Any]
) -> None:
    payload = {
        "schema": "nova.limited-shadow-oracle-overlay-error",
        "source_revision": _source_revision(),
        "completed": True,
        "render_only": True,
        "solve_calls": 0,
        "input_glob": str((SOLVE_PART_ROOT / "*/*.json").relative_to(ROOT)),
        "required_terminal_states": [
            {
                "case": case,
                "requested_cells": requested_cells,
                "source_mode": "chord",
            }
            for case, requested_cells, _display in ORACLE_REQUIRED_ROWS
        ],
        "roundoff_policy": {
            "binary64_epsilon": np.finfo(np.float64).eps,
            "quoted_default_floor_over_span": QUOTED_ROUNDOFF_FLOOR,
            "oracle_match_multiple": 100.0,
            "rule": (
                "machine epsilon times map condition; use a measured receipt "
                "floor when present"
            ),
        },
        "fixed_signed_difference_levels": SIGNED_DIFFERENCE_LEVELS.tolist(),
        "figure": {
            "filesystem_path": str(figure_path.relative_to(ROOT)),
            "project_absolute_src": f"/nova/{figure_path.relative_to(ROOT / 'docs')}",
            "sha256": _sha256(figure_path),
        },
        **measurement,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_oracle_report(
    path: Path, figure_path: Path, measurement: dict[str, Any]
) -> None:
    lines = [
        "# Limited Solovev terminal states against the analytic oracle",
        "",
        "This is a receipt-only render; it made zero solve calls. A converged "
        "whole-cell row is a self-consistent fixed point and is not an oracle "
        "match until its signed difference panel reaches the roundoff floor.",
        "",
        f"Figure: `/nova/{figure_path.relative_to(ROOT / 'docs')}`",
        "",
        "| row | max difference/span | rms difference/span | axis error (pitch) "
        "| boundary error (Wb) | boundary error/span | amplitude | converged "
        "| terminal residual | roundoff floor | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---|",
    ]
    for row in measurement["rows"]:
        lines.append(
            f"| {row['display_name']} | "
            f"{row['max_absolute_difference_over_span']:.6g} | "
            f"{row['rms_difference_over_span']:.6g} | "
            f"{row['axis_error_in_pitch']:.6g} | "
            f"{row['boundary_level_error_wb']:.6g} | "
            f"{row['boundary_level_error_over_span']:.6g} | "
            f"{row['plasma_current_amplitude']:.6g} | "
            f"{'yes' if row['converged'] else 'no'} | "
            f"{row['terminal_residual']:.6g} | "
            f"{row['roundoff_floor_over_span']:.3g} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "The three converged limited whole-cell rows retain about 0.64 pitch "
            "of axis error, boundary-level errors of 4.96, 0.14 and 0.012 Wb, "
            "and plasma-current amplitude about 0.897. Those are discretisation "
            "errors for the exact clip to remove; their small terminal residuals "
            "do not turn them into oracle matches.",
            "",
            "Persisted terminal states drawn: "
            f"{measurement['persisted_terminal_state_count']}; exact-clip "
            "terminal states discovered: "
            f"{measurement['exact_clip_terminal_state_count']}.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render paired limited-row terminal states from receipts"
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--oracle-figure", type=Path, default=DEFAULT_ORACLE_FIGURE)
    parser.add_argument("--oracle-receipt", type=Path, default=DEFAULT_ORACLE_RECEIPT)
    parser.add_argument("--oracle-report", type=Path, default=DEFAULT_ORACLE_REPORT)
    args = parser.parse_args()
    measurement = _render(args.figure)
    _write_receipt(args.receipt, args.figure, measurement)
    oracle_measurement = _render_oracle(args.oracle_figure)
    _write_oracle_receipt(args.oracle_receipt, args.oracle_figure, oracle_measurement)
    _write_oracle_report(args.oracle_report, args.oracle_figure, oracle_measurement)
    print(f"LIMITED_SHADOW_FIGURE={args.figure}", flush=True)
    print(f"LIMITED_SHADOW_RECEIPT={args.receipt}", flush=True)
    print(f"ORACLE_OVERLAY_FIGURE={args.oracle_figure}", flush=True)
    print(f"ORACLE_OVERLAY_RECEIPT={args.oracle_receipt}", flush=True)
    print(f"ORACLE_OVERLAY_REPORT={args.oracle_report}", flush=True)
    print(
        "LIMITED_SHADOW_RENDER_EXIT=0 "
        f"rows=3 oracle_rows={oracle_measurement['persisted_terminal_state_count']} "
        "solves=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
