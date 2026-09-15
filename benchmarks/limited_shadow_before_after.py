"""Render the limited-row residual-shadow comparison from persisted states.

The driver reads three paired Solovev terminal-state receipts and renders their
shadowed and unmasked flux beside the analytic field.  It does not construct a
forward operator or invoke a solve.  Shared absolute contour levels come only
from each row's persisted analytic field, and the weak-row panel marks the raw
private carriers that the former residual shadow excluded.
"""

from __future__ import annotations

import argparse
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
SOLVE_RECEIPT = FIGURE_ROOT / "solve-receipt.json"
SHADOW_RECEIPT = FIGURE_ROOT / "receipt.json"
SOURCE_PART_ROOT = (
    ROOT / "docs/figures/cut-cell-current-attribution/wholecell-rows/parts/rows"
)
UNMASKED_PART_ROOT = FIGURE_ROOT / "solve-parts/chord"
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
    figure, axes = plt.subplots(
        len(ROWS), 3, figsize=(10.8, 11.0), constrained_layout=True
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
        bbox_to_anchor=(0.5, 0.958),
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render paired limited-row terminal states from receipts"
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    measurement = _render(args.figure)
    _write_receipt(args.receipt, args.figure, measurement)
    print(f"LIMITED_SHADOW_FIGURE={args.figure}", flush=True)
    print(f"LIMITED_SHADOW_RECEIPT={args.receipt}", flush=True)
    print("LIMITED_SHADOW_RENDER_EXIT=0 rows=3 solves=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
