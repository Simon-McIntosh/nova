"""Render regenerated MAST terminal flux maps against their EFIT maps.

The renderer consumes the retained bank operands without running an equilibrium
solve.  EFIT ``efm/psirz`` is read from the same read-only shot catalogue used
by the bank producer.  Nova's total poloidal flux is divided by ``2 pi`` so
both maps are compared in Wb/rad on one EFIT-derived contour-level array per
panel.  No gauge shift, normalization, or fitted scaling is applied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import zarr

from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.imas.mast_vacuum_cohort import SHOT_STORE


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/solver-convergence-regression/bank-efit-overlays"
ROW_COUNT = 12
CONTOUR_COUNT = 13

NOVA_COLOUR = "#285f8f"
NOVA_BOUNDARY_COLOUR = "#173f5f"
EFIT_COLOUR = "#d66b18"
EFIT_BOUNDARY_COLOUR = "#8f2d0c"
WALL_COLOUR = "#252a2d"
GRID_COLOUR = "#d9dddf"


@dataclass(frozen=True)
class EfitFluxMap:
    """One finite EFIT flux plane in catalogue coordinates."""

    radius_m: np.ndarray
    height_m: np.ndarray
    flux_wb_per_radian: np.ndarray


@dataclass(frozen=True)
class ArmPlot:
    """The retained plotting state for one declared bank arm."""

    index: int
    record: dict[str, Any]
    radius_m: np.ndarray
    height_m: np.ndarray
    nova_flux_wb_per_radian: np.ndarray
    axis_m: np.ndarray
    wall_m: np.ndarray
    binding_flux_wb_per_radian: float | None
    selected_saddle_m: np.ndarray
    efit_lcfs_m: np.ndarray
    efit_x_points_m: np.ndarray

    @property
    def filename(self) -> str:
        identity = str(self.record["identity"]).replace("/", "-")
        return f"mast-{identity}-{self.record['arm']}.png"


def _finite_points(values: np.ndarray) -> np.ndarray:
    """Return finite two-dimensional points and reject malformed arrays."""

    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError(f"expected an N-by-2 point array, received {points.shape}")
    return points[np.all(np.isfinite(points), axis=1)]


def _optional_finite_scalar(value: np.ndarray) -> float | None:
    """Return one finite scalar or ``None`` for an explicitly absent value."""

    scalar = float(np.asarray(value))
    return scalar if np.isfinite(scalar) else None


def _read_arms(raw_directory: Path) -> list[ArmPlot]:
    """Read and cross-check all declared rows and their retained operands."""

    report_path = raw_directory / "current-bank.json"
    operands_path = raw_directory / "current-operands.npz"
    report = json.loads(report_path.read_text())
    records = report.get("rows")
    if not isinstance(records, list) or len(records) != ROW_COUNT:
        raise ValueError(f"current-bank.json must declare exactly {ROW_COUNT} rows")

    arms: list[ArmPlot] = []
    with np.load(operands_path, allow_pickle=False) as stored:
        declared_metadata = json.loads(str(stored["metadata"]))
        metadata_rows = declared_metadata.get("rows", [])
        if len(metadata_rows) != ROW_COUNT:
            raise ValueError("the operand metadata does not retain twelve rows")

        populated_walls = [
            _finite_points(stored[f"arm_{index:02d}_wall"])
            for index in range(ROW_COUNT)
            if stored[f"arm_{index:02d}_wall"].size
        ]
        if not populated_walls:
            raise ValueError("the operand bank retains no governed wall")
        shared_wall = populated_walls[0]
        if any(not np.array_equal(shared_wall, wall) for wall in populated_walls[1:]):
            raise ValueError("the machine-invariant wall differs between bank arms")

        for index, record in enumerate(records):
            declared = metadata_rows[index]
            identity = (record["identity"], record["arm"])
            declared_identity = (declared["identity"], declared["arm"])
            if identity != declared_identity:
                raise ValueError(
                    f"row {index} metadata mismatch: {identity} != {declared_identity}"
                )
            prefix = f"arm_{index:02d}_"
            radius = np.asarray(stored[prefix + "radius"], dtype=np.float64)
            height = np.asarray(stored[prefix + "height"], dtype=np.float64)
            flux = np.asarray(stored[prefix + "flux"], dtype=np.float64)
            if flux.size:
                expected = (height.size, radius.size)
                if flux.shape != expected or not np.all(np.isfinite(flux)):
                    raise ValueError(
                        f"row {index} Nova flux shape/finite check failed: "
                        f"{flux.shape} != {expected}"
                    )
                flux = flux / TOTAL_FLUX_FACTOR
            arms.append(
                ArmPlot(
                    index=index,
                    record=record,
                    radius_m=radius,
                    height_m=height,
                    nova_flux_wb_per_radian=flux,
                    axis_m=np.asarray(stored[prefix + "axis"], dtype=np.float64),
                    wall_m=shared_wall,
                    binding_flux_wb_per_radian=(
                        None
                        if (
                            value := _optional_finite_scalar(
                                stored[prefix + "binding_flux"]
                            )
                        )
                        is None
                        else value / TOTAL_FLUX_FACTOR
                    ),
                    selected_saddle_m=np.asarray(
                        stored[prefix + "selected_saddle"], dtype=np.float64
                    ),
                    efit_lcfs_m=_finite_points(stored[prefix + "efit_lcfs"]),
                    efit_x_points_m=_finite_points(stored[prefix + "efit_x_points"]),
                )
            )
    return arms


def _read_efit_flux_map(shot_store: Path, shot: int, slice_index: int) -> EfitFluxMap:
    """Read one finite EFM flux map from the bank producer's shot catalogue."""

    root = zarr.open_group(str(shot_store / f"{shot}.zarr"), mode="r")
    if "efm" not in root:
        raise ValueError(f"shot {shot} carries no 'efm' group")
    group = root["efm"]
    required = {"time", "gridr", "gridz", "profile_r", "psirz"}
    missing = sorted(required.difference(group.array_keys()))
    if missing:
        raise ValueError(f"shot {shot} EFM group is missing arrays {missing}")
    slice_count = int(group["time"].shape[0])
    if not 0 <= slice_index < slice_count:
        raise IndexError(
            f"slice {slice_index} is outside shot {shot} rows 0..{slice_count - 1}"
        )

    stored_radius = np.asarray(group["gridr"][:], dtype=np.float64)
    height = np.asarray(group["gridz"][:], dtype=np.float64)
    raw = np.asarray(group["psirz"][slice_index], dtype=np.float64)
    finite_columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if finite_columns.size != stored_radius.size:
        raise ValueError(
            f"shot {shot} slice {slice_index} carries {finite_columns.size} "
            f"finite radial columns, expected {stored_radius.size}"
        )
    padded_radius = np.asarray(group["profile_r"][:], dtype=np.float64)
    if not np.allclose(
        padded_radius[finite_columns], stored_radius, rtol=2.0e-7, atol=1.0e-8
    ):
        raise ValueError("finite EFM columns do not match the declared radial grid")
    flux = raw[:, finite_columns]
    if flux.shape != (height.size, stored_radius.size):
        raise ValueError(
            f"EFIT map shape {flux.shape} does not match "
            f"{(height.size, stored_radius.size)}"
        )
    return EfitFluxMap(stored_radius, height, flux)


def _contour_levels(
    efit_flux: np.ndarray,
    binding_flux_wb_per_radian: float | None,
) -> np.ndarray:
    """Return the one physical level array applied to both maps in a panel."""

    finite = np.asarray(efit_flux, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("the EFIT map contains no finite contour source")
    lower, upper = np.quantile(finite, [0.01, 0.99])
    if np.isclose(lower, upper):
        lower, upper = np.min(finite), np.max(finite)
    if np.isclose(lower, upper):
        raise ValueError("the EFIT map contains no finite contour span")
    levels = np.linspace(float(lower), float(upper), CONTOUR_COUNT)
    if binding_flux_wb_per_radian is not None:
        levels = np.append(levels, binding_flux_wb_per_radian)
    return np.unique(np.sort(levels))


def _finite_point(point: np.ndarray) -> np.ndarray | None:
    """Return one finite marker coordinate or ``None``."""

    value = np.asarray(point, dtype=np.float64).reshape(-1)
    return value if value.shape == (2,) and np.all(np.isfinite(value)) else None


def _metric(value: Any, *, unit_scale: float = 1.0, suffix: str = "") -> str:
    """Format one optional finite metric for compact panel titles."""

    if value is None:
        return "unavailable"
    try:
        numeric = float(value)
    except TypeError, ValueError:
        return "unavailable"
    if not np.isfinite(numeric):
        return "unavailable"
    return f"{numeric * unit_scale:.3g}{suffix}"


def _has_contour_segments(contours: Any) -> bool:
    """Return whether Matplotlib emitted at least one nontrivial contour path."""

    return any(
        len(segment) > 1
        for level_segments in contours.allsegs
        for segment in level_segments
    )


def _draw_panel(
    axis: Axes,
    arm: ArmPlot,
    efit_map: EfitFluxMap | None,
    efit_error: str | None,
    *,
    compact: bool,
) -> tuple[bool, bool]:
    """Draw one arm and report whether each source produced contour paths."""

    record = arm.record
    efit_contours_present = False
    nova_contours_present = False
    levels: np.ndarray | None = None
    if efit_map is not None:
        levels = _contour_levels(
            efit_map.flux_wb_per_radian, arm.binding_flux_wb_per_radian
        )
        contours = axis.contour(
            efit_map.radius_m,
            efit_map.height_m,
            efit_map.flux_wb_per_radian,
            levels=levels,
            colors=EFIT_COLOUR,
            linewidths=0.65 if compact else 0.85,
            linestyles="dashed",
            alpha=0.88,
            zorder=1,
        )
        efit_contours_present = _has_contour_segments(contours)
    if arm.nova_flux_wb_per_radian.size and levels is not None:
        contours = axis.contour(
            arm.radius_m,
            arm.height_m,
            arm.nova_flux_wb_per_radian,
            levels=levels,
            colors=NOVA_COLOUR,
            linewidths=0.6 if compact else 0.85,
            linestyles="solid",
            alpha=0.9,
            zorder=2,
        )
        nova_contours_present = _has_contour_segments(contours)
        if arm.binding_flux_wb_per_radian is not None:
            axis.contour(
                arm.radius_m,
                arm.height_m,
                arm.nova_flux_wb_per_radian,
                levels=[arm.binding_flux_wb_per_radian],
                colors=NOVA_BOUNDARY_COLOUR,
                linewidths=2.2 if compact else 2.7,
                linestyles="solid",
                zorder=5,
            )

    axis.plot(
        arm.wall_m[:, 0],
        arm.wall_m[:, 1],
        color=WALL_COLOUR,
        linewidth=1.2 if compact else 1.6,
        zorder=7,
    )
    if len(arm.efit_lcfs_m):
        axis.plot(
            arm.efit_lcfs_m[:, 0],
            arm.efit_lcfs_m[:, 1],
            color=EFIT_BOUNDARY_COLOUR,
            linewidth=2.1 if compact else 2.6,
            linestyle=(0, (5, 2)),
            zorder=6,
        )
    for point_index, point in enumerate(arm.efit_x_points_m):
        axis.scatter(
            point[0],
            point[1],
            marker="+",
            s=52 if compact else 82,
            color=EFIT_BOUNDARY_COLOUR,
            linewidths=1.7,
            zorder=9,
            label="EFIT X point" if point_index == 0 else None,
        )
    axis_point = _finite_point(arm.axis_m)
    if axis_point is not None:
        axis.scatter(
            axis_point[0],
            axis_point[1],
            marker="o",
            s=42 if compact else 64,
            facecolors="white",
            edgecolors=NOVA_BOUNDARY_COLOUR,
            linewidths=1.5,
            zorder=9,
        )
    saddle = _finite_point(arm.selected_saddle_m)
    if saddle is not None:
        axis.scatter(
            saddle[0],
            saddle[1],
            marker="X",
            s=50 if compact else 78,
            color=NOVA_BOUNDARY_COLOUR,
            linewidths=0.7,
            zorder=10,
        )

    state = "converged" if record["converged"] else record["termination_reason"]
    residual = _metric(record.get("terminal_residual"))
    lcfs_rms = _metric(
        record.get("binding_to_efit_lcfs_rms_m"), unit_scale=1.0e3, suffix=" mm"
    )
    axis.set_title(
        f"MAST {record['identity']} · {record['arm']} · {state}\n"
        f"residual {residual} · binding↔EFIT LCFS RMS {lcfs_rms}",
        loc="left",
        fontsize=7.3 if compact else 10.2,
        fontweight="semibold",
        color="#17252f",
    )

    unavailable: list[str] = []
    if efit_error is not None:
        unavailable.append(f"EFIT map unavailable: {efit_error}")
    elif not efit_contours_present:
        unavailable.append("EFIT map read, but no shared-level contour was drawable")
    if not arm.nova_flux_wb_per_radian.size:
        unavailable.append(
            "Nova terminal map/markers unavailable: "
            + str(record.get("failure_exception_class") or record["termination_reason"])
        )
    if unavailable:
        axis.text(
            0.02,
            0.02,
            "\n".join(unavailable),
            transform=axis.transAxes,
            fontsize=6.2 if compact else 8.5,
            va="bottom",
            color="#7a271a",
            bbox={"facecolor": "white", "edgecolor": "#c7c7c7", "alpha": 0.94},
            zorder=12,
        )

    x_margin = 0.06
    z_margin = 0.08
    axis.set_xlim(
        float(np.min(arm.wall_m[:, 0]) - x_margin),
        float(np.max(arm.wall_m[:, 0]) + x_margin),
    )
    axis.set_ylim(
        float(np.min(arm.wall_m[:, 1]) - z_margin),
        float(np.max(arm.wall_m[:, 1]) + z_margin),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]", fontsize=7 if compact else 9)
    axis.set_ylabel("Z [m]", fontsize=7 if compact else 9)
    axis.tick_params(labelsize=6.5 if compact else 8)
    axis.grid(color=GRID_COLOUR, linewidth=0.35, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    return efit_contours_present, nova_contours_present


def _legend_handles() -> list[Line2D]:
    """Return the shared visual-key handles used by all rendered figures."""

    return [
        Line2D([], [], color=NOVA_COLOUR, lw=1.1, label="Nova flux (solid)"),
        Line2D([], [], color=EFIT_COLOUR, lw=1.1, ls="--", label="EFIT flux (dashed)"),
        Line2D(
            [],
            [],
            color=NOVA_BOUNDARY_COLOUR,
            lw=2.6,
            label="Nova binding contour",
        ),
        Line2D(
            [],
            [],
            color=EFIT_BOUNDARY_COLOUR,
            lw=2.4,
            ls=(0, (5, 2)),
            label="EFIT LCFS",
        ),
        Line2D([], [], color=WALL_COLOUR, lw=1.5, label="governed wall"),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=NOVA_BOUNDARY_COLOUR,
            lw=0,
            label="Nova axis",
        ),
        Line2D(
            [],
            [],
            marker="X",
            color=NOVA_BOUNDARY_COLOUR,
            lw=0,
            label="Nova saddle",
        ),
        Line2D(
            [],
            [],
            marker="+",
            color=EFIT_BOUNDARY_COLOUR,
            lw=0,
            label="EFIT X point",
        ),
    ]


def render(
    raw_directory: Path,
    output_directory: Path,
    shot_store: Path = SHOT_STORE,
) -> dict[str, Any]:
    """Render the grid and every row, returning explicit coverage counts."""

    arms = _read_arms(raw_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    maps: dict[tuple[int, int], EfitFluxMap | None] = {}
    map_errors: dict[tuple[int, int], str | None] = {}
    for arm in arms:
        key = (int(arm.record["shot"]), int(arm.record["slice_index"]))
        if key in maps:
            continue
        try:
            maps[key] = _read_efit_flux_map(shot_store, *key)
            map_errors[key] = None
        except (FileNotFoundError, OSError, KeyError, IndexError, ValueError) as error:
            maps[key] = None
            map_errors[key] = f"{type(error).__name__}: {error}"

    grid, axes = plt.subplots(
        4, 3, figsize=(15.2, 17.4), constrained_layout=True, sharex=True, sharey=True
    )
    efit_count = 0
    nova_count = 0
    row_paths: list[str] = []
    for arm, axis in zip(arms, axes.ravel(), strict=True):
        key = (int(arm.record["shot"]), int(arm.record["slice_index"]))
        efit_present, nova_present = _draw_panel(
            axis, arm, maps[key], map_errors[key], compact=True
        )
        efit_count += int(efit_present)
        nova_count += int(nova_present)

        figure, row_axis = plt.subplots(figsize=(7.4, 8.6), constrained_layout=True)
        _draw_panel(row_axis, arm, maps[key], map_errors[key], compact=False)
        figure.legend(
            handles=_legend_handles(),
            loc="outside lower center",
            ncol=4,
            frameon=False,
            fontsize=8,
        )
        row_path = output_directory / arm.filename
        figure.savefig(row_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        row_paths.append(str(row_path))

    grid.suptitle(
        "Regenerated MAST bank — Nova terminal flux against EFIT",
        fontsize=15,
        fontweight="semibold",
    )
    grid.legend(
        handles=_legend_handles(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    grid_path = output_directory / "mast-bank-efit-overlays.png"
    grid.savefig(grid_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(grid)

    return {
        "rows": len(arms),
        "efit_flux_contour_panels": efit_count,
        "nova_flux_contour_panels": nova_count,
        "efit_map_unavailable": {
            f"{shot}/{slice_index}": reason
            for (shot, slice_index), reason in map_errors.items()
            if reason is not None
        },
        "grid_figure": str(grid_path),
        "row_figures": row_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render retained MAST bank flux contours against EFIT."
    )
    parser.add_argument(
        "raw_directory",
        type=Path,
        help="directory containing current-bank.json and current-operands.npz",
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shot-store", type=Path, default=SHOT_STORE)
    arguments = parser.parse_args()
    result = render(
        arguments.raw_directory, arguments.output_directory, arguments.shot_store
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
