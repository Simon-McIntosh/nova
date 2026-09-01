"""Render the recorded MAST 22086/43 cross-runtime comparison.

The committed cache did not bank flux values and the HEAD receipt did not retain a
flux grid.  Those panels therefore use the pinned-revision flux as an explicitly
labelled common spatial carrier; only the pinned panels claim a recorded flux map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PINNED_ROOT = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/"
    "nova-a0f1e0938fc2/s18-hexgrid/hdg-cache-replay"
)
BASELINE_CACHE = (
    ROOT / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)
PINNED_CACHE = (
    DEFAULT_PINNED_ROOT
    / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)
HEAD_RECEIPT = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
HEAD_MECHANISM = ROOT / "docs/figures/solver-convergence-regression/head-mechanism.json"
DEFAULT_COMPARISON = ROOT / "docs/figures/hex-cell-single-grid/pure-arm-comparison.png"
DEFAULT_RESIDUALS = (
    ROOT / "docs/figures/hex-cell-single-grid/pure-arm-comparison-residuals.png"
)

INK = "#17252f"
GRID = "#d7dce0"
WALL = "#555b60"
CPU = "#2563a8"
PINNED = "#d04a35"
HEAD = "#111111"
MIXED = "#0d7b72"


def _row(stored: np.lib.npyio.NpzFile, index: int) -> dict[str, np.ndarray]:
    prefix = f"row_{index:02d}_"
    return {
        key.removeprefix(prefix): np.asarray(stored[key])
        for key in stored.files
        if key.startswith(prefix)
    }


def _head_row(receipt: dict[str, object], arm: str) -> dict[str, object]:
    rows = [
        row
        for row in receipt["rows"]
        if row["identity"] == "22086/43" and row["arm"] == arm
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one HEAD 22086/43 {arm} row, got {len(rows)}")
    return rows[0]


def _grid(row: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = row["cell_rz"]
    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    values = row["per_cell_flux_values"]
    if values.shape != (radius.size * height.size,):
        raise RuntimeError("pinned row does not carry one flux value per cell")
    return radius, height, values.reshape((radius.size, height.size)).T


def _decorate_map(axis: plt.Axes, row: dict[str, np.ndarray]) -> None:
    wall = row["wall"]
    lcfs = row["efit_lcfs"]
    axis.plot(wall[:, 0], wall[:, 1], color=WALL, lw=1.2, label="wall")
    axis.plot(
        lcfs[:, 0], lcfs[:, 1], color="#1c8a9b", lw=1.1, ls="--", label="EFIT LCFS"
    )
    axis.set_aspect("equal")
    axis.set_xlim(0.15, 1.95)
    axis.set_ylim(-1.95, 1.95)
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.grid(color=GRID, lw=0.35, alpha=0.7)


def _mark(
    axis: plt.Axes,
    point: np.ndarray,
    marker: str,
    colour: str,
    label: str,
    *,
    filled: bool = True,
) -> None:
    point = np.asarray(point, dtype=float).reshape((-1, 2))[0]
    axis.scatter(
        point[0],
        point[1],
        marker=marker,
        s=62,
        facecolors=colour if filled else "white",
        edgecolors=colour,
        linewidths=1.5,
        zorder=6,
        label=label,
    )


def _flux_panel(
    axis: plt.Axes,
    radius: np.ndarray,
    height: np.ndarray,
    field: np.ndarray,
    levels: np.ndarray,
    row: dict[str, np.ndarray],
) -> None:
    axis.contourf(radius, height, field, levels=levels, cmap="viridis", extend="both")
    axis.contour(
        radius,
        height,
        field,
        levels=levels[::3],
        colors="white",
        linewidths=0.32,
        alpha=0.55,
    )
    _decorate_map(axis, row)


def render(comparison: Path, residuals: Path, pinned_cache: Path) -> dict[str, object]:
    head_receipt = json.loads(HEAD_RECEIPT.read_text())
    mechanism = json.loads(HEAD_MECHANISM.read_text())
    head_pure = _head_row(head_receipt, "pure")
    head_mixed = _head_row(head_receipt, "mixed")
    with (
        np.load(BASELINE_CACHE, allow_pickle=False) as baseline,
        np.load(pinned_cache, allow_pickle=False) as pinned,
    ):
        cpu_pure = _row(baseline, 10)
        cpu_mixed = _row(baseline, 11)
        gpu_pure = _row(pinned, 10)
        gpu_mixed = _row(pinned, 11)

    radius, height, pure_flux = _grid(gpu_pure)
    mixed_radius, mixed_height, mixed_flux = _grid(gpu_mixed)
    np.testing.assert_array_equal(radius, mixed_radius)
    np.testing.assert_array_equal(height, mixed_height)
    lower = float(min(pure_flux.min(), mixed_flux.min()))
    upper = float(max(pure_flux.max(), mixed_flux.max()))
    levels = np.linspace(lower, upper, 23)

    pure_o_delta = np.asarray(gpu_pure["selected_o"] - cpu_pure["selected_o"])[0]
    pure_x_delta = np.asarray(gpu_pure["selected_x"] - cpu_pure["selected_x"])[0]
    mixed_o_delta = np.asarray(gpu_mixed["selected_o"] - cpu_mixed["selected_o"])[0]
    mixed_x_delta = np.asarray(gpu_mixed["selected_x"] - cpu_mixed["selected_x"])[0]
    pure_o_mm = float(np.linalg.norm(pure_o_delta) * 1e3)
    pure_x_mm = float(np.linalg.norm(pure_x_delta) * 1e3)
    mixed_o_m = float(np.linalg.norm(mixed_o_delta))
    mixed_x_m = float(np.linalg.norm(mixed_x_delta))

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 13.2), constrained_layout=True)
    for axis in axes.flat:
        axis.set_facecolor("#fbfbf9")

    axis = axes[0, 0]
    _flux_panel(axis, radius, height, pure_flux, levels, gpu_pure)
    _mark(axis, cpu_pure["selected_o"], "o", CPU, "CPU O", filled=False)
    _mark(axis, cpu_pure["selected_x"], "X", CPU, "CPU X")
    axis.set_title("A  committed CPU bank row · converged", loc="left", color=INK)
    axis.text(
        0.02,
        0.02,
        "CPU flux was not banked\nbackground = pinned flux carrier",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        color=INK,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    axis = axes[0, 1]
    _flux_panel(axis, radius, height, pure_flux, levels, gpu_pure)
    _mark(axis, cpu_pure["selected_o"], "o", CPU, "CPU O", filled=False)
    _mark(axis, gpu_pure["selected_o"], "o", PINNED, "pinned O")
    _mark(axis, cpu_pure["selected_x"], "X", CPU, "CPU X")
    _mark(axis, gpu_pure["selected_x"], "X", PINNED, "pinned X")
    axis.plot(
        [cpu_pure["selected_o"][0, 0], gpu_pure["selected_o"][0, 0]],
        [cpu_pure["selected_o"][0, 1], gpu_pure["selected_o"][0, 1]],
        color=PINNED,
        lw=2,
    )
    axis.plot(
        [cpu_pure["selected_x"][0, 0], gpu_pure["selected_x"][0, 0]],
        [cpu_pure["selected_x"][0, 1], gpu_pure["selected_x"][0, 1]],
        color=PINNED,
        lw=2,
    )
    axis.set_title("B  pinned-revision H200 · stagnated", loc="left", color=INK)
    axis.text(
        0.02,
        0.02,
        f"recorded flux · ΔO {pure_o_mm:.2f} mm\nΔX {pure_x_mm:.2f} mm",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        color=INK,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    axis = axes[1, 0]
    _flux_panel(axis, radius, height, pure_flux, levels, gpu_pure)
    _mark(axis, cpu_pure["selected_x"], "X", CPU, "CPU X")
    _mark(axis, np.asarray(head_pure["nova_selected_saddle_m"]), "X", HEAD, "HEAD X")
    axis.set_title("C  HEAD H200 · active-set stagnation", loc="left", color=INK)
    axis.text(
        0.02,
        0.02,
        "HEAD flux grid and O point were not banked\n"
        f"X equals pinned row · residual {head_pure['terminal_residual']:.2e}",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        color=INK,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    axis = axes[1, 1]
    _flux_panel(axis, radius, height, mixed_flux, levels, gpu_mixed)
    _mark(axis, cpu_mixed["selected_o"], "o", CPU, "CPU O", filled=False)
    _mark(axis, gpu_mixed["selected_o"], "o", MIXED, "pinned O")
    _mark(axis, cpu_mixed["selected_x"], "X", CPU, "CPU X")
    _mark(axis, gpu_mixed["selected_x"], "X", MIXED, "pinned X")
    axis.set_title("D  mixed arm control · converged", loc="left", color=INK)
    axis.text(
        0.02,
        0.02,
        f"recorded flux · O {mixed_o_m:.2e} m\nX {mixed_x_m:.2e} m",
        transform=axis.transAxes,
        fontsize=9,
        va="bottom",
        color=INK,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )

    axes[0, 1].legend(loc="upper left", frameon=False, fontsize=8, ncol=2)
    colourbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(lower, upper), cmap="viridis"),
        ax=axes,
        location="right",
        shrink=0.72,
    )
    colourbar.set_label("poloidal flux per radian [Wb]")
    figure.suptitle(
        "MAST 22086/43 · recorded landmarks on one matched-level carrier",
        fontsize=18,
        color=INK,
    )
    comparison.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(comparison, dpi=190, facecolor="white")
    plt.close(figure)

    flux_difference = pure_flux - mixed_flux
    flux_limit = float(np.max(np.abs(flux_difference)))
    label_difference = (
        np.asarray(gpu_pure["domain_labels"] != cpu_pure["domain_labels"], dtype=int)
        .reshape((radius.size, height.size))
        .T
    )
    label_difference_count = int(label_difference.sum())
    residual_history = np.asarray(head_pure["active_set_residuals"], dtype=float)
    mask_differences = np.asarray(head_pure["active_set_mask_differences"], dtype=int)

    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.2), constrained_layout=True)
    axis = axes[0, 0]
    mesh = axis.pcolormesh(
        radius,
        height,
        flux_difference,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-flux_limit, vmax=flux_limit),
        shading="nearest",
    )
    _decorate_map(axis, gpu_pure)
    axis.set_title("A  pinned pure − pinned mixed flux", loc="left", color=INK)
    figure.colorbar(mesh, ax=axis, label="Δψ [Wb/rad]")

    axis = axes[0, 1]
    cmap = ListedColormap(["#f5f5f0", "#ca4b37"])
    mesh = axis.pcolormesh(
        radius,
        height,
        label_difference,
        cmap=cmap,
        norm=BoundaryNorm([-0.5, 0.5, 1.5], cmap.N),
        shading="nearest",
    )
    _decorate_map(axis, gpu_pure)
    axis.set_title(
        f"B  label changes · {label_difference_count} cells", loc="left", color=INK
    )
    figure.colorbar(mesh, ax=axis, ticks=[0, 1], label="unchanged / changed")

    axis = axes[1, 0]
    trips = np.arange(1, len(residual_history) + 1)
    axis.semilogy(trips, residual_history, marker="o", color=HEAD, lw=1.8)
    axis.axhline(float(head_pure["tolerance"]), color=PINNED, ls="--", lw=1.2)
    for trip, (residual, difference) in enumerate(
        zip(residual_history, mask_differences, strict=True), start=1
    ):
        axis.annotate(
            f"Δmask {difference}",
            (trip, residual),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_xlabel("active-set trip")
    axis.set_ylabel("relative sup residual")
    axis.set_xticks(trips)
    axis.grid(color=GRID, lw=0.45, which="both")
    axis.set_title("C  HEAD residual plateaus above tolerance", loc="left", color=INK)

    axis = axes[1, 1]
    vectors = np.stack((pure_o_delta, pure_x_delta)) * 1e3
    names = ["pure O", "pure X"]
    for vector, name in zip(vectors, names, strict=True):
        axis.arrow(
            0,
            0,
            vector[0],
            vector[1],
            color=PINNED,
            width=0.035,
            head_width=0.23,
            length_includes_head=True,
        )
        magnitude = float(np.linalg.norm(vector))
        axis.text(
            vector[0],
            vector[1],
            f"{name}  {magnitude:.3g} mm",
            color=PINNED,
            fontsize=9,
        )
    axis.scatter(0, 0, color=MIXED, s=34, zorder=4)
    axis.text(
        -2.3,
        0.72,
        f"mixed O/X ≤{max(mixed_o_m, mixed_x_m) * 1e3:.2e} mm",
        color=MIXED,
        fontsize=9,
    )
    axis.axhline(0, color=GRID, lw=0.7)
    axis.axvline(0, color=GRID, lw=0.7)
    axis.set_xlim(-2.6, 2.6)
    axis.set_ylim(-5.8, 1.2)
    axis.set_aspect("equal")
    axis.set_xlabel("ΔR [mm]")
    axis.set_ylabel("ΔZ [mm]")
    axis.set_title("D  pinned − committed primary displacement", loc="left", color=INK)

    residuals.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(residuals, dpi=190, facecolor="white")
    plt.close(figure)

    result = {
        "comparison": str(comparison),
        "residuals": str(residuals),
        "pure_o_delta_mm": pure_o_mm,
        "pure_x_delta_mm": pure_x_mm,
        "mixed_o_delta_m": mixed_o_m,
        "mixed_x_delta_m": mixed_x_m,
        "head_x_matches_pinned": bool(
            np.allclose(
                np.asarray(head_pure["nova_selected_saddle_m"]),
                gpu_pure["selected_x"][0],
                rtol=0.0,
                atol=1e-15,
            )
        ),
        "head_mixed_x_matches_pinned": bool(
            np.allclose(
                np.asarray(head_mixed["nova_selected_saddle_m"]),
                gpu_mixed["selected_x"][0],
                rtol=0.0,
                atol=1e-15,
            )
        ),
        "committed_to_pinned_label_difference_cells": label_difference_count,
        "pinned_pure_minus_mixed_flux_sup_wb_per_radian": flux_limit,
        "head_terminal_residual": float(head_pure["terminal_residual"]),
        "head_terminal_residual_over_tolerance": float(
            mechanism["bank_attribution"]["criterion_attribution"][
                "terminal_residual_over_tolerance"
            ]
        ),
        "field_availability": {
            "committed_cpu": "not banked; pinned flux shown as labelled carrier",
            "pinned_gpu": "banked schema-extension field",
            "head_gpu": "not banked; pinned flux shown as labelled carrier",
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-cache", type=Path, default=PINNED_CACHE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--residuals", type=Path, default=DEFAULT_RESIDUALS)
    arguments = parser.parse_args()
    render(arguments.comparison, arguments.residuals, arguments.pinned_cache)


if __name__ == "__main__":
    main()
