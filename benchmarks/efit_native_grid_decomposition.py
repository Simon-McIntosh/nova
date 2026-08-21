"""Bank the fitted-current EFIT decomposition on the native map grid.

The localisation path never transfers a field between grids.  It reads each
stored 65 by 65 live map, applies ``delta_star`` there, admits plasma source
cells once against the stored LCFS, retains whole linear moments for every
admitted boundary cell, and contracts those moments against fixed exact
section blocks evaluated on the same nodes.  The unmasked source arm is kept
as a counterfactual.  External flux uses ``efm/fcoil_c`` as the primary drive
and ``efm/fcoil_x`` beside it through the gain-check drive-times-response form.

Continuous map evaluation is confined to the reference qualification that
precedes decomposition: stored LCFS vertices, stored axis coordinates and
sub-grid saddle roots are checked in the reference's declared flux coordinate.
Those values never enter the residual or its implied-current localisation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import shapely
import zarr
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.interpolate import RectBivariateSpline

from benchmarks.efit_flux_decomposition import (
    _density_from_flux,
    _external_flux,
    _nearest_conductors,
    _polygon_centroid,
)
from benchmarks.efit_forward_input_census import FROZEN_SHOTS
from benchmarks.efit_roundtrip_linear_moments import (
    _contract,
    _coupling_blocks,
    _rectangles,
    _rectangular_mesh,
)
from benchmarks.efit_topology_boundary_score import (
    _live_flux_map,
    _slice_candidates,
    _stored_lcfs,
    _stored_x_points,
)
from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.imas.mast_vacuum_response import coil_sections

matplotlib.use("Agg")

DEFAULT_FIGURE_DIR = Path("docs/figures/efit-flux-decomposition")
DEFAULT_SLICES_PER_SHOT = 3
MAXIMUM_CIRCUIT_CENTROID_SEPARATION_M = 0.03
ROUND_TRIP_ORDER_FLOOR = 2.0
ROUND_TRIP_R_SQUARED_FLOOR = 0.999
CONTROL_GRID = (33, 49)
CONTROL_SUP_FRACTION = 5.152548846e-3
BANKED_EFIT_ORDER = 2.069958474
BANKED_EFIT_R_SQUARED = 0.999421810
BANKED_LINEAR_ORDER = 2.156
BANKED_LINEAR_R_SQUARED = 0.99913
CONVERGENCE_GRID = np.asarray([[17, 25], [25, 37], [33, 49], [41, 61]])
CONVERGENCE_CELL_SIZE_M = np.asarray(
    [0.129483795, 0.086322530, 0.064741897, 0.051793518]
)
CONVERGENCE_SUP_FRACTION = np.asarray(
    [8.719693412e-3, 3.614339425e-3, 2.034417834e-3, 1.307998208e-3]
)


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Return a float64 uniform axis while checking stored coordinates."""

    axis = np.asarray(values, dtype=np.float64)
    uniform = np.linspace(axis[0], axis[-1], len(axis))
    if axis.ndim != 1 or len(axis) != 65:
        raise ValueError(f"{name} must be a native 65-point axis")
    if not np.allclose(axis, uniform, rtol=2.0e-7, atol=1.0e-8):
        raise ValueError(f"{name} is not numerically uniform")
    return uniform


def _complete_rows(group: zarr.Group) -> np.ndarray:
    """Return complete map, geometry, anchor and dual-current rows."""

    candidates = _slice_candidates(group)
    row_count = len(candidates)

    def finite_rows(name: str) -> np.ndarray:
        values = np.asarray(group[name], dtype=np.float64)
        return np.all(np.isfinite(values.reshape(row_count, -1)), axis=1)

    anchors = np.isfinite(
        np.asarray(group["psi_axis"], dtype=np.float64)
    ) & np.isfinite(np.asarray(group["psi_boundary"], dtype=np.float64))
    return np.flatnonzero(
        candidates & anchors & finite_rows("fcoil_c") & finite_rows("fcoil_x")
    )


def _representative_rows(group: zarr.Group, count: int) -> np.ndarray:
    """Select fixed interior quantiles of the complete-input time sequence."""

    complete = _complete_rows(group)
    if len(complete) < count:
        raise ValueError(f"only {len(complete)} complete rows are available")
    fractions = np.linspace(0.25, 0.75, count)
    positions = np.rint(fractions * (len(complete) - 1)).astype(int)
    rows = complete[positions]
    if len(np.unique(rows)) != count:
        raise ValueError("representative complete-input rows are not distinct")
    return rows


def _map_saddles(
    flux_per_radian: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    seeds: np.ndarray,
) -> list[dict[str, float]]:
    """Refine stored X-point seeds to independently typed map saddles."""

    spline = RectBivariateSpline(height, radius, flux_per_radian, kx=3, ky=3, s=0.0)

    def gradient(point: np.ndarray) -> np.ndarray:
        r_value, z_value = point
        return np.asarray(
            [
                spline.ev(z_value, r_value, dx=0, dy=1),
                spline.ev(z_value, r_value, dx=1, dy=0),
            ],
            dtype=np.float64,
        )

    saddles: list[dict[str, float]] = []
    for seed in seeds:
        solved = optimize.root(gradient, np.asarray(seed, dtype=np.float64))
        r_value, z_value = solved.x
        if not (
            solved.success
            and radius[1] <= r_value <= radius[-2]
            and height[1] <= z_value <= height[-2]
            and np.linalg.norm(gradient(solved.x)) <= 1.0e-8
        ):
            continue
        radial = float(spline.ev(z_value, r_value, dx=0, dy=2))
        vertical = float(spline.ev(z_value, r_value, dx=2, dy=0))
        cross = float(spline.ev(z_value, r_value, dx=1, dy=1))
        determinant = radial * vertical - cross**2
        if determinant >= 0.0:
            continue
        if any(
            np.hypot(r_value - item["r_m"], z_value - item["z_m"]) <= 1.0e-5
            for item in saddles
        ):
            continue
        saddles.append(
            {
                "r_m": float(r_value),
                "z_m": float(z_value),
                "flux_wb_per_rad": float(spline.ev(z_value, r_value)),
                "gradient_norm_wb_per_rad_per_m": float(
                    np.linalg.norm(gradient(solved.x))
                ),
                "hessian_determinant": determinant,
            }
        )
    return saddles


def _reference_consistency(
    group: zarr.Group,
    row: int,
    flux_per_radian: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    lcfs: np.ndarray,
) -> dict[str, Any]:
    """Check declared LCFS and anchors against their own map before attribution."""

    axis_flux = float(group["psi_axis"][row])
    boundary_flux = float(group["psi_boundary"][row])
    declared_span = boundary_flux - axis_flux
    if declared_span == 0.0:
        raise ValueError("declared flux anchors have zero span")
    spline = RectBivariateSpline(height, radius, flux_per_radian, kx=3, ky=3, s=0.0)
    lcfs_flux = np.asarray(
        [spline.ev(z_value, r_value) for r_value, z_value in lcfs],
        dtype=np.float64,
    )
    lcfs_coordinate = (lcfs_flux - axis_flux) / declared_span
    stored_axis = np.asarray(
        [
            float(group["magnetic_axis_r"][row]),
            float(group["magnetic_axis_z"][row]),
        ]
    )
    map_axis_flux = float(spline.ev(stored_axis[1], stored_axis[0]))
    saddles = _map_saddles(
        flux_per_radian,
        radius,
        height,
        _stored_x_points(group, row),
    )
    for saddle in saddles:
        saddle["declared_coordinate"] = float(
            (saddle["flux_wb_per_rad"] - axis_flux) / declared_span
        )
    selected = (
        min(saddles, key=lambda item: abs(item["declared_coordinate"] - 1.0))
        if saddles
        else None
    )
    return {
        "checked_before_localisation": True,
        "continuous_evaluation_role": (
            "reference qualification only; no evaluated value enters decomposition"
        ),
        "normalization": (
            "(map flux - efm/psi_axis) / (efm/psi_boundary - efm/psi_axis)"
        ),
        "declared_axis_wb_per_rad": axis_flux,
        "declared_boundary_wb_per_rad": boundary_flux,
        "declared_span_wb_per_rad": declared_span,
        "stored_lcfs_against_map_contour": {
            "point_count": int(len(lcfs)),
            "declared_coordinate_sup_error_from_one": float(
                np.max(np.abs(lcfs_coordinate - 1.0))
            ),
            "declared_coordinate_rms_error_from_one": float(
                np.sqrt(np.mean((lcfs_coordinate - 1.0) ** 2))
            ),
            "median_map_flux_wb_per_rad": float(np.median(lcfs_flux)),
            "median_boundary_offset_fraction_of_declared_span": float(
                (np.median(lcfs_flux) - boundary_flux) / abs(declared_span)
            ),
        },
        "stored_axis_anchor_against_map": {
            "stored_axis_r_m": float(stored_axis[0]),
            "stored_axis_z_m": float(stored_axis[1]),
            "map_flux_wb_per_rad": map_axis_flux,
            "offset_fraction_of_declared_span": float(
                (map_axis_flux - axis_flux) / abs(declared_span)
            ),
        },
        "stored_boundary_anchor_against_map_saddle": {
            "candidate_count": len(saddles),
            "selected": selected,
            "offset_wb_per_rad": (
                None
                if selected is None
                else selected["flux_wb_per_rad"] - boundary_flux
            ),
            "offset_fraction_of_declared_span": (
                None
                if selected is None
                else (selected["flux_wb_per_rad"] - boundary_flux) / abs(declared_span)
            ),
            "classification": (
                "no typed map saddle resolved"
                if selected is None
                else "declared-boundary-vs-map-saddle offset measured"
            ),
        },
    }


def _support_mask(cells: list[np.ndarray], lcfs: np.ndarray) -> np.ndarray:
    """Admit every fixed cell with positive area inside the stored LCFS."""

    boundary = shapely.Polygon(lcfs)
    tolerance = np.finfo(np.float64).eps * max(abs(boundary.area), 1.0)
    return np.asarray(
        [
            shapely.Polygon(cell).intersection(boundary).area > tolerance
            for cell in cells
        ]
    )


def _linear_cell_vectors(
    mesh: Any, density: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact full-cell moments from native values and native gradients."""

    native_density = np.where(valid, density, 0.0)
    gradient = np.column_stack(mesh.gradient(native_density))
    area = np.asarray(mesh.cell_area, dtype=np.float64)
    return (
        native_density * area,
        gradient[:, 0] * area,
        gradient[:, 1] * area,
    )


def _restrict(
    vectors: tuple[np.ndarray, ...], admitted: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Remove exterior cells once while retaining whole admitted moments."""

    return tuple(np.where(admitted, vector, 0.0) for vector in vectors)


def _circuit_drives(
    group: zarr.Group,
    row: int,
    geometry: dict[str, Any],
    field: str,
) -> tuple[tuple[str, ...], np.ndarray, list[dict[str, Any]]]:
    """Map one stored current field to active winding packs by geometry."""

    sections = coil_sections(geometry)
    families = tuple(sorted(sections))
    family_centres = np.asarray(
        [_polygon_centroid(sections[family]) for family in families], dtype=np.float64
    )
    circuit_for_element = np.asarray(group["fcoil_circ"], dtype=int)
    element_r = np.asarray(group["fcoil_r"], dtype=np.float64)
    element_z = np.asarray(group["fcoil_z"], dtype=np.float64)
    turns = np.asarray(group["fcoil_turns"], dtype=np.float64)
    multiplier = np.asarray(group["fcoil_xmult"], dtype=np.float64)
    circuits = np.unique(circuit_for_element)[: len(families)]
    circuit_centres = []
    circuit_scale = []
    for circuit in circuits:
        selected = circuit_for_element == circuit
        weight = np.abs(turns[selected] * multiplier[selected])
        if not np.any(weight > 0.0):
            weight = np.ones(np.count_nonzero(selected))
        circuit_centres.append(
            [
                float(weight @ element_r[selected] / weight.sum()),
                float(weight @ element_z[selected] / weight.sum()),
            ]
        )
        circuit_scale.append(float(np.sum(turns[selected] * multiplier[selected])))
    circuit_centres = np.asarray(circuit_centres)
    separation = np.linalg.norm(
        family_centres[:, None, :] - circuit_centres[None, :, :], axis=2
    )
    from scipy.optimize import linear_sum_assignment

    family_rows, circuit_columns = linear_sum_assignment(separation)
    currents = np.asarray(group[field][row], dtype=np.float64)
    indices = np.asarray(group["fcoil_n"], dtype=int)
    if not np.array_equal(indices, np.arange(len(currents))):
        raise ValueError("fcoil_n does not provide zero-based stored-current order")
    drives = []
    mapping = []
    for family_index, circuit_column in zip(family_rows, circuit_columns, strict=True):
        distance = float(separation[family_index, circuit_column])
        if distance > MAXIMUM_CIRCUIT_CENTROID_SEPARATION_M:
            raise ValueError(
                f"{families[family_index]} is {distance:.6g} m from its circuit"
            )
        circuit = int(circuits[circuit_column])
        raw = float(currents[circuit - 1])
        scale = float(circuit_scale[circuit_column])
        drives.append(raw * scale)
        mapping.append(
            {
                "family": families[family_index],
                "stored_circuit": circuit,
                "centroid_separation_m": distance,
                "current_field": f"efm/{field}",
                "raw_current_a": raw,
                "filament_turn_multiplier_sum": scale,
                "effective_drive_a_turn": raw * scale,
            }
        )
    return families, np.asarray(drives), mapping


def _current_metrics(
    lattice: FluxLattice,
    flux_zr: np.ndarray,
    admitted_rz: np.ndarray,
    geometry: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read native residual current and localise its global absolute peak."""

    density_rz, valid_rz = _density_from_flux(lattice, flux_zr)
    readable = np.where(valid_rz, density_rz, 0.0)
    area = np.asarray(lattice.cell_area).reshape(lattice.shape)
    exterior = valid_rz & ~admitted_rz
    peak = np.unravel_index(int(np.argmax(np.abs(readable))), readable.shape)
    peak_r = float(lattice.radius[peak[0]])
    peak_z = float(lattice.height[peak[1]])
    nearest = _nearest_conductors(geometry, peak_r, peak_z)
    return density_rz, {
        "derivative_nodes": int(np.count_nonzero(valid_rz)),
        "signed_integral_a": float(np.sum(readable * area)),
        "absolute_integral_a": float(np.sum(np.abs(readable) * area)),
        "exterior_signed_integral_a": float(
            np.sum(readable[exterior] * area[exterior])
        ),
        "exterior_absolute_integral_a": float(
            np.sum(np.abs(readable[exterior]) * area[exterior])
        ),
        "peak_signed_a_per_m2": float(readable[peak]),
        "peak_absolute_a_per_m2": float(abs(readable[peak])),
        "peak_r_m": peak_r,
        "peak_z_m": peak_z,
        "nearest_named_conductors": nearest,
        "peak_conductor": nearest[0]["family"],
        "peak_inside_named_conductor": bool(nearest[0]["distance_m"] == 0.0),
    }


def _slice_receipt(
    *,
    shot: int,
    group: zarr.Group,
    row: int,
    radius: np.ndarray,
    height: np.ndarray,
    mesh: Any,
    cells: list[np.ndarray],
    blocks: np.ndarray,
    store_figure: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """Decompose one complete native-grid reference row."""

    flux_per_radian = _live_flux_map(group, row, len(radius))
    stored_flux = TOTAL_FLUX_FACTOR * flux_per_radian
    span = float(np.ptp(stored_flux))
    lcfs = _stored_lcfs(group, row)
    consistency = _reference_consistency(
        group, row, flux_per_radian, radius, height, lcfs
    )
    lattice = FluxLattice(radius, height)
    density_rz, valid_rz = _density_from_flux(lattice, stored_flux)
    density = density_rz.T.ravel()
    valid = valid_rz.T.ravel()
    all_vectors = _linear_cell_vectors(mesh, density, valid)
    admitted = _support_mask(cells, lcfs)
    masked_once_vectors = _restrict(all_vectors, admitted)
    plasma_flux = _contract(blocks, masked_once_vectors).reshape(
        len(height), len(radius)
    )
    unmasked_flux = _contract(blocks, all_vectors).reshape(len(height), len(radius))
    admitted_rz = admitted.reshape(len(height), len(radius)).T

    selection = MachineGeometryRegistry.default().select(shot)
    geometry = selection.configuration.geometry
    rr, zz = np.meshgrid(radius, height)
    positions = np.column_stack((rr.ravel(), zz.ravel()))
    arms: dict[str, Any] = {}
    figure_fields: dict[str, np.ndarray] = {}
    pre_external = stored_flux - plasma_flux
    _, before = _current_metrics(lattice, pre_external, admitted_rz, geometry)
    for field in ("fcoil_c", "fcoil_x"):
        families, drives, mapping = _circuit_drives(group, row, geometry, field)
        external = _external_flux(geometry, positions, families, drives).reshape(
            stored_flux.shape
        )
        residual = pre_external - external
        residual_density, current = _current_metrics(
            lattice, residual, admitted_rz, geometry
        )
        exterior_before = float(before["exterior_absolute_integral_a"])
        exterior_after = float(current["exterior_absolute_integral_a"])
        arms[field] = {
            "current_field": f"efm/{field}",
            "drive_times_response_form": (
                "effective circuit drive @ loop_response_matrix response rows"
            ),
            "circuit_mapping": mapping,
            "external_flux_sup_wb": float(np.max(np.abs(external))),
            "external_flux_sup_fraction_of_stored_span": float(
                np.max(np.abs(external)) / span
            ),
            "exterior_absolute_current_before_external_a": exterior_before,
            "exterior_absolute_current_after_external_a": exterior_after,
            "exterior_absolute_current_cancelled_fraction": float(
                1.0 - exterior_after / exterior_before
            ),
            "residual": {
                "definition": "stored - masked-once plasma - external",
                "threshold_applied_or_registered": False,
                "sup_wb": float(np.max(np.abs(residual))),
                "sup_fraction_of_stored_span": float(np.max(np.abs(residual)) / span),
                "rms_wb": float(np.sqrt(np.mean(residual**2))),
                "rms_fraction_of_stored_span": float(
                    np.sqrt(np.mean(residual**2)) / span
                ),
                "implied_current": current,
            },
        }
        if store_figure:
            figure_fields[f"{field}_external"] = external
            figure_fields[f"{field}_residual_density"] = residual_density.T

    receipt = {
        "shot": shot,
        "slice_index": row,
        "time_s": float(group["time"][row]),
        "native_grid_shape_zr": list(stored_flux.shape),
        "stored_flux_span_wb": span,
        "reference_consistency_checked_before_localisation": consistency,
        "localisation_path": {
            "interpolation_used": False,
            "grid_transfer_used": False,
            "grid": "efm/gridz by efm/gridr native live plane",
            "plasma_current": "native delta_star of efm/psirz",
            "support": "stored LCFS positive-area cell intersection",
            "masking": (
                "masked once: exterior cells removed; admitted boundary moments whole"
            ),
            "external": "native loop_response_matrix",
            "residual_current": "native delta_star of native residual",
        },
        "plasma_term": {
            "representation": (
                "cellwise-linear full rectangular moments from native values and "
                "native stencil gradients through exact fixed G0, GR, GZ blocks"
            ),
            "admitted_cell_count": int(np.count_nonzero(admitted)),
            "removed_exterior_cell_count": int(np.count_nonzero(~admitted)),
            "boundary_moments_reclipped": False,
            "masked_once_flux_sup_wb": float(np.max(np.abs(plasma_flux))),
            "unmasked_counterfactual_flux_sup_wb": float(np.max(np.abs(unmasked_flux))),
            "masked_once_minus_unmasked_sup_fraction_of_stored_span": float(
                np.max(np.abs(plasma_flux - unmasked_flux)) / span
            ),
        },
        "external_arms": arms,
    }
    return receipt, figure_fields if store_figure else None


def _convergence_receipt() -> dict[str, Any]:
    """Return the locked order criterion and banked native control point."""

    return {
        "criterion": {
            "minimum_observed_order": ROUND_TRIP_ORDER_FLOOR,
            "minimum_r_squared": ROUND_TRIP_R_SQUARED_FLOOR,
            "fixed_working_grid_error_threshold": None,
        },
        "banked_efit_series": {
            "observed_order": BANKED_EFIT_ORDER,
            "r_squared": BANKED_EFIT_R_SQUARED,
            "passes": bool(
                BANKED_EFIT_ORDER >= ROUND_TRIP_ORDER_FLOOR
                and BANKED_EFIT_R_SQUARED >= ROUND_TRIP_R_SQUARED_FLOOR
            ),
            "control_grid": list(CONTROL_GRID),
            "control_sup_fraction_of_span": CONTROL_SUP_FRACTION,
        },
        "banked_linear_series": {
            "observed_order": BANKED_LINEAR_ORDER,
            "r_squared": BANKED_LINEAR_R_SQUARED,
            "passes": bool(
                BANKED_LINEAR_ORDER >= ROUND_TRIP_ORDER_FLOOR
                and BANKED_LINEAR_R_SQUARED >= ROUND_TRIP_R_SQUARED_FLOOR
            ),
        },
        "banked_source_commits": ["e25d1eb4", "a25b3b3c", "d82587cd"],
    }


def _plot_convergence(path: Path) -> None:
    """Reproduce the banked convergence series and highlight its control."""

    figure, axis = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    axis.loglog(
        CONVERGENCE_CELL_SIZE_M,
        CONVERGENCE_SUP_FRACTION,
        "o-",
        color="#1f4e79",
        label=f"order {BANKED_EFIT_ORDER:.3f}, R² {BANKED_EFIT_R_SQUARED:.6f}",
    )
    control = 2
    axis.scatter(
        CONVERGENCE_CELL_SIZE_M[control],
        CONVERGENCE_SUP_FRACTION[control],
        s=90,
        facecolors="none",
        edgecolors="#b54a2f",
        linewidths=1.8,
        zorder=3,
    )
    axis.annotate(
        f"33×49 control\n{CONTROL_SUP_FRACTION:.3e} of span",
        (CONVERGENCE_CELL_SIZE_M[control], CONVERGENCE_SUP_FRACTION[control]),
        xytext=(18, 18),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#555"},
    )
    axis.set_xlabel("characteristic cell size (m)")
    axis.set_ylabel("round-trip sup / stored span")
    axis.grid(True, which="both", color="#ddd", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_decomposition(
    path: Path,
    figure_rows: list[tuple[int, np.ndarray, np.ndarray, dict[str, np.ndarray]]],
) -> None:
    """Place fitted and experimental external fields beside residual current."""

    figure, axes = plt.subplots(
        len(figure_rows), 3, figsize=(10.8, 2.35 * len(figure_rows)), squeeze=False
    )
    for row_axes, (shot, radius, height, fields) in zip(axes, figure_rows, strict=True):
        external_limit = max(
            float(np.max(np.abs(fields["fcoil_c_external"]))),
            float(np.max(np.abs(fields["fcoil_x_external"]))),
        )
        current = fields["fcoil_c_residual_density"]
        current_limit = float(np.max(np.abs(current)))
        panels = (
            (fields["fcoil_c_external"], external_limit, "fcoil_c external (Wb)"),
            (fields["fcoil_x_external"], external_limit, "fcoil_x external (Wb)"),
            (current, current_limit, "fcoil_c residual Δ* (A/m²)"),
        )
        for axis, (values, limit, title) in zip(row_axes, panels, strict=True):
            image = axis.pcolormesh(
                radius,
                height,
                values,
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_aspect("equal")
            axis.set_title(title, fontsize=8)
            axis.set_ylabel(f"{shot}\nZ (m)", fontsize=8)
            axis.tick_params(labelsize=7)
            figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        for axis in row_axes:
            axis.set_xlabel("R (m)", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def measure(arguments: argparse.Namespace) -> dict[str, Any]:
    """Build one native interaction bank and decompose the requested cohort."""

    shots = tuple(arguments.shots)
    first = zarr.open_group(str(arguments.store / f"{shots[0]}.zarr"), mode="r")["efm"]
    radius = _uniform_axis(np.asarray(first["gridr"]), "efm/gridr")
    height = _uniform_axis(np.asarray(first["gridz"]), "efm/gridz")
    mesh, width, vertical_extent = _rectangular_mesh(radius, height)
    cells = _rectangles(mesh.coordinate, width, vertical_extent)
    blocks = _coupling_blocks(mesh, cells, width, vertical_extent, arguments.workers)[
        :3
    ]
    rows = []
    figure_rows = []
    for shot in shots:
        group = zarr.open_group(str(arguments.store / f"{shot}.zarr"), mode="r")["efm"]
        shot_radius = _uniform_axis(np.asarray(group["gridr"]), "efm/gridr")
        shot_height = _uniform_axis(np.asarray(group["gridz"]), "efm/gridz")
        if not (
            np.array_equal(shot_radius, radius) and np.array_equal(shot_height, height)
        ):
            raise ValueError("frozen-shot native grids differ; one bank is not valid")
        selected_rows = _representative_rows(group, arguments.slices_per_shot)
        middle = int(selected_rows[len(selected_rows) // 2])
        for row in selected_rows:
            receipt, fields = _slice_receipt(
                shot=shot,
                group=group,
                row=int(row),
                radius=radius,
                height=height,
                mesh=mesh,
                cells=cells,
                blocks=blocks,
                store_figure=int(row) == middle,
            )
            rows.append(receipt)
            if fields is not None:
                figure_rows.append((shot, radius, height, fields))

    fitted_cancellation = np.asarray(
        [
            row["external_arms"]["fcoil_c"][
                "exterior_absolute_current_cancelled_fraction"
            ]
            for row in rows
        ]
    )
    experimental_cancellation = np.asarray(
        [
            row["external_arms"]["fcoil_x"][
                "exterior_absolute_current_cancelled_fraction"
            ]
            for row in rows
        ]
    )
    peak_conductors = [
        row["external_arms"]["fcoil_c"]["residual"]["implied_current"]["peak_conductor"]
        for row in rows
    ]
    peak_counts = Counter(peak_conductors)
    bank_peak = max(
        rows,
        key=lambda row: row["external_arms"]["fcoil_c"]["residual"]["implied_current"][
            "peak_absolute_a_per_m2"
        ],
    )
    bank_peak_current = bank_peak["external_arms"]["fcoil_c"]["residual"][
        "implied_current"
    ]
    modal = peak_counts.most_common(1)[0][0]
    p4_survives = bool(
        modal == "p4_lower"
        and bank_peak_current["peak_conductor"] == "p4_lower"
        and bank_peak_current["peak_inside_named_conductor"]
    )
    convergence = _convergence_receipt()
    external_fit = bool(
        convergence["banked_efit_series"]["passes"]
        and convergence["banked_linear_series"]["passes"]
        and np.all(fitted_cancellation > 0.0)
    )
    result = {
        "receipt": "native-grid fitted-current EFIT flux decomposition",
        "cohort": {
            "shots": list(shots),
            "slices_per_shot": arguments.slices_per_shot,
            "slice_count": len(rows),
            "selection": "25%, 50% and 75% quantiles of complete-input rows",
            "all_native_shapes_zr": sorted(
                {tuple(row["native_grid_shape_zr"]) for row in rows}
            ),
        },
        "round_trip_criterion": convergence,
        "masking_contract": {
            "semantics": (
                "masked once: exterior cells removed and admitted boundary "
                "moments whole"
            ),
            "unmasked_counterfactual_reported_per_slice": True,
            "banked_measurement_commit": "d82587cd",
        },
        "localisation_interpolation_audit": {
            "interpolation_anywhere_in_localisation_path": False,
            "native_grid_transfer_count": 0,
            "qualification_interpolant_enters_localisation": False,
        },
        "aggregate_external_field": {
            "primary_current_field": "efm/fcoil_c",
            "comparison_current_field": "efm/fcoil_x",
            "method": "drive @ loop_response_matrix.T on native nodes",
            "fcoil_c_cancellation_fraction": {
                "minimum": float(np.min(fitted_cancellation)),
                "median": float(np.median(fitted_cancellation)),
                "maximum": float(np.max(fitted_cancellation)),
            },
            "fcoil_x_cancellation_fraction": {
                "minimum": float(np.min(experimental_cancellation)),
                "median": float(np.median(experimental_cancellation)),
                "maximum": float(np.max(experimental_cancellation)),
            },
        },
        "plain_verdict": {
            "external_field_fit_for_free_boundary_condition": external_fit,
            "external_field": "FIT" if external_fit else "NOT FIT",
            "external_field_basis": (
                "both banked order criteria pass and fitted-current exterior "
                "absolute-current cancellation is positive on every sampled slice; "
                "no residual threshold is applied"
            ),
            "p4_lower_peak": "SURVIVES" if p4_survives else "RETRACTS",
            "p4_lower_rule": (
                "survives only if p4_lower is the modal fitted-current peak "
                "conductor and contains the bank-global absolute peak"
            ),
            "fitted_current_peak_conductor_counts": dict(sorted(peak_counts.items())),
            "bank_global_peak": {
                "shot": bank_peak["shot"],
                "slice_index": bank_peak["slice_index"],
                "time_s": bank_peak["time_s"],
                **bank_peak_current,
            },
        },
        "residual_policy": {
            "threshold_applied_or_registered": False,
            "reported_quantities": (
                "sup and RMS fractions of stored span; signed and absolute current "
                "integrals; signed peak density and native R,Z localisation"
            ),
        },
        "slices": rows,
    }
    arguments.figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_convergence(arguments.figure_dir / "roundtrip-convergence.png")
    _plot_decomposition(
        arguments.figure_dir / "native-grid-decomposition.png", figure_rows
    )
    output = arguments.figure_dir / "native-grid-decomposition.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(output),
        "figures": [
            str(arguments.figure_dir / "native-grid-decomposition.png"),
            str(arguments.figure_dir / "roundtrip-convergence.png"),
        ],
        "slice_count": len(rows),
        "plain_verdict": result["plain_verdict"],
        "aggregate_external_field": result["aggregate_external_field"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--shots", type=int, nargs="+", default=list(FROZEN_SHOTS))
    parser.add_argument("--slices-per-shot", type=int, default=DEFAULT_SLICES_PER_SHOT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser


def main() -> None:
    """Print a compact summary while banking the full receipt and figures."""

    print(json.dumps(measure(_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
