"""Decompose a stored EFIT flux map with Nova's physical operators.

The stored map and LCFS remain the referee.  EFIT flux per toroidal radian is
converted to Nova total flux, and the conservation ``delta_star`` operator is
read on the stored grid.  Only current inside the stored LCFS is interpolated
to and driven through Nova's plasma-current-to-grid coupling.  Exterior
current never enters the plasma term and is reported independently.

The external term uses the physical-registry winding packs and
``loop_response_matrix``.  Its drive is the experimental ``fcoil_x`` row,
mapped through the stored circuit geometry and filament multipliers.  The
contraction is the gain-check prediction form: drive row times response rows.
EFIT's fitted ``fcoil_c`` output is deliberately not read.

Residual norms cover the complete Nova grid without fitting or exclusion.
Only current-density diagnostics are restricted to nodes with a complete
central-difference stencil, because the conservation operator is undefined at
the finite grid boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import zarr
from scipy.constants import mu_0
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.equilibrium.conservation import FluxLattice, delta_star
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.imas.mast_vacuum_response import coil_sections, loop_response_matrix

DEFAULT_SHOT = 21978
DEFAULT_TIME_S = 0.245
OPERATOR_ROUND_TRIP_BOUND = 1.0e-6
OPERATOR_STENCIL_MARGIN = 2
MAXIMUM_CIRCUIT_CENTROID_SEPARATION_M = 0.03


@dataclass(frozen=True)
class StoredSlice:
    """One usable EFIT slice and its stored physical metadata."""

    index: int
    time_s: float
    radius_m: np.ndarray
    height_m: np.ndarray
    total_flux_wb: np.ndarray
    lcfs_radius_m: np.ndarray
    lcfs_height_m: np.ndarray
    plasma_current_a: float


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    """Return a numerically uniform axis without retaining float32 jitter."""

    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate axis")
    uniform = np.linspace(axis[0], axis[-1], axis.size)
    if not np.allclose(axis, uniform, rtol=2.0e-7, atol=1.0e-8):
        raise ValueError(f"{name} is not uniformly spaced")
    return uniform


def _stored_lcfs(group: zarr.Group, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite stored LCFS vertices for one row."""

    count_value = float(group["lcfsn_c"][index])
    if not np.isfinite(count_value) or count_value != int(count_value):
        raise ValueError(f"stored LCFS point count is invalid: {count_value}")
    count = int(count_value)
    radius = np.asarray(group["lcfs_r"][index], dtype=float)[:count]
    height = np.asarray(group["lcfs_z"][index], dtype=float)[:count]
    if count < 3 or not np.all(np.isfinite(radius)) or not np.all(np.isfinite(height)):
        raise ValueError("stored LCFS does not contain a finite polygon")
    return radius, height


def _read_stored_slice(group: zarr.Group, requested_time_s: float) -> StoredSlice:
    """Read the nearest usable finite map, removing only alignment padding."""

    times = np.asarray(group["time"], dtype=float)
    stored_radius = np.asarray(group["gridr"], dtype=float)
    radius = _uniform_axis(stored_radius, "gridr")
    height = _uniform_axis(np.asarray(group["gridz"], dtype=float), "gridz")
    profile_radius = np.asarray(group["profile_r"], dtype=float)
    for index in np.argsort(np.abs(times - requested_time_s)):
        aligned = np.asarray(group["psirz"][int(index)], dtype=float)
        finite_columns = np.flatnonzero(np.all(np.isfinite(aligned), axis=0))
        if finite_columns.size != radius.size:
            continue
        if not np.allclose(
            profile_radius[finite_columns], stored_radius, rtol=2.0e-7, atol=1.0e-8
        ):
            continue
        per_radian = aligned[:, finite_columns]
        if per_radian.shape != (height.size, radius.size):
            continue
        lcfs_radius, lcfs_height = _stored_lcfs(group, int(index))
        plasma_current = float(group["plasma_current_c"][int(index)])
        if not np.isfinite(plasma_current) or plasma_current == 0.0:
            continue
        return StoredSlice(
            index=int(index),
            time_s=float(times[index]),
            radius_m=radius,
            height_m=height,
            total_flux_wb=TOTAL_FLUX_FACTOR * per_radian,
            lcfs_radius_m=lcfs_radius,
            lcfs_height_m=lcfs_height,
            plasma_current_a=plasma_current,
        )
    raise ValueError(f"no usable EFIT slice exists near {requested_time_s:g} s")


def _lcfs_mask(
    radius: np.ndarray,
    height: np.ndarray,
    lcfs_radius: np.ndarray,
    lcfs_height: np.ndarray,
) -> np.ndarray:
    """Return LCFS containment in ``(radius, height)`` lattice order."""

    radius_grid, height_grid = np.meshgrid(radius, height, indexing="ij")
    return inside_polygon(
        radius_grid.ravel(),
        height_grid.ravel(),
        lcfs_radius,
        lcfs_height,
    ).reshape(radius_grid.shape)


def _density_from_flux(
    lattice: FluxLattice, flux_zr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply conservation delta-star and return density in lattice order."""

    elliptic = np.asarray(delta_star(lattice, flux_zr.T.ravel()), dtype=float)
    density = -elliptic / (TOTAL_FLUX_FACTOR * mu_0 * np.asarray(lattice.node_radius))
    valid = np.asarray(lattice.interior(margin=OPERATOR_STENCIL_MARGIN), dtype=bool)
    return density.reshape(lattice.shape), valid.reshape(lattice.shape)


def _interpolator(
    source_r: np.ndarray, source_z: np.ndarray, values_zr: np.ndarray
) -> RegularGridInterpolator:
    """Build the named bilinear structured-grid interpolation used here."""

    return RegularGridInterpolator(
        (source_z, source_r), values_zr, method="linear", bounds_error=True
    )


def _evaluate_on_grid(
    interpolation: RegularGridInterpolator,
    target_r: np.ndarray,
    target_z: np.ndarray,
) -> np.ndarray:
    """Evaluate a structured-grid interpolation in ``(height, radius)`` order."""

    radius, height = np.meshgrid(target_r, target_z)
    return interpolation(np.column_stack((height.ravel(), radius.ravel()))).reshape(
        height.shape
    )


def _flux_interpolation_receipt(
    stored: StoredSlice,
    nova_r: np.ndarray,
    nova_z: np.ndarray,
    nova_flux: np.ndarray,
) -> dict[str, float | int | str]:
    """Measure the bilinear forward/reverse error over the shared grid extent."""

    reverse = RegularGridInterpolator(
        (nova_z, nova_r), nova_flux, method="linear", bounds_error=False
    )
    height, radius = np.meshgrid(stored.height_m, stored.radius_m, indexing="ij")
    overlap = (
        (radius >= nova_r[0])
        & (radius <= nova_r[-1])
        & (height >= nova_z[0])
        & (height <= nova_z[-1])
    )
    returned = reverse(np.column_stack((height[overlap], radius[overlap])))
    error = returned - stored.total_flux_wb[overlap]
    span = float(np.ptp(stored.total_flux_wb))
    return {
        "method": "bilinear regular-grid EFIT-to-Nova-to-EFIT round trip",
        "overlap_points": int(np.count_nonzero(overlap)),
        "sup_error_wb": float(np.max(np.abs(error))),
        "sup_error_fraction_of_stored_span": float(np.max(np.abs(error)) / span),
    }


def _plasma_flux(
    profile: Any, lattice: FluxLattice, density_rz: np.ndarray
) -> np.ndarray:
    """Drive Nova's production plasma-current-to-grid coupling."""

    cell_current_zr = (density_rz * lattice.cell_area.reshape(lattice.shape)).T
    return (
        np.asarray(profile.plasma_to_grid, dtype=float) @ cell_current_zr.ravel()
    ).reshape(profile.grid_z.size, profile.grid_r.size)


def _polygon_centroid(vertices: tuple[np.ndarray, ...]) -> tuple[float, float]:
    """Return an area-weighted winding-pack centroid."""

    polygons = [shapely.Polygon(part) for part in vertices]
    areas = np.asarray([abs(polygon.area) for polygon in polygons], dtype=float)
    radius = np.asarray([polygon.centroid.x for polygon in polygons], dtype=float)
    height = np.asarray([polygon.centroid.y for polygon in polygons], dtype=float)
    return float(areas @ radius / areas.sum()), float(areas @ height / areas.sum())


def _experimental_circuit_drives(
    group: zarr.Group,
    row: int,
    geometry: dict[str, Any],
) -> tuple[tuple[str, ...], np.ndarray, list[dict[str, float | int | str]]]:
    """Map experimental fcoil currents to described winding packs by geometry."""

    sections = coil_sections(geometry)
    families = tuple(sorted(sections))
    family_centroids = np.asarray(
        [_polygon_centroid(sections[family]) for family in families], dtype=float
    )

    circuit_for_element = np.asarray(group["fcoil_circ"], dtype=int)
    element_r = np.asarray(group["fcoil_r"], dtype=float)
    element_z = np.asarray(group["fcoil_z"], dtype=float)
    turns = np.asarray(group["fcoil_turns"], dtype=float)
    multiplier = np.asarray(group["fcoil_xmult"], dtype=float)
    all_circuits = np.unique(circuit_for_element)
    if all_circuits.size < len(families):
        raise ValueError("stored circuit table has fewer circuits than active families")
    circuits = all_circuits[: len(families)]
    circuit_centroids = []
    circuit_scale = []
    for circuit in circuits:
        selected = circuit_for_element == circuit
        weight = np.abs(turns[selected] * multiplier[selected])
        if not np.any(weight > 0.0):
            weight = np.ones(np.count_nonzero(selected))
        circuit_centroids.append(
            [
                float(weight @ element_r[selected] / weight.sum()),
                float(weight @ element_z[selected] / weight.sum()),
            ]
        )
        circuit_scale.append(float(np.sum(turns[selected] * multiplier[selected])))
    circuit_centroids_array = np.asarray(circuit_centroids)
    separation = np.hypot(
        family_centroids[:, None, 0] - circuit_centroids_array[None, :, 0],
        family_centroids[:, None, 1] - circuit_centroids_array[None, :, 1],
    )
    family_rows, circuit_columns = linear_sum_assignment(separation)
    if not np.array_equal(family_rows, np.arange(len(families))):
        raise ValueError("physical-family assignment did not cover every family")

    experimental = np.asarray(group["fcoil_x"][row], dtype=float)
    fcoil_index = np.asarray(group["fcoil_n"], dtype=int)
    if not np.array_equal(fcoil_index, np.arange(experimental.size)):
        raise ValueError(
            "fcoil_n does not provide the expected zero-based current order"
        )
    drives = []
    mapping = []
    for family_index, circuit_column in zip(family_rows, circuit_columns, strict=True):
        distance = float(separation[family_index, circuit_column])
        if distance > MAXIMUM_CIRCUIT_CENTROID_SEPARATION_M:
            raise ValueError(
                f"active family {families[family_index]!r} is {distance:.6g} m "
                "from its nearest stored circuit"
            )
        circuit = int(circuits[circuit_column])
        raw_current = float(experimental[circuit - 1])
        scale = float(circuit_scale[circuit_column])
        effective_drive = raw_current * scale
        drives.append(effective_drive)
        mapping.append(
            {
                "family": families[family_index],
                "stored_circuit": circuit,
                "centroid_separation_m": distance,
                "fcoil_x_a": raw_current,
                "filament_turn_multiplier_sum": scale,
                "effective_drive_a_turn": effective_drive,
            }
        )
    return families, np.asarray(drives), mapping


def _external_flux(
    geometry: dict[str, Any],
    positions: np.ndarray,
    families: tuple[str, ...],
    drives: np.ndarray,
) -> np.ndarray:
    """Apply the described response with the gain-check prediction contraction."""

    response = loop_response_matrix(geometry, positions, families=families)
    return drives @ response.T


def _nearest_conductors(
    geometry: dict[str, Any], radius: float, height: float, count: int = 3
) -> list[dict[str, float | str]]:
    """Name the physical winding packs nearest one grid point."""

    point = shapely.Point(radius, height)
    distances = []
    for family, parts in coil_sections(geometry).items():
        distance = min(point.distance(shapely.Polygon(part)) for part in parts)
        distances.append({"family": family, "distance_m": float(distance)})
    return sorted(distances, key=lambda item: float(item["distance_m"]))[:count]


def _current_receipt(
    lattice: FluxLattice,
    flux_zr: np.ndarray,
    lcfs_rz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return residual current plus complete and exterior diagnostics."""

    density, valid = _density_from_flux(lattice, flux_zr)
    read_density = np.where(valid, density, 0.0)
    cell_area = lattice.cell_area.reshape(lattice.shape)
    exterior = valid & ~lcfs_rz
    return (
        density,
        valid,
        {
            "derivative_nodes": int(np.count_nonzero(valid)),
            "exterior_nodes": int(np.count_nonzero(exterior)),
            "signed_integral_a": float(np.sum(read_density * cell_area)),
            "absolute_integral_a": float(np.sum(np.abs(read_density) * cell_area)),
            "exterior_signed_integral_a": float(
                np.sum(read_density[exterior] * cell_area[exterior])
            ),
            "exterior_absolute_integral_a": float(
                np.sum(np.abs(read_density[exterior]) * cell_area[exterior])
            ),
        },
    )


def decompose(
    *,
    shot: int,
    requested_time_s: float,
    store: Path,
    artifact_cache: Path,
    artifact_digest: str,
) -> dict[str, Any]:
    """Run the measured decomposition and return a JSON-compatible receipt."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    stored = _read_stored_slice(group, requested_time_s)
    stored_span = float(np.ptp(stored.total_flux_wb))
    stored_lattice = FluxLattice(stored.radius_m, stored.height_m)
    stored_density, stored_valid = _density_from_flux(
        stored_lattice, stored.total_flux_wb
    )
    stored_lcfs = _lcfs_mask(
        stored.radius_m,
        stored.height_m,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    plasma_cells = stored_valid & stored_lcfs
    exterior_cells = stored_valid & ~stored_lcfs
    cell_area = stored_lattice.cell_area.reshape(stored_lattice.shape)
    plasma_density = np.where(plasma_cells, stored_density, 0.0)
    exterior_signed = float(
        np.sum(stored_density[exterior_cells] * cell_area[exterior_cells])
    )
    exterior_absolute = float(
        np.sum(np.abs(stored_density[exterior_cells]) * cell_area[exterior_cells])
    )

    chain = build_mast_parity_chain(
        shot,
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
    )
    profile = chain.profile_solver
    nova_r = np.asarray(profile.grid_r, dtype=float)
    nova_z = np.asarray(profile.grid_z, dtype=float)
    nova_lattice = FluxLattice(nova_r, nova_z)
    nova_lcfs = _lcfs_mask(
        nova_r,
        nova_z,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    nova_flux = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored.total_flux_wb),
        nova_r,
        nova_z,
    )
    interpolation = _flux_interpolation_receipt(stored, nova_r, nova_z, nova_flux)
    interpolated_plasma_density = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, plasma_density.T),
        nova_r,
        nova_z,
    ).T
    interpolated_plasma_density = np.where(nova_lcfs, interpolated_plasma_density, 0.0)
    plasma_flux = _plasma_flux(profile, nova_lattice, interpolated_plasma_density)

    repeated_density, repeated_valid = _density_from_flux(nova_lattice, plasma_flux)
    repeated_density = np.where(repeated_valid & nova_lcfs, repeated_density, 0.0)
    repeated_plasma_flux = _plasma_flux(profile, nova_lattice, repeated_density)
    operator_error = float(np.max(np.abs(plasma_flux - repeated_plasma_flux)))
    operator_fraction = operator_error / stored_span

    selection = MachineGeometryRegistry.default().select(shot)
    geometry = selection.configuration.geometry
    families, experimental_drives, circuit_mapping = _experimental_circuit_drives(
        group, stored.index, geometry
    )
    stored_grid_radius, stored_grid_height = np.meshgrid(
        stored.radius_m, stored.height_m
    )
    stored_positions = np.column_stack(
        (stored_grid_radius.ravel(), stored_grid_height.ravel())
    )
    stored_external_flux = _external_flux(
        geometry, stored_positions, families, experimental_drives
    ).reshape(stored.total_flux_wb.shape)
    _, _, stored_after_external_current = _current_receipt(
        stored_lattice,
        stored.total_flux_wb - stored_external_flux,
        stored_lcfs,
    )
    stored_exterior_after = float(
        stored_after_external_current["exterior_absolute_integral_a"]
    )
    stored_exterior_cancellation = 1.0 - stored_exterior_after / exterior_absolute

    grid_radius, grid_height = np.meshgrid(nova_r, nova_z)
    positions = np.column_stack((grid_radius.ravel(), grid_height.ravel()))
    external_flux = _external_flux(
        geometry, positions, families, experimental_drives
    ).reshape(nova_flux.shape)
    external_sup = float(np.max(np.abs(external_flux)))

    pre_external_residual = nova_flux - plasma_flux
    residual = pre_external_residual - external_flux
    residual_sup = float(np.max(np.abs(residual)))
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    _, _, before_external_current = _current_receipt(
        nova_lattice, pre_external_residual, nova_lcfs
    )
    residual_density, residual_valid, residual_current = _current_receipt(
        nova_lattice, residual, nova_lcfs
    )
    exterior_before = float(before_external_current["exterior_absolute_integral_a"])
    exterior_after = float(residual_current["exterior_absolute_integral_a"])
    exterior_cancellation = (
        1.0 - exterior_after / exterior_before
        if exterior_before > 0.0
        else float("nan")
    )
    read_residual_density = np.where(residual_valid, residual_density, 0.0)
    peak_index = np.unravel_index(
        int(np.argmax(np.abs(read_residual_density))), residual_density.shape
    )
    peak_radius = float(nova_r[peak_index[0]])
    peak_height = float(nova_z[peak_index[1]])

    operator_pass = operator_fraction <= OPERATOR_ROUND_TRIP_BOUND
    forward_reasons = []
    if not operator_pass:
        forward_reasons.append("plasma coupling fails the masked delta-star bound")
    if stored_exterior_cancellation <= 0.0:
        forward_reasons.append(
            "experimental-current external field does not cancel exterior current"
        )

    return {
        "source": {
            "shot": shot,
            "requested_time_s": requested_time_s,
            "efit_time_index": stored.index,
            "efit_time_s": stored.time_s,
            "flux_conversion": "stored Wb/rad multiplied by 2*pi to total Wb",
            "stored_flux_span_wb": stored_span,
            "stored_plasma_current_a": stored.plasma_current_a,
            "experimental_conductor_current_field": "efm/fcoil_x",
            "fitted_conductor_current_field_read": False,
            "geometry_identity": selection.configuration.physical_digest,
        },
        "grid": {
            "stored_shape_zr": list(stored.total_flux_wb.shape),
            "nova_shape_zr": list(nova_flux.shape),
            "interpolation": interpolation,
        },
        "operator_round_trip": {
            "method": (
                "stored delta_star -> stored-LCFS mask -> bilinear density mapping "
                "-> Nova plasma_to_grid -> Nova delta_star -> same LCFS mask -> "
                "Nova plasma_to_grid"
            ),
            "stored_complete_stencil_nodes": int(np.count_nonzero(stored_valid)),
            "stored_lcfs_plasma_nodes": int(np.count_nonzero(plasma_cells)),
            "nova_lcfs_plasma_nodes": int(np.count_nonzero(nova_lcfs)),
            "sup_error_wb": operator_error,
            "sup_error_fraction_of_stored_span": operator_fraction,
            "required_bound": OPERATOR_ROUND_TRIP_BOUND,
            "passes": operator_pass,
            "interpretation": (
                "operator round trip is within bound"
                if operator_pass
                else (
                    "coupling-operator or grid defect; decomposition is diagnostic only"
                )
            ),
        },
        "exterior_current": {
            "definition": (
                "stored delta-star current on complete-stencil nodes outside the "
                "stored LCFS; never supplied to the plasma coupling"
            ),
            "exterior_nodes": int(np.count_nonzero(exterior_cells)),
            "signed_total_a": exterior_signed,
            "absolute_total_a": exterior_absolute,
            "signed_fraction_of_stored_plasma_current": (
                exterior_signed / abs(stored.plasma_current_a)
            ),
            "absolute_fraction_of_stored_plasma_current": (
                exterior_absolute / abs(stored.plasma_current_a)
            ),
        },
        "external_reconstruction": {
            "method": (
                "loop_response_matrix with gain-check drive @ response-row prediction"
            ),
            "current_field": "efm/fcoil_x experimental input",
            "fitted_current_field_used": False,
            "circuit_selection": (
                "the first 13 stored fcoil circuits are the active block; every "
                "assignment to a registry family is checked by centroid"
            ),
            "physical_family_count": len(families),
            "circuit_mapping": circuit_mapping,
            "stored_grid_sup_norm_wb": float(np.max(np.abs(stored_external_flux))),
            "stored_grid_sup_norm_fraction_of_stored_span": float(
                np.max(np.abs(stored_external_flux)) / stored_span
            ),
            "nova_grid_sup_norm_wb": external_sup,
            "nova_grid_sup_norm_fraction_of_stored_span": external_sup / stored_span,
            "stored_grid_exterior_cancellation": {
                "absolute_current_before_external_a": exterior_absolute,
                "absolute_current_after_external_a": stored_exterior_after,
                "cancelled_fraction": stored_exterior_cancellation,
                "interpretation": (
                    "experimental-current external field cancels most of the "
                    "stored exterior delta-star signal"
                ),
            },
        },
        "residual": {
            "definition": (
                "interpolated stored flux - masked plasma flux - external flux"
            ),
            "norm_domain": "all Nova grid nodes, without fit, mask, or exclusion",
            "sup_norm_wb": residual_sup,
            "sup_norm_fraction_of_stored_span": residual_sup / stored_span,
            "rms_wb": residual_rms,
            "rms_fraction_of_stored_span": residual_rms / stored_span,
            "implied_missing_current": {
                **residual_current,
                "peak_absolute_a_per_m2": float(np.max(np.abs(read_residual_density))),
                "peak_r_m": peak_radius,
                "peak_z_m": peak_height,
                "nearest_named_conductors": _nearest_conductors(
                    geometry, peak_radius, peak_height
                ),
            },
            "nova_grid_exterior_cancellation": {
                "absolute_current_before_external_a": exterior_before,
                "absolute_current_after_external_a": exterior_after,
                "cancelled_fraction": exterior_cancellation,
            },
        },
        "forward_boundary_condition": {
            "fit": not forward_reasons,
            "verdict": "FIT" if not forward_reasons else "NOT FIT",
            "reasons": forward_reasons,
            "qualification": (
                "residual norms have no acceptance threshold; the verdict uses only "
                "the specified operator bound and the sign of measured exterior-"
                "current cancellation; interpolation is reported separately for "
                "attribution"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--time", type=float, default=DEFAULT_TIME_S)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    return parser


def main() -> None:
    """Print the decomposition receipt as stable JSON."""

    arguments = _parser().parse_args()
    result = decompose(
        shot=arguments.shot,
        requested_time_s=arguments.time,
        store=arguments.store,
        artifact_cache=arguments.artifact_cache,
        artifact_digest=arguments.artifact_digest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
