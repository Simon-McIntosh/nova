"""Measure flux convergence from clipped-source coupling near an LCFS.

The production arm contracts each clipped cell's exact zeroth and first
current moments against the three fixed parent-cell response blocks.  The
repair arm instead integrates every clipped support directly with the exact
polygon flux kernel.  Both arms use the same piecewise-linear separatrix and
the same conserved uniform current density, so their difference isolates the
parent-cell source representation rather than clipping or normalisation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import stats

from nova.biot.polygonanalytic import polygon_analytic_flux_moments
from nova.equilibrium.separatrix_clip import AtomicCellMesh


DEFAULT_OUTPUT = Path(
    "docs/figures/coefficient-space-newton/clipped-coupling-order.json"
)
MESH_SPACINGS_M = (0.24, 0.16, 0.12, 0.08, 0.06, 0.04)
SOURCE_CENTRE_M = np.asarray([1.65, 0.0])
SOURCE_RADII_M = np.asarray([0.48, 0.64])
CURRENT_DENSITY = 1.0e6
REFERENCE_VERTEX_COUNT = 2048
TARGET_ANGLE_COUNT = 16
SEPARATRIX_PSI_N = (0.96, 1.0, 1.04)
CLOSED_PSI_N = (0.2, 0.5, 0.8)


@dataclass(frozen=True)
class TargetSet:
    """Fixed target coordinates and their regional membership."""

    coordinate: np.ndarray
    region_indices: dict[str, np.ndarray]


def _ellipse_polygon(vertex_count: int) -> np.ndarray:
    """Return a counter-clockwise polygon converging spectrally to the LCFS."""
    angle = 2.0 * np.pi * (np.arange(vertex_count) + 0.371) / vertex_count
    return SOURCE_CENTRE_M + SOURCE_RADII_M * np.column_stack(
        (np.cos(angle), np.sin(angle))
    )


def _polygon_area(vertices: np.ndarray) -> float:
    """Return the unsigned shoelace area of one polygon."""
    following = np.roll(vertices, -1, axis=0)
    return float(
        0.5
        * abs(
            np.sum(vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1])
        )
    )


def _targets() -> TargetSet:
    """Return fixed samples split by analytic normalised-flux region."""
    angle = 2.0 * np.pi * (np.arange(TARGET_ANGLE_COUNT) + 0.173) / TARGET_ANGLE_COUNT
    coordinates = []
    region_indices: dict[str, list[int]] = {
        "separatrix_band": [],
        "closed_flux_region": [],
    }
    for region, levels in (
        ("separatrix_band", SEPARATRIX_PSI_N),
        ("closed_flux_region", CLOSED_PSI_N),
    ):
        for psi_norm in levels:
            radius = np.sqrt(psi_norm)
            ring = SOURCE_CENTRE_M + radius * SOURCE_RADII_M * np.column_stack(
                (np.cos(angle), np.sin(angle))
            )
            start = len(coordinates)
            coordinates.extend(ring)
            region_indices[region].extend(range(start, start + len(ring)))
    return TargetSet(
        coordinate=np.asarray(coordinates),
        region_indices={
            name: np.asarray(indices, dtype=np.intp)
            for name, indices in region_indices.items()
        },
    )


def _parent_cells(spacing: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Tile a support-complete rectangular source mesh at one spacing."""
    radial_half_count = int(np.ceil(SOURCE_RADII_M[0] / spacing)) + 1
    vertical_half_count = int(np.ceil(SOURCE_RADII_M[1] / spacing)) + 1
    radial_index = np.arange(-radial_half_count, radial_half_count + 1)
    vertical_index = np.arange(-vertical_half_count, vertical_half_count + 1)
    rr, zz = np.meshgrid(
        SOURCE_CENTRE_M[0] + spacing * radial_index,
        SOURCE_CENTRE_M[1] + spacing * vertical_index,
    )
    centres = np.column_stack((rr.ravel(), zz.ravel()))
    half = 0.5 * spacing
    offset = np.asarray([[-half, -half], [half, -half], [half, half], [-half, half]])
    return [centre + offset for centre in centres], centres


def _ellipse_level(points: np.ndarray) -> np.ndarray:
    """Return a signed level that is positive inside the analytic LCFS."""
    normalised = (points - SOURCE_CENTRE_M) / SOURCE_RADII_M
    return 1.0 - np.sum(normalised**2, axis=1)


def _fixed_parent_response(
    targets: np.ndarray,
    parent: np.ndarray,
    centre: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the shipped three-block response of one fixed parent cell."""
    return polygon_analytic_flux_moments(
        targets[:, 0],
        targets[:, 1],
        parent,
        expansion_point=centre,
    )


def _mesh_flux(spacing: float, targets: np.ndarray) -> dict[str, object]:
    """Evaluate shipped and exact-clipped flux on one source mesh."""
    parents, centres = _parent_cells(spacing)
    atomic = AtomicCellMesh.from_cells(parents, centroids=centres)
    clipped = atomic.clip(_ellipse_level(atomic.node_coordinates))
    shipped_flux = np.zeros(len(targets))
    repaired_flux = np.zeros(len(targets))
    included = np.flatnonzero(clipped.included)
    boundary = np.flatnonzero(clipped.boundary)
    second_mean = spacing**2 / 12.0

    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        for source in included:
            current = CURRENT_DENSITY * clipped.area[source]
            first = CURRENT_DENSITY * clipped.first_area_moment[source]
            uniform, radial, vertical = _fixed_parent_response(
                targets, parents[source], centres[source]
            )
            shipped_flux += (
                current * uniform
                + (first[0] / second_mean) * radial
                + (first[1] / second_mean) * vertical
            )

            vertex_count = int(clipped.vertex_count[source])
            support = clipped.support_vertices[source, :vertex_count]
            exact_uniform = polygon_analytic_flux_moments(
                targets[:, 0], targets[:, 1], support
            )[0]
            repaired_flux += current * exact_uniform

    if not np.all(np.isfinite(shipped_flux)) or not np.all(np.isfinite(repaired_flux)):
        raise RuntimeError("a coupling arm produced non-finite flux")
    return {
        "spacing_m": spacing,
        "parent_cell_count": len(parents),
        "included_cell_count": int(len(included)),
        "cut_cell_count": int(len(boundary)),
        "clipped_area_m2": float(clipped.patch_area_sum),
        "area_error_fraction": float(
            (clipped.patch_area_sum - np.pi * np.prod(SOURCE_RADII_M))
            / (np.pi * np.prod(SOURCE_RADII_M))
        ),
        "shipped_flux_wb": shipped_flux,
        "exact_clipped_polygon_flux_wb": repaired_flux,
    }


def _relative_errors(
    computed: np.ndarray, reference: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    """Return relative RMS and sup flux errors on one target region."""
    delta = computed[indices] - reference[indices]
    selected = reference[indices]
    return {
        "relative_rms": float(np.linalg.norm(delta) / np.linalg.norm(selected)),
        "relative_sup": float(np.max(np.abs(delta)) / np.max(np.abs(selected))),
        "absolute_rms_wb": float(np.sqrt(np.mean(delta**2))),
        "absolute_sup_wb": float(np.max(np.abs(delta))),
    }


def _order_fit(spacing: np.ndarray, error: np.ndarray) -> dict[str, float | int]:
    """Fit ``error = scale * spacing**order`` with residual diagnostics."""
    if np.any(error <= 0.0):
        raise ValueError("convergence errors must be strictly positive")
    log_spacing = np.log(spacing)
    log_error = np.log(error)
    fitted = stats.linregress(log_spacing, log_error)
    prediction = fitted.intercept + fitted.slope * log_spacing
    residual = log_error - prediction
    return {
        "order": float(fitted.slope),
        "order_standard_error": float(fitted.stderr),
        "log_fit_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "log_fit_residual_max_abs": float(np.max(np.abs(residual))),
        "r_squared": float(fitted.rvalue**2),
        "prefactor": float(np.exp(fitted.intercept)),
        "points": int(len(spacing)),
    }


def run_study(spacings: tuple[float, ...] = MESH_SPACINGS_M) -> dict[str, object]:
    """Run the declared mesh ladder and form the convergence receipt."""
    if len(spacings) < 4:
        raise ValueError("the convergence ladder requires at least four spacings")
    spacing = np.asarray(spacings, dtype=float)
    if np.any(spacing <= 0.0) or not np.all(np.diff(spacing) < 0.0):
        raise ValueError("mesh spacings must be positive and strictly decreasing")

    targets = _targets()
    reference_polygon = _ellipse_polygon(REFERENCE_VERTEX_COUNT)
    reference_area = _polygon_area(reference_polygon)
    reference_flux = (
        CURRENT_DENSITY
        * reference_area
        * polygon_analytic_flux_moments(
            targets.coordinate[:, 0], targets.coordinate[:, 1], reference_polygon
        )[0]
    )
    mesh_results = [_mesh_flux(value, targets.coordinate) for value in spacing]

    arms: dict[str, dict[str, object]] = {}
    for arm in ("shipped_clipped_moments", "exact_clipped_polygon"):
        flux_key = (
            "shipped_flux_wb"
            if arm == "shipped_clipped_moments"
            else "exact_clipped_polygon_flux_wb"
        )
        regions: dict[str, dict[str, object]] = {}
        for region, indices in targets.region_indices.items():
            errors = [
                _relative_errors(result[flux_key], reference_flux, indices)
                for result in mesh_results
            ]
            relative_rms = np.asarray([row["relative_rms"] for row in errors])
            regions[region] = {
                "target_count": int(len(indices)),
                "errors_by_spacing": [
                    {"spacing_m": float(value), **error}
                    for value, error in zip(spacing, errors, strict=True)
                ],
                "relative_rms_order_fit": _order_fit(spacing, relative_rms),
            }
        arms[arm] = {
            "method": (
                "exact clipped zeroth and first current moments contracted "
                "against uniform and two linear response blocks of the fixed "
                "parent cell"
                if arm == "shipped_clipped_moments"
                else "uniform current integrated directly over each clipped support "
                "with polygon_analytic_flux_moments"
            ),
            "regions": regions,
        }

    shipped_order = arms["shipped_clipped_moments"]["regions"]["separatrix_band"][
        "relative_rms_order_fit"
    ]
    repaired_order = arms["exact_clipped_polygon"]["regions"]["separatrix_band"][
        "relative_rms_order_fit"
    ]
    banked_order = 0.97
    shipped_difference = abs(shipped_order["order"] - banked_order)
    shipped_uncertainty = max(
        shipped_order["order_standard_error"],
        shipped_order["log_fit_residual_rms"],
    )
    repair_gain = repaired_order["order"] - shipped_order["order"]
    accounts_for_banked = (
        shipped_difference <= 2.0 * shipped_uncertainty and repair_gain > 0.5
    )
    if accounts_for_banked:
        adjudication = (
            "The shipped band order is compatible with the banked 0.97 within "
            "twice its declared fit uncertainty, and direct clipped-polygon "
            "integration raises the order by more than 0.5. The repaired source "
            "coupling therefore accounts for the banked first-order deficit in "
            "this controlled refinement study."
        )
    else:
        adjudication = (
            "The exact clipped-polygon arm does not both reproduce the banked "
            "0.97 in the shipped control and raise its order by more than 0.5. "
            "This study leaves a residual deficit that another term must explain."
        )

    serialisable_mesh = []
    for result in mesh_results:
        serialisable_mesh.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"shipped_flux_wb", "exact_clipped_polygon_flux_wb"}
            }
        )
    return {
        "schema": "nova.clipped-coupling-order.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "measurement": "flux-field convergence under clipped-source coupling",
        "source": {
            "shape": "ellipse",
            "centre_m": SOURCE_CENTRE_M.tolist(),
            "semi_axes_m": SOURCE_RADII_M.tolist(),
            "current_density_a_per_m2": CURRENT_DENSITY,
            "density_model": "uniform inside the LCFS and zero outside",
            "separatrix_model": (
                "linear interpolation of the analytic ellipse level on shared "
                "parent-cell edges"
            ),
        },
        "mesh_ladder": {
            "declared_spacings_m": spacing.tolist(),
            "rungs": serialisable_mesh,
        },
        "targets": {
            "angle_count_per_ring": TARGET_ANGLE_COUNT,
            "separatrix_band": {
                "definition": "analytic psi_N in [0.96, 1.04]",
                "sampled_psi_N": list(SEPARATRIX_PSI_N),
                "target_count": int(len(targets.region_indices["separatrix_band"])),
            },
            "closed_flux_region": {
                "definition": "analytic psi_N in {0.2, 0.5, 0.8}",
                "sampled_psi_N": list(CLOSED_PSI_N),
                "target_count": int(len(targets.region_indices["closed_flux_region"])),
            },
        },
        "reference": {
            "method": (
                "exact polygon flux kernel on one 2048-edge ellipse approximation"
            ),
            "polygon_vertices": REFERENCE_VERTEX_COUNT,
            "area_m2": reference_area,
            "analytic_ellipse_area_m2": float(np.pi * np.prod(SOURCE_RADII_M)),
            "area_error_fraction": float(
                (reference_area - np.pi * np.prod(SOURCE_RADII_M))
                / (np.pi * np.prod(SOURCE_RADII_M))
            ),
        },
        "arms": arms,
        "banked_order_adjudication": {
            "banked_observed_order": banked_order,
            "shipped_band_order_difference": float(shipped_difference),
            "declared_shipped_fit_uncertainty": float(shipped_uncertainty),
            "repair_order_gain": float(repair_gain),
            "repaired_order_accounts_for_banked_0_97": bool(accounts_for_banked),
            "statement": adjudication,
        },
    }


def main() -> None:
    """Write one JSON convergence receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = run_study()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")
    shipped = receipt["arms"]["shipped_clipped_moments"]["regions"]
    repaired = receipt["arms"]["exact_clipped_polygon"]["regions"]
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "shipped_band_order": shipped["separatrix_band"][
                    "relative_rms_order_fit"
                ]["order"],
                "repaired_band_order": repaired["separatrix_band"][
                    "relative_rms_order_fit"
                ]["order"],
                "shipped_closed_order": shipped["closed_flux_region"][
                    "relative_rms_order_fit"
                ]["order"],
                "repaired_closed_order": repaired["closed_flux_region"][
                    "relative_rms_order_fit"
                ]["order"],
                "accounts_for_banked_0_97": receipt["banked_order_adjudication"][
                    "repaired_order_accounts_for_banked_0_97"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
