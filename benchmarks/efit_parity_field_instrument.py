"""Compare solved and reference field energy with one Nova instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np
import zarr
from matplotlib.path import Path as MplPath

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    GRID_STRIDE,
    _stored_map,
)
from benchmarks.efit_parity_boundary_volume import (  # noqa: E402
    _polygon_measure,
    _verify_protected_artifacts,
)
from benchmarks.efit_parity_inductance_partition import (  # noqa: E402
    BOUNDARY_RECEIPT,
    MOMENT_RECEIPT,
    OUTPUT_RECEIPT as PARTITION_RECEIPT,
    SOURCE_RECEIPT,
    TARGET_SHOT,
    _boundary_moments,
    _cell_partition,
    _terminal_observation,
)
from benchmarks.efit_topology_boundary_score import _stored_lcfs  # noqa: E402
from nova.equilibrium.conservation import poloidal_field  # noqa: E402
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

OUTPUT_DIRECTORY = Path("docs/figures/efit-forward-parity")
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "field-energy-instrument-control.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "field-energy-instrument-control.png"
EXPECTED_SLICE_INDEX = 43
PUBLISHED_REFERENCE_FIELD_ENERGY = 0.496621310710907
BANKED_SOLVED_FIELD_ENERGY = 0.28949138263178675
BANKED_STORED_BOUNDARY_AREA = 1.837030238598465
BANKED_STORED_BOUNDARY_VOLUME = 8.792322015190356
EXPECTED_REFERENCE_MAP_FIELD_ENERGY = 0.2989777277348594
EXPECTED_REBASELINED_SOLVED_FIELD_ENERGY = 0.29880628232236717


def _relative_deviation(observed: float, expected: float) -> float:
    return observed / expected - 1.0


def _source_digests(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _field_magnitude(profile, flux: np.ndarray) -> np.ndarray:
    radial, vertical = poloidal_field(profile.lattice, jnp.asarray(flux))
    return np.hypot(
        np.asarray(radial, dtype=np.float64),
        np.asarray(vertical, dtype=np.float64),
    )


def _plot_fields(fields: dict[str, Any], receipt: dict[str, Any], path: Path) -> None:
    radius = fields["radius"]
    height = fields["height"]
    contour = fields["stored_contour"]
    inside = fields["inside"].reshape(fields["shape"])
    solved = np.where(inside, fields["solved_field"].reshape(fields["shape"]), np.nan)
    reference = np.where(
        inside, fields["reference_field"].reshape(fields["shape"]), np.nan
    )
    field_limit = float(np.nanmax(np.stack((solved, reference))))
    comparison = receipt["field_energy_comparison_table"]

    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True, constrained_layout=True
    )
    image = None
    for axis, values, title, energy in (
        (
            axes[0],
            solved,
            "Nova solved terminal map",
            comparison[1]["field_energy_t2_m3"],
        ),
        (
            axes[1],
            reference,
            "Reference own map",
            comparison[0]["field_energy_t2_m3"],
        ),
    ):
        image = axis.pcolormesh(
            radius,
            height,
            values.T,
            shading="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=field_limit,
        )
        axis.plot(contour[:, 0], contour[:, 1], color="white", lw=1.1)
        axis.set_title(f"{title}\n∫|Bp|² dV = {energy:.6f} T² m³")
        axis.set_xlabel("R [m]")
        axis.set_aspect("equal")
    axes[0].set_ylabel("Z [m]")
    figure.colorbar(image, ax=axes, label="|Bp| [T]", shrink=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _support_receipt(
    profile,
    equilibrium,
    solved_contour: np.ndarray,
    stored_contour: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    solved_partition = _cell_partition(profile, equilibrium, solved_contour)
    coordinates = np.asarray(profile.lattice.coordinate, dtype=np.float64)
    stored_inside = MplPath(stored_contour, closed=True).contains_points(
        coordinates, radius=1.0e-12
    )
    solved_inside = solved_partition["inside"]
    cell_area = np.asarray(profile.operator.area, dtype=np.float64)
    cell_radius = np.asarray(profile.lattice.node_radius, dtype=np.float64)
    cell_volume = 2.0 * np.pi * cell_radius * cell_area
    stored_area = float(np.sum(cell_area[stored_inside]))
    stored_volume = float(np.sum(cell_volume[stored_inside]))
    solved_area = float(np.sum(cell_area[solved_inside]))
    stored_only = stored_inside & ~solved_inside
    solved_only = solved_inside & ~stored_inside
    net_area_change = stored_area - solved_area
    support = {
        "definition": "mesh-cell centroids inside the stored LCFS contour",
        "authoritative_cell_count": int(np.count_nonzero(stored_inside)),
        "banked_solved_branch_cell_count": int(np.count_nonzero(solved_inside)),
        "overlap_cell_count": int(np.count_nonzero(stored_inside & solved_inside)),
        "symmetric_difference_cell_count": int(
            np.count_nonzero(stored_inside ^ solved_inside)
        ),
        "stored_only_cell_count": int(np.count_nonzero(stored_only)),
        "solved_only_cell_count": int(np.count_nonzero(solved_only)),
        "cell_quadrature_poloidal_area_m2": stored_area,
        "cell_quadrature_toroidal_volume_m3": stored_volume,
        "cell_quadrature_area_relative_error_from_exact_contour": (
            _relative_deviation(stored_area, BANKED_STORED_BOUNDARY_AREA)
        ),
        "cell_quadrature_volume_relative_error_from_exact_contour": (
            _relative_deviation(stored_volume, BANKED_STORED_BOUNDARY_VOLUME)
        ),
        "net_area_change_from_banked_support_m2": net_area_change,
        "net_area_change_fraction_of_stored_support": net_area_change / stored_area,
        "net_area_change_fraction_of_banked_support": net_area_change / solved_area,
        "geometric_support_scale_statement": (
            "The net area change is 3.85 percent of the banked support. The "
            "directly rebaselined field-energy shift below is the controlled "
            "measure of its effect on this nonuniform integrand."
        ),
    }
    partition = {
        "inside": stored_inside,
        "cell_current": solved_partition["cell_current"],
    }
    return partition, support


def _outcome(
    reference_energy: float,
    solved_energy: float,
    accuracy_floor: float,
) -> dict[str, Any]:
    instrument_ratio = reference_energy / PUBLISHED_REFERENCE_FIELD_ENERGY
    physics_ratio = solved_energy / reference_energy
    total_ratio = solved_energy / PUBLISHED_REFERENCE_FIELD_ENERGY
    near_published = abs(instrument_ratio - 1.0) <= accuracy_floor
    near_solved = abs(reference_energy / solved_energy - 1.0) <= accuracy_floor
    if near_published:
        verdict = "INSTRUMENT_EXONERATED_GENUINE_CURRENT_DISTRIBUTION_DIFFERENCE"
        disposition = "RETAIN_AS_PHYSICS_DIFFERENCE"
        reading = (
            "Nova's operator reproduces the reference publication on the "
            "reference map within the control floor. The instrument is "
            "exonerated and the corrected solved/reference deficit is physical."
        )
    elif near_solved:
        verdict = "INSTRUMENT_DIFFERENCE"
        disposition = "RETRACT"
        reading = (
            "Nova's operator returns the solved-map energy on the reference "
            "map within the control floor. The row retracts and carries no "
            "current-distribution claim."
        )
    else:
        verdict = "MIXED_INSTRUMENT_AND_CURRENT_DISTRIBUTION_DIFFERENCE"
        disposition = "RETAIN_WITH_SPLIT_ATTRIBUTION"
        reading = (
            "The reference-map value is near neither pure endpoint at the "
            "control floor. Instrument and current distribution both contribute, "
            "and the two reported ratios quantify the multiplicative split."
        )
    total_deficit = PUBLISHED_REFERENCE_FIELD_ENERGY - solved_energy
    instrument_deficit = PUBLISHED_REFERENCE_FIELD_ENERGY - reference_energy
    physics_deficit = reference_energy - solved_energy
    return {
        "control_accuracy_floor_relative": accuracy_floor,
        "nova_operator_on_reference_over_reference_published": instrument_ratio,
        "instrument_signed_relative_deviation": instrument_ratio - 1.0,
        "nova_solved_over_nova_operator_on_reference": physics_ratio,
        "physics_signed_relative_deviation_after_instrument_division": (
            physics_ratio - 1.0
        ),
        "nova_solved_over_reference_published": total_ratio,
        "total_signed_relative_deviation": total_ratio - 1.0,
        "multiplicative_closure_residual": instrument_ratio * physics_ratio
        - total_ratio,
        "deficit_split_on_published_energy_scale": {
            "total_t2_m3": total_deficit,
            "instrument_t2_m3": instrument_deficit,
            "physics_t2_m3": physics_deficit,
            "instrument_fraction": instrument_deficit / total_deficit,
            "physics_fraction": physics_deficit / total_deficit,
            "additive_closure_residual_t2_m3": (
                instrument_deficit + physics_deficit - total_deficit
            ),
        },
        "verdict": verdict,
        "row_disposition": disposition,
        "reading": reading,
    }


def measure(
    source_path: Path = SOURCE_RECEIPT,
    moment_path: Path = MOMENT_RECEIPT,
    boundary_path: Path = BOUNDARY_RECEIPT,
    partition_path: Path = PARTITION_RECEIPT,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the one-region, one-operator field-energy control."""
    configure_dtypes()
    source = json.loads(source_path.read_text())
    moment = json.loads(moment_path.read_text())
    boundary = json.loads(boundary_path.read_text())
    banked_partition = json.loads(partition_path.read_text())
    integrity = _verify_protected_artifacts(source)

    profile, equilibrium, solved_contour, solved_state = _terminal_observation(
        source, store, bank
    )
    group = zarr.open_group(str(store / f"{TARGET_SHOT}.zarr"), mode="r")["efm"]
    stored_contour = _stored_lcfs(group, EXPECTED_SLICE_INDEX)
    full_radius, full_height, reference_full = _stored_map(group, EXPECTED_SLICE_INDEX)
    reference_flux = reference_full[::GRID_STRIDE, ::GRID_STRIDE]
    if not np.array_equal(full_radius[::GRID_STRIDE], profile.lattice.radius):
        raise RuntimeError("the reference and Nova radial axes differ")
    if not np.array_equal(full_height[::GRID_STRIDE], profile.lattice.height):
        raise RuntimeError("the reference and Nova vertical axes differ")

    contour_measure = _polygon_measure(stored_contour)
    banked_contour = next(
        row
        for row in boundary["contour_comparison_table"]
        if row["contour"] == "stored_lcfs"
    )
    if banked_contour["poloidal_area_m2"] != BANKED_STORED_BOUNDARY_AREA:
        raise RuntimeError("the stored-boundary area bank drifted")
    if banked_contour["exact_solid_of_revolution_m3"] != BANKED_STORED_BOUNDARY_VOLUME:
        raise RuntimeError("the stored-boundary volume bank drifted")
    if contour_measure["poloidal_area_m2"] != BANKED_STORED_BOUNDARY_AREA:
        raise RuntimeError("the reconstructed stored-boundary area differs")
    if contour_measure["exact_solid_of_revolution_m3"] != BANKED_STORED_BOUNDARY_VOLUME:
        raise RuntimeError("the reconstructed stored-boundary volume differs")

    partition, support = _support_receipt(
        profile, equilibrium, solved_contour, stored_contour
    )
    solved_moments = _boundary_moments(
        profile, equilibrium, solved_state, partition, moment
    )
    reference_moments = _boundary_moments(
        profile, equilibrium, reference_flux.ravel(), partition, moment
    )
    solved_energy = float(
        solved_moments["integrals"]["poloidal_field_squared_volume_integral_t2_m3"]
    )
    reference_energy = float(
        reference_moments["integrals"]["poloidal_field_squared_volume_integral_t2_m3"]
    )
    if not np.isclose(
        reference_energy,
        EXPECTED_REFERENCE_MAP_FIELD_ENERGY,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("the reference-map field energy drifted")
    if not np.isclose(
        solved_energy,
        EXPECTED_REBASELINED_SOLVED_FIELD_ENERGY,
        rtol=2.0e-15,
        atol=0.0,
    ):
        raise RuntimeError("the rebaselined solved-map field energy drifted")
    banked_energy = float(
        banked_partition["boundary_enclosed_moment_rescore"]["integrals"][
            "poloidal_field_squared_volume_integral_t2_m3"
        ]
    )
    if banked_energy != BANKED_SOLVED_FIELD_ENERGY:
        raise RuntimeError("the banked solved field energy drifted")
    accuracy_floor = max(
        abs(support["cell_quadrature_area_relative_error_from_exact_contour"]),
        abs(support["cell_quadrature_volume_relative_error_from_exact_contour"]),
    )
    outcome = _outcome(reference_energy, solved_energy, accuracy_floor)
    support_energy_shift = _relative_deviation(
        solved_energy, BANKED_SOLVED_FIELD_ENERGY
    )
    prior_deficit = abs(
        BANKED_SOLVED_FIELD_ENERGY / PUBLISHED_REFERENCE_FIELD_ENERGY - 1.0
    )
    support["banked_solved_field_energy_t2_m3"] = BANKED_SOLVED_FIELD_ENERGY
    support["rebaselined_solved_field_energy_t2_m3"] = solved_energy
    support["rebaselined_over_banked"] = solved_energy / BANKED_SOLVED_FIELD_ENERGY
    support["signed_relative_field_energy_shift"] = support_energy_shift
    support["prior_published_deficit_magnitude"] = prior_deficit
    support["support_shift_cannot_explain_prior_deficit"] = bool(
        abs(support_energy_shift) < prior_deficit
    )

    source_paths = (source_path, moment_path, boundary_path, partition_path)
    receipt = {
        "receipt": {
            "kind": "field_energy_instrument_control",
            "shot": TARGET_SHOT,
            "slice_index": EXPECTED_SLICE_INDEX,
            "source_receipts": [str(path) for path in source_paths],
            "source_receipt_sha256": _source_digests(source_paths),
        },
        "outcomes_declared_before_measurement": {
            "reference_map_near_reference_published": {
                "verdict": (
                    "INSTRUMENT_EXONERATED_GENUINE_CURRENT_DISTRIBUTION_DIFFERENCE"
                ),
                "row_disposition": "RETAIN_AS_PHYSICS_DIFFERENCE",
            },
            "reference_map_near_nova_solved": {
                "verdict": "INSTRUMENT_DIFFERENCE",
                "row_disposition": "RETRACT",
            },
            "reference_map_near_neither_endpoint": {
                "verdict": "MIXED_INSTRUMENT_AND_CURRENT_DISTRIBUTION_DIFFERENCE",
                "row_disposition": "RETAIN_WITH_SPLIT_ATTRIBUTION",
            },
            "near_relative_tolerance": accuracy_floor,
            "near_tolerance_source": (
                "maximum absolute stored-LCFS cell-quadrature area or volume "
                "reproduction error"
            ),
        },
        "execution_contract": {
            "nonlinear_solve_calls": 0,
            "equilibrium_observations_from_serialised_state": 1,
            "reference_flux_source": "2*pi*efm/psirz",
            "reference_native_grid_shape": list(reference_full.shape),
            "nova_operator_grid_shape": list(reference_flux.shape),
            "gradient_operator": "nova.equilibrium.conservation.poloidal_field",
            "cell_quadrature": "2*pi*R*cell_area",
            "imported_integrator": (
                "benchmarks.efit_parity_inductance_partition._boundary_moments"
            ),
            "support_policy": "one 243-centroid stored-LCFS mask for both fields",
        },
        "controlled_region": {
            "contour": "stored_lcfs",
            "contour_point_count": int(contour_measure["point_count"]),
            "exact_poloidal_area_m2": float(contour_measure["poloidal_area_m2"]),
            "exact_toroidal_volume_m3": float(
                contour_measure["exact_solid_of_revolution_m3"]
            ),
            **support,
        },
        "field_energy_comparison_table": [
            {
                "field": "reference_own_map",
                "operator": "nova",
                "support": "stored_lcfs_243_centroids",
                "field_energy_t2_m3": reference_energy,
            },
            {
                "field": "nova_solved_terminal_map",
                "operator": "nova",
                "support": "stored_lcfs_243_centroids",
                "field_energy_t2_m3": solved_energy,
            },
            {
                "field": "reference_own_map",
                "operator": "reference_published",
                "support": "reference_published",
                "field_energy_t2_m3": PUBLISHED_REFERENCE_FIELD_ENERGY,
            },
        ],
        "instrument_control": outcome,
        "protected_banked_artifacts": integrity,
        "figure": str(OUTPUT_FIGURE),
    }
    fields = {
        "radius": np.asarray(profile.lattice.radius),
        "height": np.asarray(profile.lattice.height),
        "shape": profile.lattice.shape,
        "inside": partition["inside"],
        "stored_contour": stored_contour,
        "solved_field": _field_magnitude(
            profile, solved_state[: profile.lattice.node_count]
        ),
        "reference_field": _field_magnitude(profile, reference_flux.ravel()),
    }
    return receipt, fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    arguments = parser.parse_args()
    receipt, fields = measure()
    _plot_fields(fields, receipt, arguments.figure)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
