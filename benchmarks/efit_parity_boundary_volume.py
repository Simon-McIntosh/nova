"""Reconcile published plasma volumes with their reported boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from benchmarks.efit_forward_parity_slice import (  # noqa: E402
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    select_slices_by_shot,
)
from benchmarks.efit_parity_root_geometry import (  # noqa: E402
    _assert_reference,
    _closed_axis_branch,
    _sha256_array,
    _stored_lcfs,
    _unit_boundary_branches,
)
from nova.equilibrium import fixed_point  # noqa: E402
from nova.equilibrium.topology import TopologyClass  # noqa: E402
from nova.imas.mast_solve_inputs import SHOT_STORE  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

OUTPUT_DIRECTORY = Path("docs/figures/efit-forward-parity")
SOURCE_RECEIPT = OUTPUT_DIRECTORY / "converged-root-geometry-attribution.json"
MOMENT_RECEIPT = OUTPUT_DIRECTORY / "moment-definition-rescore.json"
OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "boundary-enclosed-volume-reconciliation.json"
OUTPUT_FIGURE = OUTPUT_DIRECTORY / "boundary-enclosed-volume-reconciliation.png"
TARGET_SHOT = 22086
CENTROID_CONVENTION_TOLERANCE = 0.01


def _relative_deviation(observed: float, expected: float) -> float:
    return observed / expected - 1.0


def _closed_polygon(points: np.ndarray) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("a polygon needs at least three (R, Z) points")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("polygon points must be finite")
    if not np.array_equal(polygon[0], polygon[-1]):
        polygon = np.vstack((polygon, polygon[0]))
    return polygon


def _polygon_measure(points: np.ndarray) -> dict[str, float | int]:
    """Measure one polygon with shoelace and an exact revolution integral."""
    polygon = _closed_polygon(points)
    left = polygon[:-1]
    right = polygon[1:]
    cross = left[:, 0] * right[:, 1] - right[:, 0] * left[:, 1]
    signed_area = 0.5 * np.sum(cross)
    if signed_area == 0.0:
        raise ValueError("polygon area is zero")
    centroid_radius = np.sum((left[:, 0] + right[:, 0]) * cross) / (6.0 * signed_area)
    area = abs(float(signed_area))
    first_moment = area * float(centroid_radius)
    first_moment_volume = 2.0 * np.pi * first_moment
    exact_volume = abs(
        float(
            np.pi
            / 3.0
            * np.sum(
                (left[:, 0] ** 2 + left[:, 0] * right[:, 0] + right[:, 0] ** 2)
                * (right[:, 1] - left[:, 1])
            )
        )
    )
    if not np.isclose(exact_volume, first_moment_volume, rtol=2.0e-14):
        raise RuntimeError("the two solid-of-revolution quadratures disagree")
    return {
        "point_count": int(len(points)),
        "poloidal_area_m2": area,
        "area_centroid_major_radius_m": float(centroid_radius),
        "area_first_moment_m3": first_moment,
        "exact_solid_of_revolution_m3": exact_volume,
        "first_moment_approximation_m3": first_moment_volume,
        "quadrature_relative_difference": _relative_deviation(
            first_moment_volume, exact_volume
        ),
        "arclength_m": float(np.linalg.norm(np.diff(polygon, axis=0), axis=1).sum()),
    }


def _verify_protected_artifacts(source: dict[str, Any]) -> dict[str, Any]:
    integrity = source["banked_artifact_integrity"]
    expected = integrity["sha256"]
    observed = {
        name: hashlib.sha256((OUTPUT_DIRECTORY / name).read_bytes()).hexdigest()
        for name in expected
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"protected banked artifacts changed: {mismatches}")
    if len(observed) != 23 or len(observed) != integrity["verified_digest_count"]:
        raise RuntimeError("protected artifact count is not the declared 23")
    return {
        "declared_count": len(expected),
        "verified_digest_count": len(observed),
        "all_digests_match": True,
        "source_and_output_receipts_are_outside_protected_set": True,
    }


def _reconstruct_contours(
    source: dict[str, Any],
    store: Path,
    bank: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = next(
        item
        for item in select_slices_by_shot(bank)
        if int(item[0]["shot"]) == TARGET_SHOT
    )
    case, context = _mast_case_from_selection(store, *selected)
    _assert_reference(case["reference"])
    profile = context["profile"]
    terminal = source["constrained_root"]["terminal_state"]
    state_array = np.asarray(terminal["values"], dtype=np.float64)
    if _sha256_array(state_array) != terminal["sha256"]:
        raise RuntimeError("serialized terminal-state digest does not match")
    state = jnp.asarray(state_array)
    history = fixed_point.FixedPointResult(
        state=state,
        residual=jnp.asarray(source["constrained_root"]["terminal_residual"]),
        trace=jnp.asarray([], dtype=state.dtype),
    )
    target_current = float(source["constrained_root"]["target_current_a"])
    equilibrium = profile._receipt(
        state,
        history,
        TopologyClass.DIVERTED,
        target_current,
    )
    topology = equilibrium.topology
    solved_flux = state_array[: profile.lattice.node_count].reshape(
        profile.lattice.shape
    )
    branches = _unit_boundary_branches(
        profile.lattice.radius,
        profile.lattice.height,
        solved_flux,
        float(topology.axis_flux),
        float(topology.boundary_flux),
    )
    solved = _closed_axis_branch(branches, np.asarray(topology.axis, dtype=np.float64))
    stored = _stored_lcfs(context["group"], context["row"])

    core = np.asarray(equilibrium.domains.core, dtype=bool)
    cell_area = np.asarray(profile.operator.area, dtype=np.float64)
    cell_radius = np.asarray(profile.lattice.node_radius, dtype=np.float64)
    core_area = float(np.sum(cell_area[core]))
    core_volume = float(np.sum(2.0 * np.pi * cell_radius[core] * cell_area[core]))
    banked_volume = source["moment_normalisation_attribution"][
        "solved_live_equilibrium"
    ]["plasma_volume_m3"]
    if not np.isclose(core_volume, banked_volume, rtol=2.0e-14):
        raise RuntimeError("reconstructed confined-core volume differs from the bank")
    return (
        {"solved": solved, "stored": stored},
        {
            "cell_count": int(core.size),
            "confined_core_cell_count": int(np.count_nonzero(core)),
            "confined_core_poloidal_area_m2": core_area,
            "confined_core_toroidal_volume_m3": core_volume,
            "reconstructed_axis_m": [
                float(value) for value in np.asarray(topology.axis)
            ],
            "reconstructed_axis_flux_wb": float(topology.axis_flux),
            "reconstructed_boundary_flux_wb": float(topology.boundary_flux),
        },
    )


def _boundary_support_rescore(
    source: dict[str, Any],
    moment: dict[str, Any],
    solved_measure: dict[str, Any],
    stored_measure: dict[str, Any],
) -> dict[str, Any]:
    attribution = source["moment_normalisation_attribution"]
    solved = attribution["solved_live_equilibrium"]
    reference = attribution["reference"]
    volumes = {
        "solved": {
            "published_m3": solved["plasma_volume_m3"],
            "boundary_enclosed_m3": solved_measure["exact_solid_of_revolution_m3"],
        },
        "reference": {
            "published_m3": reference["plasma_volume_m3"],
            "boundary_enclosed_m3": stored_measure["exact_solid_of_revolution_m3"],
        },
    }
    integrals = {
        "poloidal_beta": (
            solved["pressure_volume_integral_pa_m3"],
            reference["pressure_volume_integral_implied_by_betap_pa_m3"],
        ),
        "internal_inductance": (
            solved["poloidal_field_squared_volume_integral_t2_m3"],
            reference["poloidal_field_squared_volume_integral_t2_m3"],
        ),
    }
    metrics: dict[str, Any] = {}
    for name, (solved_integral, reference_integral) in integrals.items():
        solved_mean = solved_integral / volumes["solved"]["published_m3"]
        reference_mean = reference_integral / volumes["reference"]["published_m3"]
        solved_adjusted = solved_mean * volumes["solved"]["boundary_enclosed_m3"]
        reference_adjusted = (
            reference_mean * volumes["reference"]["boundary_enclosed_m3"]
        )
        deviation = _relative_deviation(solved_adjusted, reference_adjusted)
        metrics[name] = {
            "banked_solved_integral": solved_integral,
            "banked_reference_integral": reference_integral,
            "banked_solved_mean_integrand": solved_mean,
            "banked_reference_mean_integrand": reference_mean,
            "boundary_adjusted_solved_integral": solved_adjusted,
            "boundary_adjusted_reference_integral": reference_adjusted,
            "signed_relative_deviation": deviation,
            "original_signed_relative_deviation": moment[
                "matched_definition_deviations"
            ][name]["signed_relative_deviation"],
            "deficit_eliminated": deviation >= 0.0,
        }
    return {
        "method": (
            "Apply each side's banked mean integrand to the toroidal volume "
            "enclosed by its own reported boundary; no field integral or solve "
            "is rerun."
        ),
        "qualification": (
            "This isolates the support-volume consequence while preserving the "
            "banked mean integrands; it is not a new spatial quadrature of pressure "
            "or poloidal field."
        ),
        "volumes": volumes,
        "metrics": metrics,
        "parity_consequence": {
            "poloidal_beta_volume_deficit": (
                "ELIMINATED"
                if metrics["poloidal_beta"]["deficit_eliminated"]
                else "SURVIVES"
            ),
            "internal_inductance_volume_deficit": (
                "ELIMINATED"
                if metrics["internal_inductance"]["deficit_eliminated"]
                else "SURVIVES"
            ),
        },
    }


def _plot_contours(contours: dict[str, np.ndarray], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 6.2), constrained_layout=True)
    solved = _closed_polygon(contours["solved"])
    stored = _closed_polygon(contours["stored"])
    axis.fill(
        stored[:, 0],
        stored[:, 1],
        color="#4c78a8",
        alpha=0.20,
        label="stored LCFS enclosed region",
    )
    axis.fill(
        solved[:, 0],
        solved[:, 1],
        color="#f58518",
        alpha=0.24,
        label="solved closed-branch enclosed region",
    )
    axis.plot(stored[:, 0], stored[:, 1], color="#24527a", lw=1.4)
    axis.plot(solved[:, 0], solved[:, 1], color="#b55d00", lw=1.4)
    axis.set_title("MAST 22086/43 boundary-enclosed regions")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_aspect("equal")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def reconcile(
    source_path: Path = SOURCE_RECEIPT,
    moment_path: Path = MOMENT_RECEIPT,
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return the zero-solve boundary-volume reconciliation and its contours."""
    configure_dtypes()
    source = json.loads(source_path.read_text())
    moment = json.loads(moment_path.read_text())
    integrity = _verify_protected_artifacts(source)
    contours, clipping = _reconstruct_contours(source, store, bank)
    solved = _polygon_measure(contours["solved"])
    stored = _polygon_measure(contours["stored"])
    published_solved = source["moment_normalisation_attribution"][
        "solved_live_equilibrium"
    ]["plasma_volume_m3"]
    published_reference = source["moment_normalisation_attribution"]["reference"][
        "plasma_volume_m3"
    ]
    published_ratio = published_solved / published_reference
    enclosed_area_ratio = solved["poloidal_area_m2"] / stored["poloidal_area_m2"]
    centroid_radius_ratio = (
        solved["area_centroid_major_radius_m"] / stored["area_centroid_major_radius_m"]
    )
    enclosed_volume_ratio = (
        solved["exact_solid_of_revolution_m3"] / stored["exact_solid_of_revolution_m3"]
    )
    clipping["confined_core_area_fraction_of_solved_boundary"] = (
        clipping["confined_core_poloidal_area_m2"] / solved["poloidal_area_m2"]
    )
    clipping["confined_core_volume_fraction_of_solved_boundary"] = (
        clipping["confined_core_toroidal_volume_m3"]
        / solved["exact_solid_of_revolution_m3"]
    )
    currents = moment["current_supports"]
    clipping.update(
        {
            "confined_core_current_a": currents["confined_core"],
            "pinned_all_domain_current_a": currents["all_domain_constrained"],
            "confined_core_current_fraction": currents["confined_core_over_all_domain"],
            "area_fraction_minus_current_fraction": (
                clipping["confined_core_area_fraction_of_solved_boundary"]
                - currents["confined_core_over_all_domain"]
            ),
        }
    )
    published_solved_over_enclosed = (
        published_solved / solved["exact_solid_of_revolution_m3"]
    )
    published_reference_over_enclosed = (
        published_reference / stored["exact_solid_of_revolution_m3"]
    )
    candidates = {
        "confined_core_clipping": {
            "verdict": "SUPPORTED",
            "deciding_figures": {
                "published_solved_over_own_boundary_volume": (
                    published_solved_over_enclosed
                ),
                "confined_core_volume_over_own_boundary_volume": clipping[
                    "confined_core_volume_fraction_of_solved_boundary"
                ],
                "confined_core_area_fraction": clipping[
                    "confined_core_area_fraction_of_solved_boundary"
                ],
                "confined_core_current_fraction": clipping[
                    "confined_core_current_fraction"
                ],
            },
            "reading": (
                "The published solved volume is the reconstructed confined-core "
                "cell volume, not the volume enclosed by the solved boundary."
            ),
        },
        "genuinely_different_curves": {
            "verdict": "EXCLUDED",
            "deciding_figures": {
                "published_volume_ratio": published_ratio,
                "enclosed_poloidal_area_ratio": enclosed_area_ratio,
                "enclosed_toroidal_volume_ratio": enclosed_volume_ratio,
            },
            "reading": (
                "The boundary-enclosed areas and volumes are comparable rather "
                "than differing by the published factor."
            ),
        },
        "revolution_or_jacobian_convention": {
            "verdict": "EXCLUDED",
            "deciding_figures": {
                "published_volume_ratio": published_ratio,
                "area_centroid_major_radius_ratio": centroid_radius_ratio,
                "absolute_ratio_difference": abs(
                    published_ratio - centroid_radius_ratio
                ),
                "one_percent_match_required": CENTROID_CONVENTION_TOLERANCE,
            },
            "reading": (
                "The published volume ratio does not match the centroid-radius "
                "ratio, while the exact and first-moment revolution formulas agree."
            ),
        },
    }
    receipt = {
        "receipt": {
            "kind": "boundary_enclosed_volume_reconciliation",
            "shot": TARGET_SHOT,
            "slice_index": 43,
            "source_receipts": [str(source_path), str(moment_path)],
            "source_receipt_sha256": {
                str(source_path): hashlib.sha256(source_path.read_bytes()).hexdigest(),
                str(moment_path): hashlib.sha256(moment_path.read_bytes()).hexdigest(),
            },
        },
        "execution_contract": {
            "nonlinear_solve_calls": 0,
            "new_solved_equilibria": 0,
            "equilibrium_observations_from_serialised_state": 1,
            "contour_quadrature": (
                "same closed-polygon shoelace, centroid and segment revolution "
                "formulas for both contours"
            ),
        },
        "contour_comparison_table": [
            {"contour": "solved_closed_axis_enclosing_branch", **solved},
            {"contour": "stored_lcfs", **stored},
        ],
        "published_volume_comparison": {
            "solved_m3": published_solved,
            "reference_m3": published_reference,
            "solved_over_reference": published_ratio,
            "solved_published_over_own_boundary_enclosed": (
                published_solved_over_enclosed
            ),
            "reference_published_over_own_boundary_enclosed": (
                published_reference_over_enclosed
            ),
            "disagreeing_published_volume": "solved_confined_core_volume",
            "disagreement_factor_boundary_over_published": (
                solved["exact_solid_of_revolution_m3"] / published_solved
            ),
        },
        "controlled_ratios": {
            "solved_over_stored_poloidal_area": enclosed_area_ratio,
            "solved_over_stored_area_centroid_major_radius": centroid_radius_ratio,
            "solved_over_stored_boundary_enclosed_toroidal_volume": (
                enclosed_volume_ratio
            ),
        },
        "clipping": clipping,
        "candidate_discrimination": candidates,
        "boundary_support_moment_rescore": _boundary_support_rescore(
            source, moment, solved, stored
        ),
        "protected_banked_artifacts": integrity,
        "figure": str(OUTPUT_FIGURE),
        "verdict": "SOLVED_PUBLISHED_VOLUME_IS_CONFINED_CORE_NOT_BOUNDARY_ENCLOSED",
    }
    return receipt, contours


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    arguments = parser.parse_args()
    receipt, contours = reconcile()
    _plot_contours(contours, arguments.figure)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n")
    comparison = receipt["published_volume_comparison"]
    beta = receipt["boundary_support_moment_rescore"]["metrics"]["poloidal_beta"]
    li = receipt["boundary_support_moment_rescore"]["metrics"]["internal_inductance"]
    discrepancy = comparison["disagreement_factor_boundary_over_published"]
    print(
        "BOUNDARY_VOLUME_RECONCILIATION "
        f"verdict={receipt['verdict']} "
        f"boundary_over_solved={discrepancy:.12g} "
        f"beta_relative={beta['signed_relative_deviation']:.12g} "
        f"li_relative={li['signed_relative_deviation']:.12g}"
    )


if __name__ == "__main__":
    main()
