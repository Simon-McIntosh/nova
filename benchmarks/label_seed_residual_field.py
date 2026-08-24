"""Bank current-constrained map residual fields for the frozen MAST cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    DEFAULT_OUTPUT,
    FIXED_POINT_CRITERION,
    FROZEN_SCORECARD_RECEIPT_NAME,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    _baseline_by_shot,
    _mast_case_from_selection,
    _passive_inclusive_case,
    _passive_inclusive_solve,
    select_slices_by_shot,
)
from benchmarks import mast_response_carrier_warm as response_carrier
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes

DEFAULT_RECEIPT = Path(
    "docs/figures/plateau-input-attribution/label-seed-residual-field.json"
)
SINGLE_REFERENCE_RECEIPT = DEFAULT_OUTPUT / "reference-seeded-forward-slice.json"
FROZEN_BASELINE_RECEIPT = DEFAULT_OUTPUT / FROZEN_SCORECARD_RECEIPT_NAME
PINNING_COMMIT = "cf812416343bbc821757fa2382d1e333d55e9f4f"
SINGLE_REFERENCE_SOURCE_COMMIT = "57a1bfb987f2fa3f0d126f6a0b04e1e06bbf0e61"
FROZEN_BASELINE_SOURCE_COMMIT = "f771f90db43f946dd72a71b9d7decbb2acd8dc36"


def _archive_scalar(archive: Any, name: str) -> str:
    """Read one required scalar string from a persisted response carrier."""
    values = np.asarray(archive[name])
    if values.shape != ():
        raise ValueError(f"persisted {name} must be scalar")
    return str(values.item())


def _persisted_response_cache(
    carrier: Path,
    carrier_receipt: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact response carrier after an isolated fail-closed check."""
    command = [
        sys.executable,
        str(Path(response_carrier.__file__).resolve()),
        "check",
        "--carrier",
        str(carrier),
        "--receipt",
        str(carrier_receipt),
    ]
    checked = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    response, metadata = response_carrier.load_carrier(carrier)
    with np.load(carrier, allow_pickle=False) as archive:
        input_digests = json.loads(_archive_scalar(archive, "input_digests_json"))
        audit = json.loads(_archive_scalar(archive, "audit_json"))

    if input_digests.get("combined_sha256") != metadata["semantic_response_identity"]:
        raise ValueError("persisted response input ledger changed identity")
    required_audit = {
        "active_circuit_count",
        "passive_or_vessel_circuit_count",
        "section_kernel_evaluations",
        "passive_registry_minimum_overlap_fraction",
        "passive_registry_maximum_separation_m",
    }
    missing_audit = required_audit.difference(audit)
    if missing_audit:
        raise ValueError(
            "persisted response audit is incomplete: "
            + ", ".join(sorted(missing_audit))
        )
    if int(audit["active_circuit_count"]) != 13:
        raise ValueError("persisted response has the wrong active-circuit inventory")
    if int(audit["passive_or_vessel_circuit_count"]) != 88:
        raise ValueError("persisted response has the wrong passive-circuit inventory")
    checked_stdout = checked.stdout.strip()
    if "direct_builders=0 verdict=PASS" not in checked_stdout:
        raise RuntimeError("named carrier check did not report a cache-only pass")

    cache = {
        "response": response,
        "input_digests": input_digests,
        "audit": {
            "stored_circuit_count": metadata["stored_circuit_count"],
            **audit,
        },
    }
    evidence = {
        "loaded_from_persisted_carrier": True,
        "carrier": metadata,
        "named_cache_only_check": {
            "command": command,
            "stdout": checked_stdout,
            "direct_green_operator_builder_entries": 0,
            "passes": True,
        },
    }
    return cache, evidence


def _source_revision() -> str:
    """Return the committed tree used by this invocation."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _norms(values: np.ndarray, span: float) -> dict[str, float]:
    """Return absolute sup and Euclidean magnitudes with stable scaling."""
    field = np.asarray(values, dtype=np.float64)
    sup = float(np.max(np.abs(field))) if field.size else 0.0
    l2 = float(np.linalg.norm(field))
    return {
        "sup_wb": sup,
        "l2_wb": l2,
        "rms_wb": float(l2 / np.sqrt(max(field.size, 1))),
        "sup_fraction_of_label_span": float(sup / span),
        "l2_fraction_of_label_span": float(l2 / span),
    }


def _distance_to_boundary(coordinates: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Return point-to-segment distances from grid nodes to a closed boundary."""
    start = np.asarray(boundary, dtype=np.float64)
    end = np.roll(start, -1, axis=0)
    segment = end - start
    denominator = np.sum(segment * segment, axis=1)
    offset = np.asarray(coordinates, dtype=np.float64)[:, None, :] - start[None]
    fraction = np.divide(
        np.sum(offset * segment[None], axis=2),
        denominator[None],
        out=np.zeros((coordinates.shape[0], start.shape[0]), dtype=np.float64),
        where=denominator[None] > 0.0,
    )
    separation = offset - np.clip(fraction, 0.0, 1.0)[:, :, None] * segment[None]
    return np.min(np.linalg.norm(separation, axis=2), axis=1)


def _region_record(
    field: np.ndarray,
    selected: np.ndarray,
    span: float,
    grid_squared_magnitude: float,
) -> dict[str, Any]:
    """Describe one spatial selection without changing the field denominator."""
    values = np.asarray(field, dtype=np.float64)[selected]
    squared = float(np.sum(values**2))
    return {
        "node_count": int(values.size),
        **_norms(values, span),
        "squared_magnitude_wb2": squared,
        "fraction_of_grid_squared_magnitude": float(
            squared / max(grid_squared_magnitude, np.finfo(float).tiny)
        ),
    }


def _profile_explained_fraction(
    normalized_flux: np.ndarray, residual: np.ndarray, selected: np.ndarray
) -> float:
    """Measure how much core residual energy is smooth in labelled flux."""
    coordinate = np.asarray(normalized_flux, dtype=np.float64)[selected]
    values = np.asarray(residual, dtype=np.float64)[selected]
    if values.size < 4:
        return 0.0
    edges = np.linspace(0.0, 1.0, 13)
    bins = np.clip(np.digitize(coordinate, edges[1:-1]), 0, len(edges) - 2)
    prediction = np.empty_like(values)
    for index in range(len(edges) - 1):
        members = bins == index
        if np.any(members):
            prediction[members] = np.mean(values[members])
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total <= np.finfo(float).tiny:
        return 0.0
    unexplained = float(np.sum((values - prediction) ** 2))
    return float(np.clip(1.0 - unexplained / total, 0.0, 1.0))


def _stencil_oscillation(
    coordinates: np.ndarray, field: np.ndarray, stencil_width: float
) -> dict[str, float | int]:
    """Measure sign alternation and neighbour-scale variation on grid edges."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    distance, neighbours = cKDTree(coordinates).query(coordinates, k=5)
    pairs: set[tuple[int, int]] = set()
    for source in range(coordinates.shape[0]):
        for target, separation in zip(
            np.atleast_1d(neighbours[source, 1:]),
            np.atleast_1d(distance[source, 1:]),
            strict=True,
        ):
            if separation <= 1.01 * stencil_width:
                pairs.add(tuple(sorted((source, int(target)))))
    if not pairs:
        return {
            "neighbour_pair_count": 0,
            "sign_change_fraction": 0.0,
            "normalised_difference_energy": 0.0,
        }
    indices = np.asarray(sorted(pairs), dtype=int)
    first = values[indices[:, 0]]
    second = values[indices[:, 1]]
    difference_energy = float(np.sum((first - second) ** 2))
    level_energy = float(np.sum(first**2 + second**2))
    return {
        "neighbour_pair_count": int(indices.shape[0]),
        "sign_change_fraction": float(np.mean(first * second < 0.0)),
        "normalised_difference_energy": float(
            difference_energy
            / max(difference_energy + level_energy, np.finfo(float).tiny)
        ),
    }


def _field_receipt(
    case: dict[str, Any], profile, mapped: jax.Array
) -> tuple[dict[str, Any], dict[str, float]]:
    """Retain the exact residual field and its spatial pattern descriptors."""
    state = np.asarray(case["state"], dtype=np.float64)
    image = np.asarray(mapped, dtype=np.float64)
    residual = image - state
    grid_count = profile.operator.grid.node_number
    wall_count = profile.operator.wall.node_number
    if residual.size != grid_count + wall_count:
        raise RuntimeError("the mapped state does not match the physical grid and wall")

    grid = np.asarray(case["grid_coordinate"], dtype=np.float64)
    wall = np.asarray(case["wall_coordinate"], dtype=np.float64)
    boundary = np.asarray(case["boundary"], dtype=np.float64)
    grid_residual = residual[:grid_count]
    wall_residual = residual[grid_count:]
    span = float(case["span_wb"])
    stencil_width = float(cKDTree(grid).query(grid, k=2)[0][:, 1].min())
    inside = MplPath(boundary, closed=True).contains_points(grid, radius=1.0e-12)
    boundary_band = _distance_to_boundary(grid, boundary) <= stencil_width
    sol = ~inside
    closed_core = inside & ~boundary_band
    sol_away_from_boundary = sol & ~boundary_band
    grid_squared = float(np.sum(grid_residual**2))
    wall_squared = float(np.sum(wall_residual**2))
    full_squared = grid_squared + wall_squared

    operator = profile.operator
    prescribed = operator.prescribed_current_field
    if prescribed is None:
        raise RuntimeError("the labelled map has no prescribed current response")
    circuit_response = np.asarray(prescribed.response, dtype=np.float64)
    if circuit_response.shape != (residual.size, 101):
        raise RuntimeError("the labelled map has the wrong circuit-response shape")
    response_squared = np.sum(circuit_response**2, axis=0)
    fitted_amplitude = np.divide(
        circuit_response.T @ residual,
        response_squared,
        out=np.zeros(circuit_response.shape[1], dtype=np.float64),
        where=response_squared > 0.0,
    )
    projection_squared = fitted_amplitude**2 * response_squared
    best_circuit = int(np.argmax(projection_squared))
    circuit_fraction = float(
        projection_squared[best_circuit] / max(full_squared, np.finfo(float).tiny)
    )
    normalized_flux = (state[:grid_count] - float(operator.declared_axis_flux)) / (
        float(operator.declared_boundary_flux) - float(operator.declared_axis_flux)
    )
    profile_fraction = _profile_explained_fraction(
        normalized_flux, grid_residual, closed_core
    )
    oscillation = _stencil_oscillation(grid, grid_residual, stencil_width)
    boundary_squared = float(np.sum(grid_residual[boundary_band] ** 2))
    core_squared = float(np.sum(grid_residual[closed_core] ** 2))
    candidate_scores = {
        "wall": float(
            (boundary_squared + wall_squared) / max(full_squared, np.finfo(float).tiny)
        ),
        "conductor_current_wiring": circuit_fraction,
        "profiles": float(
            core_squared / max(full_squared, np.finfo(float).tiny) * profile_fraction
        ),
        "discretisation": float(
            grid_squared
            / max(full_squared, np.finfo(float).tiny)
            * float(oscillation["normalised_difference_energy"])
        ),
    }
    implicated = max(candidate_scores, key=candidate_scores.get)
    region_masks = {
        "closed_flux_region": inside,
        "scrape_off_layer": sol,
        "within_one_stencil_width_of_boundary": boundary_band,
    }
    exclusive_masks = {
        "closed_flux_core_away_from_boundary": closed_core,
        "scrape_off_layer_away_from_boundary": sol_away_from_boundary,
        "boundary_band": boundary_band,
    }
    record = {
        "definition": (
            "ForwardProfile.flux_map(diverted, target_current)(label) - label"
        ),
        "application_count": 1,
        "label_state_source": "efm/psirz in total Wb on the selected frozen row",
        "gauge_adjustment": "none; this is the map residual scored by the solver",
        "state_node_count": int(residual.size),
        "grid_node_count": grid_count,
        "wall_node_count": wall_count,
        "norms": {
            "full_state": _norms(residual, span),
            "grid": _norms(grid_residual, span),
            "wall": _norms(wall_residual, span),
        },
        "regional_squared_magnitude": {
            "denominator": (
                "grid residual squared magnitude; boundary band overlaps the "
                "closed-flux and scrape-off masks"
            ),
            **{
                name: _region_record(grid_residual, mask, span, grid_squared)
                for name, mask in region_masks.items()
            },
        },
        "exclusive_regional_partition": {
            "denominator": (
                "grid residual squared magnitude; these three masks are disjoint "
                "and exhaustive"
            ),
            **{
                name: _region_record(grid_residual, mask, span, grid_squared)
                for name, mask in exclusive_masks.items()
            },
        },
        "stencil_width_m": stencil_width,
        "profile_extraction_pattern": {
            "method": "twelve fixed normalized-flux bins over the closed-flux core",
            "explained_core_variance_fraction": profile_fraction,
        },
        "single_circuit_green_pattern": {
            "method": (
                "least-squares projection of the full residual onto each of 101 "
                "single-circuit response columns"
            ),
            "best_response_column_zero_based": best_circuit,
            "best_stored_circuit_one_based": best_circuit + 1,
            "equivalent_current_correction_a": float(fitted_amplitude[best_circuit]),
            "fraction_of_full_squared_magnitude_explained": circuit_fraction,
        },
        "stencil_pattern": oscillation,
        "wall_state_fraction_of_full_squared_magnitude": float(
            wall_squared / max(full_squared, np.finfo(float).tiny)
        ),
        "candidate_pattern_scores": {
            "method": (
                "wall is boundary-band plus wall-state energy; conductor wiring "
                "is the best single-circuit projection; profiles are closed-core "
                "energy explained by labelled flux; discretisation is neighbour-"
                "difference energy. Scores describe patterns and are not "
                "acceptance criteria."
            ),
            "scores": candidate_scores,
            "most_implicated_candidate": implicated,
        },
        "spatial_field": {
            "grid_coordinates_m": grid.tolist(),
            "grid_residual_wb": grid_residual.tolist(),
            "wall_coordinates_m": wall.tolist(),
            "wall_residual_wb": wall_residual.tolist(),
            "stored_lcfs_coordinates_m": boundary.tolist(),
        },
    }
    return record, candidate_scores


def _terminal_classification(solve: dict[str, Any]) -> str:
    """Classify the terminal solve without treating current pinning as convergence."""
    branch = solve["forward_branch_receipt"]
    terminal = solve["terminal_state"]
    reference_current = abs(float(terminal["reference_plasma_current_a"]))
    current_fraction = abs(float(terminal["plasma_current_a"])) / reference_current
    if branch["converged"] and current_fraction >= 0.01:
        return "converged_plasma_root"
    if current_fraction < 0.01:
        return "vacuum_collapse"
    return "bounded_non_convergence"


def _historical_context() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read the two immutable receipts that current pinning supersedes."""
    single = json.loads(SINGLE_REFERENCE_RECEIPT.read_text())
    baseline = _baseline_by_shot(json.loads(FROZEN_BASELINE_RECEIPT.read_text()))
    primary = single["primary"]
    single_reading = {
        "receipt": str(SINGLE_REFERENCE_RECEIPT),
        "authored_on": "2026-08-21",
        "source_commit": SINGLE_REFERENCE_SOURCE_COMMIT,
        "reference": "MAST 21985/51",
        "classification": primary["branch"]["classification"],
        "fixed_point_defect": primary["solver"]["fixed_point_defect"],
        "plasma_current_fraction_of_label": primary["metrics"]["plasma_current"][
            "absolute_fraction_of_reference"
        ],
        "superseded": True,
        "reason": "the reading predates declared-current map pinning",
    }
    return single_reading, baseline


def run(
    store: Path = SHOT_STORE,
    bank: Path = DECOMPOSITION_BANK,
    output: Path = DEFAULT_RECEIPT,
    carrier: Path = response_carrier.DEFAULT_CARRIER,
    carrier_receipt: Path = response_carrier.DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Measure one map application and one pinned solve for every frozen row."""
    configure_dtypes()
    source_revision = _source_revision()
    single_reading, historical_baseline = _historical_context()
    response_cache, carrier_evidence = _persisted_response_cache(
        carrier,
        carrier_receipt,
    )
    selected = select_slices_by_shot(bank)
    records = []
    candidate_rows = []
    direct_builder_entries = 0
    for selected_row, qualification in selected:
        case, context = _mast_case_from_selection(
            store,
            selected_row,
            qualification,
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        direct_builder_entries += int(policy["section_kernel_evaluations_this_shot"])

        reference = passive_case["reference"]
        target_current = abs(float(reference["plasma_current_a"]))
        label = jnp.asarray(passive_case["state"])
        mapped = profile.flux_map(
            requested_class=TopologyClass.DIVERTED,
            target_current=target_current,
        )(label)
        mapped = jax.block_until_ready(mapped)
        field, scores = _field_receipt(passive_case, profile, mapped)
        solve, _trace, _branch = _passive_inclusive_solve(
            passive_case,
            context,
            profile,
            newton_budget=NEWTON_STEPS,
            target_current=target_current,
        )
        terminal = solve["terminal_state"]
        branch = solve["forward_branch_receipt"]
        current_fraction = abs(float(terminal["plasma_current_a"])) / abs(
            float(terminal["reference_plasma_current_a"])
        )
        historical = historical_baseline[str(reference["shot"])]["solve_outcome"]
        records.append(
            {
                "reference": reference,
                "one_application_residual": field,
                "label_seeded_newton_solve": {
                    "source_revision": source_revision,
                    "route": "ForwardProfile.solve_branch newton_krylov",
                    "target_current_a": target_current,
                    "requested_branch": branch["requested_class"],
                    "achieved_branch": branch["achieved_class"],
                    "topology_consistent": branch["topology_consistent"],
                    "converged": branch["converged"],
                    "iterations": branch["iterations"],
                    "terminal_relative_residual": branch["residual"],
                    "terminal_branch_classification": _terminal_classification(solve),
                    "terminal_plasma_current_a": terminal["plasma_current_a"],
                    "label_plasma_current_a": terminal["reference_plasma_current_a"],
                    "terminal_plasma_current_fraction_of_label": current_fraction,
                    "normalisation_policy": terminal["normalisation_policy"],
                    "normalisation_amplitude": terminal["normalisation_amplitude"],
                },
                "superseded_comparison": {
                    "frozen_row_unpinned_reading": {
                        "receipt": str(FROZEN_BASELINE_RECEIPT),
                        "authored_on": "2026-08-21",
                        "source_commit": FROZEN_BASELINE_SOURCE_COMMIT,
                        "outcome_class": historical["outcome_class"],
                        "converged": historical["converged"],
                        "terminal_residual": historical["terminal_residual"],
                        "terminal_plasma_current_a": historical[
                            "terminal_plasma_current_a"
                        ],
                        "superseded": True,
                    },
                    "quoted_vacuum_reading": single_reading,
                    "superseding_pinning_commit": {
                        "commit": PINNING_COMMIT,
                        "authored_on": "2026-08-22",
                        "mechanism": (
                            "target current divided by unscaled cell-current sum "
                            "renormalises every current moment"
                        ),
                    },
                },
                "prescribed_current_policy": policy,
            }
        )
        candidate_rows.append(scores)

    if len(records) != 6:
        raise RuntimeError(f"expected six frozen references, measured {len(records)}")
    if direct_builder_entries != 0:
        raise RuntimeError(
            "persisted-carrier run entered the direct Green response builder"
        )
    mean_scores = {
        name: float(np.mean([row[name] for row in candidate_rows]))
        for name in (
            "wall",
            "conductor_current_wiring",
            "profiles",
            "discretisation",
        )
    }
    implicated = max(mean_scores, key=mean_scores.get)
    verdict = {
        "implicated_candidate": implicated,
        "cohort_mean_pattern_scores": mean_scores,
        "statement": (
            f"The one-application residual structure most strongly implicates "
            f"{implicated.replace('_', ' ')} among wall, conductor-current wiring, "
            "profiles and discretisation; this is a spatial attribution, not proof "
            "that changing that input produces a converged parity root. The current "
            "production circuit is closed, so its score diagnoses a wiring-shaped "
            "residual and does not authorise fitted currents."
        ),
        "qualification": (
            "The scores are commensurate fractions of full residual energy after "
            "the profile and stencil descriptors weight their respective domains; "
            "no achieved solve value defines an acceptance threshold."
        ),
    }
    receipt = {
        "receipt": "MAST label-seed one-application residual fields",
        "source_revision": source_revision,
        "backend": "JAX_PLATFORMS=cpu",
        "execution_contract": {
            "frozen_reference_count": 6,
            "selection": "lowest worst-fraction qualified row per frozen shot",
            "map": "all 101 fitted circuits with declared plasma-current pinning",
            "map_application_count_per_reference": 1,
            "newton_route": "newton_krylov",
            "newton_promotions": NEWTON_STEPS,
            "gmres_iterations_per_promotion": GMRES_ITERATIONS,
            "fixed_point_criterion": FIXED_POINT_CRITERION,
            "residual_and_solve_share_source_revision": True,
            "persisted_response_carrier": carrier_evidence,
            "direct_green_operator_builder_entries": direct_builder_entries,
        },
        "historical_context": {
            "quoted_2026_08_21_vacuum_reading": single_reading,
            "pinning_commit_that_supersedes_it": {
                "commit": PINNING_COMMIT,
                "authored_on": "2026-08-22",
            },
        },
        "per_reference": records,
        "aggregate": {
            "reference_count": len(records),
            "terminal_branch_counts": {
                name: sum(
                    row["label_seeded_newton_solve"]["terminal_branch_classification"]
                    == name
                    for row in records
                )
                for name in (
                    "converged_plasma_root",
                    "bounded_non_convergence",
                    "vacuum_collapse",
                )
            },
            "all_terminal_currents_pinned_to_label": bool(
                all(
                    abs(
                        row["label_seeded_newton_solve"][
                            "terminal_plasma_current_fraction_of_label"
                        ]
                        - 1.0
                    )
                    <= 1.0e-12
                    for row in records
                )
            ),
            "verdict": verdict,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Run the frozen-cohort measurement and print its compact verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--bank", type=Path, default=DECOMPOSITION_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    parser.add_argument(
        "--carrier-receipt", type=Path, default=response_carrier.DEFAULT_RECEIPT
    )
    arguments = parser.parse_args()
    receipt = run(
        arguments.store,
        arguments.bank,
        arguments.output,
        arguments.carrier,
        arguments.carrier_receipt,
    )
    aggregate = receipt["aggregate"]
    print(
        "LABEL_SEED_RESIDUAL_FIELD "
        f"references={aggregate['reference_count']} "
        f"branches={aggregate['terminal_branch_counts']} "
        f"candidate={aggregate['verdict']['implicated_candidate']}"
    )


if __name__ == "__main__":
    main()
