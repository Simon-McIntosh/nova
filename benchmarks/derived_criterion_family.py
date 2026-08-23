"""Derive three consumer criteria from committed measurement receipts.

The response, DIII-D convergence, and compiled-parity criteria are rebuilt
from their measurement domains.  No equilibrium is constructed or solved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

OUTPUT_PATH = Path("docs/figures/forward-operator-refinement/criterion-family.json")

COUPLED_TRACE_SOURCE = Path(
    "docs/figures/forward-operator-refinement/passive-closure-trace.json"
)
STABILITY_CONTROL_SOURCE = Path(
    "docs/figures/forward-operator-refinement/passive-closure-stability-control.json"
)
MODE_SOURCE = Path(
    "docs/figures/forward-operator-refinement/coupled-map-mode-identification.json"
)
MESH_SOURCE = Path(
    "docs/figures/diiid-forward-onboarding/topology-qualified-mesh-convergence.json"
)
DIIID_REGISTRATION_SOURCE = Path(
    "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_preregistration.json"
)
PARITY_GATE_SOURCE = Path(
    "docs/figures/mast-catalog-gpu-solve/jitted-eager-parity-gate.json"
)
PARITY_ATTRIBUTION_SOURCE = Path(
    "docs/figures/mast-catalog-gpu-solve/parity-divergence-attribution.json"
)
EVENT_SOURCE = Path(
    "docs/figures/forward-operator-refinement/event-resolved-amplification.json"
)

COUPLED_CONSUMER = Path("tests/test_equilibrium_forward_reference.py")
DIIID_CONSUMER = Path("benchmarks/diiid_forward_gs_match.py")
PARITY_CONSUMER = Path("benchmarks/jitted_eager_parity_gate.py")

JSON_SOURCES = (
    COUPLED_TRACE_SOURCE,
    STABILITY_CONTROL_SOURCE,
    MODE_SOURCE,
    MESH_SOURCE,
    DIIID_REGISTRATION_SOURCE,
    PARITY_GATE_SOURCE,
    PARITY_ATTRIBUTION_SOURCE,
    EVENT_SOURCE,
)
CONSUMER_SOURCES = (COUPLED_CONSUMER, DIIID_CONSUMER, PARITY_CONSUMER)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=0.0):
        raise RuntimeError(f"banked {label} changed: {left!r} != {right!r}")


def _load_inputs() -> tuple[dict[Path, dict[str, Any]], dict[Path, str]]:
    receipts = {path: json.loads(path.read_text()) for path in JSON_SOURCES}
    consumers = {path: path.read_text() for path in CONSUMER_SOURCES}
    return receipts, consumers


def _coupled_response_criterion(
    trace: dict[str, Any],
    control: dict[str, Any],
    mode: dict[str, Any],
    consumer_source: str,
) -> dict[str, Any]:
    if "PASSIVE_REPRODUCTION_MOVE_CEILING = 0.15" not in consumer_source:
        raise RuntimeError("the coupled-response consumer registration changed")

    trace_comparators = trace["comparators"]
    control_comparators = control["comparators"]
    elongated = trace["terminal_decomposition"]
    stable = control["terminal_decomposition"]
    documented_direct = float(trace_comparators["documented_direct_response_points"])
    measured_direct = max(
        float(elongated["direct_external_peak_points"]),
        float(stable["direct_external_peak_points"]),
    )
    direct_anchor = max(documented_direct, measured_direct)

    stable_mode = mode["stable_control"]
    leading = stable_mode["dominant_eigenpairs"][0]
    leading_comparison = stable_mode["leading_mode_comparison"]
    elongated_mode = mode["elongated_cross_carrier"]

    configurations = [
        {
            "configuration": "elongated_reference",
            "boundary_elongation": float(
                control_comparators["elongated_boundary_elongation"]
            ),
            "channel_identity": "radial-dominant expanding axis-displacement channel",
            "absolute_radial_over_vertical_motion": float(
                elongated_mode["late_vector_localisation"]["axis_motion"][
                    "absolute_radial_over_vertical_motion"
                ]
            ),
            "gain": float(elongated["root_response_over_direct_peak"]),
            "derived_bound_points": (
                direct_anchor * float(elongated["root_response_over_direct_peak"])
            ),
            "measured_root_response_peak_points": float(
                elongated["root_response_peak_points"]
            ),
            "mode_rate_reading": {
                "kind": "expanding terminal Rayleigh quotient",
                "value": float(
                    elongated_mode["passive_driven_tangent"][
                        "terminal_rayleigh_quotient"
                    ]
                ),
            },
            "receipt_inputs": [str(COUPLED_TRACE_SOURCE), str(MODE_SOURCE)],
        },
        {
            "configuration": "stable_near_circular_control",
            "boundary_elongation": float(stable_mode["carrier"]["boundary_elongation"]),
            "channel_identity": (
                "stable vertical free-boundary axis-displacement channel"
            ),
            "absolute_radial_over_vertical_motion": float(
                leading["localisation"]["axis_motion"][
                    "absolute_radial_over_vertical_motion"
                ]
            ),
            "gain": float(stable["root_response_over_direct_peak"]),
            "derived_bound_points": (
                direct_anchor * float(stable["root_response_over_direct_peak"])
            ),
            "measured_root_response_peak_points": float(
                stable["root_response_peak_points"]
            ),
            "mode_rate_reading": {
                "kind": "leading eigenvalue",
                "value": float(leading["eigenvalue"]["magnitude"]),
                "ritz_residual_l2": float(leading["ritz_residual_l2"]),
                "scalar_modal_factor": float(
                    leading_comparison["scalar_modal_resolvent_factor"]
                ),
                "measured_gain_over_scalar_modal_factor": float(
                    leading_comparison["measured_gain_over_scalar_modal_factor"]
                ),
            },
            "receipt_inputs": [
                str(STABILITY_CONTROL_SOURCE),
                str(MODE_SOURCE),
            ],
        },
    ]
    for row in configurations:
        row["measured_response_within_derived_bound"] = (
            row["measured_root_response_peak_points"] <= row["derived_bound_points"]
        )
        row["bound_headroom_points"] = (
            row["derived_bound_points"] - row["measured_root_response_peak_points"]
        )

    _close(configurations[0]["gain"], 6.5998576353591725, label="elongated gain")
    _close(configurations[1]["gain"], 106.99077030932676, label="stable gain")
    _close(
        configurations[1]["mode_rate_reading"]["value"],
        0.9874290999339198,
        label="stable leading eigenvalue",
    )
    return {
        "consumer": str(COUPLED_CONSUMER),
        "registered_reading_replaced_points": 0.15,
        "formula": "B(c, k_c) = D_ref * G(c, k_c)",
        "formula_terms": {
            "D_ref": ("max(documented direct response, measured direct external peak)"),
            "D_ref_points": direct_anchor,
            "G(c, k_c)": (
                "measured root-response peak divided by direct external peak for "
                "configuration c and its identified physical channel k_c"
            ),
        },
        "configuration_table": configurations,
        "domain_of_validity": (
            "The relation is anchored only on the two measured carriers and their "
            "different identified channels. It is not an interpolation in elongation, "
            "and it must not be extrapolated to another configuration until that "
            "configuration's channel identity and gain are measured."
        ),
        "single_scalar_bound_allowed": False,
        "receipt_inputs": [
            str(COUPLED_TRACE_SOURCE),
            str(STABILITY_CONTROL_SOURCE),
            str(MODE_SOURCE),
            str(COUPLED_CONSUMER),
        ],
    }


def _diiid_criterion(
    mesh: dict[str, Any],
    registration: dict[str, Any],
    consumer_source: str,
) -> dict[str, Any]:
    if "REGISTERED_RESIDUAL_TOLERANCE = 1.0e-5" not in consumer_source:
        raise RuntimeError("the DIII-D registered residual reading changed")
    if "GATE_RESIDUAL_TOLERANCE = 1.0e-6" not in consumer_source:
        raise RuntimeError("the DIII-D hard-coded gate reading changed")

    rungs = mesh["rungs"]
    coarse = float(rungs[0]["solver"]["terminal_relative_residual"])
    fine = float(rungs[1]["solver"]["terminal_relative_residual"])
    observed_order = float(mesh["verdict"]["observed_order"])
    candidates = [1.0e-6, 1.0e-5]
    eligible = [candidate for candidate in candidates if candidate <= fine]
    if not eligible:
        raise RuntimeError("no declared DIII-D residual candidate is below the floor")
    selected = max(eligible)

    bar_basis = registration["score"]["bar_basis"]
    registered_bar = float(
        registration["score"]["registered_median_interior_r_squared_bar"]
    )
    expected_bar = float(
        bar_basis["measured_label_representability_median_r_squared"]
    ) * float(bar_basis["fraction_of_measured_ceiling_retained"])
    _close(registered_bar, expected_bar, label="DIII-D reference-accuracy bar")
    _close(fine, 7.930534999195602e-5, label="DIII-D fine-mesh floor")

    return {
        "consumer": str(DIIID_CONSUMER),
        "formula": (
            "tau_DIII-D = max{t in T_declared : t <= E_disc,fine}, "
            "T_declared = {1e-6, 1e-5}"
        ),
        "selected_relative_residual_bound": selected,
        "candidate_table": [
            {
                "candidate": candidate,
                "candidate_over_fine_mesh_floor": candidate / fine,
                "fine_mesh_floor_over_candidate": fine / candidate,
                "below_discretisation_floor": candidate <= fine,
                "adjudication": (
                    "REJECTED_PRECISION_WITHOUT_ACCURACY"
                    if candidate < selected
                    else "SURVIVES_WITH_DERIVED_READING"
                ),
            }
            for candidate in candidates
        ],
        "mesh_evidence": {
            "coarse_relative_residual": coarse,
            "fine_relative_residual": fine,
            "fine_to_coarse_ratio": fine / coarse,
            "observed_order": observed_order,
            "classification": mesh["verdict"]["classification"],
            "krylov_action_qualification": [
                rung["solver"]["krylov_action_qualification"] for rung in rungs
            ],
        },
        "reference_accuracy_trace": {
            "measured_label_representability_median_r_squared": float(
                bar_basis["measured_label_representability_median_r_squared"]
            ),
            "retained_fraction": float(
                bar_basis["fraction_of_measured_ceiling_retained"]
            ),
            "registered_median_interior_r_squared_bar": registered_bar,
            "strict_gs_residual_attributed_to_irreducible_non_gs_content": float(
                bar_basis["strict_gs_residual_attributed_to_irreducible_non_gs_content"]
            ),
            "normalisation_adjudication": (
                "R-squared reference accuracy and fixed-point relative residual "
                "have different normalisations, so they are traced together but "
                "not numerically equated. Only the same-norm discretisation floor "
                "selects the residual bound."
            ),
        },
        "surviving_value_justification": (
            "The 1e-5 candidate is the least over-solving declared candidate: it "
            "remains 7.93 times below the measured same-norm fine-mesh floor. The "
            "1e-6 candidate is 79.3 times below that floor and cannot improve the "
            "independently limited reference accuracy."
        ),
        "receipt_inputs": [
            str(MESH_SOURCE),
            str(DIIID_REGISTRATION_SOURCE),
            str(DIIID_CONSUMER),
        ],
    }


def _terminal_observable_bounds(
    terminal_quantities: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name, measurement in sorted(terminal_quantities.items()):
        absolute = float(measurement["maximum_absolute_difference"])
        relative = float(measurement["maximum_relative_difference"])
        dtype = str(measurement["dtype"])
        if dtype.startswith(("bool", "int", "uint")) or absolute == 0.0:
            criterion_kind = "exact_equality"
            formula = "compiled(q) == eager(q)"
            bounds: dict[str, float] = {}
        else:
            criterion_kind = "banked_dual_envelope"
            formula = "delta_abs(q) <= A_q and delta_rel(q) <= R_q"
            bounds = {
                "absolute_bound": absolute,
                "relative_bound": relative,
            }
        rows.append(
            {
                "observable": name,
                "dtype": dtype,
                "shape": measurement["shape"],
                "criterion_kind": criterion_kind,
                "formula": formula,
                **bounds,
                "calibration_maximum_absolute_difference": absolute,
                "calibration_maximum_relative_difference": relative,
                "receipt_inputs": [str(PARITY_GATE_SOURCE)],
            }
        )
    return rows


def _terminal_parity_criterion(
    gate: dict[str, Any],
    attribution: dict[str, Any],
    event: dict[str, Any],
    consumer_source: str,
) -> dict[str, Any]:
    if "REGISTERED_TOLERANCE = 1.0e-10" not in consumer_source:
        raise RuntimeError("the terminal-parity consumer registration changed")
    comparisons = gate["comparisons"]
    if set(comparisons) != {"moment_seed_traced_image", "profile_solve"}:
        raise RuntimeError("the banked parity comparison structure changed")

    one_map = comparisons["moment_seed_traced_image"]
    terminal = comparisons["profile_solve"]
    if len(one_map["quantities"]) != 4 or len(terminal["quantities"]) != 69:
        raise RuntimeError("the banked parity quantity cohort changed")

    cause = attribution["attribution"]["causes"]["AMPLIFIED_REPRESENTATION_DIFFERENCE"]
    epsilon = float(cause["float64_epsilon"])
    roundoff_bound = float(cause["roundoff_band_upper_bound"])
    _close(roundoff_bound, 16.0 * epsilon, label="float64 roundoff band")
    gate_one_map_relative = float(
        one_map["quantities"]["flux"]["maximum_relative_difference"]
    )
    attributed_one_map_relative = float(cause["maximum_single_map_relative_difference"])
    direction = event["predictions"]["different_seed_direction"]
    observable_bounds = _terminal_observable_bounds(terminal["quantities"])
    exact_count = sum(
        row["criterion_kind"] == "exact_equality" for row in observable_bounds
    )

    return {
        "consumer": str(PARITY_CONSUMER),
        "one_map_bound": {
            "formula": (
                "delta_map = max|M_compiled(s)-M_eager(s)| / "
                "max(max|M_eager(s)|, tiny_float64) <= 16*epsilon_float64"
            ),
            "relative_bound": roundoff_bound,
            "float64_epsilon": epsilon,
            "banked_maximum_relative_difference": max(
                gate_one_map_relative, attributed_one_map_relative
            ),
            "bound_over_banked_maximum": (
                roundoff_bound / max(gate_one_map_relative, attributed_one_map_relative)
            ),
            "machine_precision_demonstrated": True,
            "receipt_inputs": [
                str(PARITY_GATE_SOURCE),
                str(PARITY_ATTRIBUTION_SOURCE),
            ],
        },
        "seed_alignment_criterion": {
            "formula": "||s_compiled - s_eager||_2 = 0",
            "equivalent_reading": (
                "identical shape, dtype, values, branch construction, and solver "
                "budget before terminal parity is evaluated"
            ),
            "numeric_tolerance": 0.0,
            "conditional_terminal_parity": True,
            "derivation": (
                "The alternate real-LCFS direction has nonzero Euclidean separation "
                "0.7665 and moves the burst set from [3, 8, 12] to empty while "
                "changing cumulative separation growth from 5.38e9 to 0.0619. "
                "The bank contains no calibrated nonzero safe radius, so exact "
                "seed identity is the only non-arbitrary alignment criterion."
            ),
            "direction_dependence_measurement": {
                "seed_direction_l2": float(direction["seed_direction_l2"]),
                "seed_direction_peak": float(direction["seed_direction_peak"]),
                "baseline_burst_updates": direction["baseline_burst_updates"],
                "alternate_burst_updates": direction["alternate_burst_updates"],
                "baseline_cumulative_separation_growth": float(
                    direction["baseline_cumulative_separation_growth"]
                ),
                "alternate_cumulative_separation_growth": float(
                    direction["alternate_cumulative_separation_growth"]
                ),
            },
            "receipt_inputs": [str(EVENT_SOURCE)],
        },
        "terminal_observable_registration": {
            "formula": (
                "A_q = max_calibration delta_abs(q), "
                "R_q = max_calibration delta_rel(q); future aligned pairs require "
                "both bounds, while exact calibration leaves require equality"
            ),
            "calibration_cohort": "six frozen MAST references",
            "observable_count": len(observable_bounds),
            "exact_equality_count": exact_count,
            "dual_envelope_count": len(observable_bounds) - exact_count,
            "bounds": observable_bounds,
            "calibration_limit": (
                "These are frozen empirical envelopes for subsequent aligned-seed "
                "evaluations, not physics tolerances and not an independent verdict "
                "on the calibration cohort."
            ),
            "receipt_inputs": [str(PARITY_GATE_SOURCE)],
        },
        "terminal_parity_rule": (
            "Terminal parity is evaluated only after exact seed alignment; the "
            "one-map representation bound and every terminal-observable bound are "
            "reported separately, never collapsed to one terminal-state scalar."
        ),
        "receipt_inputs": [
            str(PARITY_GATE_SOURCE),
            str(PARITY_ATTRIBUTION_SOURCE),
            str(EVENT_SOURCE),
            str(PARITY_CONSUMER),
        ],
    }


def build_receipt_from_data(
    receipts: dict[Path, dict[str, Any]], consumers: dict[Path, str]
) -> dict[str, Any]:
    """Build the family from already loaded banked evidence."""
    coupled = _coupled_response_criterion(
        receipts[COUPLED_TRACE_SOURCE],
        receipts[STABILITY_CONTROL_SOURCE],
        receipts[MODE_SOURCE],
        consumers[COUPLED_CONSUMER],
    )
    diiid = _diiid_criterion(
        receipts[MESH_SOURCE],
        receipts[DIIID_REGISTRATION_SOURCE],
        consumers[DIIID_CONSUMER],
    )
    parity = _terminal_parity_criterion(
        receipts[PARITY_GATE_SOURCE],
        receipts[PARITY_ATTRIBUTION_SOURCE],
        receipts[EVENT_SOURCE],
        consumers[PARITY_CONSUMER],
    )
    observable_count = parity["terminal_observable_registration"]["observable_count"]

    return {
        "receipt": {
            "kind": "derived_criterion_family",
            "status": "complete",
            "execution_mode": "banked-measurements-only",
            "equilibrium_solves_run": 0,
            "registered_consumer_count": 3,
            "derived_bound_group_count": 3,
            "coupled_configuration_count": 2,
            "terminal_observable_bound_count": observable_count,
        },
        "criterion_family": {
            "coupled_response": coupled,
            "diiid_forward_gate": diiid,
            "terminal_compiled_parity": parity,
        },
        "inherited_constants_replaced": [
            {
                "consumer": str(COUPLED_CONSUMER),
                "inherited_reading": "PASSIVE_REPRODUCTION_MOVE_CEILING = 0.15 points",
                "replacement_reading": (
                    "B(c, k_c) = 0.098 * G(c, k_c): 0.646786 points for the "
                    "elongated radial-dominant channel and 10.4851 points for the "
                    "stable vertical control channel"
                ),
            },
            {
                "consumer": str(DIIID_CONSUMER),
                "inherited_reading": "GATE_RESIDUAL_TOLERANCE = 1e-6",
                "replacement_reading": (
                    "rejected: 79.305 times below the measured fine-mesh floor"
                ),
            },
            {
                "consumer": str(DIIID_CONSUMER),
                "inherited_reading": "REGISTERED_RESIDUAL_TOLERANCE = 1e-5",
                "replacement_reading": (
                    "1e-5 survives numerically, re-derived as the largest declared "
                    "candidate below the 7.930534999195602e-5 same-norm floor"
                ),
            },
            {
                "consumer": str(PARITY_CONSUMER),
                "inherited_reading": "atol = rtol = 1e-10 at terminal state",
                "replacement_reading": (
                    "one-map relative bound 16*epsilon_float64 = "
                    "3.552713678800501e-15 plus 69 separately registered terminal "
                    "observable bounds, conditional on exact seed alignment"
                ),
            },
        ],
        "claim_bounds": {
            "registered_source_constants_changed": False,
            "new_equilibrium_solve": False,
            "banked_measurements_only": True,
            "coupled_relation_interpolates_between_configurations": False,
            "diiid_reference_accuracy_equated_to_solver_residual": False,
            "terminal_observable_envelopes_are_physics_tolerances": False,
            "terminal_parity_without_seed_alignment_claimed": False,
        },
    }


def build_receipt() -> dict[str, Any]:
    """Build the criterion family and attach exact source digests."""
    receipts, consumers = _load_inputs()
    receipt = build_receipt_from_data(receipts, consumers)
    receipt["sources"] = {
        str(path): _sha256(path) for path in (*JSON_SOURCES, *CONSUMER_SOURCES)
    }
    return receipt


def write_receipt(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """Write and return the derived criterion-family receipt."""
    receipt = build_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    arguments = parser.parse_args()
    receipt = write_receipt(arguments.output)
    family = receipt["criterion_family"]
    terminal_count = family["terminal_compiled_parity"][
        "terminal_observable_registration"
    ]["observable_count"]
    print(
        "consumers=3 "
        f"configurations={len(family['coupled_response']['configuration_table'])} "
        "diiid=1e-5 "
        f"terminal_observables={terminal_count} "
        "solves=0"
    )


if __name__ == "__main__":
    main()
