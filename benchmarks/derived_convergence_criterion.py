"""Derive a fixed-point criterion from held-out banked mesh ladders.

The criterion for a frozen target is fitted only from other references in the
same pre-registered topology stratum.  The target's closest residual and its
own mesh pair are excluded, so the threshold cannot reproduce the quantity it
is intended to test by construction.  This module reads committed receipts and
does not construct or solve an equilibrium.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.efit_operator_consistency_order import _fit_power_order
from benchmarks.efit_parity_criterion_provenance import richardson_fine_error

OUTPUT_PATH = Path(
    "docs/figures/scoring-criteria-derivation/derived-convergence-criterion.json"
)
THREE_SPACING_OUTPUT_PATH = Path(
    "docs/figures/forward-operator-refinement/three-spacing-criterion-refresh.json"
)
MESH_SOURCE = Path(
    "docs/figures/moment-conditioned-basin-entry/stall-mesh-sensitivity.json"
)
THREE_SPACING_SOURCE = Path(
    "docs/figures/forward-operator-refinement/reference-native-resolution-default.json"
)
TOPOLOGY_SOURCE = Path(
    "docs/figures/efit-forward-parity/tared-plasma-support-solve.json"
)
GATED_RESIDUAL_SOURCE = Path(
    "docs/figures/efit-forward-parity/passive-inclusive-frozen-six-scorecard.json"
)
BENCHMARK_SOURCE = Path("benchmarks/efit_forward_parity_slice.py")

REGISTERED_CRITERION = 1.0e-8
TARGET_MESH_SPACING_M = 0.125
MESH_RATIO = 2.0
EXPECTED_REFERENCES = {
    (21978, 35),
    (21983, 35),
    (21985, 51),
    (21986, 46),
    (21989, 55),
    (22086, 43),
}
EXPECTED_STRATA = {
    "closed-axis": {(21983, 35), (21985, 51)},
    "confinement-construction": {
        (21978, 35),
        (21986, 46),
        (21989, 55),
        (22086, 43),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_key(reference: dict[str, Any]) -> tuple[int, int]:
    return int(reference["shot"]), int(reference["slice_index"])


def _reference_label(key: tuple[int, int]) -> str:
    return f"{key[0]}/{key[1]}"


def _load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        json.loads(MESH_SOURCE.read_text()),
        json.loads(TOPOLOGY_SOURCE.read_text()),
        json.loads(GATED_RESIDUAL_SOURCE.read_text()),
    )


def _topology_strata(topology: dict[str, Any]) -> dict[tuple[int, int], str]:
    strata: dict[tuple[int, int], str] = {}
    for row in topology["per_shot"]:
        key = _reference_key(row["reference"])
        status = row["instrument_controlled_rows"]["lcfs_closed_branch"]["status"]
        if status == "scoreable":
            strata[key] = "closed-axis"
        elif status == "unscoreable_no_closed_axis_branch":
            strata[key] = "confinement-construction"
        else:
            raise ValueError(f"unsupported banked topology status {status!r}")

    if set(strata) != EXPECTED_REFERENCES:
        raise RuntimeError("the frozen six-reference cohort changed")
    for name, expected in EXPECTED_STRATA.items():
        observed = {key for key, stratum in strata.items() if stratum == name}
        if observed != expected:
            raise RuntimeError(f"the banked {name} stratum changed")
    return strata


def _mesh_pairs(mesh: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    pairs = {}
    for source_row in mesh["per_reference"]:
        key = _reference_key(source_row)
        row = copy.deepcopy(source_row)
        coarse = row["mesh_levels"]["coarse"]
        fine = row["mesh_levels"]["fine"]
        if not math.isclose(
            float(coarse["mesh_spacing_m"]) / float(fine["mesh_spacing_m"]),
            MESH_RATIO,
        ):
            raise RuntimeError("the banked mesh-spacing ratio changed")
        pairs[key] = row
    if set(pairs) != EXPECTED_REFERENCES - {(21986, 46)}:
        raise RuntimeError("the banked five-reference mesh cohort changed")
    return pairs


def _gated_rows(scorecard: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    rows = {_reference_key(row["reference"]): row for row in scorecard["per_shot"]}
    if set(rows) != EXPECTED_REFERENCES:
        raise RuntimeError("the frozen passive-inclusive cohort changed")
    return rows


def _finite_interval(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return (
        [float(value) for value in values] if all(map(math.isfinite, values)) else None
    )


def _fit_held_out_ladder(
    *,
    target: tuple[int, int],
    stratum: str,
    strata: dict[tuple[int, int], str],
    pairs: dict[tuple[int, int], dict[str, Any]],
    supplemental_levels: dict[tuple[int, int], list[dict[str, float]]] | None = None,
) -> dict[str, Any]:
    peer_keys = sorted(key for key in pairs if key != target and strata[key] == stratum)
    if not peer_keys:
        raise RuntimeError(
            f"no independent mesh pair remains for {_reference_label(target)}"
        )

    cell_sizes: list[float] = []
    residuals: list[float] = []
    fit_levels: list[dict[str, Any]] = []
    for key in peer_keys:
        pair = pairs[key]
        peer_levels: dict[float, float] = {}
        for level in ("coarse", "fine"):
            measurement = pair["mesh_levels"][level]
            spacing = float(measurement["mesh_spacing_m"])
            peer_levels[spacing] = float(measurement["terminal_residual"])
        if supplemental_levels and key in supplemental_levels:
            for measurement in supplemental_levels[key]:
                spacing = float(measurement["mesh_spacing_m"])
                residual = float(measurement["terminal_residual"])
                if spacing in peer_levels and not math.isclose(
                    peer_levels[spacing], residual, rel_tol=1.0e-12, abs_tol=0.0
                ):
                    raise RuntimeError(
                        f"supplemental ladder disagrees at {spacing} m for "
                        f"{_reference_label(key)}"
                    )
                peer_levels[spacing] = residual
        for spacing, residual in sorted(peer_levels.items(), reverse=True):
            cell_sizes.append(spacing)
            residuals.append(residual)
            fit_levels.append(
                {
                    "reference": _reference_label(key),
                    "mesh_spacing_m": spacing,
                    "terminal_residual": residual,
                }
            )

    fit = _fit_power_order(
        np.asarray(cell_sizes, dtype=float), np.asarray(residuals, dtype=float)
    )
    order = float(fit["observed_order"])
    coefficient = float(fit["coefficient"])
    criterion = coefficient * TARGET_MESH_SPACING_M**order
    order_interval = _finite_interval(fit["order_95_percent_interval"])
    result = {
        "formula": "tau_i(h_i) = C_(s,-i) * h_i**p_(s,-i)",
        "criterion": float(criterion),
        "target_mesh_spacing_m": TARGET_MESH_SPACING_M,
        "held_out_target": _reference_label(target),
        "fit_reference_count": len(peer_keys),
        "fit_sample_count": len(residuals),
        "fit_unique_mesh_spacing_count": len(set(cell_sizes)),
        "fit_references": [_reference_label(key) for key in peer_keys],
        "fit_cell_sizes_m": cell_sizes,
        "fit_residuals": residuals,
        "observed_order": order,
        "coefficient": coefficient,
        "order_standard_error": float(fit["order_standard_error"]),
        "order_95_percent_interval": order_interval,
        "log_residual_rms": float(fit["log_residual_rms"]),
        "fraction_residual_rms": float(fit["fraction_residual_rms"]),
        "r_squared": float(fit["r_squared"]),
        "target_residual_used_in_fit": False,
        "target_mesh_pair_used_in_fit": False,
        "uncertainty_qualification": _fit_qualification(
            peer_count=len(peer_keys),
            unique_spacing_count=len(set(cell_sizes)),
        ),
    }
    if supplemental_levels is not None:
        result["fit_levels"] = fit_levels
    return result


def _fit_qualification(*, peer_count: int, unique_spacing_count: int) -> str:
    if unique_spacing_count >= 3:
        return (
            "THREE-SPACING FIT: a peer reference supplies a third distinct mesh "
            "spacing, so the two-spacing asymptotic qualification is lifted for "
            "this target; the fit still has no independent fourth-spacing check."
        )
    if peer_count == 1:
        return (
            "EXACT-PAIR-ONLY: one peer at two mesh spacings determines the power "
            "law exactly; there is no fit error bar or third-mesh confirmation."
        )
    return (
        "Peer references provide repeated samples at only two distinct mesh "
        "spacings; cross-reference scatter is measured, but there is no "
        "third-mesh confirmation of an asymptotic regime."
    )


def _three_spacing_levels(
    source: dict[str, Any],
) -> dict[tuple[int, int], list[dict[str, float]]]:
    selection = source["selection"]
    key = int(selection["shot"]), int(selection["slice_index"])
    if key != (21978, 35) or selection["qualification_passes"] is not True:
        raise RuntimeError("the banked three-spacing control selection changed")

    rows = source["resolution_rows"]
    level_sources = (
        rows["banked_33_point"],
        rows["fresh_65_point"]["metrics"],
        rows["fresh_95_point_reference_native"]["metrics"],
    )
    expected_points = (33, 65, 95)
    levels = []
    for points, row in zip(expected_points, level_sources, strict=True):
        if row["grid_shape"] != f"{points} by {points} rectangular benchmark lattice":
            raise RuntimeError("the banked three-spacing grid ladder changed")
        if row["controlled_lcfs_status"] != "unscoreable_no_closed_axis_branch":
            raise RuntimeError("the banked ladder topology stratum changed")
        levels.append(
            {
                "mesh_spacing_m": max(
                    abs(float(row["radial_step_m"])),
                    abs(float(row["vertical_step_m"])),
                ),
                "terminal_residual": float(row["terminal_fixed_point_residual"]),
            }
        )
    if len({level["mesh_spacing_m"] for level in levels}) != 3:
        raise RuntimeError("the banked control does not contain three mesh spacings")
    return {key: levels}


def circular_richardson_collapse(
    coarse_residual: float, fine_residual: float, mesh_ratio: float = MESH_RATIO
) -> dict[str, float]:
    """Show the identity produced when a pair supplies its own order."""
    if not (
        math.isfinite(coarse_residual)
        and math.isfinite(fine_residual)
        and coarse_residual > fine_residual > 0.0
        and mesh_ratio > 1.0
    ):
        raise ValueError("collapse inputs require coarse > fine > 0 and ratio > 1")
    order = math.log(coarse_residual / fine_residual) / math.log(mesh_ratio)
    estimate = richardson_fine_error(coarse_residual, fine_residual, order, mesh_ratio)
    return {
        "same_pair_order": order,
        "richardson_estimate": estimate,
        "fine_residual": fine_residual,
        "estimate_over_fine_residual": estimate / fine_residual,
    }


def build_receipt_from_data(
    mesh: dict[str, Any],
    topology: dict[str, Any],
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    """Build the criterion receipt from already loaded banked data."""
    strata = _topology_strata(topology)
    pairs = _mesh_pairs(mesh)
    gated = _gated_rows(scorecard)

    rows = []
    for key in sorted(EXPECTED_REFERENCES):
        stratum = strata[key]
        fit = _fit_held_out_ladder(
            target=key,
            stratum=stratum,
            strata=strata,
            pairs=pairs,
        )
        gated_row = gated[key]
        low_order_qualification = None
        if key == (21978, 35):
            low_order_qualification = (
                "LEAST-TRUSTWORTHY TARGET: its banked same-reference order is "
                "0.966596902394278 and was measured outside a confirmed "
                "asymptotic regime; that pair is excluded from this fit."
            )
        elif key == (22086, 43):
            low_order_qualification = (
                "LEAST-TRUSTWORTHY TARGET: its banked same-reference order is "
                "1.6371714162964488 and was measured outside a confirmed "
                "asymptotic regime; that pair is excluded from this fit."
            )
        rows.append(
            {
                "reference": _reference_label(key),
                "stratum": stratum,
                "derived_criterion": fit["criterion"],
                "criterion_unit": "relative sup norm of fixed-point defect",
                "gated_closest_residual_display_only": float(
                    gated_row["closest_approach"]["residual"]
                ),
                "gated_outcome_display_only": gated_row["solve_outcome"][
                    "outcome_class"
                ],
                "fit": fit,
                "target_qualification": low_order_qualification,
            }
        )

    circular_rows = []
    for key, pair in sorted(pairs.items()):
        coarse = float(pair["mesh_levels"]["coarse"]["terminal_residual"])
        fine = float(pair["mesh_levels"]["fine"]["terminal_residual"])
        circular_rows.append(
            {
                "reference": _reference_label(key),
                **circular_richardson_collapse(coarse, fine),
            }
        )

    stratum_rows = {}
    for name, members in EXPECTED_STRATA.items():
        measured_orders = [
            float(pairs[key]["observed_mesh_order"])
            for key in sorted(members & pairs.keys())
        ]
        stratum_rows[name] = {
            "frozen_references": [_reference_label(key) for key in sorted(members)],
            "mesh_order_reference_count": len(measured_orders),
            "banked_observed_order_range": [
                min(measured_orders),
                max(measured_orders),
            ],
            "domain_of_validity": (
                "Only the pre-registered axis-enclosing branch stratum on the "
                "banked 0.125 m and 0.0625 m mesh spacings. Leave-one-out fits "
                "have one peer pair and therefore no error bar."
                if name == "closed-axis"
                else (
                    "Only the pre-registered no-axis-enclosing-branch stratum on "
                    "the banked 0.125 m and 0.0625 m mesh spacings. The order "
                    "range is broad and two spacings do not establish an "
                    "asymptotic regime."
                )
            ),
        }

    return {
        "receipt": {
            "kind": "held_out_mesh_power_law_convergence_criterion",
            "status": "complete",
            "execution_mode": "banked-data-only-no-equilibrium-solves",
            "equilibrium_solves_run": 0,
            "reference_count": len(rows),
        },
        "criterion": {
            "formula": "tau_i(h_i) = C_(s,-i) * h_i**p_(s,-i)",
            "fit_model": "R_j(h) = C_(s,-i) * h**p_(s,-i), j != i",
            "target_mesh_spacing_m": TARGET_MESH_SPACING_M,
            "target_mesh_provenance": (
                "The gated scorecard is constructed on the benchmark's 33 by 33 "
                "rectangular lattice. The banked mesh ladder identifies that "
                "coarse lattice by its common 0.125 m limiting spacing."
            ),
            "stratification": (
                "pre-registered closed-axis and confinement-construction strata; "
                "never pooled"
            ),
            "independence_argument": (
                "For target i, C_(s,-i) and p_(s,-i) are fitted only from other "
                "references j in the same pre-registered stratum. The gated "
                "closest residual and target i's own mesh pair are excluded."
            ),
            "inputs": [
                "peer-reference coarse and fine banked fixed-point residuals",
                "peer-reference banked mesh spacings",
                "pre-registered topology stratum",
                "the target benchmark mesh spacing",
            ],
            "excluded_inputs": [
                "the gated target closest residual",
                "the target reference's own coarse/fine mesh pair",
            ],
            "registered_criterion_retained": REGISTERED_CRITERION,
            "registered_tolerance_changed": False,
            "fitness_verdict": (
                "Reuse _fit_power_order for the peer-only log-space power law. "
                "Do not reuse same-pair Richardson as a gate because its order "
                "and estimate contain the target fine residual."
            ),
        },
        "strata": stratum_rows,
        "per_reference": rows,
        "excluded_circular_estimator": {
            "order_formula": "p_i = log(R_coarse_i / R_fine_i) / log(r)",
            "candidate_formula": ("E_f_i = (R_coarse_i - R_fine_i) / (r**p_i - 1)"),
            "algebraic_collapse": [
                "r**p_i = R_coarse_i / R_fine_i",
                ("E_f_i = (R_coarse_i - R_fine_i) / (R_coarse_i / R_fine_i - 1)"),
                "E_f_i = R_fine_i",
            ],
            "reason_excluded": (
                "The target fine residual supplies both the fitted order and the "
                "estimate, so E_f_i <= E_f_i is not an independent test."
            ),
            "per_reference_numeric_check": circular_rows,
        },
        "claim_bounds": {
            "criterion_scope": (
                "A leave-one-reference-out interpolation of banked mesh floors, "
                "not a production solver stopping tolerance."
            ),
            "new_equilibrium_solve": False,
            "banked_residuals_only": True,
            "third_mesh_available": False,
            "independent_asymptotic_confirmation": False,
            "new_reliability_verdict_claimed": False,
        },
    }


def build_receipt() -> dict[str, Any]:
    """Build the complete receipt from committed evidence inputs."""
    mesh, topology, scorecard = _load_sources()
    receipt = build_receipt_from_data(mesh, topology, scorecard)
    receipt["sources"] = {
        str(MESH_SOURCE): _sha256(MESH_SOURCE),
        str(TOPOLOGY_SOURCE): _sha256(TOPOLOGY_SOURCE),
        str(GATED_RESIDUAL_SOURCE): _sha256(GATED_RESIDUAL_SOURCE),
        str(BENCHMARK_SOURCE): _sha256(BENCHMARK_SOURCE),
    }
    return receipt


def build_three_spacing_receipt_from_data(
    mesh: dict[str, Any],
    topology: dict[str, Any],
    scorecard: dict[str, Any],
    three_spacing_source: dict[str, Any],
) -> dict[str, Any]:
    """Refresh held-out criteria where an independent third rung is banked."""
    two_spacing_receipt = build_receipt_from_data(mesh, topology, scorecard)
    two_spacing_rows = {
        row["reference"]: row for row in two_spacing_receipt["per_reference"]
    }
    strata = _topology_strata(topology)
    pairs = _mesh_pairs(mesh)
    gated = _gated_rows(scorecard)
    supplemental_levels = _three_spacing_levels(three_spacing_source)

    rows = []
    for key in sorted(EXPECTED_REFERENCES):
        label = _reference_label(key)
        stratum = strata[key]
        fit = _fit_held_out_ladder(
            target=key,
            stratum=stratum,
            strata=strata,
            pairs=pairs,
            supplemental_levels=supplemental_levels,
        )
        two_spacing_criterion = float(two_spacing_rows[label]["derived_criterion"])
        refreshed_criterion = float(fit["criterion"])
        third_spacing_used = fit["fit_unique_mesh_spacing_count"] == 3
        rows.append(
            {
                "reference": label,
                "stratum": stratum,
                "two_spacing_criterion": two_spacing_criterion,
                "refreshed_criterion": refreshed_criterion,
                "refreshed_over_two_spacing": (
                    refreshed_criterion / two_spacing_criterion
                ),
                "criterion_unit": "relative sup norm of fixed-point defect",
                "gated_closest_residual_display_only": float(
                    gated[key]["closest_approach"]["residual"]
                ),
                "fit": fit,
                "target_qualification": two_spacing_rows[label]["target_qualification"],
                "third_spacing_used": third_spacing_used,
                "qualification_status": (
                    "THREE_SPACING_QUALIFICATION_LIFTED"
                    if third_spacing_used
                    else "TWO_SPACING_QUALIFICATION_RETAINED"
                ),
                "qualification_reason": (
                    "The banked 21978/35 control ladder is an independent peer "
                    "input for this target."
                    if third_spacing_used
                    else (
                        "The only banked third rung belongs to this target and is "
                        "held out with its mesh pair."
                        if key == (21978, 35)
                        else "No peer in this target's stratum has a third rung."
                    )
                ),
            }
        )

    stratum_rows = {}
    for name, members in EXPECTED_STRATA.items():
        member_rows = [row for row in rows if row["stratum"] == name]
        lifted = [row["reference"] for row in member_rows if row["third_spacing_used"]]
        retained = [
            row["reference"] for row in member_rows if not row["third_spacing_used"]
        ]
        stratum_rows[name] = {
            "frozen_references": [_reference_label(key) for key in sorted(members)],
            "qualification_status": (
                "LIFTED"
                if not retained
                else ("PARTIALLY_LIFTED" if lifted else "RETAINED")
            ),
            "targets_with_three_spacing_fit": lifted,
            "targets_retaining_two_spacing_qualification": retained,
            "two_spacing_order_range": [
                min(
                    two_spacing_rows[row["reference"]]["fit"]["observed_order"]
                    for row in member_rows
                ),
                max(
                    two_spacing_rows[row["reference"]]["fit"]["observed_order"]
                    for row in member_rows
                ),
            ],
            "refreshed_order_range": [
                min(row["fit"]["observed_order"] for row in member_rows),
                max(row["fit"]["observed_order"] for row in member_rows),
            ],
            "qualification_reason": (
                "No closed-axis peer has a banked 95-point rung."
                if name == "closed-axis"
                else (
                    "The 21978/35 ladder lifts the qualification for the other "
                    "three targets; its own criterion must exclude that ladder."
                )
            ),
        }

    return {
        "receipt": {
            "kind": "three_spacing_held_out_criterion_refresh",
            "status": "complete",
            "execution_mode": "banked-data-only-no-equilibrium-solves",
            "equilibrium_solves_run": 0,
            "reference_count": len(rows),
            "refreshed_reference_count": sum(row["third_spacing_used"] for row in rows),
            "retained_reference_count": sum(
                not row["third_spacing_used"] for row in rows
            ),
        },
        "criterion": {
            "formula": "tau_i(h_i) = C_(s,-i) * h_i**p_(s,-i)",
            "fit_model": "R_j(h) = C_(s,-i) * h**p_(s,-i), j != i",
            "target_mesh_spacing_m": TARGET_MESH_SPACING_M,
            "independence_argument": (
                "For target i, all fit levels belong to other references j in "
                "the same pre-registered topology stratum. The gated closest "
                "residual and every level in target i's mesh ladder are excluded."
            ),
            "registered_criterion_retained": REGISTERED_CRITERION,
            "registered_tolerance_changed": False,
        },
        "banked_three_spacing_ladder": {
            "reference": "21978/35",
            "stratum": "confinement-construction",
            "spacing_definition": (
                "maximum absolute rectangular-axis step, matching the limiting "
                "mesh spacing used by the banked two-spacing pairs"
            ),
            "levels": supplemental_levels[(21978, 35)],
        },
        "strata": stratum_rows,
        "per_reference": rows,
        "claim_bounds": {
            "new_equilibrium_solve": False,
            "banked_residuals_only": True,
            "blanket_asymptotic_confirmation": False,
            "reason": (
                "One banked third-spacing ladder can independently refresh only "
                "targets outside that ladder and inside its topology stratum."
            ),
            "new_reliability_verdict_claimed": False,
        },
    }


def build_three_spacing_receipt() -> dict[str, Any]:
    """Build the three-spacing refresh from committed evidence inputs."""
    mesh, topology, scorecard = _load_sources()
    three_spacing_source = json.loads(THREE_SPACING_SOURCE.read_text())
    receipt = build_three_spacing_receipt_from_data(
        mesh, topology, scorecard, three_spacing_source
    )
    receipt["sources"] = {
        str(MESH_SOURCE): _sha256(MESH_SOURCE),
        str(THREE_SPACING_SOURCE): _sha256(THREE_SPACING_SOURCE),
        str(TOPOLOGY_SOURCE): _sha256(TOPOLOGY_SOURCE),
        str(GATED_RESIDUAL_SOURCE): _sha256(GATED_RESIDUAL_SOURCE),
        str(BENCHMARK_SOURCE): _sha256(BENCHMARK_SOURCE),
    }
    return receipt


def write_receipt(path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """Write and return the criterion receipt."""
    receipt = build_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def write_three_spacing_receipt(
    path: Path = THREE_SPACING_OUTPUT_PATH,
) -> dict[str, Any]:
    """Write and return the three-spacing criterion refresh."""
    receipt = build_three_spacing_receipt()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--three-spacing-refresh", action="store_true")
    arguments = parser.parse_args()
    if arguments.three_spacing_refresh:
        receipt = write_three_spacing_receipt(
            arguments.output or THREE_SPACING_OUTPUT_PATH
        )
        criteria = [row["refreshed_criterion"] for row in receipt["per_reference"]]
    else:
        receipt = write_receipt(arguments.output or OUTPUT_PATH)
        criteria = [row["derived_criterion"] for row in receipt["per_reference"]]
    print(
        f"references={len(criteria)} criterion_min={min(criteria):.9e} "
        f"criterion_max={max(criteria):.9e} solves=0"
    )


if __name__ == "__main__":
    main()
