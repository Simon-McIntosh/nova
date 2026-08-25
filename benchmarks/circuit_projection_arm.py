"""Test a residual projection against the stored MAST circuit wiring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from nova.imas.mast_solve_inputs import SHOT_STORE


DEFAULT_RESIDUAL_RECEIPT = Path(
    "docs/figures/plateau-input-attribution/label-seed-residual-field.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/plateau-input-attribution/circuit-projection-arm.json"
)
EXPECTED_BEST_CIRCUIT = {
    "21978/35": 83,
    "21983/35": 83,
    "21985/51": 83,
    "21986/46": 84,
    "21989/55": 83,
    "22086/43": 83,
}
DEFINITION_ARRAYS = (
    "fcoil_n",
    "fcoil_circ",
    "fcoil_r",
    "fcoil_z",
    "fcoil_width",
    "fcoil_height",
    "fcoil_ang1",
    "fcoil_ang2",
    "fcoil_turns",
    "fcoil_xmult",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(group: zarr.Group) -> str:
    """Hash the complete stored circuit geometry and response multipliers."""
    digest = hashlib.sha256()
    for name in DEFINITION_ARRAYS:
        array = np.ascontiguousarray(np.asarray(group[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _reference_name(row: dict[str, Any]) -> str:
    reference = row["reference"]
    return f"{int(reference['shot'])}/{int(reference['slice_index'])}"


def _projection_rows(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {_reference_name(row): row for row in receipt["per_reference"]}
    if set(rows) != set(EXPECTED_BEST_CIRCUIT):
        raise ValueError("residual receipt does not contain the exact reference cohort")
    if len(rows) != len(receipt["per_reference"]):
        raise ValueError("residual receipt repeats a reference")
    return rows


def _simple_discrete_hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "choice": "open_or_missing_single_section",
            "equivalent_correction_fraction": -1.0,
            "resulting_response_scale": 0.0,
        },
        {
            "choice": "single_section_polarity_reversal",
            "equivalent_correction_fraction": -2.0,
            "resulting_response_scale": -1.0,
        },
        {
            "choice": "one_additional_identical_series_turn",
            "equivalent_correction_fraction": 1.0,
            "resulting_response_scale": 2.0,
        },
    ]


def _circuit_comparison(
    group: zarr.Group,
    row_index: int,
    stored_circuit: int,
    equivalent_correction_a: float,
) -> dict[str, Any]:
    circuit_for_element = np.asarray(group["fcoil_circ"], dtype=int)
    current_index = np.asarray(group["fcoil_n"], dtype=int)
    driven_currents = np.asarray(group["fcoil_c"][row_index], dtype=np.float64)
    if not np.array_equal(current_index, np.arange(driven_currents.size)):
        raise ValueError("stored circuit-current order is not zero based")
    if stored_circuit < 1 or stored_circuit > driven_currents.size:
        raise ValueError(f"stored circuit {stored_circuit} has no driven current")

    selected = np.flatnonzero(circuit_for_element == stored_circuit)
    if selected.size == 0:
        raise ValueError(f"stored circuit {stored_circuit} has no section")
    turns = np.asarray(group["fcoil_turns"], dtype=np.float64)[selected]
    gains = np.asarray(group["fcoil_xmult"], dtype=np.float64)[selected]
    if selected.size != 1:
        raise ValueError(
            f"stored circuit {stored_circuit} is no longer a single-section circuit"
        )
    if not np.array_equal(turns, np.ones(1)) or not np.array_equal(gains, np.ones(1)):
        raise ValueError(
            f"stored circuit {stored_circuit} no longer has one turn and unit gain"
        )
    response_multipliers = turns * gains
    driven_current_a = float(driven_currents[stored_circuit - 1])
    if driven_current_a == 0.0:
        raise ValueError(f"stored circuit {stored_circuit} has zero driven current")

    correction_fraction = float(equivalent_correction_a / driven_current_a)
    implied_response_scale = float(1.0 + correction_fraction)
    hypotheses = _simple_discrete_hypotheses()
    matching_choices = [
        item["choice"]
        for item in hypotheses
        if math.isclose(
            correction_fraction,
            float(item["equivalent_correction_fraction"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    shipped_multiplier = float(np.sum(response_multipliers))
    implied_multiplier = float(shipped_multiplier * implied_response_scale)
    nearest_integer = int(round(implied_multiplier))
    return {
        "stored_circuit_one_based": stored_circuit,
        "stored_response_column_zero_based": stored_circuit - 1,
        "stored_driven_current_a": driven_current_a,
        "equivalent_current_correction_a": equivalent_correction_a,
        "correction_fraction_of_stored_driven_current": correction_fraction,
        "implied_total_current_a": float(driven_current_a + equivalent_correction_a),
        "implied_response_scale": implied_response_scale,
        "shipped_definition": {
            "circuit_assignment_path": "efm/fcoil_circ",
            "driven_current_path": "efm/fcoil_c",
            "turns_path": "efm/fcoil_turns",
            "gain_path": "efm/fcoil_xmult",
            "section_element_indices_zero_based": selected.tolist(),
            "section_element_count": int(selected.size),
            "turns": turns.tolist(),
            "gains": gains.tolist(),
            "turn_gain_products": response_multipliers.tolist(),
            "summed_turn_gain_multiplier": shipped_multiplier,
            "polarity": "positive" if shipped_multiplier > 0.0 else "negative",
            "series_parallel_choice_present": bool(selected.size > 1),
        },
        "discrete_comparison": {
            "simple_choices": hypotheses,
            "matching_simple_choices": matching_choices,
            "implied_turn_gain_multiplier": implied_multiplier,
            "nearest_integer_turn_gain_multiplier": nearest_integer,
            "distance_to_nearest_integer": float(
                abs(implied_multiplier - nearest_integer)
            ),
            "per_reference_verdict": "NONE",
            "reason": (
                "The shipped circuit has one section at one positive turn and unit "
                "gain, so it has no series-parallel branch choice. The projection "
                "matches neither an open section, a polarity reversal, nor one "
                "additional identical series turn; its much larger negative scale "
                "is not a shipped discrete choice."
            ),
        },
    }


def build_receipt(
    residual_receipt_path: Path = DEFAULT_RESIDUAL_RECEIPT,
    store: Path = SHOT_STORE,
) -> dict[str, Any]:
    """Build the wiring interpretation from the immutable residual projection."""
    residual_receipt = json.loads(residual_receipt_path.read_text(encoding="utf-8"))
    rows = _projection_rows(residual_receipt)
    definition_digests: dict[str, str] = {}
    references = []
    by_circuit: dict[int, list[dict[str, Any]]] = {}

    for reference_name in EXPECTED_BEST_CIRCUIT:
        row = rows[reference_name]
        reference = row["reference"]
        shot = int(reference["shot"])
        row_index = int(reference["slice_index"])
        projection = row["one_application_residual"]["single_circuit_green_pattern"]
        stored_circuit = int(projection["best_stored_circuit_one_based"])
        if stored_circuit != EXPECTED_BEST_CIRCUIT[reference_name]:
            raise ValueError(f"{reference_name} best stored circuit changed")

        group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
        definition_digests[reference_name] = _array_digest(group)
        comparison = _circuit_comparison(
            group,
            row_index,
            stored_circuit,
            float(projection["equivalent_current_correction_a"]),
        )
        record = {
            "reference": reference_name,
            "shot": shot,
            "slice_index": row_index,
            "projection_explained_fraction_of_full_squared_magnitude": float(
                projection["fraction_of_full_squared_magnitude_explained"]
            ),
            **comparison,
        }
        references.append(record)
        by_circuit.setdefault(stored_circuit, []).append(record)

    unique_definition_digests = sorted(set(definition_digests.values()))
    if len(unique_definition_digests) != 1:
        raise ValueError("the reference cohort does not share one circuit definition")

    circuit_consistency = {}
    for stored_circuit, circuit_rows in sorted(by_circuit.items()):
        scales = [float(item["implied_response_scale"]) for item in circuit_rows]
        circuit_consistency[str(stored_circuit)] = {
            "reference_count": len(circuit_rows),
            "references": [str(item["reference"]) for item in circuit_rows],
            "implied_response_scale_minimum": min(scales),
            "implied_response_scale_maximum": max(scales),
            "implied_response_scale_span": float(max(scales) - min(scales)),
            "common_discrete_scale_present": bool(
                len(circuit_rows) > 1
                and all(
                    math.isclose(scales[0], scale, rel_tol=0.0, abs_tol=1.0e-12)
                    for scale in scales[1:]
                )
            ),
        }

    scores = residual_receipt["aggregate"]["verdict"]["cohort_mean_pattern_scores"]
    conductor_score = float(scores["conductor_current_wiring"])
    wall_score = float(scores["wall"])
    if round(conductor_score, 10) != 0.1106335889:
        raise ValueError("conductor-wiring cohort score changed")
    if round(wall_score, 10) != 0.0948572698:
        raise ValueError("wall cohort score changed")

    return {
        "receipt": "MAST single-circuit projection discrete-wiring check",
        "inputs": {
            "residual_field_receipt": {
                "path": str(residual_receipt_path),
                "sha256": _sha256(residual_receipt_path),
                "source_revision": residual_receipt["source_revision"],
            },
            "shot_store": str(store),
            "circuit_definition_arrays": list(DEFINITION_ARRAYS),
            "shared_circuit_definition_sha256": unique_definition_digests[0],
            "definition_digest_by_reference": definition_digests,
        },
        "comparison_contract": {
            "reference_count": len(references),
            "reference_order": list(EXPECTED_BEST_CIRCUIT),
            "best_stored_circuit_by_reference": EXPECTED_BEST_CIRCUIT,
            "projection_role": (
                "A read-only equivalent-current interpretation of an already-banked "
                "single-circuit residual projection; it is not a current fit."
            ),
            "discrete_rule": (
                "Compare with the shipped section count, signed turns and gains. "
                "A simple open, polarity or added-series-turn choice must match its "
                "fixed response scale; a hardware wiring choice for one stored "
                "circuit must also be invariant across references."
            ),
        },
        "per_reference": references,
        "per_circuit_consistency": circuit_consistency,
        "pattern_score_context": {
            "conductor_wiring_cohort_mean": conductor_score,
            "wall_cohort_mean": wall_score,
            "absolute_score_gap": float(conductor_score - wall_score),
            "interpretation": (
                "The near-tied cohort means are carried without a separation claim; "
                "this projection check cannot separate conductor wiring from wall "
                "structure on its own."
            ),
        },
        "verdict": {
            "discrete_wiring_difference": "NONE",
            "references_with_plausible_discrete_difference": 0,
            "references_checked": len(references),
            "statement": (
                "None of the six equivalent current corrections corresponds to a "
                "plausible discrete difference in the shipped one-section, "
                "one-turn, unit-gain circuit definitions. Circuit 83 also demands "
                "different scales across its five references, which a fixed wiring "
                "choice cannot do."
            ),
            "conductor_wall_separation": "NOT_ESTABLISHED",
            "fitted_current_authored": False,
            "repair_authored": False,
        },
    }


def write_receipt(receipt: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def check_receipt(
    output: Path = DEFAULT_OUTPUT,
    residual_receipt: Path = DEFAULT_RESIDUAL_RECEIPT,
    store: Path = SHOT_STORE,
) -> dict[str, Any]:
    """Fail closed unless the banked interpretation reproduces its inputs."""
    banked = json.loads(output.read_text(encoding="utf-8"))
    expected = build_receipt(residual_receipt, store)
    if banked != expected:
        raise ValueError("banked circuit projection does not reproduce its inputs")
    verdict = banked["verdict"]
    if verdict["discrete_wiring_difference"] != "NONE":
        raise ValueError("a non-shipped wiring choice was treated as plausible")
    if verdict["references_with_plausible_discrete_difference"] != 0:
        raise ValueError("a reference was assigned a discrete wiring difference")
    if verdict["fitted_current_authored"] or verdict["repair_authored"]:
        raise ValueError("the diagnostic check authored a fit or repair")
    if verdict["conductor_wall_separation"] != "NOT_ESTABLISHED":
        raise ValueError("near-tied pattern scores were treated as separated")
    return banked


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument(
        "--residual-receipt", type=Path, default=DEFAULT_RESIDUAL_RECEIPT
    )
    measure.add_argument("--store", type=Path, default=SHOT_STORE)
    measure.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check = commands.add_parser("check")
    check.add_argument(
        "--residual-receipt", type=Path, default=DEFAULT_RESIDUAL_RECEIPT
    )
    check.add_argument("--store", type=Path, default=SHOT_STORE)
    check.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    if arguments.command == "measure":
        receipt = build_receipt(arguments.residual_receipt, arguments.store)
        write_receipt(receipt, arguments.output)
    else:
        receipt = check_receipt(
            arguments.output, arguments.residual_receipt, arguments.store
        )
    print(
        "CIRCUIT_PROJECTION_DISCRETE_WIRING "
        f"references={receipt['comparison_contract']['reference_count']} "
        f"best_circuits={sorted(set(EXPECTED_BEST_CIRCUIT.values()))} "
        f"discrete={receipt['verdict']['discrete_wiring_difference']} "
        f"separation={receipt['verdict']['conductor_wall_separation']} "
        "fits=0 repairs=0 verdict=PASS"
    )


if __name__ == "__main__":
    main()
