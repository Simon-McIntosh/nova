"""Recheck banked DIII-D terminal saddles against wall containment.

The banked receipt remains read-only.  The before read is serialized evidence;
the after read applies the production wall-polygon mask from the containment
repair before repeating the exact normalized-level and wall-shadow reductions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import diiid_forward_gs_match as forward_match
from nova.equilibrium.connectivity_boundary import _points_inside_polygon


DEFAULT_INPUT = Path(
    "docs/figures/plateau-input-attribution/margin-frame-remeasure.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/plateau-input-attribution/diiid-containment-recheck.json"
)
CONTAINMENT_REPAIR_COMMIT = "5acfe07bd589534d70dd88ce98516889ff5f7db0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extended(value: float) -> tuple[float | None, str | None]:
    if math.isfinite(value):
        return value, None
    if math.isinf(value):
        return None, "positive_infinity" if value > 0.0 else "negative_infinity"
    return None, "not_a_number"


def _selection_state(
    candidates: list[dict[str, Any]],
    eligible: np.ndarray,
    axis_z: float,
    wall_coordinate: list[float],
    wall_level: float,
) -> dict[str, Any]:
    levels = np.asarray(
        [candidate["normalized_flux_operand"] for candidate in candidates],
        dtype=np.float64,
    )
    selected_index = int(np.argmin(np.where(eligible, levels, np.inf)))
    if not eligible[selected_index]:
        raise RuntimeError("containment removed every typed saddle candidate")
    selected = candidates[selected_index]
    eligible_heights = np.asarray(
        [
            candidate["coordinate_m"][1]
            for candidate, is_eligible in zip(candidates, eligible, strict=True)
            if is_eligible
        ],
        dtype=np.float64,
    )
    low = float(np.min(eligible_heights))
    high = float(np.max(eligible_heights))
    if low > axis_z:
        low = -np.inf
    if high < axis_z:
        high = np.inf
    wall_z = float(wall_coordinate[1])
    shadowed = wall_z < low or wall_z > high
    class_margin = np.inf if shadowed else wall_level - levels[selected_index]
    finite_margin, margin_nonfinite = _extended(float(class_margin))
    return {
        "selected_x_coordinate_m": selected["coordinate_m"],
        "selected_normalized_level": float(levels[selected_index]),
        "wall_shadow_verdict": "SHADOWED" if shadowed else "ADMITTED",
        "class_margin": finite_margin,
        "class_margin_nonfinite": margin_nonfinite,
        "achieved_class": "diverted" if class_margin >= 0.0 else "limited",
    }


def _wall_for_arm(
    arm: str,
    shot: str,
    physical_wall: np.ndarray,
    axes_by_shot: dict[str, tuple[np.ndarray, np.ndarray]],
    data: Path,
) -> np.ndarray:
    if arm == "physical_ring":
        return physical_wall
    if arm != "pseudo_wall":
        raise RuntimeError(f"unexpected topology surface {arm!r}")
    if shot not in axes_by_shot:
        row = forward_match._read(data / shot, ("efit_grid_R", "efit_grid_Z"))
        axes_by_shot[shot] = forward_match.canonical_axes(row)
    return forward_match.pseudo_wall(
        *axes_by_shot[shot],
        forward_match.REGISTERED_BASELINE_PSEUDO_WALL_EXPANSION,
    )


def measure(input_path: Path, data: Path) -> dict[str, Any]:
    banked = json.loads(input_path.read_text())
    physical_wall, physical_receipt = forward_match._physical_wall_ring(
        forward_match.DEFAULT_MACHINE_ARTIFACT_CACHE,
        forward_match.DEFAULT_MACHINE_ARTIFACT_DIGEST,
    )
    axes_by_shot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    before_unreachable = 0
    after_without_support = 0

    for arm, arm_receipt in banked["arms"].items():
        for record in arm_receipt["frame_records"]:
            terminal = record["margin_graded"]
            diagnostic = terminal["terminal_xpoint_diagnostics"]
            candidates = diagnostic["typed_saddle_candidates"]
            if not candidates:
                raise RuntimeError("banked terminal has no typed saddle candidate")
            shot = record["shot"]
            selected_before = np.asarray(
                [candidate["selected"] for candidate in candidates], dtype=bool
            )
            if np.count_nonzero(selected_before) != 1:
                raise RuntimeError("banked terminal does not name exactly one saddle")

            points = np.asarray(
                [candidate["coordinate_m"] for candidate in candidates],
                dtype=np.float64,
            )
            inside_vessel = np.asarray(
                _points_inside_polygon(
                    points[:, 0],
                    points[:, 1],
                    physical_wall[:, 0],
                    physical_wall[:, 1],
                ),
                dtype=bool,
            )
            selection_was_inside_vessel = bool(inside_vessel[selected_before][0])

            selector_wall = _wall_for_arm(arm, shot, physical_wall, axes_by_shot, data)
            eligible_after = np.asarray(
                _points_inside_polygon(
                    points[:, 0],
                    points[:, 1],
                    selector_wall[:, 0],
                    selector_wall[:, 1],
                ),
                dtype=bool,
            )
            wall_operand = diagnostic["wall_operand"]
            wall_level = wall_operand["normalized_flux_before_shadow"]
            if wall_level is None:
                raise RuntimeError("banked wall operand is not finite before shadow")

            before = {
                "selected_x_coordinate_m": diagnostic["selected_x_coordinate_m"],
                "selected_normalized_level": diagnostic[
                    "selected_x_normalized_flux_operand"
                ],
                "wall_shadow_verdict": (
                    "SHADOWED" if wall_operand["shadowed"] else "ADMITTED"
                ),
                "class_margin": (
                    terminal["terminal_class_margin"]
                    if math.isfinite(terminal["terminal_class_margin"])
                    else None
                ),
                "class_margin_nonfinite": (
                    diagnostic["wall_operand"]["normalized_flux_nonfinite"]
                    if not math.isfinite(terminal["terminal_class_margin"])
                    else None
                ),
                "achieved_class": terminal["terminal_topology_class"],
            }
            after = _selection_state(
                candidates,
                eligible_after,
                float(terminal["terminal_axis_rz_m"][1]),
                wall_operand["coordinate_m"],
                float(wall_level),
            )
            changed = before != after
            connectivity_count = diagnostic["connectivity_admission"][
                "admitted_candidate_count"
            ]
            before_unreachable += diagnostic["selection_status"] == (
                "selected_typed_saddle_not_connectivity_reachable"
            )
            after_without_support += connectivity_count == 0
            rows.append(
                {
                    "shot": shot,
                    "frame": int(record["frame"]),
                    "arm": arm,
                    "changed": changed,
                    "pre_repair_selection_inside_diiid_wall": (
                        selection_was_inside_vessel
                    ),
                    "before": before,
                    "after": after,
                }
            )

    if len(rows) != 10:
        raise RuntimeError(f"expected ten terminal rows, found {len(rows)}")
    out_of_vessel = sum(
        not row["pre_repair_selection_inside_diiid_wall"] for row in rows
    )
    changed = sum(row["changed"] for row in rows)
    class_flips = sum(
        row["before"]["achieved_class"] != row["after"]["achieved_class"]
        for row in rows
    )
    pattern_survives_as_evidence = out_of_vessel == 0
    return {
        "artifact": "DIII-D typed-saddle containment recheck",
        "measurement_contract": {
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "after_authority_commit": CONTAINMENT_REPAIR_COMMIT,
            "after_selector": (
                "argmin normalized level over finite typed saddles inside the "
                "active topology-surface polygon"
            ),
            "vessel_test": (
                "production fixed-shape ray crossing with boundary inclusion "
                "against the governed DIII-D wall polygon; no bounding box"
            ),
            "physical_wall": physical_receipt,
            "bank_preservation": (
                "margin-frame-remeasure.json and existing figures were read only"
            ),
        },
        "summary": {
            "terminal_count": len(rows),
            "changed_terminal_count": changed,
            "unchanged_terminal_count": len(rows) - changed,
            "pre_repair_selection_inside_diiid_wall_count": len(rows) - out_of_vessel,
            "pre_repair_selection_outside_diiid_wall_count": out_of_vessel,
            "achieved_class_flip_count": class_flips,
            "before_not_connectivity_reachable_count": before_unreachable,
            "before_connectivity_support_count": len(rows) - before_unreachable,
            "after_without_connectivity_support_count": after_without_support,
            "after_connectivity_support_count": len(rows) - after_without_support,
            "ten_of_ten_reachability_claim_survives_as_evidence": (
                pattern_survives_as_evidence
            ),
            "reachability_claim_verdict": (
                "SURVIVES"
                if pattern_survives_as_evidence
                else "WITHDRAW: the cohort includes out-of-vessel selected saddles"
            ),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=forward_match.DEFAULT_DATA)
    arguments = parser.parse_args()
    receipt = measure(arguments.input, arguments.data)
    arguments.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
