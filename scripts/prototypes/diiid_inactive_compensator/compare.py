"""Read the matched masked and unmasked arms into one attributed table.

Every row is solved twice from identical seeds and source inputs: once with the
inactive compensator's image suppressed and once with the withdrawn unmasked
rule restored. This reads both receipts and both inner-endpoint reads, and
reports per row the convergence, the lower-saddle distance to EFIT's lower null,
the masked-minus-unmasked shift of that distance, and the inactive
compensator's field at the unmasked terminal state.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
SHIFT_LIMIT_MM = 1.0


def reason_names(tree):
    """Return the termination-reason vocabulary the measured tree declares."""
    module = ast.parse((tree / "nova/equilibrium/fixed_point.py").read_text())
    definition = None
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "FixedPointTerminationReason":
                definition = node
    assert definition is not None
    names = {}
    for node in definition.body:
        if isinstance(node, ast.Assign):
            column = node.targets[0]
            names[ast.literal_eval(node.value)] = column.id
    return names


def case_name(frame, branch):
    return f"frame-{frame}-{branch}"


def read_arm(folder, names):
    """Return one arm's outcome, or raise when its receipt is incomplete."""
    receipt = json.loads((folder / "receipt.json").read_text())
    trips = json.loads((folder / "trips.json").read_text())
    topology = receipt["terminal_topology"]
    terminal = trips["terminal"]
    return dict(
        converged=bool(receipt["converged"]),
        residual=float(receipt["residual"]),
        termination_reason=names[receipt["terminal_reason"]],
        lower_saddle_distance_m=float(topology["lower_saddle_distance_m"]),
        lower_saddle_rz_m=list(topology["lower_saddle_rz_m"]),
        efit_lower_null_rz_m=list(receipt["efit_x_point_rz_m"]),
        terminal_pin_field_mt=float(terminal["pin_field_mt"]),
        terminal_pin_unknown=list(terminal["pin_unknown"]),
        terminal_pin_image_sup_wb=float(terminal["pin_image_sup_wb"]),
        support=int(receipt["support"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = reason_names(args.tree)
    rows = {}
    for frame, branch in ROWS:
        name = case_name(frame, branch)
        masked = read_arm(args.input / "masked" / name, names)
        unmasked = read_arm(args.input / "unmasked" / name, names)
        shift_mm = 1e3 * (
            masked["lower_saddle_distance_m"] - unmasked["lower_saddle_distance_m"]
        )
        rows[name] = dict(
            masked=masked,
            unmasked=unmasked,
            masked_minus_unmasked_null_shift_mm=shift_mm,
            shift_within_one_mm=abs(shift_mm) <= SHIFT_LIMIT_MM,
        )
        print(
            "ROW "
            + json.dumps(
                dict(
                    row=name,
                    masked_converged=masked["converged"],
                    masked_residual=masked["residual"],
                    unmasked_converged=unmasked["converged"],
                    unmasked_residual=unmasked["residual"],
                    masked_lower_null_mm=1e3 * masked["lower_saddle_distance_m"],
                    unmasked_lower_null_mm=1e3 * unmasked["lower_saddle_distance_m"],
                    shift_mm=shift_mm,
                    unmasked_terminal_pin_field_mt=unmasked["terminal_pin_field_mt"],
                    unmasked_pin_image_sup_wb=unmasked["terminal_pin_image_sup_wb"],
                    masked_terminal_pin_field_mt=masked["terminal_pin_field_mt"],
                )
            ),
            flush=True,
        )
    verdict = dict(
        all_masked_converged=all(row["masked"]["converged"] for row in rows.values()),
        every_shift_within_one_mm=all(
            row["shift_within_one_mm"] for row in rows.values()
        ),
        rows=rows,
    )
    args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print(
        "VERDICT "
        + json.dumps(
            dict(
                all_masked_converged=verdict["all_masked_converged"],
                every_shift_within_one_mm=verdict["every_shift_within_one_mm"],
            )
        )
    )
    if not verdict["all_masked_converged"]:
        raise SystemExit("a masked row did not converge")
    if not verdict["every_shift_within_one_mm"]:
        raise SystemExit("a masked row moved its lower null by more than 1 mm")


if __name__ == "__main__":
    main()
