"""Require complete, spatially bound receipts and test the failure premise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = ((137, "diverted"), (146, "diverted"), (146, "limited"), (179, "limited"))
EXPECTED_REASON = {
    (137, "diverted"): 12,
    (146, "diverted"): 1,
    (146, "limited"): 7,
    (179, "limited"): 12,
}


def completion(input_root, folder, spatial):
    """Distinguish a completed trace from its optional counterfactual tail."""
    log = (folder / "row.log").read_text()
    if log.endswith("EXIT=0\n"):
        return dict(process_exit=0, basis="row completed normally")
    # A tail import failure occurs after the spatial summary has been flushed.
    # Qualify that record explicitly and preserve the original nonzero exit.
    assert folder.name == "frame-179-limited" and log.endswith("EXIT=1\n")
    failed = input_root / "inactive-field-import-failure/control.log"
    tail = failed.read_text()
    assert tail.endswith("EXIT=1\n")
    assert "AttributeError: module 'measure' has no attribute 'load'" in tail
    assert "PROFILE_ARM " not in tail and "REQUEST " not in tail
    full_log = (input_root.parent / "titan-trace.log").read_text()
    records = [
        json.loads(line.removeprefix("SPATIAL_SUMMARY "))
        for line in full_log.splitlines()
        if line.startswith("SPATIAL_SUMMARY ")
    ]
    matched = [row for row in records if row["terminal"] == spatial["terminal"]]
    assert len(matched) == 1
    assert matched[0]["trips"] == len(spatial["trips"])
    assert matched[0]["proposals"] == len(spatial["proposals"])
    assert len(spatial["directional_discriminator"]) == 8
    assert spatial["stages"]
    return dict(
        process_exit=1,
        basis="complete spatial receipt precedes optional pre-solver import failure",
        preserved_failure_log=str(failed),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for frame, branch in ROWS:
        folder = args.input / "edge" / f"frame-{frame}-{branch}"
        receipt = json.loads((folder / "receipt.json").read_text())
        spatial = json.loads((folder / "spatial.json").read_text())
        completed = completion(args.input, folder, spatial)
        tables = json.loads((folder / "source-inputs.json").read_text())
        for table in tables.values():
            assert len(table["coordinate"]) == 29
            assert table["coordinate"][-1] == 1.0
            assert table["values"][-1] != 0
        assert len(spatial["trips"]) == receipt["counts"]["inner"] > 0
        assert spatial["terminal"]["support"] == receipt["support"] > 0
        iterations = [row for trip in spatial["trips"] for row in trip["iterations"]]
        assert iterations, "instrument must see executed Newton proposals"
        accepted = sum(bool(row["inner_iteration_accepted"]) for row in iterations)
        assert accepted == sum(trip["inner"]["accepted"] for trip in spatial["trips"])
        failed = [row for row in iterations if not row["inner_iteration_accepted"]]
        changed = set()
        for trip in spatial["trips"]:
            changed.update(trip["entering_indices"])
            changed.update(trip["leaving_indices"])
        result = dict(
            completion=completed,
            frame=frame,
            branch=branch,
            residual=receipt["residual"],
            converged=receipt["converged"],
            reason=receipt["terminal_reason"],
            expected_reason=EXPECTED_REASON[frame, branch],
            premise_reproduced=receipt["terminal_reason"]
            == EXPECTED_REASON[frame, branch],
            support=receipt["support"],
            trips=len(spatial["trips"]),
            accepted=accepted,
            refused=len(failed),
            changed_cells=len(changed),
            refusals={
                name: sum(row["decision_name"] == name for row in failed)
                for name in sorted({row["decision_name"] for row in failed})
            },
            krylov={
                name: sum(row["krylov_name"] == name for row in failed)
                for name in sorted({row["krylov_name"] for row in failed})
            },
            terminal_boundary=spatial["terminal"],
            directional_discriminator=spatial.get("directional_discriminator", []),
        )
        rows.append(result)
        print("ROW " + json.dumps(result), flush=True)
    control = next(
        row for row in rows if row["frame"] == 146 and row["branch"] == "diverted"
    )
    result = dict(
        rows=rows,
        all_premises_reproduced=all(row["premise_reproduced"] for row in rows),
        control_converged=control["converged"],
    )
    folder = args.input / "inactive-field-suppressed"
    assert (folder / "control.log").read_text().endswith("EXIT=0\n")
    counterfactual = json.loads((folder / "receipt.json").read_text())
    assert counterfactual["constraint_rows"] == 1
    assert abs(counterfactual["constraint_residual"][0]) < 1e-12
    original = args.input / "edge/frame-146-limited/source-inputs.json"
    original_tables = json.loads(original.read_text())
    paired_tables = json.loads((folder / "source-inputs.json").read_text())
    result["cross_device_profile_difference"] = {}
    for key in original_tables:
        values = np.asarray(original_tables[key]["values"])
        delta = values - paired_tables[key]["values"]
        result["cross_device_profile_difference"][key] = dict(
            maximum_absolute=float(np.max(np.abs(delta))),
            maximum_relative=float(
                np.max(np.abs(delta) / np.maximum(abs(values), 1e-30))
            ),
        )
    result["inactive_field_counterfactual"] = {
        key: counterfactual[key]
        for key in (
            "residual",
            "converged",
            "terminal_reason",
            "support",
            "axis_rz_m",
            "constraint_residual",
        )
    }
    reference_folder = args.input / "inactive-field-reference"
    assert (reference_folder / "reference.log").read_text().endswith("EXIT=0\n")
    reference = json.loads((reference_folder / "receipt.json").read_text())
    assert (
        json.loads((reference_folder / "source-inputs.json").read_text())
        == paired_tables
    )
    reference_seed = np.load(reference_folder / "request-seed.npz")["state"]
    counterfactual_seed = np.load(folder / "request-seed.npz")["state"]
    np.testing.assert_allclose(
        reference_seed, counterfactual_seed, rtol=2e-14, atol=2e-14
    )
    result["paired_reference"] = {
        key: reference[key] for key in result["inactive_field_counterfactual"]
    }
    result["paired_seed_difference_sup_wb"] = float(
        np.max(np.abs(reference_seed - counterfactual_seed))
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert np.isfinite([row["residual"] for row in rows]).all()
    failed_tails = sum(row["completion"]["process_exit"] != 0 for row in rows)
    print(
        f"VERIFIED {len(rows)} spatial receipts; "
        f"failure premise reproduced={result['all_premises_reproduced']}; "
        f"control converged={result['control_converged']}; "
        f"completed traces with failed optional tail={failed_tails}"
    )


if __name__ == "__main__":
    main()
