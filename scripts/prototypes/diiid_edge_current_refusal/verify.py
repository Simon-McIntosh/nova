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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for frame, branch in ROWS:
        folder = args.input / "edge" / f"frame-{frame}-{branch}"
        assert (folder / "row.log").read_text().endswith("EXIT=0\n")
        receipt = json.loads((folder / "receipt.json").read_text())
        spatial = json.loads((folder / "spatial.json").read_text())
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
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert np.isfinite([row["residual"] for row in rows]).all()
    print(
        f"VERIFIED {len(rows)} spatial receipts; "
        f"failure premise reproduced={result['all_premises_reproduced']}; "
        f"control converged={result['control_converged']}"
    )


if __name__ == "__main__":
    main()
