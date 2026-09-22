"""Refuse publication unless every declared numerical check is satisfied."""
# ruff: noqa: E501 -- Captions and persisted HTML retain their literal text.

import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / "acceptance.json").read_text())
failures = []
for row in data["rows"]:
    if not row.get("completed"):
        failures.append(
            f"{row['requested_cells']} requested cells: terminal receipt incomplete"
        )
        continue
    if not row["residual_pass"] or not row["converged"]:
        failures.append(
            f"{row['realised_cells']} cells: residual {row['terminal_residual']:.14g} exceeds 1e-12; converged={row['converged']}"
        )
    if not row["position_pass"]:
        failures.append(
            f"{row['realised_cells']} cells: axis outside one lattice pitch"
        )
    if not row["outside_fraction_pass"]:
        failures.append(
            f"{row['realised_cells']} cells: outside-centroid fraction {100 * row['outside_centroid_fraction']:.10g}% exceeds 4.9%"
        )
if not data["complete_module_delta"]:
    failures.append("The complete imported-module baseline/after delta is unverified")
failures.extend(data["added_failures"])
for failure in failures:
    print("REFUSED: " + failure)
print(json.dumps({"accepted": not failures, "failures": failures}, indent=2))
raise SystemExit(bool(failures))
