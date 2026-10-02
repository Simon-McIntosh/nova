"""Annotate strict-exit-incidence.json with the figure's missing-member note.

Adds one derived, figure-scoped field recording what the paired-latency figure
discloses. It reads only ``completion.{declared_member_count,missing_members}``
and derives nothing the receipt did not already carry.
"""

from __future__ import annotations

import json
from pathlib import Path

RECEIPT = Path("docs/figures/millisecond-converged-solve/strict-exit-incidence.json")


def main() -> None:
    payload = json.loads(RECEIPT.read_text())
    completion = payload["completion"]
    declared = completion["declared_member_count"]
    missing = completion["missing_member_count"]
    identities = ", ".join(row["identity"] for row in completion["missing_members"])
    completion["figure_disclosure"] = (
        f"{missing} of {declared} declared members missing ({identities}); "
        "no strict-exit incidence value was persisted for any member, so the "
        "figure draws no incidence panel."
    )
    RECEIPT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("ANNOTATED", RECEIPT)
    print(completion["figure_disclosure"])


if __name__ == "__main__":
    main()
