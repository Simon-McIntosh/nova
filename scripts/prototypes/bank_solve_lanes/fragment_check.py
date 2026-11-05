"""Check the record's two-step claim and its quoted step vectors against receipts.

Two statements in the record are claims about numbers held elsewhere, and the
prose cannot reveal when they drift from their source: the level every lane
reaches after two Newton steps, which the per-lane table and
``lanes-paths.json`` both hold, and the step vectors quoted from the runs.

So this reads the receipt and the fragment and requires them to agree. Each
quoted sixteen-element step vector must also appear verbatim in a retained log,
because a vector spliced from two jobs cannot be reproduced from any file.

Exit status is nonzero on any disagreement, so the check can be pointed at a
candidate record and will fail on one that overstates the measurements.
"""

import argparse
import json
import re
from pathlib import Path

VECTOR = re.compile(r"\[(?:\s*-?\d+\s*,){15}\s*-?\d+\s*\]")


def two_step_split(payload):
    """How the lanes' residuals fall at iterate 2: high level, low level."""
    residual = payload["paths"]["residual"]
    lane_residuals = [row[2] for row in residual]
    high = max(lane_residuals)
    low = min(lane_residuals)
    assert high / low > 10
    at_high = sum(1 for value in lane_residuals if value > (high * low) ** 0.5)
    return high, low, at_high, len(lane_residuals) - at_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fragment", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--log", type=Path, action="append", default=[])
    args = parser.parse_args()
    text = args.fragment.read_text()
    payload = json.loads(args.receipt.read_text())
    high, low, at_high, at_low = two_step_split(payload)
    print(f"HIGH {high:.3e} LOW {low:.3e} LANES_AT_HIGH {at_high} AT_LOW {at_low}")
    claim = f"nine of them exactly at that level and seven already between"
    if claim not in text:
        raise SystemExit("the record does not state the two-level split at iterate 2")
    if (at_high, at_low) != (9, 7):
        raise SystemExit(f"the receipt splits {at_high}/{at_low} at iterate 2, not 9/7")
    logs = [path.read_text() for path in args.log]
    missing = [
        vector
        for vector in VECTOR.findall(text)
        if not any(vector.replace(" ", "") in log.replace(" ", "") for log in logs)
    ]
    if missing:
        raise SystemExit(f"step vectors not found in any retained log: {missing}")
    print(f"VECTORS {len(VECTOR.findall(text))} all present in a retained log")


if __name__ == "__main__":
    main()
