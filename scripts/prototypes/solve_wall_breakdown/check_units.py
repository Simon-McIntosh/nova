"""Check that the fragment's millisecond prose agrees with the job receipt.

The receipt records seconds. The fragment states milliseconds. A factor-of-1000
slip between the two reads as a passing table and a wrong sentence, so every
`N ms` figure in the fragment must match a receipt value converted to
milliseconds, and every program's warm wall must appear.

Exit status 1 with the offending values listed means the prose and the receipt disagree.
"""

import argparse
import json
from pathlib import Path
import re
import sys


PROGRAM_KEYS = {
    "topology_read": ["topology_read_ms", "topology_read"],
    "booking": ["booking_ms", "booking"],
    "map": ["map_ms", "map"],
}


def receipt_milliseconds(receipt):
    """Return every millisecond value the receipt states for the warm step."""
    values = {}
    for program in receipt.get("programs", []):
        if program["program"] in PROGRAM_KEYS and "warm_median_seconds" in program:
            values[program["program"]] = 1000.0 * program["warm_median_seconds"]
    for key, row in (receipt.get("steps") or {}).items():
        if not isinstance(row, dict):
            continue
        for name, value in row.items():
            if name.endswith("_ms") and isinstance(value, (int, float)):
                values[f"batch{key}:{name}"] = float(value)
    return values


def fragment_values(text):
    """Return the numbers the fragment states in milliseconds, in order."""
    return [(float(match.group(1)), match.group(0)) for match in re.finditer(
        r"([0-9]+(?:\.[0-9]+)?)\s?ms\b", text
    )]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    values = receipt_milliseconds(receipt)
    if not values:
        print("receipt states no warm wall; nothing to check against", file=sys.stderr)
        return 2

    text = args.fragment.read_text()
    stated = fragment_values(text)
    allowed = [1000.0, 1.0]
    problems = []
    matched = set()
    for value, raw in stated:
        candidates = list(values.values()) + allowed
        close = [name for name, expected in values.items()
                 if abs(value - expected) <= max(0.06, 0.01 * expected)]
        if close:
            matched.update(close)
            continue
        if any(abs(value - target) <= 1e-9 for target in allowed):
            continue
        problems.append((raw, sorted(set(values.values()))))

    for name, expected in values.items():
        if name in matched:
            continue
        if name.startswith("batch"):
            continue
        problems.append((f"{name} = {expected:.1f} ms is stated nowhere", []))

    if problems:
        print("UNIT CHECK FAILED", file=sys.stderr)
        for raw, expected in problems:
            print(f"  fragment states {raw}; receipt offers {expected}", file=sys.stderr)
        return 1
    print(f"UNIT CHECK OK: {len(stated)} stated values agree with the receipt")
    print(json.dumps({name: round(value, 3) for name, value in values.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())