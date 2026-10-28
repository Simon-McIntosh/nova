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
    return [(float(match.group(1).replace(",", "")), match.group(0)) for match in re.finditer(
        r"([0-9][0-9,]*(?:\.[0-9]+)?)\s?ms\b", text
    )]


def precision_tolerance(raw):
    """Half a unit in the last place the fragment states.

    A fixed absolute tolerance cannot catch a seconds-for-milliseconds slip:
    0.7357 s is numerically the millisecond value over 1000, and a 0.06 ms
    allowance would let it match an unrelated study constant. Comparing at the
    precision actually written makes the two distinguishable.
    """
    numeric = re.match(r"([0-9][0-9,]*(?:\.[0-9]+)?)", raw)
    clean = numeric.group(1).replace(",", "") if numeric else raw
    decimals = len(clean.split(".")[1]) if "." in clean else 0
    return 0.5 * 10.0 ** (-decimals) + 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cold-receipt", type=Path)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    values = receipt_milliseconds(receipt)
    if not values:
        print("receipt states no warm wall; nothing to check against", file=sys.stderr)
        return 2
    cold = {}
    if args.cold_receipt:
        cold = receipt_milliseconds(json.loads(args.cold_receipt.read_text()))
        cold = {f"cold:{name}": value for name, value in cold.items() if ":" not in name}

    text = args.fragment.read_text()
    stated = fragment_values(text)
    derived = {}
    if all(f"cold:{name}" in cold for name in ("topology_read", "booking", "map")):
        derived["three programs only"] = sum(cold[f"cold:{name}"] for name in ("topology_read", "booking", "map"))
    if "batch1:map_ms" in values and "batch1:topology_read_ms" in values:
        derived["map less read"] = values["batch1:map_ms"] - values["batch1:topology_read_ms"]
    if "batch1:jacobian_block_ms" in values:
        step_total = sum(value for name, value in values.items()
                         if name.startswith("batch1:") and not name.endswith("_p95_ms"))
        derived["step less jacobian"] = step_total - values["batch1:jacobian_block_ms"]
        if "batch1:map_ms" in values:
            derived["step less map"] = step_total - values["batch1:map_ms"]
        derived["step total"] = step_total
    external = [
        1000.0,
        1.0,
        6.120,   # the 282-cell lean-map study's connected read
        0.743,   # the study's two field-map contractions
        10.176,  # the study's fused map
        406.1,   # the study's isolated device contour tree
    ]
    allowed = external + list(derived.values())
    pool = dict(values)
    pool.update(cold)
    problems = []
    matched = set()
    for value, raw in stated:
        tolerance = precision_tolerance(raw)
        close = [name for name, expected in pool.items()
                 if abs(value - expected) <= tolerance]
        if close:
            matched.update(close)
            continue
        if any(abs(value - target) <= tolerance for target in allowed):
            continue
        problems.append((raw, sorted(set(values.values()))))

    for name, expected in values.items():
        if name in matched or not name.startswith("batch1:") or name.endswith("_p95_ms"):
            continue
        problems.append((f"{name} = {expected:.1f} ms is stated nowhere", []))

    if problems:
        print("UNIT CHECK FAILED", file=sys.stderr)
        for raw, expected in problems:
            print(f"  fragment states {raw}; receipt offers {expected}", file=sys.stderr)
        return 1
    print(f"UNIT CHECK OK: {len(stated)} stated values agree with the receipt")
    print(json.dumps({name: round(value, 3) for name, value in values.items()}, indent=2))
    print(json.dumps({name: round(value, 3) for name, value in derived.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())