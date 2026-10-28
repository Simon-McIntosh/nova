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
    return [
        (float(match.group(1).replace(",", "")), match.group(0))
        for match in re.finditer(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s?ms\b", text)
    ]


def fragment_seconds(text):
    """Return the numbers the fragment states in seconds, in order."""
    return [
        (float(match.group(1).replace(",", "")), match.group(0))
        for match in re.finditer(r"(?<!m)\b([0-9][0-9,]*(?:\.[0-9]+)?)\s?s\b", text)
    ]


def cold_seconds(receipt):
    """Every second-valued stage the cold receipt records, and their sums.

    The cold table's own rows and the summary sentence must both come from
    these, so a decomposition that does not add up cannot be stated.
    """
    stages = (
        "trace_seconds",
        "lower_seconds",
        "xla_compile_seconds",
        "first_execute_seconds",
    )
    values = {}
    for program in receipt.get("programs", []):
        for stage in stages:
            if stage in program:
                values[f"{program['program']}.{stage}"] = float(program[stage])
    for stage in stages:
        rows = [value for name, value in values.items() if name.endswith("." + stage)]
        if rows:
            values[f"sum {stage}"] = sum(rows)
    totals = [values[name] for name in values if name.startswith("sum ")]
    if totals:
        values["sum every stage"] = sum(totals)
    compile_totals = [values.get(f"sum {stage}") for stage in stages[:3]]
    if all(value is not None for value in compile_totals):
        values["sum trace lowering compile"] = sum(compile_totals)
    for name in ("carrier_load_seconds", "operand_load_seconds"):
        if name in receipt:
            values[name] = float(receipt[name])
    jets = receipt.get("jets") or {}
    for name in ("load_seconds", "stored_build_seconds"):
        if jets.get(name):
            values[f"jets {name}"] = float(jets[name])
    return values


def headline_cold_total(text, expected):
    """Check the plan-index headline states the cold total, not one of its parts.

    The meta summary is what a plan index renders, so a decomposition corrected
    in the body but not here keeps the pre-correction number in front of the
    reader. The figure named as the cold compile is the sum of every cold stage.
    """
    match = re.search(r'<meta name="plan-summary" content="([^"]*)"', text)
    if not match:
        return None
    stated = re.search(r"compile[^.]*?is ([0-9][0-9,]*(?:\.[0-9]+)?) s", match.group(1))
    if not stated:
        return None
    value = float(stated.group(1).replace(",", ""))
    if abs(value - expected) > precision_tolerance(stated.group(1)):
        return f"plan-summary states a cold compile of {stated.group(1)} s; the receipt's cold stages sum to {expected:.3f} s"
    return None


def containment_claims(text):
    """Sentences asserting the map contains the read stage, which the code contradicts.

    map_moments calls kernel.read over roots it solves itself, while the step's
    read stage runs nova.equilibrium.topology.read; they are different programs
    over different inputs, so the map's wall cannot contain the read's.
    """
    patterns = [
        r"map contains this read",
        r"carries the read inside it",
        r"contains the read",
        r"the read inside the map",
    ]
    return [pattern for pattern in patterns if re.search(pattern, text)]


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
    parser.add_argument("--trace-receipt", type=Path)
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
        cold = {
            f"cold:{name}": value for name, value in cold.items() if ":" not in name
        }

    text = args.fragment.read_text()
    stated = fragment_values(text)
    derived = {}
    if all(f"cold:{name}" in cold for name in ("topology_read", "booking", "map")):
        derived["three programs only"] = sum(
            cold[f"cold:{name}"] for name in ("topology_read", "booking", "map")
        )
    if args.trace_receipt:
        traces = json.loads(args.trace_receipt.read_text())
        for label, record in traces.items():
            repetitions = 20.0
            derived[f"{label} device"] = record["device_us_total"] / repetitions / 1000
            derived[f"{label} memcpy"] = record["memcpy_us_total"] / repetitions / 1000
            tree = sum(
                value
                for name, value in record["top_kernels"]
                if "build_contour_tree" in name
            )
            derived[f"{label} tree"] = tree / repetitions / 1000
            if label == "map" and "batch1:map_ms" in values:
                derived["map less its own tree"] = (
                    values["batch1:map_ms"] - tree / repetitions / 1000
                )
                step_total = sum(
                    value
                    for name, value in values.items()
                    if name.startswith("batch1:") and not name.endswith("_p95_ms")
                )
                derived["step less the map's own tree"] = (
                    step_total - tree / repetitions / 1000
                )
            ranks = [
                value
                for name, value in record["top_kernels"]
                if "build_contour_tree" in name
            ]
            if ranks:
                derived[f"{label} largest tree rank"] = max(ranks) / repetitions / 1000
            derived[f"{label} launch issue"] = record["kernels"] / repetitions * 0.0015
    if "batch1:map_ms" in values and "batch1:topology_read_ms" in values:
        derived["map less read"] = (
            values["batch1:map_ms"] - values["batch1:topology_read_ms"]
        )
    if "batch1:jacobian_block_ms" in values:
        step_total = sum(
            value
            for name, value in values.items()
            if name.startswith("batch1:") and not name.endswith("_p95_ms")
        )
        derived["step less jacobian"] = step_total - values["batch1:jacobian_block_ms"]
        if "batch1:map_ms" in values:
            derived["step less map"] = step_total - values["batch1:map_ms"]
        derived["step total"] = step_total
    external = [
        1000.0,
        1.0,
        6.120,  # the 282-cell lean-map study's connected read
        0.743,  # the study's two field-map contractions
        10.176,  # the study's fused map
        406.1,  # the study's isolated device contour tree
    ]
    allowed = external + list(derived.values())
    pool = dict(values)
    pool.update(cold)
    problems = []
    matched = set()
    for value, raw in stated:
        tolerance = precision_tolerance(raw)
        close = [
            name
            for name, expected in pool.items()
            if abs(value - expected) <= tolerance
        ]
        if close:
            matched.update(close)
            continue
        if any(abs(value - target) <= tolerance for target in allowed):
            continue
        problems.append((raw, sorted(set(values.values()))))

    for name, expected in values.items():
        if (
            name in matched
            or not name.startswith("batch1:")
            or name.endswith("_p95_ms")
        ):
            continue
        problems.append((f"{name} = {expected:.1f} ms is stated nowhere", []))

    seconds = {}
    if args.cold_receipt:
        seconds = cold_seconds(json.loads(args.cold_receipt.read_text()))
    seconds.update(
        {
            f"step arm {name}": value
            for name, value in cold_seconds(receipt).items()
            if name.startswith(("operand_load", "carrier_load", "jets "))
        }
    )
    for value, raw in fragment_seconds(text):
        tolerance = precision_tolerance(raw)
        if any(abs(value - expected) <= tolerance for expected in seconds.values()):
            continue
        problems.append((raw, sorted(round(v, 3) for v in seconds.values())))

    if args.cold_receipt and "sum every stage" in seconds:
        headline = headline_cold_total(text, seconds["sum every stage"])
        if headline:
            problems.append((headline, []))

    for pattern in containment_claims(text):
        problems.append(
            (
                f"fragment asserts containment ({pattern}), contradicted by map_moments "
                "calling kernel.read rather than the measured topology.read",
                [],
            )
        )

    if problems:
        print("UNIT CHECK FAILED", file=sys.stderr)
        for raw, expected in problems:
            print(
                f"  fragment states {raw}; receipt offers {expected}", file=sys.stderr
            )
        return 1
    print(f"UNIT CHECK OK: {len(stated)} stated values agree with the receipt")
    print(
        json.dumps({name: round(value, 3) for name, value in values.items()}, indent=2)
    )
    print(
        json.dumps({name: round(value, 3) for name, value in derived.items()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
