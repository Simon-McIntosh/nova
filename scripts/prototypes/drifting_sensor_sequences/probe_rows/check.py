"""Assert the two clauses this node is measured on, from the reduced summary.

Clause one: for each code and each noise sequence the arm carries at least the
two hundred slices it is stated over.  Clause two: the control holds -- the
white draw's lag-one correlation sits within the analysis module's limit for
both codes, and the 1/f sequence's lag-one sits above its floor.  Every number
comes from ``analyse.py``; this module only refuses a record whose own tables
do not meet the clauses, and prints the rows it judged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analyse  # noqa: E402

SLICES = 200
STREAMS = ("axis", "flux", "q95")


def row(code: str, sequence: str, arm: dict) -> str:
    return (
        "%s %-5s slices=%d boundary=%d axis_rms=%.4f flux_rms=%.4f q95_rms=%.4f "
        "lag1=(%.3f, %.3f, %.3f)"
        % (
            code,
            sequence,
            arm["slices"],
            arm["slices_with_boundary"],
            arm["axis_rms"],
            arm["flux_rms"],
            arm["q95_rms"],
            arm["axis_lag1"],
            arm["flux_lag1"],
            arm["q95_lag1"],
        )
    )


def check(payload: dict) -> tuple[list[str], int, list[str]]:
    failures = []
    performed = 0
    lines = []
    for code in ("nova", "efit"):
        streams = payload["control"].get(code, {})
        for sequence in ("white", "drift"):
            arm = streams.get(sequence)
            performed += 1
            if arm is None:
                failures.append("%s: no %s arm was reduced" % (code, sequence))
                continue
            lines.append(row(code, sequence, arm))
            performed += 1
            if arm["slices"] < SLICES:
                failures.append(
                    "%s %s: %d slices is short of %d"
                    % (code, sequence, arm["slices"], SLICES)
                )
            performed += 1
            if arm["slices_with_boundary"] < SLICES // 2:
                failures.append(
                    "%s %s: a boundary on only %d slices"
                    % (code, sequence, arm["slices_with_boundary"])
                )
        performed += 1
        if streams.get("white_lag1_pass") is not True:
            failures.append("%s: the white lag-one control failed" % code)
        performed += 1
        if streams.get("drift_lag1_pass") is not True:
            failures.append("%s: the 1/f lag-one clause failed" % code)
    for code in ("nova", "efit"):
        streams = payload["control"].get(code, {})
        for sequence in ("white", "drift"):
            arm = streams.get(sequence)
            if arm is None:
                continue
            for stream in STREAMS:
                value = arm[stream + "_lag1"]
                performed += 1
                if not abs(value) <= analyse.LAG_ONE_LIMIT and sequence == "white":
                    failures.append(
                        "%s %s %s lag-one %.3f outside +/-%.2f"
                        % (code, sequence, stream, value, analyse.LAG_ONE_LIMIT)
                    )
    return failures, performed, lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(analyse.STUDY / "summary.json"))
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.summary).read_text())
    failures, performed, lines = check(payload)
    for line in lines:
        print(line)
    for note in failures:
        print("CHECK-FAILED " + note)
    if failures:
        print("%d of %d checks failed" % (len(failures), performed))
        return 1
    print("CHECKS-PASSED %d" % performed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())