"""Check a drifting-sensor record's arm table against the arms' summaries.

Each row of the fragment's arm table must carry, per sequence, the lag-one
triplet its own arm's ``summary.json`` holds — axis, boundary flux over the
generating state's fixed span, and q95 — and the nova paragraph's imas-efit
numbers must be the axis RMSE recomputed on the slice indices nova returned.
The convention is the one :mod:`analyse` applies; a constant divisor leaves a
correlation unchanged, so the fixed-span fraction and the raw boundary flux in
Wb give the same figure and only one of them may be quoted.

Run as a script over one fragment:

    python lag_table.py <fragment.html> [<runs directory>]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

GENERATING_AXIS = np.array([6.2886334, 0.4982048])
CURRENT_RUN = "r-20261119T065220776269-drifting-sensor-probe-rows"
PRIOR_RUN = "r-20261115T202148674467-drifting-sensor-sequences"

# (row label in the table, sequence, which run's summary, key in that summary)
ROWS = (
    (
        "24 loops + 48 poloidal-probe rows, IMAS convention",
        "white",
        "current",
        "efit-white-200-0.0100",
    ),
    (
        "24 loops + 48 poloidal-probe rows, IMAS convention",
        "drift",
        "current",
        "efit-drift-200-0.0100",
    ),
    ("24 loops alone", "white", "prior", "efit-white-200-0.0100-loops"),
    ("24 loops alone", "drift", "prior", "efit-drift-200-0.0100-loops"),
    ("24 loops + 48 probe rows, prior head", "white", "prior", "efit-white-200-0.0100"),
    ("24 loops + 48 probe rows, prior head", "drift", "prior", "efit-drift-200-0.0100"),
)


def triplet(entry: dict, key: str) -> list[float]:
    """Return the lag-one triplet a summary records for one arm."""

    return [
        float(entry[key]["axis_lag1"]),
        float(entry[key]["flux_lag1"]),
        float(entry[key]["q95_lag1"]),
    ]


def matched_rmse(sequence: str, runs: Path, indices=(0, 2)) -> str:
    """Return imas-efit's axis RMSE on the slices nova returned."""

    study = runs / CURRENT_RUN / "study"
    rows = {
        json.loads(line)["slice"]: json.loads(line)
        for line in open(study / ("efit-%s-200-0.0100.jsonl" % sequence))
        if line.strip()
    }
    errors = np.array(
        [
            np.linalg.norm(np.array(rows[index]["axis"]) - GENERATING_AXIS)
            for index in indices
        ]
    )
    return "%.5f" % float(np.sqrt(np.mean(errors**2)))


def failures(fragment: Path, runs: Path) -> list[str]:
    """Return one line per table cell the arms' summaries do not support."""

    text = fragment.read_text()
    current = json.loads((runs / CURRENT_RUN / "study/summary.json").read_text())
    prior = json.loads((runs / PRIOR_RUN / "study/summary.json").read_text())
    summaries = {"current": current["reference"], "prior": prior["reference"]}
    found = []
    for label, sequence, source, key in ROWS:
        pattern = re.compile(
            r"<tr><td>%s</td><td>%s</td>.*?<td>([-0-9., ]+)</td></tr>"
            % (re.escape(label), sequence)
        )
        match = pattern.search(text)
        if match is None:
            found.append("%s/%s row not found" % (label, sequence))
            continue
        written = [float(value) for value in match.group(1).split(",")]
        expected = triplet(summaries[source], key)
        if len(written) != 3 or any(
            abs(a - b) > 5.1e-4 for a, b in zip(written, expected)
        ):
            found.append(
                "%s/%s carries %s, its summary records [%.4f, %.4f, %.4f]"
                % (label, sequence, match.group(1).strip(), *expected)
            )
    for sequence in ("white", "drift"):
        value = matched_rmse(sequence, runs)
        if value not in text:
            found.append("slice-matched %s RMSE %s absent" % (sequence, value))
    return found


def main(argv: list[str]) -> int:
    fragment = Path(argv[1])
    runs = Path(argv[2]) if len(argv) > 2 else Path.home() / ".config/reckon/crew/runs"
    found = failures(fragment, runs)
    for line in found:
        print("FAIL " + line)
    print("checked %d rows in %s" % (len(ROWS), fragment))
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
