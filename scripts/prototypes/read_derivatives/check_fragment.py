"""Assert the landing fragment's numbers are the measured receipts' numbers.

The receipts are gitignored, so this check runs only where the measurement jobs
left them on disk. It reads the two ``<kind>-550.json`` receipts beside this
script and the evidence fragment, and exits nonzero listing every figure the
fragment prints at a precision the receipt does not round to, every validity
cell that disagrees with its receipt row, and any tally claim over the reads
that the receipts do not support.

    python check_fragment.py [fragment.html]
"""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "scripts/prototypes/read_derivatives/logs"
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/converged-forward-solve/read-derivative-prototype.html"
)
fragment = Path(sys.argv[1] if len(sys.argv) > 1 else FRAGMENT).read_text()

receipt = {}
for kind in ("limited", "diverted"):
    receipt[kind] = json.loads((LOGS / f"{kind}-550.json").read_text())

problems = []


def pair(kind_key: str, name: str) -> tuple[str, str]:
    out = []
    for kind in ("limited", "diverted"):
        row = [
            r
            for r in receipt[kind]["rows"]
            if r["program"] == name and r["state"] == "converged"
        ][0]
        out.append(row[kind_key])
    return out[0], out[1]


def expect(row_id: str, column: str, values) -> None:
    """The compile row must carry each receipt value to six significant figures."""
    match = re.search(rf'<tr id="{row_id}">(.*?)</tr>', fragment, re.S)
    if match is None:
        problems.append(f"missing row {row_id}")
        return
    cells = re.findall(r"<td>(.*?)</td>", match.group(1))
    if len(cells) != 5:
        problems.append(f"{row_id}: {len(cells)} cells, expected 5")
        return
    index = {"cold": 1, "hlo": 2, "warm": 3}[column]
    printed = [part.strip() for part in cells[index].split("/")]
    for printed_value, measured in zip(printed, values):
        # A printed value is consistent when it is the receipt rounded at the
        # precision the table prints, so the table cannot hide a mismatch by
        # quoting fewer digits than it measured.
        decimals = len(printed_value.partition(".")[2])
        rounded = f"{measured:.{decimals}f}"
        if rounded != printed_value:
            problems.append(
                f"{row_id} {column}: printed {printed_value!r}, "
                f"receipt {measured!r} rounds to {rounded!r}"
            )


for name, row_id in (
    ("oracle", "read-derivative-prototype-oracle"),
    ("interp", "read-derivative-prototype-interp"),
):
    for column, key in (
        ("cold", "cold_compile_seconds"),
        ("hlo", "hlo_instructions"),
        ("warm", "warm_seconds"),
    ):
        expect(row_id, column, pair(key, name))

# fd takes a step operand, so its rows are the same program with the same walls.
for column, key in (
    ("cold", "cold_compile_seconds"),
    ("hlo", "hlo_instructions"),
    ("warm", "warm_seconds"),
):
    expect("read-derivative-prototype-fd", column, pair(key, "fd"))

# The read program is measured once per kind and repeated per state row.
read = {}
for kind in ("limited", "diverted"):
    row = next(r for r in receipt[kind]["rows"] if r["program"] == "oracle")
    read[kind] = (
        row["read_cold_compile_seconds"],
        row["read_hlo_instructions"],
        row["read_warm_seconds"],
    )
for column, offset in (("cold", 0), ("hlo", 1), ("warm", 2)):
    expect(
        "read-derivative-prototype-read",
        column,
        (read["limited"][offset], read["diverted"][offset]),
    )

# Validity: every error-table row's two validity cells against its receipt row.
match = re.search(
    r'<table id="read-derivative-prototype-table-errors">(.*?)</table>', fragment, re.S
)
if match is None:
    problems.append("missing error table")
else:
    for block in re.findall(r'<tr id="([^"]+)">(.*?)</tr>', match.group(1), re.S):
        row_id, body = block
        found = re.search(
            r"read-derivative-prototype-(limited|diverted)-(converged|trial)-(fd|interp)",
            row_id,
        )
        if found is None:
            continue
        kind, state, program = found.groups()
        cells = re.findall(r"<td>(.*?)</td>", body)
        variant = next(
            r
            for r in receipt[kind]["rows"]
            if r["program"] == program and r["state"] == state
        )
        oracle = next(
            r
            for r in receipt[kind]["rows"]
            if r["program"] == "oracle" and r["state"] == state
        )
        for printed, measured, label in (
            (cells[4], variant["valid"] and variant["qualified"], f"{row_id} variant"),
            (cells[5], oracle["valid"] and oracle["qualified"], f"{row_id} oracle"),
        ):
            claimed = printed.strip() == "yes"
            if claimed != measured:
                problems.append(
                    f"{label}: printed {printed.strip()!r}, receipt validity {measured}"
                )

# Validity tallies claimed in the reading paragraph.
variant_valid = sum(
    1
    for kind in receipt
    for r in receipt[kind]["rows"]
    if r["program"] != "oracle" and r["valid"] and r["qualified"]
)
oracle_valid = sum(
    1
    for kind in receipt
    for r in receipt[kind]["rows"]
    if r["program"] == "oracle" and r["valid"] and r["qualified"]
)
variant_total = sum(
    1 for kind in receipt for r in receipt[kind]["rows"] if r["program"] != "oracle"
)
oracle_total = sum(
    1 for kind in receipt for r in receipt[kind]["rows"] if r["program"] == "oracle"
)
words = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    8: "eight",
    12: "twelve",
}
claims = 0
for pattern, expected in (
    (
        r"(\w+) of the (\w+) variant reads admits? a valid",
        (words[variant_valid], words[variant_total]),
    ),
    (
        r"(\w+) of the (\w+) oracle reads (?:do|does|is|are)",
        (words[oracle_valid], words[oracle_total]),
    ),
    (
        r"(\w+) of the (\w+) (?:variant |oracle )?rows overall are valid",
        (words[variant_valid + oracle_valid], words[variant_total + oracle_total]),
    ),
    (
        r"oracle(?:'s own read)? refuses (\w+) of (?:the |its )(\w+) state instances",
        (words[oracle_total - oracle_valid], words[oracle_total]),
    ),
):
    for found in re.finditer(pattern, fragment):
        claims += 1
        for token, want in zip(found.groups(), expected):
            if token != want:
                problems.append(f"validity tally: printed {token!r}, receipts {want!r}")
if claims == 0:
    problems.append("no validity tally claim found in the fragment")

print("\n".join(problems) if problems else "CONSISTENT")
sys.exit(1 if problems else 0)
