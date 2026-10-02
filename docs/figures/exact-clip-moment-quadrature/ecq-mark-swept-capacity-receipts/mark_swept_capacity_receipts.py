"""Mark the swept exact_support_capacity in the clip and scan receipts as historical.

The traced support capacity derives from the realised layout through
traced_polygon_vertex_capacity: spline_chain_samples_per_chord plus
atomic_support_capacity, one traced arc plus the straight edge chain. Before
that derivation landed (revision f7fefbd780aae7ef3d9707722fc4d80a1f0742cd) the
bound was the swept product atomic_support_capacity times
spline_chain_samples_per_chord, 128 samples reserved for every straight slot.

This driver adds a dimensions_provenance block beside every recorded dimensions
mapping in the targeted receipts and splices the same marker into the fenced
receipt quoted in the low-state-amplitude report. Every recorded numeric value
is left byte-identical; only provenance keys are added.

Run from the login node with the shared environment:

    PYTHONPATH="$PWD" ~/Code/nova/.venv/bin/python \
        docs/figures/exact-clip-moment-quadrature/ecq-mark-swept-capacity-receipts/mark_swept_capacity_receipts.py
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from nova.equilibrium.separatrix_clip import traced_polygon_vertex_capacity

REPO = Path(__file__).resolve().parents[4]
FIGURE = REPO / "docs" / "figures"
NODE = FIGURE / "exact-clip-moment-quadrature" / "ecq-mark-swept-capacity-receipts"

RETIRED_BY = "f7fefbd780aae7ef3d9707722fc4d80a1f0742cd"
SWEPT = (
    "atomic_support_capacity * spline_chain_samples_per_chord, 128 times every "
    "straight slot"
)
CURRENT = (
    "spline_chain_samples_per_chord + atomic_support_capacity, one traced arc "
    "plus the straight edge chain"
)
TARGET_VALUES = {3072, 3840, 2816}
CAP_RE = re.compile(r'"exact_support_capacity"\s*:\s*(-?\d+)')

JSON_TARGETS = [
    FIGURE / "exact-clip-capacity" / "memory-base.json",
    FIGURE / "exact-clip-capacity" / "memory-current.json",
] + sorted(
    (FIGURE / "millisecond-converged-solve" / "scan-prototype" / "parts").glob("*.json")
)

REPORT_MD = FIGURE / "figure-and-solver-audit" / "low-state-amplitude-nan" / "report.md"


def provenance(dimensions: dict) -> dict:
    straight = int(dimensions["atomic_support_capacity"])
    chain = int(dimensions["spline_chain_samples_per_chord"])
    recorded = int(dimensions["exact_support_capacity"])
    block = {
        "exact_support_capacity": {
            "status": "historical",
            "recorded_value": recorded,
            "superseded_expression": SWEPT,
            "retired_by_revision": RETIRED_BY,
            "current_expression": CURRENT,
            "current_derived_value_for_this_layout": traced_polygon_vertex_capacity(
                straight
            ),
        },
        "measurements": (
            "compile wall, solve, stages and allocator fields record this run as "
            "measured and are unchanged by the capacity retirement"
        ),
        "chain_samples_reserved_for_every_slot": chain,
    }
    if "exact_quadrature_points_per_cell" in dimensions:
        block["exact_quadrature_points_per_cell"] = {
            "status": "historical",
            "recorded_value": int(dimensions["exact_quadrature_points_per_cell"]),
            "superseded_expression": (
                "(exact_support_capacity - 2) * quadrature_nodes_per_triangle at "
                "the swept capacity {0}".format(recorded)
            ),
            "retired_by_revision": RETIRED_BY,
            "current_expression": (
                "the exact route's closed-form polygon moments evaluate no "
                "per-cell quadrature points"
            ),
        }
    return block


def annotate(obj, paths, prefix=""):
    if isinstance(obj, dict):
        if isinstance(obj.get("dimensions"), dict):
            out = {}
            for key, value in obj.items():
                if key == "dimensions":
                    out[key] = value
                    if "dimensions_provenance" in obj:
                        out["dimensions_provenance"] = obj["dimensions_provenance"]
                    else:
                        out["dimensions_provenance"] = provenance(value)
                    paths.append(prefix + "/dimensions_provenance")
                elif key == "dimensions_provenance":
                    continue
                else:
                    out[key] = annotate(value, paths, prefix + "/" + key)
            return out
        return {k: annotate(v, paths, prefix + "/" + k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [
            annotate(v, paths, prefix + "[{0}]".format(i)) for i, v in enumerate(obj)
        ]
    return obj


def numeric_paths(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_paths(v, prefix + "/" + k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(numeric_paths(v, prefix + "[{0}]".format(i)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


def annotate_json(path):
    before = json.loads(path.read_text())
    nums_before = numeric_paths(before)
    paths = []
    after = annotate(before, paths)
    path.write_text(json.dumps(after, indent=2) + "\n")
    nums_after = numeric_paths(after)
    changed = {
        key: [nums_before[key], nums_after[key]]
        for key in nums_before
        if nums_after.get(key) != nums_before[key]
    }
    return {
        "file": str(path.relative_to(REPO)),
        "provenance_paths": paths,
        "numeric_fields_before": len(nums_before),
        "numeric_changed": changed,
    }


def annotate_report():
    text = REPORT_MD.read_text()
    if '"dimensions_provenance"' in text:
        return {"file": str(REPORT_MD.relative_to(REPO)), "status": "already-marked"}
    snippet = (
        '"atomic_support_capacity": 24,\n'
        '"cells": 342,\n'
        '"cut_cell_bank_capacity": 342,\n'
        '"exact_support_capacity": 3072,\n'
    )
    if snippet not in text:
        raise SystemExit("report.md fenced block not found verbatim")
    block = provenance(
        {
            "atomic_support_capacity": 24,
            "spline_chain_samples_per_chord": 128,
            "exact_support_capacity": 3072,
            "exact_quadrature_points_per_cell": 196480,
        }
    )
    encoded = json.dumps(block, indent=2)
    inner = encoded.split("\n")[1:-1]
    marker = '"dimensions_provenance": {\n' + "\n".join(inner) + "\n}"
    REPORT_MD.write_text(text.replace(snippet, snippet + marker + ",\n"))
    return {"file": str(REPORT_MD.relative_to(REPO)), "status": "marked"}


def swept_files():
    names = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "docs/figures"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hits = {}
    for name in names:
        content = (REPO / name).read_text(errors="replace")
        values = {int(v) for v in CAP_RE.findall(content)}
        if values & TARGET_VALUES:
            hits[name] = {
                "values": sorted(values & TARGET_VALUES),
                "marked": '"dimensions_provenance"' in content,
            }
    return hits


def main():
    receipt = {"json": [], "report": None}
    for path in JSON_TARGETS:
        if path.exists():
            receipt["json"].append(annotate_json(path))
    receipt["report"] = annotate_report()
    hits = swept_files()
    receipt["sweep"] = {
        "target_value_files": len(hits),
        "marked": sorted(n for n, v in hits.items() if v["marked"]),
        "unmarked": sorted(n for n, v in hits.items() if not v["marked"]),
    }
    out = NODE / "annotation-receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("WROTE", out.relative_to(REPO))
    for entry in receipt["json"]:
        print("ANNOTATED", len(entry["provenance_paths"]), entry["file"])
    print("REPORT", receipt["report"])
    print("SWEEP target files:", receipt["sweep"]["target_value_files"])
    print("SWEEP unmarked:")
    for name in receipt["sweep"]["unmarked"]:
        print("   ", name)


if __name__ == "__main__":
    main()
