"""Mark the swept exact_support_capacity in the cut-cell clip and memory
receipts as historical.

The traced support capacity derives from the realised layout through
traced_polygon_vertex_capacity: spline_chain_samples_per_chord plus
atomic_support_capacity, one traced arc plus the straight edge chain. Before
that derivation landed (revision f7fefbd780aae7ef3d9707722fc4d80a1f0742cd) the
bound was the swept product atomic_support_capacity times
spline_chain_samples_per_chord, 128 samples reserved for every straight slot.

Committed receipts under cut-cell-current-attribution/clip-quadrature and
exact-clip-memory carry a swept capacity in {2560, 3072, 3840, 2816}: four
layouts, one per atomic support capacity 20, 24, 30 and 22. Most record the
dimensions in a ``dimensions`` mapping and one in a ``dimension_context``
mapping, both beside the same atomic capacity, chord samples and companion
quadrature count. Each gains a ``dimensions_provenance`` block in the form the
predecessor node established, inserted by the same annotator it used, and the
block states the bound the current code derives for that layout:
148 = 128 + 20, 150 = 128 + 22, 152 = 128 + 24 and 158 = 128 + 30. The
committed receipts carry twenty-three blocks in total: nine for capacity 3072,
five for 2560, five for 3840 and four for 2816.

Commit sizes, each by git diff --shortstat rather than the receipt subset:
df8f2989b changed 17 files with 823 insertions; 4d3e05548 changed 1 file with
71 insertions.

This driver imports that annotator by path and reuses ``provenance``,
``annotate``, ``numeric_paths`` and ``swept_files`` unchanged; it only points
the enumeration at the two cut-cell directories and adds a context-mapping arm
for the one receipt that records its dimensions under a different key. Every
recorded numeric value is left byte-identical; only provenance keys are added.

Run from the login node with the shared environment:

    PYTHONPATH="$PWD" ~/Code/nova/.venv/bin/python \
        docs/figures/exact-clip-moment-quadrature/ecq-mark-cut-cell-capacity-receipts/mark_cut_cell_capacity_receipts.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIGURE = REPO / "docs" / "figures"
NODE = FIGURE / "exact-clip-moment-quadrature" / "ecq-mark-cut-cell-capacity-receipts"
REFERENCE = (
    FIGURE
    / "exact-clip-moment-quadrature"
    / "ecq-mark-swept-capacity-receipts"
    / "mark_swept_capacity_receipts.py"
)

TARGET_DIRS = [
    FIGURE / "cut-cell-current-attribution" / "clip-quadrature",
    FIGURE / "cut-cell-current-attribution" / "exact-clip-memory",
]

CONTEXT_KEY = "dimension_context"
CONTEXT_QUADRATURE_KEY = "quadrature_points_per_cell"

TARGET_VALUES = {2560, 3072, 3840, 2816}


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "mark_swept_capacity_receipts", REFERENCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_provenance(ref, context: dict) -> dict:
    """The same provenance block, read through the recording key names.

    The context mapping stores the chord samples as ``spline_chain_samples``
    and the companion count as ``quadrature_points_per_cell``; the canonical
    names are supplied so the block states the same values the annotator does
    for a ``dimensions`` mapping.
    """
    shaped = {
        "atomic_support_capacity": context["atomic_support_capacity"],
        "spline_chain_samples_per_chord": context["spline_chain_samples"],
        "exact_support_capacity": context["exact_support_capacity"],
    }
    block = ref.provenance(shaped)
    if CONTEXT_QUADRATURE_KEY in context:
        block["quadrature_points_per_cell"] = {
            "status": "historical",
            "recorded_value": int(context[CONTEXT_QUADRATURE_KEY]),
            "superseded_expression": (
                "(exact_support_capacity - 2) * quadrature_nodes_per_triangle "
                "at the swept capacity {0}".format(
                    int(context["exact_support_capacity"])
                )
            ),
            "retired_by_revision": ref.RETIRED_BY,
            "current_expression": (
                "the exact route's closed-form polygon moments evaluate no "
                "per-cell quadrature points"
            ),
        }
    return block


def annotate_context(ref, obj, paths, prefix=""):
    """Insert a provenance block beside each ``dimension_context`` mapping.

    Mirrors ``ref.annotate`` for the context key: the block is written once,
    immediately after the mapping, and every pre-existing value is untouched.
    """
    if isinstance(obj, str) or not isinstance(obj, (dict, list)):
        return obj
    if isinstance(obj, list):
        return [
            annotate_context(ref, v, paths, prefix + "[{0}]".format(i))
            for i, v in enumerate(obj)
        ]
    if isinstance(obj.get(CONTEXT_KEY), dict):
        out = {}
        for key, value in obj.items():
            if key == CONTEXT_KEY:
                out[key] = value
                if "dimensions_provenance" in obj:
                    out["dimensions_provenance"] = obj["dimensions_provenance"]
                else:
                    out["dimensions_provenance"] = context_provenance(ref, value)
                paths.append(prefix + "/dimensions_provenance")
            elif key == "dimensions_provenance":
                continue
            else:
                out[key] = annotate_context(ref, value, paths, prefix + "/" + key)
        return out
    return {
        k: annotate_context(ref, v, paths, prefix + "/" + k) for k, v in obj.items()
    }


def contains_mapping(obj, key) -> bool:
    if isinstance(obj, dict):
        if isinstance(obj.get(key), dict):
            return True
        return any(contains_mapping(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_mapping(v, key) for v in obj)
    return False


def target_files(ref) -> list:
    names = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "docs/figures"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    hits = []
    for name in names:
        path = REPO / name
        if path.suffix != ".json":
            continue
        if not any(target in path.parents for target in TARGET_DIRS):
            continue
        values = {int(v) for v in ref.CAP_RE.findall(path.read_text(errors="replace"))}
        if values & TARGET_VALUES:
            hits.append(path)
    return hits


def annotate_file(ref, path, arm) -> dict:
    before = json.loads(path.read_text())
    nums_before = ref.numeric_paths(before)
    paths = []
    if arm == "context":
        after = annotate_context(ref, before, paths)
    else:
        after = ref.annotate(before, paths)
    path.write_text(json.dumps(after, indent=2) + "\n")
    nums_after = ref.numeric_paths(after)
    changed = {
        key: [nums_before[key], nums_after[key]]
        for key in nums_before
        if nums_after.get(key) != nums_before[key]
    }
    return {
        "file": str(path.relative_to(REPO)),
        "arm": arm,
        "provenance_paths": paths,
        "numeric_fields_before": len(nums_before),
        "numeric_changed": changed,
    }


def main():
    ref = load_reference()
    receipt = {"json": [], "context_arm": None, "sweep": None}
    for path in target_files(ref):
        obj = json.loads(path.read_text())
        if contains_mapping(obj, CONTEXT_KEY):
            arm = "context"
        elif contains_mapping(obj, "dimensions"):
            arm = "dimensions"
        else:
            receipt["json"].append(
                {
                    "file": str(path.relative_to(REPO)),
                    "arm": "unmapped",
                    "provenance_paths": [],
                    "numeric_fields_before": len(ref.numeric_paths(obj)),
                    "numeric_changed": {},
                }
            )
            continue
        entry = annotate_file(ref, path, arm)
        receipt["json"].append(entry)
        if arm == "context":
            receipt["context_arm"] = entry
    ref.TARGET_VALUES = TARGET_VALUES
    hits = ref.swept_files()
    per_value = {}
    for value in sorted(TARGET_VALUES):
        found = {n: h for n, h in hits.items() if value in h["values"]}
        per_value[str(value)] = {
            "marked": sorted(n for n, h in found.items() if h["marked"]),
            "unmarked": sorted(n for n, h in found.items() if not h["marked"]),
            "marked_count": sum(1 for h in found.values() if h["marked"]),
            "unmarked_count": sum(1 for h in found.values() if not h["marked"]),
        }
    receipt["sweep"] = {
        "target_value_files": len(hits),
        "marked": sorted(n for n, v in hits.items() if v["marked"]),
        "unmarked": sorted(n for n, v in hits.items() if not v["marked"]),
        "per_value": per_value,
    }
    out = NODE / "annotation-receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("WROTE", out.relative_to(REPO))
    for entry in receipt["json"]:
        print("ANNOTATED", len(entry["provenance_paths"]), entry["arm"], entry["file"])
        if entry["numeric_changed"]:
            print("   NUMERIC CHANGED:", entry["numeric_changed"])
    print("SWEEP target files:", receipt["sweep"]["target_value_files"])
    print("SWEEP marked:", len(receipt["sweep"]["marked"]))
    print("SWEEP unmarked:", len(receipt["sweep"]["unmarked"]))
    for value, entry in receipt["sweep"]["per_value"].items():
        print(
            "   value",
            value,
            "marked",
            entry["marked_count"],
            "unmarked",
            entry["unmarked_count"],
        )


if __name__ == "__main__":
    main()
