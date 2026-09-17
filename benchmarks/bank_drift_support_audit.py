"""Audit the constraint support each tree carries into the banked solve.

The bank drift shows the operand partition as the first stage that differs on
every paired arm: the producer tree and the current tree emit the same
profile_support digest but different partition_structure digests. Before any
solver repair this audit establishes whether that difference is a missing or
misrouted support row, or a rerouting of identical rows.

Per arm and per tree the audit tables two families of columns, and says which
file each family came from:

* the solve receipt -- converged, termination reason, terminal residual, taken
  from the arm receipts under ``arms/`` and from the emissions;
* the stage digests and the partition structure summary -- taken from the
  ``stages`` block carried on each *row* of an emission or receipt file, which
  is where they live; the payload-level ``stages`` key does not exist and
  reading it instead silently yields an empty comparison.

The row inventory is the load-bearing half and is read from each tree's
operator source. Both trees declare the same three specialisation rows and the
bank builder writes them with the same expressions, so the question is whether
the operator asks for those rows to enter the trace as data or holds them as
host constants. ``--traced`` reports the source evidence for that.

Reads are source-level and receipt-level only: no geometry is built, no data is
opened, nothing compiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORT_ROWS = (
    "declared_axis_flux",
    "declared_boundary_flux",
    "declared_support",
)
OPERATOR_REL = "nova/equilibrium/forward_operator.py"
CONSTRAINT_REL = "nova/equilibrium/constraint.py"
# The tree label each side of the drift writes into every emission.
TREE_ALIASES = {
    "fae50f15": "old",
    "faefloor": "old",
    "main": "new",
    "head": "new",
    "current": "new",
}
RECEIPT_FIELDS = (
    "converged",
    "termination_reason",
    "terminal_residual",
    "stage_exception",
)
STAGE_FIELDS = ("structure", "values", "support")


def tree_of(payload: dict, path: Path) -> str:
    label = str(payload.get("tree_label") or "").strip().lower()
    if label in TREE_ALIASES:
        return TREE_ALIASES[label]
    if label:
        return label
    name = path.name
    if name.startswith("old-"):
        return "old"
    if name.startswith("new-"):
        return "new"
    return "unknown"


def rows_of(payload: dict) -> list[dict]:
    return [row for row in (payload.get("rows") or []) if isinstance(row, dict)]


def stages_of(row: dict) -> dict:
    return row.get("stages") or {}


def digest_fields(row: dict) -> dict:
    """Digests and partition structure summary, read from the row's stages."""
    stages = stages_of(row)
    structure = stages.get("partition_structure") or {}
    summary = structure.get("summary") or {}
    support = stages.get("profile_support") or {}
    values = stages.get("partition_values") or {}
    dtypes = summary.get("leaf_dtypes")
    return {
        "structure_digest": str(structure.get("digest", "")),
        "values_digest": str(values.get("digest", "")),
        "support_digest": str(support.get("digest", "")),
        "support_parts": {
            str(key): str(value) for key, value in (support.get("parts") or {}).items()
        },
        "leaf_count": summary.get("leaf_count"),
        "tree_node_count": summary.get("tree_node_count"),
        "is_pytree": summary.get("is_pytree"),
        "operator_type": str(summary.get("operator_type", "")),
        "leaf_dtypes": tuple(dtypes) if isinstance(dtypes, list) else (),
    }


def receipt_fields(receipt: dict) -> dict:
    residual = receipt.get("terminal_residual")
    try:
        formatted = "%.6g" % float(residual) if residual is not None else ""
    except TypeError, ValueError:
        formatted = repr(residual)
    return {
        "converged": receipt.get("converged"),
        "termination_reason": str(receipt.get("termination_reason", "")),
        "terminal_residual": formatted,
        "stage_exception": str(receipt.get("stage_exception", "")),
    }


def collect(reports_root: Path) -> dict:
    """Merge every emission and receipt into one record per arm, per tree.

    The pairing key is the identity the emissions carry (``21978/35`` style) and
    the arm label, never the file stem -- the stem carries the tree prefix, so
    keying on it pairs nothing.
    """
    records: dict[tuple[str, str], dict] = {}
    skipped: list[str] = []
    for path in sorted(reports_root.glob("bank-drift*/**/*.json")):
        if "faefloor-tree" in path.parts:
            continue
        relative = str(path.relative_to(reports_root))
        if "arc-tracer" in relative:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            skipped.append("%s (%s)" % (relative, error))
            continue
        tree = tree_of(payload, path)
        for row in rows_of(payload):
            identity = str(row.get("identity") or "").strip()
            if not identity:
                continue
            staged = digest_fields(row)
            for label, receipt in (row.get("arms") or {}).items():
                key = (identity, str(label))
                record = records.setdefault(key, {})
                entry = record.setdefault(
                    tree,
                    {
                        "receipt_source": "",
                        "stage_source": "",
                        "identity": identity,
                        "arm_label": str(label),
                        "tree": tree,
                    },
                )
                if isinstance(receipt, dict) and not entry["receipt_source"]:
                    entry.update(receipt_fields(receipt))
                    entry["receipt_source"] = relative
                if staged["structure_digest"] and not entry["stage_source"]:
                    entry.update(staged)
                    entry["stage_source"] = relative
    return {"records": records, "skipped": skipped}


def row_inventory(tree_root: Path) -> dict:
    """Rows and unknowns one tree declares, read from source only."""
    operator = tree_root / OPERATOR_REL
    inventory = {
        "tree_root": str(tree_root),
        "operator_present": operator.exists(),
        "pytree_registered": False,
        "trace_hook": False,
        "traced_rows": [],
        "declared_rows": [],
        "compensating_unknowns": [],
        "notes": [],
    }
    if not operator.exists():
        inventory["notes"].append("missing " + OPERATOR_REL)
        return inventory
    source = operator.read_text()
    inventory["pytree_registered"] = "register_pytree_node_class" in source
    inventory["trace_hook"] = "_dynamic_extra_names" in source
    inventory["traced_rows"] = [
        name for name in SUPPORT_ROWS if ('"%s"' % name) in source
    ]
    inventory["declared_rows"] = [name for name in SUPPORT_ROWS if name in source]
    constraints = tree_root / CONSTRAINT_REL
    if constraints.exists():
        lines = constraints.read_text().splitlines()
        inventory["compensating_unknowns"] = [
            line.split("class ", 1)[1].split("(", 1)[0].strip()
            for line in lines
            if line.startswith("class ") and "CompensatingUnknown" in line
        ]
    return inventory


def cell(value: object) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if text else "-"


def receipt_cell(entry: dict, field: str) -> str:
    """A receipt field, or a loud refusal naming the source it came from.

    A silent blank here reads as "the emission did not report a residual" and
    is indistinguishable from a field the merge never populated, so the blank
    is made to speak.
    """
    if field not in entry:
        return "NO-FIELD"
    value = entry.get(field)
    if value not in (None, ""):
        return str(value)
    source = entry.get("receipt_source") or "(no source)"
    return "EMPTY@%s" % source


def print_per_arm(records: dict) -> None:
    print("== per arm and tree (receipt columns from the receipt source):")
    print(
        "%-10s %-6s %-4s %-6s %-12s %-32s %-8s %-6s"
        % (
            "identity",
            "arm",
            "tree",
            "conv",
            "residual",
            "termination reason",
            "pytree",
            "leaves",
        )
    )
    for key in sorted(records):
        for tree in ("old", "new"):
            entry = records[key].get(tree)
            if entry is None:
                print("%-10s %-6s %-4s %s" % (key[0], key[1], tree, "(no record)"))
                continue
            print(
                "%-10s %-6s %-4s %-6s %-12s %-32s %-8s %-6s"
                % (
                    key[0],
                    key[1],
                    tree,
                    cell(entry.get("converged")),
                    receipt_cell(entry, "terminal_residual"),
                    cell(entry.get("termination_reason")),
                    cell(entry.get("is_pytree")),
                    cell(entry.get("leaf_count")),
                )
            )


def print_support(records: dict) -> dict:
    print("\n== per arm and tree (stage columns from the stage source):")
    print(
        "%-10s %-6s %-4s %-18s %-18s %-18s %s"
        % (
            "identity",
            "arm",
            "tree",
            "profile_support",
            "partition_structure",
            "partition_values",
            "support parts",
        )
    )
    tally = {
        "paired": 0,
        "support_same": 0,
        "support_differ": 0,
        "structure_same": 0,
        "structure_differ": 0,
        "missing": 0,
    }
    for key in sorted(records):
        for tree in ("old", "new"):
            entry = records[key].get(tree)
            if entry is None:
                continue
            print(
                "%-10s %-6s %-4s %-18s %-18s %-18s %s"
                % (
                    key[0],
                    key[1],
                    tree,
                    cell(entry.get("support_digest")),
                    cell(entry.get("structure_digest")),
                    cell(entry.get("values_digest")),
                    json.dumps(entry.get("support_parts") or {}, sort_keys=True),
                )
            )
    print("\n== paired support comparison (the fence's question):")
    for key in sorted(records):
        old = records[key].get("old")
        new = records[key].get("new")
        if old is None or new is None:
            tally["missing"] += 1
            print(
                "%-10s %-6s UNPAIRED old=%s new=%s"
                % (key[0], key[1], old is not None, new is not None)
            )
            continue
        tally["paired"] += 1
        support_same = old.get("support_digest") and old.get(
            "support_digest"
        ) == new.get("support_digest")
        structure_same = old.get("structure_digest") and old.get(
            "structure_digest"
        ) == new.get("structure_digest")
        if not (old.get("support_digest") and new.get("support_digest")):
            tally["missing"] += 1
            print(
                "%-10s %-6s NO STAGE EVIDENCE on one side -- not a pass"
                % (key[0], key[1])
            )
            continue
        tally["support_same"] += int(bool(support_same))
        tally["support_differ"] += int(not support_same)
        tally["structure_same"] += int(bool(structure_same))
        tally["structure_differ"] += int(not structure_same)
        print(
            "%-10s %-6s support=%-8s structure=%-8s old[%s/%s] new[%s/%s]"
            % (
                key[0],
                key[1],
                "same" if support_same else "DIFFERS",
                "same" if structure_same else "DIFFERS",
                cell(old.get("termination_reason")),
                cell(old.get("terminal_residual")),
                cell(new.get("termination_reason")),
                cell(new.get("terminal_residual")),
            )
        )
    return tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        default="/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local",
    )
    parser.add_argument("--old-tree", required=True, help="producer tree root")
    parser.add_argument("--new-tree", required=True, help="current tree root")
    parser.add_argument(
        "--json-out", default=None, help="write the merged table as JSON here"
    )
    args = parser.parse_args()

    collected = collect(Path(args.reports_root))
    records = collected["records"]
    print("records (identity, arm): %d" % len(records))
    if collected["skipped"]:
        print("skipped: %s" % "; ".join(collected["skipped"]))
    print()

    print_per_arm(records)
    tally = print_support(records)

    print("\n== row inventory (source only, no compile)")
    for label, tree_root in (("old", args.old_tree), ("new", args.new_tree)):
        found = row_inventory(Path(tree_root))
        print(
            "%-4s pytree=%-5s trace_hook=%-5s traced_rows=%s declared_rows=%s "
            "unknowns=%s"
            % (
                label,
                found["pytree_registered"],
                found["trace_hook"],
                found["traced_rows"],
                found["declared_rows"],
                found["compensating_unknowns"],
            )
        )

    print(
        "\nsummary: paired=%d support_identical=%d support_differs=%d "
        "structure_identical=%d structure_differs=%d unpaired_or_missing=%d"
        % (
            tally["paired"],
            tally["support_same"],
            tally["support_differ"],
            tally["structure_same"],
            tally["structure_differ"],
            tally["missing"],
        )
    )

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(
            json.dumps(
                {str(key): value for key, value in sorted(records.items())},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
