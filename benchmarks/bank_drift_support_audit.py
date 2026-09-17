"""Audit the constraint support each tree carries into the banked solve.

The bank drift shows the operand partition as the first stage that differs on
every paired arm: the producer tree and the current tree emit the same
profile_support digest but different partition_structure digests. Before any
solver repair this audit establishes whether that difference is a missing or
misrouted support row, or a rerouting of identical rows.

Per arm and per tree the audit reads the emission receipt for the termination
reason and terminal residual, the partition structure summary for the leaf
count, pytree flag and leaf dtypes, and the row inventory from each tree's
operator source. The inventory is the load-bearing half: both trees declare the
same three specialisation rows and the bank builder writes them with the same
expressions, so the question is whether the operator asks for those rows to
enter the trace as data or bakes them in as host constants.

Reads are source-level only: no geometry is built, no data is opened, nothing
compiles. The merged receipt is optional; when absent the audit states that and
pairs the arms from the emissions.
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
ARMS_DIRS = ("bank-drift/arms", "bank-drift-remainder/arms")


def load_emission(path: Path) -> dict:
    return json.loads(path.read_text())


def arm_rows(payload: dict) -> dict[str, dict]:
    """Map arm label to its solve receipt for one emission payload."""
    out: dict[str, dict] = {}
    for row in payload.get("rows") or []:
        for label, receipt in (row.get("arms") or {}).items():
            out[str(label)] = receipt or {}
    return out


def structure_summary(payload: dict) -> dict:
    stages = payload.get("stages") or {}
    structure = stages.get("partition_structure") or {}
    summary = structure.get("summary") or {}
    return {
        "digest": str(structure.get("digest", "")),
        "leaf_count": summary.get("leaf_count"),
        "tree_node_count": summary.get("tree_node_count"),
        "is_pytree": summary.get("is_pytree"),
        "operator_type": str(summary.get("operator_type", "")),
        "leaf_dtypes": tuple(summary.get("leaf_dtypes") or ()),
    }


def digests(payload: dict) -> dict:
    stages = payload.get("stages") or {}
    return {
        "structure": str((stages.get("partition_structure") or {}).get("digest", "")),
        "values": str((stages.get("partition_values") or {}).get("digest", "")),
        "support": str((stages.get("profile_support") or {}).get("digest", "")),
        "support_parts": {
            str(key): str(value)
            for key, value in (
                (stages.get("profile_support") or {}).get("parts") or {}
            ).items()
        },
    }


def collect(reports_root: Path) -> dict:
    """Key emission records by (arm, arm label) and confirm the tree pairing."""
    records: dict[tuple[str, str], dict] = {}
    for relative in ARMS_DIRS:
        directory = reports_root / relative
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = load_emission(path)
            tree = str(payload.get("tree_label", ""))
            if not tree:
                tree = "old" if path.name.startswith("old-") else "new"
            if tree in ("fae50f15", "fae-floor", "faefloor"):
                tree = "old"
            summary = structure_summary(payload)
            digest = digests(payload)
            for label, receipt in arm_rows(payload).items():
                residual = receipt.get("terminal_residual")
                records[(path.stem, label)] = {
                    "tree": tree,
                    "path": str(path),
                    "requested_arm": str(payload.get("requested_arm", path.stem)),
                    "converged": receipt.get("converged"),
                    "termination_reason": str(receipt.get("termination_reason", "")),
                    "terminal_residual": (
                        "" if residual is None else "%.6g" % robust_float(residual)
                    ),
                    "stage_exception": str(receipt.get("stage_exception", "")),
                    "structure": summary,
                    "digests": digest,
                }
    return records


def robust_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


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
        text = constraints.read_text()
        inventory["compensating_unknowns"] = [
            line.split("class ", 1)[1].split("(", 1)[0].strip()
            for line in text.splitlines()
            if line.startswith("class ") and "CompensatingUnknown" in line
        ]
    return inventory


def pair_records(records: dict) -> list:
    """Pair each arm label across the two trees, keeping the pairing explicit."""
    grouped: dict[tuple[str, str], dict] = {}
    for (stem, label), record in records.items():
        grouped.setdefault((stem, label), {})[record["tree"]] = record
    return sorted(grouped.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        default="/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local",
    )
    parser.add_argument("--old-tree", required=True)
    parser.add_argument("--new-tree", required=True)
    parser.add_argument("--receipt", default="/bank-drift/receipt.json")
    args = parser.parse_args()

    root = Path(args.reports_root)
    records = collect(root)

    print("== emission inventory (per arm and tree)")
    for (stem, label), record in sorted(records.items()):
        summary = record["structure"]
        print(
            "%-30s %-6s %-4s conv=%-5s res=%-14s reason=%-28s "
            "leaves=%-4s nodes=%-4s pytree=%-5s support_digest=%s"
            % (
                stem,
                label,
                record["tree"],
                str(record["converged"]),
                record["terminal_residual"],
                record["termination_reason"] or "(none)",
                summary["leaf_count"],
                summary["tree_node_count"],
                summary["is_pytree"],
                record["digests"]["support"],
            )
        )

    print("\n== row inventory (source only, no compile)")
    inventory = {}
    for label, tree_root in (("old", args.old_tree), ("new", args.new_tree)):
        inventory[label] = row_inventory(Path(tree_root))
        found = inventory[label]
        print(
            "%s: op=%s pytree=%s trace_hook=%s traced_rows=%s "
            "declared_rows=%s unknowns=%s"
            % (
                label,
                found["tree_root"],
                found["pytree_registered"],
                found["trace_hook"],
                found["traced_rows"],
                found["declared_rows"],
                found["compensating_unknowns"],
            )
        )

    print("\n== paired support comparison")
    same_support = 0
    differing_structure = 0
    unpaired = 0
    for (stem, label), trees in pair_records(records):
        old = trees.get("old")
        new = trees.get("new")
        if old is None or new is None:
            unpaired += 1
            print("%-30s %-6s UNPAIRED old=%s new=%s"
                  % (stem, label, old is not None, new is not None))
            continue
        support_same = old["digests"]["support"] == new["digests"]["support"]
        structure_same = old["digests"]["structure"] == new["digests"]["structure"]
        same_support += int(support_same)
        differing_structure += int(not structure_same)
        print(
            "%-30s %-6s support=%-6s structure=%-6s "
            "old[%s/%s] new[%s/%s]"
            % (
                stem,
                label,
                "same" if support_same else "DIFFERS",
                "same" if structure_same else "DIFFERS",
                old["termination_reason"] or "n/a",
                old["terminal_residual"] or "n/a",
                new["termination_reason"] or "n/a",
                new["terminal_residual"] or "n/a",
            )
        )

    receipt = Path(args.receipt)
    if not receipt.is_absolute():
        receipt = root / str(receipt).lstrip("/")
    print("\n== merged receipt")
    if receipt.exists():
        payload = load_emission(receipt)
        print("present: %s (%d bytes)" % (receipt, receipt.stat().st_size))
        print(json.dumps(payload, indent=2, sort_keys=True)[:1500])
    else:
        print("not available: %s" % receipt)
        print("pairing derived from the emissions; stated, not silently absorbed")

    print(
        "\nsummary: paired_arms=%d support_identical=%d structure_differs=%d "
        "unpaired=%d" % (len(pair_records(records)), same_support,
                         differing_structure, unpaired)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())