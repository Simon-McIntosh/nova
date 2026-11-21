"""Run paired CPU gates and controls for the WEST refinement measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.iron_graded_refinement.gate import GATE, arm, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-tree", "head-tree", "output", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("base-revision", "head-revision"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, tree, revision in (
        ("gate-base", args.base_tree, args.base_revision),
        ("gate-head", args.head_tree, args.head_revision),
    ):
        prepare(tree, revision, args.source)
        record[name] = arm(name, tree, revision, GATE, args.output)
    control = args.head_tree.parent / "control"
    shutil.copytree(args.head_tree, control, ignore=shutil.ignore_patterns(".git"))
    script = control / "scripts/west_iron_linear/measure.py"
    data = script.read_text()
    interaction_guard = (
        'max(row["frobenius_relative"], row["maximum_entry_relative"]) <= 1e-12'
    )
    refinement_guard = "np.all(np.abs(change) < 0.01)"
    assert data.count(interaction_guard) == data.count(refinement_guard) == 1
    data = data.replace(interaction_guard, interaction_guard.replace("1e-12", "1.0"))
    data = data.replace(refinement_guard, refinement_guard.replace("0.01", "0.02"))
    script.write_text(data)
    arguments = [
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "tests/test_magnetisation.py",
        "-k",
        "west_route_comparison or west_refinement",
    ]
    record["negative-control"] = arm(
        "negative-control", control, args.head_revision, arguments, args.output
    )
    (args.output / "gate.json").write_text(json.dumps(record, indent=2) + "\n")
    assert record["negative-control"]["exit"] != 0, "mutated guards were not detected"
    return int(any(record[name]["exit"] != 0 for name in ("gate-base", "gate-head")))


if __name__ == "__main__":
    raise SystemExit(main())
