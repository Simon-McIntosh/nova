"""Use the shared two-tree gate without redundant filtered suite arms."""

import argparse
import json
from pathlib import Path

from scripts.iron_graded_refinement.gate import GATE, arm, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-tree", "head-tree", "output", "source"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("base-revision", "head-revision"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, tree, revision in (
        ("gate-base", args.base_tree, args.base_revision),
        ("gate-head", args.head_tree, args.head_revision),
    ):
        prepare(tree, revision, args.source)
        rows[name] = arm(name, tree, revision, GATE, args.output)
    (args.output / "gate.json").write_text(json.dumps(rows, indent=2) + "\n")
    return int(any(r["exit"] != 0 for r in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
