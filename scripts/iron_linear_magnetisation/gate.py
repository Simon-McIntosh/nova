"""Reuse the exported-tree gate runner for affine iron tests and suite arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from scripts.iron_graded_refinement.gate import GATE, arm, prepare

SUITE = ["-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/test_magnetisation.py"]


def receipt(row, revision):
    text = Path(row["log"]).read_text()
    failures = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
    return dict(
        revision=revision,
        command=row["command"],
        exit_status=row["exit"],
        completed=bool(re.search(r"\d+ (?:passed|failed).* in ", text)),
        failure_count=len(failures),
        failure_ids=failures,
        log_path=row["log"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tree", type=Path, required=True)
    parser.add_argument("--head-tree", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "suites"), default="fast")
    parser.add_argument("--arm", choices=("baseline", "after", "both"), default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, tree, revision in (
        ("baseline", args.base_tree, args.base_revision),
        ("after", args.head_tree, args.head_revision),
    ):
        if args.arm != "both" and args.arm != name:
            continue
        prepare(tree, revision, args.source)
        row = arm(
            name + "-" + args.mode,
            tree,
            revision,
            GATE if args.mode == "fast" else SUITE,
            args.output,
        )
        record[name] = receipt(row, revision)
        (args.output / (args.mode + ".json")).write_text(
            json.dumps(record, indent=2) + "\n"
        )
    if args.mode == "fast" and args.arm in {"both", "after"}:
        path = args.head_tree / "nova/biot/tiledassembly.py"
        source = path.read_text()
        original = "    x, y = jnp.asarray(x.reshape(-1)), jnp.asarray(y.reshape(-1))"
        if source.count(original) != 1:
            raise SystemExit("quadrature mutation did not identify one rule")
        path.write_text(
            source.replace(
                original,
                original
                + "\n    if linear:\n"
                "        weight = weight.at[weight.size // 2].multiply(1.1)",
            )
        )
        row = arm(
            "negative-control",
            args.head_tree,
            args.head_revision,
            [*SUITE, "-k", "linear_shell_operator_pin", "-s"],
            args.output,
        )
        record["negative_control"] = receipt(row, args.head_revision)
        (args.output / (args.mode + ".json")).write_text(
            json.dumps(record, indent=2) + "\n"
        )
        if row["exit"] == 0 or not record["negative_control"]["failure_ids"]:
            raise SystemExit(
                "quadrature negative control did not produce a failing test id"
            )
    return int(
        any(
            record[name]["exit_status"] != 0
            for name in ("baseline", "after")
            if name in record
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
