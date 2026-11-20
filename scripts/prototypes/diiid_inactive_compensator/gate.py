"""Measure CPU arms and refuse a restored inactive source image in one job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.prototypes.diiid_core_hold.regressions import extract, gate as core_gate

PIN = ["tests/equilibrium/test_solve_vertical_pin.py"]
FOCUSED = [
    "tests/test_equilibrium_constraint_protocol.py",
    "tests/test_constraint_residual_algebra.py",
    "tests/test_fixed_point.py",
    "tests/test_fixed_point_stationary_mask.py",
    "tests/test_reduced_newton_constraints.py",
]


def arm(tree, revision, output, name, targets, extra=()):
    log = output / f"{name}.log"
    with tempfile.TemporaryDirectory(
        prefix=f"inactive-{name}-", dir=os.environ.get("TMPDIR")
    ) as temporary:
        record = core_gate(tree, revision, [*targets, *extra], log, Path(temporary))
    lines = log.read_text().splitlines()
    record.update(
        measurement_module=next(
            line.removeprefix("IMPORTED_MODULE=")
            for line in lines
            if line.startswith("IMPORTED_MODULE=")
        ),
        measurement_cwd=next(
            line.removeprefix("RESOLVED_CWD=")
            for line in lines
            if line.startswith("RESOLVED_CWD=")
        ),
    )
    assert Path(record["measurement_module"]).resolve().is_relative_to(tree.resolve())
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(name, json.dumps(record), flush=True)
    return record


def negative_control(tree, revision, output, scratch, after):
    """Restore the inactive image on the augmented-map proposal route only."""
    candidate = scratch / "negative"
    extract(candidate, revision, tree)
    source = candidate / "nova/equilibrium/constraint.py"
    original = source.read_text()
    marker = "delta = pair.flux_delta(profile, context, value)"
    assert original.count(marker) == 1, "one augmented-map image call site"
    source.write_text(
        original.replace(
            marker,
            marker + "\n"
            "                delta = delta + pair.unknown.flux_delta("
            "profile, context, pair.functional, pair.binding.payload, value)",
        )
    )
    record = arm(
        candidate, revision, output, "negative-radial", PIN, ("-k", "compensator")
    )
    failures = set(record["failure_ids"]) - set(after["failure_ids"])
    assert record["completed"] and record["exit_status"] != 0 and failures
    assert failures, "the restored image must redden the route test"
    assert all(
        "test_compensator_image_and_tangent_on_every_proposal_route[False-" in failure
        for failure in failures
    ), failures
    source.write_text(original)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tree = args.tree.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="inactive-image-", dir=os.environ["TMPDIR"]
    ) as work:
        scratch = Path(work)
        baseline_tree = scratch / "base"
        extract(baseline_tree, args.base, tree)
        baseline = arm(baseline_tree, args.base, args.output, "baseline", PIN)
        after = arm(tree, head, args.output, "after", PIN)
        negative_control(tree, head, args.output, scratch, after)
        assert baseline["completed"] and after["completed"]
        assert not (set(after["failure_ids"]) - set(baseline["failure_ids"]))
        # The wider consumer group aborts in the compiler on both arms of this
        # host, so it is recorded as measured and never required to complete;
        # a completed pair must still add no failure.
        baseline_focused = arm(
            baseline_tree, args.base, args.output, "baseline-focused", FOCUSED
        )
        after_focused = arm(tree, head, args.output, "after-focused", FOCUSED)
        if baseline_focused["completed"] and after_focused["completed"]:
            assert not (
                set(after_focused["failure_ids"]) - set(baseline_focused["failure_ids"])
            )
    print("GATE: no added failures; restored inactive image refused", flush=True)


if __name__ == "__main__":
    main()
