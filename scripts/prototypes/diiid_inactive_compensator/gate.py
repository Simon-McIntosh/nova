"""Measure CPU arms and refuse restored inactive source images in one job."""

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
    """Delegate execution and receipt parsing to the shared CPU arm owner."""
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


def negative_controls(tree, revision, output, scratch, after, reverse_commit=None):
    """Mutate each source and the shadow mask independently in an extracted head."""
    candidate = scratch / "negative"
    extract(candidate, revision, tree)
    source = candidate / "nova/equilibrium/constraint.py"
    original = source.read_text()
    marker = "normalized = jnp.where(self.binding.payload, normalized, 0.0)"
    assert original.count(marker) == 1
    records = {}
    for kind, unknown in (
        ("radial", "UniformRadialFieldUnknown"),
        ("circuit", "CircuitCurrentUnknown"),
    ):
        source.write_text(
            original.replace(
                marker,
                "normalized = (normalized if "
                f"isinstance(self.unknown, {unknown}) "
                "else jnp.where(self.binding.payload, normalized, 0.0))",
            )
        )
        record = arm(
            candidate, revision, output, f"negative-{kind}", PIN, ("-k", "compensator")
        )
        failures = set(record["failure_ids"]) - set(after["failure_ids"])
        assert record["completed"] and record["exit_status"] != 0 and failures
        assert any(kind in failure for failure in failures)
        if kind == "circuit":
            assert any("test_reduced_compensator" in failure for failure in failures)
        records[kind] = record
    source.write_text(original)
    tests = candidate / PIN[0]
    checks = tests.read_text()
    marker = "jnp.zeros_like(state, dtype=bool).at[4].set(True)"
    assert checks.count(marker) == 1
    tests.write_text(checks.replace(marker, "jnp.zeros_like(state, dtype=bool)"))
    record = arm(
        candidate, revision, output, "negative-shadow", PIN, ("-k", "compensator")
    )
    failures = set(record["failure_ids"]) - set(after["failure_ids"])
    assert record["completed"] and record["exit_status"] != 0 and failures
    assert all(
        "test_compensator_image_and_tangent_on_every_proposal_route" in f
        for f in failures
    )
    records["shadow"] = record
    tests.write_text(checks)
    if reverse_commit:
        patch = subprocess.check_output(
            ["git", "-C", str(tree), "diff", reverse_commit + "^", reverse_commit]
        )
        subprocess.run(
            ["git", "apply", "--reverse"],
            input=patch,
            cwd=candidate,
            check=True,
        )
        # The regression oracle stays present when the answering change is absent.
        tests.write_text(checks)
        record = arm(
            candidate, revision, output, "negative-reversed", PIN, ("-k", "compensator")
        )
        failures = set(record["failure_ids"]) - set(after["failure_ids"])
        assert record["completed"] and record["exit_status"] != 0 and failures
        assert any("test_reduced_compensator" in f for f in failures)
        assert any("test_compensator_gate_delegates" in f for f in failures)
        records["reversed"] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-commit")
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
        baseline_focused = arm(
            baseline_tree, args.base, args.output, "baseline-focused", FOCUSED
        )
        after = arm(tree, head, args.output, "after", PIN)
        after_focused = arm(tree, head, args.output, "after-focused", FOCUSED)
        negative_controls(tree, head, args.output, scratch, after, args.reverse_commit)
        assert all(
            item["completed"]
            for item in (baseline, baseline_focused, after, after_focused)
        )
        assert not (set(after["failure_ids"]) - set(baseline["failure_ids"]))
        assert not (
            set(after_focused["failure_ids"]) - set(baseline_focused["failure_ids"])
        )
    print(
        "GATE: no added failures; radial/circuit images and false shadows refused",
        flush=True,
    )


if __name__ == "__main__":
    main()
