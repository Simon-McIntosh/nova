"""Measure base, head and a restored inactive radial image in one CPU job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
PIN = ["tests/equilibrium/test_solve_vertical_pin.py"]
FOCUSED = [
    "tests/test_equilibrium_constraint_protocol.py",
    "tests/test_constraint_residual_algebra.py",
    "tests/test_fixed_point.py",
    "tests/test_fixed_point_stationary_mask.py",
]


def arm(tree, revision, output, name, targets, extra=()):
    """Persist the complete command, imported source, summary and exit."""
    command = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(
        [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", *targets, *extra]
    )
    environment = dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu")
    module = subprocess.check_output(
        [PYTHON, "-c", "import nova.equilibrium.constraint as m; print(m.__file__)"],
        cwd=tree,
        env=environment,
        text=True,
    ).strip()
    log = output / f"{name}.log"
    with log.open("w") as stream:
        stream.write(f"# revision: {revision}; tree: {tree}; command: {command}\n")
        stream.write(f"# module: {module}; cwd: {tree.resolve()}\n")
        stream.flush()
        result = subprocess.run(
            command,
            shell=True,
            cwd=tree,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.write(f"EXIT={result.returncode}\n")
    text = log.read_text()
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.MULTILINE)
    summary = [
        line
        for line in text.splitlines()
        if re.search(r"\d+ (?:passed|failed|error).* in ", line)
    ]
    record = dict(
        revision=revision,
        command=command,
        exit_status=result.returncode,
        log_path=str(log),
        completed=bool(summary),
        failure_count=len(set(failures)),
        failure_ids=sorted(set(failures)),
        summary=summary[-1] if summary else None,
        measurement_module=module,
        measurement_cwd=str(tree.resolve()),
    )
    (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(name, json.dumps(record), flush=True)
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
        scratch = Path(work) / "base"
        scratch.mkdir()
        archive = Path(work) / "base.tar"
        subprocess.run(
            ["git", "-C", str(tree), "archive", args.base, "--output", str(archive)],
            check=True,
        )
        subprocess.run(["tar", "-xf", str(archive), "-C", str(scratch)], check=True)
        baseline = arm(scratch, args.base, args.output, "baseline", PIN)
        baseline_focused = arm(
            scratch, args.base, args.output, "baseline-focused", FOCUSED
        )
        after = arm(tree, head, args.output, "after", PIN)
        after_focused = arm(tree, head, args.output, "after-focused", FOCUSED)
        for relative in ("nova/equilibrium/constraint.py", *PIN):
            (scratch / relative).write_bytes((tree / relative).read_bytes())
        source = scratch / "nova/equilibrium/constraint.py"
        text = source.read_text()
        marker = "image_value = jnp.where(pair.binding.payload, value, 0.0)"
        assert text.count(marker) == 1
        source.write_text(
            text.replace(
                marker,
                "image_value = (value if "
                "isinstance(pair.unknown, UniformRadialFieldUnknown) "
                "else jnp.where(pair.binding.payload, value, 0.0))",
            )
        )
        negative = arm(
            scratch,
            head,
            args.output,
            "negative-control",
            PIN,
            ("-k", "test_compensator_image_and_tangent_on_every_proposal_route"),
        )
        assert negative["completed"] and negative["exit_status"] != 0
        assert set(negative["failure_ids"]) - set(after["failure_ids"])
        assert all("radial" in failure for failure in negative["failure_ids"])
        assert all(
            item["completed"]
            for item in (baseline, baseline_focused, after, after_focused)
        )
        assert not (set(after["failure_ids"]) - set(baseline["failure_ids"]))
        assert not (
            set(after_focused["failure_ids"]) - set(baseline_focused["failure_ids"])
        )
    print("GATE: no added failures; restored inactive radial image refused", flush=True)


if __name__ == "__main__":
    main()
