"""Persist serial CPU gates and the declared seed-hold mutation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.diiid_core_hold.regressions import (
    ROOT as CORE_ROOT,
    extract,
    gate as core_gate,
)

ROOT = CORE_ROOT
SCOPED = (
    "tests/test_shadow_follows_image.py",
    "tests/test_xpoint_attribution_image_read.py",
)


def gate(tree, revision, paths, log, scratch):
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nova.equilibrium import fixed_point; "
            "import os; print(fixed_point.__file__); print(os.getcwd())",
        ],
        cwd=tree,
        env=dict(os.environ, PYTHONPATH="."),
        capture_output=True,
        text=True,
        check=True,
    )
    module, cwd = probe.stdout.strip().splitlines()[-2:]
    assert Path(module).resolve().is_relative_to(tree.resolve())
    result = core_gate(tree, revision, paths, log, scratch)
    header, body = log.read_text().split("\n", 1)
    log.write_text(header + f"\n# module: {module}; cwd: {cwd}\n" + body)
    result.update(measurement_module=module, measurement_cwd=cwd)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", type=Path)
    parser.add_argument("--mutation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    with tempfile.TemporaryDirectory(
        prefix="nova-shadow-gate-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        tree = scratch / "tree"
        extract(tree, args.revision, ROOT)
        if args.mutation:
            source = tree / "nova/equilibrium/forward_operator.py"
            text = source.read_text()
            marker = "            jnp.zeros_like(psi, dtype=bool),\n"
            marker += "            partition,\n"
            assert text.count(marker) == 1
            source.write_text(
                text.replace(
                    marker,
                    "            self.residual_shadow_mask(psi, requested_class)\n"
                    "            if partition is None else partition.residual_shadow,\n"
                    "            partition,\n",
                )
            )
            paths = [SCOPED[0]]
            results["negative_control"] = gate(
                tree,
                args.revision,
                paths,
                args.output / "negative-control.log",
                scratch,
            )
        elif args.files:
            paths = json.loads(args.files.read_text())
            for path in paths:
                if not (tree / path).is_file():
                    continue
                folder = scratch / Path(path).stem
                folder.mkdir()
                results[path] = gate(
                    tree,
                    args.revision,
                    [path],
                    args.output / (path.replace("/", "-") + ".log"),
                    folder,
                )
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print("REGRESSION", path, json.dumps(results[path]), flush=True)
        else:
            paths = [path for path in SCOPED if (tree / path).is_file()]
            if paths:
                results["scope"] = gate(
                    tree, args.revision, paths, args.output / "suite.log", scratch
                )
            else:
                log = args.output / "empty-baseline.log"
                log.write_text(
                    f"# revision: {args.revision}; tree: {tree}\n"
                    "No scoped test file exists at this revision; no pytest invoked.\n"
                    "EXIT=0\n"
                )
                results["scope"] = dict(
                    revision=args.revision,
                    command="no applicable scoped test files",
                    completed=True,
                    exit_status=0,
                    failure_count=0,
                    failure_ids=[],
                    log_path=str(log),
                    measured_nothing=True,
                )
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.mutation:
        assert results["negative_control"]["exit_status"] != 0


if __name__ == "__main__":
    main()
