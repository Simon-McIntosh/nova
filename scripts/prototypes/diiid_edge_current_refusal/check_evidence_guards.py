"""Exercise receipt guards on captured evidence and isolated corruptions."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


@contextmanager
def changed(path):
    """Restore a test-owned copy after one mutation, including failed checks."""
    content = path.read_bytes()
    try:
        yield path
    finally:
        path.write_bytes(content)


def update_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tree", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.tree / "scripts/prototypes/diiid_edge_current_refusal"
    environment = dict(os.environ, PYTHONPATH=str(args.tree), JAX_PLATFORMS="cpu")
    results = []

    def expect(name, command, error=None):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            cwd=args.tree,
            timeout=120,
            check=False,
        )
        output = result.stdout + result.stderr
        passed = (
            result.returncode == 0
            if error is None
            else result.returncode == 1 and error in output
        )
        log = args.output / f"{name}.log"
        log.write_text(
            f"# tree: {args.tree}; command: {shlex.join(command)}\n"
            + output
            + f"EXIT={result.returncode}\n"
        )
        row = dict(case=name, passed=passed, exit=result.returncode, log=str(log))
        results.append(row)
        print("EVIDENCE_GUARD " + json.dumps(row), flush=True)

    with tempfile.TemporaryDirectory(prefix="edge-receipt-guards-") as temporary:
        scratch = Path(temporary)
        cohort = scratch / "cohort"
        shutil.copytree(args.input, cohort)
        shutil.copy2(args.input.parent / "titan-trace.log", scratch / "titan-trace.log")
        figures = scratch / "figures"
        figures.mkdir()
        for svg in (args.tree / "docs/figures/diiid-edge-current-refusal").glob(
            "*.svg"
        ):
            shutil.copy2(svg, figures / svg.name)
        verify = [
            sys.executable,
            str(source / "verify.py"),
            "--input",
            str(cohort),
            "--output",
            str(scratch / "summary.json"),
        ]
        audit = [sys.executable, str(source / "audit.py"), "--input", str(cohort)]
        markers = [
            sys.executable,
            str(source / "marker_visibility.py"),
            "--input",
            str(cohort),
            "--figures",
            str(figures),
            "--only",
            "frame-137-diverted",
            "frame-146-diverted",
            "frame-146-limited",
            "frame-179-limited",
            "inactive-field-reference",
            "inactive-field-suppressed",
        ]
        expect("captured-cohort", verify)
        expect("bound-endpoints", audit)
        expect("visible-markers-and-empty-control", markers)

        folder = cohort / "edge/frame-137-diverted"
        with changed(folder / "receipt.json") as path:
            update_json(path, lambda row: row.update(terminal_reason=1))
            expect("wrong-termination", verify, "failure premise was not reproduced")
        with changed(cohort / "edge/frame-146-diverted/receipt.json") as path:
            update_json(path, lambda row: row.update(converged=False))
            expect("unconverged-control", verify, "convergence contradicts")
        with changed(folder / "diagnostics-0.npz") as path:
            with np.load(path) as archive:
                arrays = dict(archive)
            arrays["inner_iteration_decisions"][0] = 999
            np.savez_compressed(path, **arrays)
            expect(
                "changed-solver-diagnostic", verify, "differs from solver diagnostics"
            )
        with changed(folder / "receipt.json") as path:

            def corrupt_count(row):
                row["changed_mask_entries"][0] += 1

            update_json(path, corrupt_count)
            expect("wrong-shadow-count", audit, "shadow changes, reported")
        with changed(folder / "inner-0.npz") as path:
            with np.load(path) as archive:
                arrays = dict(archive)
            arrays["initial"] += 1.0
            arrays["state"] += 1.0
            np.savez_compressed(path, **arrays)
            expect("unbound-endpoint", audit, "cannot bind the selected endpoint")

        # Remove the evidence and its drawn circles together: a self-consistent
        # empty figure still has to disagree with the captured ladder inventory.
        svg = figures / "frame-137-diverted.svg"
        with changed(folder / "spatial.json") as path, changed(svg):
            update_json(path, lambda row: row.update(proposals=[]))
            root = ET.parse(svg).getroot()
            namespace = "{http://www.w3.org/2000/svg}"
            link = "{http://www.w3.org/1999/xlink}href"
            definitions = {node.get("id"): node for node in root.iter()}
            panel = definitions["axes_2"]
            for group in panel.iter():
                for mark in list(group):
                    if (
                        mark.tag == namespace + "use"
                        and "stroke: #bf3446" in mark.get("style", "")
                        and "C" in definitions[mark.get(link)[1:]].get("d", "")
                    ):
                        group.remove(mark)
            for node in root.iter(namespace + "text"):
                if (
                    node.text
                    and "cells change in terminal refused proposals" in node.text
                ):
                    node.text = "0 cells change in terminal refused proposals"
            ET.ElementTree(root).write(svg)
            expect("emptied-evidence-and-figure", markers, "spatial proposal inventory")
        print(
            f"Removing isolated receipt and figure copies under {scratch}", flush=True
        )
    print(f"Removed {scratch}; captured evidence remains untouched", flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    failures = sum(not row["passed"] for row in results)
    print(f"EVIDENCE_GUARD_SUMMARY cases={len(results)} failed={failures}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
