"""Replay the numerical controls and compare them with retained baseline rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from scripts.prototypes.diiid_diverted_branch import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    results = {}
    with tempfile.TemporaryDirectory(
        dir=os.environ["TMPDIR"], prefix="branch-controls-"
    ) as scratch:
        tree = Path(scratch) / "head"
        measure.admission.extract(tree, args.head, repository)
        for frame, branch, case in (
            (35, "mast", "mast-21978-35"),
            (179, "limited", "frame-179-limited"),
        ):
            output = args.output / case
            output.mkdir()
            command = [
                sys.executable,
                str(Path(__file__).with_name("measure.py")),
                "--row",
                "--tree",
                str(tree),
                "--revision",
                args.head,
                "--frame",
                str(frame),
                "--branch",
                branch,
                "--output",
                str(output),
            ]
            log = output / "row.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.head}; tree: {tree}; "
                    f"command: {shlex.join(command)}\n"
                )
                stream.flush()
                run = subprocess.run(
                    command,
                    cwd=tree,
                    env=dict(os.environ, PYTHONPATH=str(tree)),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                stream.write(f"EXIT={run.returncode}\n")
            results[case] = dict(exit_status=run.returncode, log_path=str(log))
            (args.output / "job-results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            if run.returncode:
                raise SystemExit(run.returncode)
            base = args.baseline / case
            original = json.loads((base / "receipt.json").read_text())
            repaired = json.loads((output / "receipt.json").read_text())
            assert (
                original["base_revision"] == "04337b991e8aed9d0904494437e02a1b66fa24dc"
            )
            seed = original["seed_admission"][0]["sha256"]
            assert seed == repaired["seed_admission"][0]["sha256"]
            assert repaired["augmented_topology_reader_connected"] is True
            with (
                np.load(base / "terminal.npz") as b,
                np.load(output / "terminal.npz") as h,
            ):
                difference = float(np.max(np.abs(b["state"] - h["state"])))
                scale = max(1.0, float(np.max(np.abs(b["state"]))))
            tolerance = 8 * np.finfo(float).eps * scale
            passed = (
                difference <= tolerance
                and repaired["converged"] == original["converged"]
            )
            results[case].update(
                seed_sha256=seed,
                max_abs_state_difference_wb=difference,
                roundoff_tolerance_wb=tolerance,
                passed=bool(passed),
                residual=[original["residual"], repaired["residual"]],
                converged=[original["converged"], repaired["converged"]],
                branch_warmup=repaired["warmup_branch_refused_proposals"],
                branch_newton=repaired["newton_branch_refused_proposals"],
            )
            (args.output / "job-results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            print("CONTROL_RESULT " + json.dumps({case: results[case]}), flush=True)
    raise SystemExit(0 if all(item["passed"] for item in results.values()) else 1)


if __name__ == "__main__":
    main()
