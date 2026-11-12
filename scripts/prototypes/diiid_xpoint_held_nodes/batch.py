"""Measure both immutable physical-image arms in one GPU allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from scripts.prototypes.diiid_core_hold.batch import CASES, case_name, extract, gate

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    with tempfile.TemporaryDirectory(
        prefix="nova-shadow-cohort-", dir=os.environ["TMPDIR"]
    ) as temporary:
        for arm, revision in (("base", args.base), ("head", args.head)):
            tree = Path(temporary) / arm
            extract(tree, revision, ROOT)
            for frame, branch in CASES:
                name = case_name(frame, branch)
                output = args.output / arm / name
                output.mkdir(parents=True)
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("measure.py")),
                    "--tree",
                    str(tree),
                    "--revision",
                    revision,
                    "--frame",
                    str(frame),
                    "--branch",
                    branch,
                    "--output",
                    str(output),
                    "--trace-helper",
                    str(args.trace_helper),
                ]
                log = output / "row.log"
                with log.open("w") as stream:
                    stream.write(
                        f"# revision: {revision}; tree: {tree}; "
                        f"command: {shlex.join(command)}\n"
                    )
                    stream.flush()
                    process = subprocess.run(
                        command,
                        cwd=tree,
                        env=dict(os.environ, PYTHONPATH=str(tree)),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.write(f"EXIT={process.returncode}\n")
                results[arm + ":" + name] = dict(
                    revision=revision, exit_status=process.returncode, log_path=str(log)
                )
                if process.returncode == 0:
                    results[arm + ":" + name].update(
                        json.loads((output / "shadow-image.json").read_text())
                    )
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
                print("ROW", arm, name, process.returncode, flush=True)
                if process.returncode:
                    raise SystemExit(process.returncode)
            result = gate(
                tree,
                revision,
                "tests/test_equilibrium_forward_solve.py",
                args.output / (arm + "-forward.log"),
                platforms=os.environ["JAX_PLATFORMS"],
            )
            results[arm + ":forward"] = result
            (args.output / "results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
