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

from provenance import TRACE_HELPER

from scripts.prototypes.diiid_core_hold.batch import (
    CASES,
    case_name,
    extract,
    gate,
    resume_plan,
)

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    revisions = (("base", args.base), ("head", args.head))
    results, retained = {}, {}
    if args.resume_from:
        previous = json.loads((args.resume_from / "results.json").read_text())
        (args.output / "previous-attempts.json").write_text(
            json.dumps(previous, indent=2) + "\n"
        )
        results, retained = resume_plan(args.resume_from, revisions)
        for key, folder in retained.items():
            arm, case = key.split(":", 1)
            for name in (
                "receipt.json",
                "solve-receipt.json",
                "shadow-image.json",
                "stages.json",
                "trips.json",
            ):
                json.loads((folder / name).read_text())
            shadow = json.loads((folder / "shadow-image.json").read_text())
            solve = json.loads((folder / "solve-receipt.json").read_text())
            assert shadow["revision"] == solve["revision"] == dict(revisions)[arm]
            assert shadow["converged"] == solve["converged"]
            target = args.output / arm / case
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(folder.resolve(), target_is_directory=True)
    plan = dict(
        retained={key: str(folder) for key, folder in retained.items()},
        pending=[
            arm + ":" + case_name(frame, branch)
            for arm, _ in revisions
            for frame, branch in CASES
            if arm + ":" + case_name(frame, branch) not in retained
        ],
        pending_forward=[
            arm for arm, _ in revisions if arm + ":forward" not in results
        ],
        trace_helper=str(TRACE_HELPER),
    )
    (args.output / "continuation-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    print("CONTINUATION_PLAN " + json.dumps(plan), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.plan_only:
        return
    with tempfile.TemporaryDirectory(
        prefix="nova-shadow-cohort-", dir=os.environ["TMPDIR"]
    ) as temporary:
        for arm, revision in revisions:
            tree = Path(temporary) / arm
            extract(tree, revision, ROOT)
            for frame, branch in CASES:
                name = case_name(frame, branch)
                if not args.preflight and arm + ":" + name in retained:
                    print("RETAINED", arm, name, flush=True)
                    continue
                if args.preflight:
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("preflight.py")),
                        "--tree",
                        str(tree),
                        "--frame",
                        str(frame),
                        "--branch",
                        branch,
                    ]
                    log = args.output / (arm + "-" + name + "-setup.log")
                    with log.open("w") as stream:
                        stream.write(
                            f"# revision: {revision}; tree: {tree}; "
                            f"command: {shlex.join(command)}\n"
                        )
                        stream.flush()
                        process = subprocess.run(
                            command,
                            cwd=tree,
                            env=dict(
                                os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu"
                            ),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                        stream.write(f"EXIT={process.returncode}\n")
                    setup = None
                    if process.returncode == 0:
                        text = log.read_text()
                        assert "EXTERNAL_TRACE_REFUSED" in text
                        setup = next(
                            json.loads(line.removeprefix("ROW_SETUP_READY "))
                            for line in text.splitlines()
                            if line.startswith("ROW_SETUP_READY ")
                        )
                    results[arm + ":" + name] = dict(
                        setup_only=True,
                        setup_receipt=setup,
                        revision=revision,
                        exit_status=process.returncode,
                        log_path=str(log),
                    )
                    (args.output / "results.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )
                    if process.returncode:
                        raise SystemExit(process.returncode)
                    print("SETUP_COMPLETE", arm, name, flush=True)
                    continue
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
                    str(TRACE_HELPER),
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
            if args.preflight or arm + ":forward" in results:
                continue
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
