"""Run paired immutable source arms, persisting every completed row."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_warmup_admission"))
from measure import extract  # noqa: E402
from regression import gate  # noqa: E402

CASES = (
    (137, "diverted"),
    (146, "diverted"),
    (146, "limited"),
    (179, "limited"),
    (35, "mast"),
)


def case_name(frame, branch):
    return "mast-21978-35" if branch == "mast" else f"frame-{frame}-{branch}"


def completed_row(record, revision):
    if not record or record.get("exit_status") != 0:
        return None
    log = Path(record["log_path"])
    if not log.is_file() or not log.read_text().endswith("EXIT=0\n"):
        return None
    folder = log.parent
    for name in (
        "receipt.json",
        "core-analysis.json",
        "terminal.npz",
        "source-inputs.json",
    ):
        if not (folder / name).is_file():
            raise ValueError(f"completed row has no {name}: {folder}")
    receipt = json.loads((folder / "receipt.json").read_text())
    analysis = json.loads((folder / "core-analysis.json").read_text())
    if "maxima_count" not in analysis or "source_centre_to_shoulder" not in analysis:
        raise ValueError(
            f"completed row has no critical-point/source analysis: {folder}"
        )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "--quiet",
            receipt["base_revision"],
            revision,
            "--",
            "nova",
            "tests",
        ],
        cwd=ROOT,
        check=True,
    )
    return folder.resolve()


def resume_plan(previous, revisions):
    summary = json.loads((previous / "results.json").read_text())
    retained = {}
    for arm, revision in revisions:
        for frame, branch in CASES:
            key = arm + ":" + case_name(frame, branch)
            folder = completed_row(summary.get(key), revision)
            if folder is not None:
                retained[key] = folder
            else:
                summary.pop(key, None)
        key = arm + ":forward"
        record = summary.get(key)
        if record:
            log = Path(record["log_path"])
            if not (
                record.get("completed")
                and log.is_file()
                and log.read_text().endswith(f"EXIT={record['exit_status']}\n")
            ):
                summary.pop(key)
    return summary, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-helper", type=Path, required=True)
    parser.add_argument("--cpu-files", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    revisions = (("base", args.base), ("head", args.head))
    summary, retained = {}, {}
    if args.resume_from:
        if args.cpu_files:
            raise ValueError("solver continuation cannot include CPU regression files")
        summary, retained = resume_plan(args.resume_from, revisions)
    plan = dict(
        retained={key: str(path) for key, path in retained.items()},
        pending=[
            arm + ":" + case_name(frame, branch)
            for arm, _ in revisions
            for frame, branch in CASES
            if arm + ":" + case_name(frame, branch) not in retained
        ],
        pending_forward=[
            arm for arm, _ in revisions if arm + ":forward" not in summary
        ],
    )
    print("CONTINUATION_PLAN " + json.dumps(plan), flush=True)
    if args.plan_only:
        return
    args.output.mkdir(parents=True, exist_ok=not bool(args.resume_from))
    for key, folder in retained.items():
        arm, case = key.split(":", 1)
        target = args.output / arm / case
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(folder, target_is_directory=True)
    (args.output / "continuation-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with tempfile.TemporaryDirectory(
        prefix="nova-core-hold-", dir=os.environ["TMPDIR"]
    ) as scratch:
        for arm, revision in revisions:
            if (
                all(
                    arm + ":" + case_name(frame, branch) in retained
                    for frame, branch in CASES
                )
                and arm + ":forward" in summary
            ):
                continue
            tree = Path(scratch) / arm
            extract(tree, revision, ROOT)
            if args.cpu_files:
                files = json.loads(args.cpu_files.read_text())
                for file in files:
                    if not (tree / file).is_file():
                        continue
                    result = gate(
                        tree,
                        revision,
                        file,
                        args.output / (arm + "-" + Path(file).stem + ".log"),
                        quiet=True,
                    )
                    summary[arm + ":" + file] = result
                    (args.output / "results.json").write_text(
                        json.dumps(summary, indent=2) + "\n"
                    )
                continue
            for frame, branch in CASES:
                case = case_name(frame, branch)
                if arm + ":" + case in retained:
                    continue
                output = args.output / arm / case
                output.mkdir(parents=True, exist_ok=True)
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
                    result = subprocess.run(
                        command,
                        cwd=tree,
                        env=dict(os.environ, PYTHONPATH=str(tree)),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    stream.write(f"EXIT={result.returncode}\n")
                summary[arm + ":" + case] = dict(
                    exit_status=result.returncode, log_path=str(log)
                )
                (args.output / "results.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print("ROW", arm, case, result.returncode, flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)
            if arm + ":forward" in summary:
                continue
            result = gate(
                tree,
                revision,
                "tests/test_equilibrium_forward_solve.py",
                args.output / (arm + "-forward.log"),
                platforms=os.environ["JAX_PLATFORMS"],
            )
            summary[arm + ":forward"] = result
            (args.output / "results.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
