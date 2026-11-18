"""Run the four gate rows at the base and the head inside one allocation.

Both arms are extracted from the repository at their own revision, so the
product code each row imports is that revision's and nothing is patched. Rows
are solved one at a time in this allocation (the job holds one card), and each
row's receipt is persisted as it lands so an expiry loses one row and not the
run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROWS = (
    (137, "diverted", "137D"),
    (146, "diverted", "146D"),
    (146, "limited", "146L"),
    (179, "limited", "179L"),
)


def extract(tree, revision, repository):
    """Write one revision's tracked files into a scratch tree."""

    tree.mkdir()
    archive = subprocess.Popen(
        ["git", "-C", str(repository), "archive", revision],
        stdout=subprocess.PIPE,
    )
    unpack = subprocess.run(
        ["tar", "-x", "-C", str(tree)],
        stdin=archive.stdout,
        check=False,
    )
    archive.stdout.close()
    if archive.wait() or unpack.returncode:
        raise SystemExit(f"git archive {revision} did not extract into {tree}")


def completed(folder):
    """A row landed only when its solve and its analysis receipts are on disk."""

    log = folder / "row.log"
    if not log.is_file():
        return False
    for name in ("receipt.json", "core-analysis.json", "built-profile.json"):
        if not (folder / name).is_file():
            return False
    return log.read_text().endswith("EXIT=0\n")


def solve_row(python, tree, revision, frame, branch, output, trace_helper):
    output.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(HERE / "measure.py"),
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
        str(trace_helper),
    ]
    log = output / "row.log"
    with log.open("w") as stream:
        stream.write(
            f"# revision: {revision}; tree: {tree}; command: {' '.join(command)}\n"
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
    return output, result.returncode, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trace-helper",
        type=Path,
        default=None,
        help="the repository source reader, which the row driver loads itself",
    )
    args = parser.parse_args()
    if args.trace_helper is None:
        args.trace_helper = (
            args.repository / "scripts/prototypes/diiid_hollow_current/trace.py"
        )
    python = sys.executable
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix="finite-edge-flux-", dir=os.environ["TMPDIR"])
    )
    trees = {}
    for arm, revision in (("base", args.base), ("head", args.head)):
        tree = scratch / arm
        extract(tree, revision, args.repository)
        trees[arm] = (tree, revision)
        print(f"TREE {arm} {revision} {tree}", flush=True)
    comparison = []
    for arm, (tree, revision) in trees.items():
        for frame, branch, label in ROWS:
            folder = args.output / arm / f"frame-{frame}-{branch}"
            if completed(folder):
                print(f"RETAINED {arm} {label}", flush=True)
            else:
                folder, code, log = solve_row(
                    python,
                    tree,
                    revision,
                    frame,
                    branch,
                    folder,
                    args.trace_helper,
                )
                print(f"ROW {arm} {label} EXTRACT_ID={code} LOG={log}", flush=True)
                if code:
                    raise SystemExit(code)
            receipt = json.loads((folder / "receipt.json").read_text())
            built = json.loads((folder / "built-profile.json").read_text())
            comparison.append(
                dict(
                    arm=arm,
                    label=label,
                    frame=frame,
                    branch=branch,
                    **receipt,
                    built_profile=built,
                )
            )
            (args.output / "comparison.json").write_text(
                json.dumps(comparison, indent=2) + "\n"
            )
    print(
        "BATCH_COMPLETE "
        + json.dumps([row["label"] + ":" + row["arm"] for row in comparison]),
        flush=True,
    )


if __name__ == "__main__":
    main()
