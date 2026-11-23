"""Run one suite arm per revision, through the package's own arm runner.

Each arm runs in a tree exported from the arm's own revision, so its command
resolves over that revision rather than over the checkout's head, and each is run
and logged by ``run_suite_arm.sh``, the one writer of the header and trailer that
``records.suite_arm_log`` reads back. Every log is read through that reader after
the arm finishes, so a format the package cannot parse fails here rather than at
promotion.

The exported trees are made under ``$TMPDIR`` (``/tmp`` when unset), which is the
node-local disk a job's scratch already points at, and each is removed once its
arm's log has been read.

    suite_arms.py <worktree> <log directory> <base revision> <head revision>

Prints one ``SUITE_ARM`` line per arm naming its revision, tree, passed count and
exit status, and writes the same facts to ``<log directory>/suite-arms.json``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import records  # noqa: E402

FILES = (
    "tests/test_stored_state_refusals.py",
    "scripts/prototypes/stored_state_refusals/test_panel_visibility.py",
    "scripts/prototypes/stored_state_refusals/test_receipt_durability.py",
    "scripts/prototypes/stored_state_refusals/test_record_claims.py",
)
RUNNER = HERE / "run_suite_arm.sh"


def _export(worktree: Path, revision: str, destination: Path) -> None:
    """Write the tree of one revision into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(worktree), "archive", revision],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-C", str(destination), "-x"], input=archive.stdout, check=True
    )


def _run_arm(
    worktree: Path, revision: str, label: str, log_dir: Path, root: Path
) -> dict:
    """Export one revision, run the arm through the wrapper, and read its log."""
    tree = root / label
    _export(worktree, revision, tree)
    targets = [name for name in FILES if (tree / name).is_file()]
    if not targets:
        raise SystemExit(f"{label}: none of this node's test files exist at {revision}")
    log = log_dir / f"suite-{label}.log"
    if log.exists():
        log.unlink()
    subprocess.run(
        [
            str(RUNNER),
            "--log",
            str(log),
            "--tree",
            str(tree),
            "--revision",
            revision,
            "-p",
            "no:cacheprovider",
            "-q",
            *targets,
        ],
        capture_output=True,
        text=True,
    )
    read = records.suite_arm_log(log)
    if read["revision"] != revision:
        raise SystemExit("log names revision %s, not %s" % (read["revision"], revision))
    if read["tree"] != str(tree):
        raise SystemExit("log names tree %s, not %s" % (read["tree"], tree))
    missing = [name for name in targets if name not in read["command"]]
    if missing:
        raise SystemExit("log's command omits %s" % missing)
    passed = records.suite_arm_passed(read)
    print(
        "# revision %s\n# command: %s\n# tree: %s"
        % (read["revision"], read["command"], read["tree"]),
        flush=True,
    )
    for line in read["body"][-6:]:
        print(line, flush=True)
    print(f"EXIT={read['exit_status']}", flush=True)
    return dict(
        label=label,
        revision=revision,
        tree=str(tree),
        command=read["command"],
        targets=targets,
        exit_status=read["exit_status"],
        passed=passed,
        log=str(log),
    )


def main() -> int:
    worktree, log_dir, base, head = sys.argv[1:5]
    worktree = Path(worktree).resolve()
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("TMPDIR") or "/tmp")
    root = Path(tempfile.mkdtemp(prefix="suite-arms-", dir=scratch))
    try:
        arms = [
            _run_arm(worktree, base, "base", log_dir, root),
            _run_arm(worktree, head, "head", log_dir, root),
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    (log_dir / "suite-arms.json").write_text(
        json.dumps(dict(scratch=str(scratch), arms=arms), indent=1) + "\n"
    )
    for arm in arms:
        print(
            "SUITE_ARM %s revision=%s passed=%s exit=%d tree=%s log=%s"
            % (
                arm["label"],
                arm["revision"],
                arm["passed"],
                arm["exit_status"],
                arm["tree"],
                arm["log"],
            ),
            flush=True,
        )
    bad = [arm for arm in arms if arm["exit_status"] != 0 or arm["passed"] is None]
    if bad:
        print("SUITE_ARMS_FAILED " + json.dumps([a["label"] for a in bad]), flush=True)
        return 1
    print("SUITE_ARMS_FINISHED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
