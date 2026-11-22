"""Run the suite arm for every test file this record's node touches, at two revisions.

The base arm runs in a tree exported from the base revision and the after arm in
a tree exported from the head, so each arm's command resolves over that arm's
own tree. Each log names the arm's revision, its tree and the exact command on
one line, carries the pytest output, and ends with the command's own exit
status; the same block is echoed to stdout so the submitting job's log carries
it too.

    suite_arms.py <worktree> <log directory> <base revision> <head revision>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FILES = (
    "tests/test_stored_state_refusals.py",
    "scripts/prototypes/stored_state_refusals/test_panel_visibility.py",
    "scripts/prototypes/stored_state_refusals/test_receipt_durability.py",
)
PYTHON = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")


def _export(worktree: Path, revision: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(worktree), "archive", revision],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        input=archive.stdout,
        check=True,
    )


def _run_arm(worktree: Path, revision: str, label: str, log_dir: Path, root: Path):
    tree = root / label
    _export(worktree, revision, tree)
    targets = [name for name in FILES if (tree / name).is_file()]
    if not targets:
        raise SystemExit(f"{label}: none of this node's test files exist at {revision}")
    command = " ".join(
        [str(PYTHON), "-m", "pytest", "-p", "no:cacheprovider", "-q", *targets]
    )
    header = f"# arm={label} revision={revision} tree={tree} command: {command}"
    environment = {
        "PYTHONPATH": str(tree),
        "JAX_PLATFORMS": "cpu",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "TMPDIR": "/tmp",
    }
    result = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-p", "no:cacheprovider", "-q", *targets],
        cwd=tree,
        env=environment,
        capture_output=True,
        text=True,
    )
    body = result.stdout + result.stderr
    log = log_dir / f"suite-{label}.log"
    log.write_text(header + "\n" + body + f"EXIT={result.returncode}\n")
    print(header, flush=True)
    for line in body.splitlines()[-6:]:
        print(line, flush=True)
    print(f"EXIT={result.returncode}", flush=True)
    return dict(
        label=label,
        revision=revision,
        tree=str(tree),
        command=command,
        targets=targets,
        exit=result.returncode,
        log=str(log),
        passed=_passed(body),
    )


def _passed(body: str) -> int | None:
    for line in reversed(body.splitlines()):
        if " passed" in line:
            token = line.split(" passed")[0].split()[-1]
            try:
                return int(token)
            except ValueError:
                continue
    return None


def main() -> int:
    worktree, log_dir, base, head = sys.argv[1:5]
    worktree = Path(worktree).resolve()
    log_dir = Path(log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="suite-arms-", dir="/tmp"))
    try:
        arms = [
            _run_arm(worktree, base, "base", log_dir, root),
            _run_arm(worktree, head, "head", log_dir, root),
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for arm in arms:
        print(
            "SUITE_ARM %s revision=%s passed=%s exit=%d log=%s"
            % (arm["label"], arm["revision"], arm["passed"], arm["exit"], arm["log"]),
            flush=True,
        )
    return 1 if any(arm["exit"] != 0 for arm in arms) else 0


if __name__ == "__main__":
    sys.exit(main())
