"""Run the promotion gate at the base and the head revision.

    python scripts/iron_graded_refinement/gate.py --base-tree <dir> \
        --head-tree <dir> --base-revision <sha> --head-revision <sha> \
        --output <dir>

Each arm runs in its own extracted tree, with ``PYTHONPATH`` making that tree
the code imported. An arm's log carries its revision, the module actually
imported, its working directory and the command as a shell reads it, so a
reader can reproduce it from the tree alone.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
GATE = [
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "-q",
    "tests/test_artifacts_carry_no_worktree_paths.py",
    "tests/test_imas_iron_read.py",
    "tests/test_magnetisation.py",
    "-m",
    "not slow",
]
SUITE = [
    "-m",
    "pytest",
    "-p",
    "no:cacheprovider",
    "-q",
    "tests/test_magnetisation.py",
    "-m",
    "not slow",
]


def prepare(tree: Path, revision: str, source: Path) -> None:
    """Give an extracted tree an index naming the revision's tracked files.

    The artifact test enumerates candidates with ``git grep``, which needs a
    repository. The tree is not one, so it gets a local index built from the
    source checkout's object store: the object directories are borrowed through
    ``objects/info/alternates`` so nothing is cross-revision-wrong, and
    ``read-tree`` records exactly the paths and blobs the revision commits.
    """

    def run(*arguments: str) -> None:
        subprocess.run(
            ["git", "-C", str(tree), *arguments], check=True, capture_output=True
        )

    objects = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            "objects",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not (tree / ".git").exists():
        run("init", "--quiet")
        alternates = tree / ".git" / "objects" / "info" / "alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(objects + "\n")
    run("read-tree", revision)


def environment(tree: Path) -> dict:
    scratch = os.environ.get("TMPDIR", "/tmp")
    return dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS="cpu",
        MPLCONFIGDIR=str(Path(scratch) / "matplotlib"),
    )


def arm(
    name: str, tree: Path, revision: str, arguments: list[str], output: Path
) -> dict:
    """Run one pytest arm in its own tree and return its receipt."""
    log = output / f"{name}.log"
    env = environment(tree)
    probe = subprocess.run(
        [PYTHON, "-c", "import nova.biot.magnetisation as m; print(m.__file__)"],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
    )
    command = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join([PYTHON, *arguments])
    with log.open("w") as stream:
        stream.write(f"# revision: {revision}\n")
        stream.write(f"# tree: {tree}\n")
        stream.write(f"# module: {probe.stdout.strip()}\n")
        stream.write(f"# cwd: {tree}\n")
        stream.write(f"# command: {command}\n")
        stream.flush()
        done = subprocess.run(
            [PYTHON, *arguments],
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    with log.open("a") as stream:
        stream.write(f"EXIT={done.returncode}\n")
    print(f"{name}: EXIT={done.returncode} -> {log}", flush=True)
    return dict(command=command, exit=done.returncode, log=str(log), tree=str(tree))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-tree", type=Path, required=True)
    parser.add_argument("--head-tree", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--head-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args.base_tree, args.base_revision, args.source)
    prepare(args.head_tree, args.head_revision, args.source)
    record = {}
    for name, tree, revision, arguments in (
        ("gate-base", args.base_tree, args.base_revision, GATE),
        ("gate-head", args.head_tree, args.head_revision, GATE),
        ("baseline-suite", args.base_tree, args.base_revision, SUITE),
        ("after-suite", args.head_tree, args.head_revision, SUITE),
    ):
        if not (tree / "tests" / "test_magnetisation.py").is_file():
            print(f"{name}: tests/test_magnetisation.py absent from {tree}", flush=True)
            continue
        record[name] = arm(name, tree, revision, arguments, args.output)
    (args.output / "gate.json").write_text(json.dumps(record, indent=2) + "\n")
    failed = [name for name, row in record.items() if row["exit"] != 0]
    print("GATE FAILED: " + ", ".join(failed) if failed else "GATE PASSED", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
