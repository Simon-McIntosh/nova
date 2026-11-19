"""Record the suite arms and the mutation control in one allocation.

The node's test file lives in the prototype directory, so it exists at the head
revision and not at the base revision.  Each arm is therefore run with the same
template resolved at its own revision: the base arm runs the path inside a tree
extracted from the base commit, where pytest reports the file absent, and the
head arm runs it in the job's own checkout.  The mutation control then shows
the psi-identity guard refusing the input the finding describes.

Run on a compute node with the shared interpreter:

    python scripts/prototypes/polygon_symmetry_violation/arms.py --worktree <path>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

BASE_REVISION = "7ee4fa80433045b2d3482cdc48ae3dac6bb5ae47"
TEST = "scripts/prototypes/polygon_symmetry_violation/test_prototypes.py"
CONTROL = "scripts/prototypes/polygon_symmetry_violation/mutation_control.py"


def command(tree, target):
    """The template, resolved at ``tree``."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        target,
    ]


def environment(tree):
    return dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu")


def run(tree, target, label):
    """Run one arm and print its literal command, output and exit status."""
    argv = command(tree, target)
    print("ARM %s" % label, flush=True)
    print(
        "COMMAND PYTHONPATH=. JAX_PLATFORMS=cpu %s" % " ".join(argv[1:]),
        flush=True,
    )
    completed = subprocess.run(
        argv, cwd=str(tree), env=environment(tree), capture_output=True, text=True
    )
    print(completed.stdout.strip(), flush=True)
    if completed.stderr.strip():
        print(completed.stderr.strip(), flush=True)
    print("EXIT=%d" % completed.returncode, flush=True)
    return completed.returncode


def extract(worktree, revision, destination):
    """Extract one revision's tree without touching the working copy."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(worktree), "archive", revision],
        check=True,
        capture_output=True,
    )
    path = destination / "tree.tar"
    path.write_bytes(archive.stdout)
    with tarfile.open(path) as tar:
        tar.extractall(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--directory", default="arms")
    args = parser.parse_args()

    head = Path.cwd()
    base = extract(Path(args.worktree), args.base_revision, Path(args.directory))
    base_status = run(base, TEST, "base")
    head_status = run(head, TEST, "head")

    print("ARM control", flush=True)
    control = subprocess.run(
        [sys.executable, CONTROL],
        cwd=str(head),
        env=environment(head),
        capture_output=True,
        text=True,
    )
    print(control.stdout.strip(), flush=True)
    if control.stderr.strip():
        print(control.stderr.strip(), flush=True)
    print("CONTROL_EXIT=%d" % control.returncode, flush=True)

    print(
        "HEADLINE base_exit %d head_exit %d control_exit %d"
        % (base_status, head_status, control.returncode)
    )


if __name__ == "__main__":
    main()
