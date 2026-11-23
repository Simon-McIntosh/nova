"""Stamp the revision a measurement ran at, refusing a tree it cannot vouch for.

A recorded revision only describes the code that ran when the tree it names
was clean at the start and had not moved by the time the record was composed.
Both halves are checked here rather than assumed, so a measurement cannot
publish a revision that belongs to some other commit.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _git(tree, *arguments):
    completed = subprocess.run(
        ["git", "-C", str(tree), *arguments],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {tree}: {completed.stderr.strip()}"
        )
    return completed.stdout


def head(tree):
    return _git(tree, "rev-parse", "HEAD").strip()


def tracked_changes(tree):
    return _git(tree, "status", "--porcelain", "--untracked-files=no").strip()


def stamped_revision(tree):
    changes = tracked_changes(tree)
    if changes:
        raise RuntimeError(
            "the tracked tree is dirty, so a revision recorded now would not be "
            f"the code that runs:\n{changes}"
        )
    return head(tree)


def unchanged(tree, expected):
    current = head(tree)
    if current != expected:
        raise RuntimeError(
            f"HEAD moved during the measurement, {expected} -> {current}, so the "
            "record would name a revision that did not compose it"
        )
    return current


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("--expect", default=None)
    args = parser.parse_args()
    try:
        if args.expect:
            print(unchanged(args.tree, args.expect))
        else:
            print(stamped_revision(args.tree))
    except RuntimeError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
