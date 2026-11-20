"""Run the matched masked and unmasked cohort in one allocation and read it.

The batch solves 137D, 146D, 146L and 179L on the finite-edge head twice from
identical seeds and source inputs, with the inactive compensator's image
suppressed and restored. This driver then compares the two arms and renders the
masked terminal flux panels, so the whole measurement is one job.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]


def revision(tree):
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()


def run(command, tree):
    print("RUN " + shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=tree)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=TREE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--cards", type=int, default=1)
    args = parser.parse_args()
    tree = args.tree.resolve()
    revision_sha = revision(tree)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = "none"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    args.output.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(HERE / "batch.py"),
            "--tree",
            str(tree),
            "--revision",
            revision_sha,
            "--output",
            str(args.output),
            "--trace-helper",
            str(tree / "scripts/prototypes/diiid_hollow_current/trace.py"),
            "--slots",
            str(args.slots),
            "--cards",
            str(args.cards),
        ],
        tree,
    )
    run(
        [
            sys.executable,
            str(HERE / "compare.py"),
            "--input",
            str(args.output),
            "--tree",
            str(tree),
            "--output",
            str(args.output / "comparison.json"),
        ],
        tree,
    )
    run(
        [
            sys.executable,
            str(HERE / "render.py"),
            "--input",
            str(args.output),
            "--output",
            str(args.figures),
            "--previews",
            str(args.previews),
        ],
        tree,
    )


if __name__ == "__main__":
    main()
