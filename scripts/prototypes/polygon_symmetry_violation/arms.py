"""Record the node's suite arms and close with a verdict line.

Both arms extract the repository at their own revision, so a test file the
change adds is absent from the base arm's command and the head arm runs the
path it names. Each arm writes its own log opening with the revision, the tree
and the literal command and closing with the runner's exit status; an arm whose
whole set is absent records that instead of running an empty command.

The driver's own exit status is the verdict: zero when every arm is green or
absent where ``--absent-ok`` says absence is expected, one otherwise, so a log
that ends ``EXIT=1`` is a red run and a gate reading it cannot mistake one for
a pass. Scratch trees are extracted under ``$TMPDIR`` so they die with the run's
scratch instead of landing in the working directory.

Run on a compute node with the shared interpreter:

    python scripts/prototypes/polygon_symmetry_violation/arms.py \
        --repository <worktree> --base <sha> --head <sha> \
        --tests scripts/prototypes/polygon_symmetry_violation/test_prototypes.py \
        --absent-ok base --output <run>/suite-arms
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
ABSENT = 4


def extract(tree, revision, repository):
    """Extract one revision into a fresh directory under the run's scratch."""
    tree.mkdir()
    archive = subprocess.Popen(
        ["git", "-C", str(repository), "archive", revision],
        stdout=subprocess.PIPE,
    )
    unpack = subprocess.run(
        ["tar", "-x", "-C", str(tree)], stdin=archive.stdout, check=False
    )
    archive.stdout.close()
    if archive.wait() or unpack.returncode:
        raise SystemExit(f"git archive {revision} did not extract into {tree}")


def arm(name, tree, revision, output, tests):
    """Run one arm and return its record; the log ends with its exit status."""
    present = [path for path in tests if (tree / path).is_file()]
    command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", *present]
    log = output / f"{name}-suite.log"
    if not present:
        log.write_text(
            f"# revision: {revision}; arm {name}; cwd: {tree}\n"
            f"# ABSENT: no test file of this set exists at this revision\n"
            f"EXIT={ABSENT}\n"
        )
        return dict(
            revision=revision,
            command=None,
            exit_status=ABSENT,
            absent=True,
            log_path=str(log),
        )
    with log.open("w") as stream:
        stream.write(f"# revision: {revision}; arm {name}\n")
        stream.write(f"# command: {' '.join(command)}\n")
        stream.write(f"# scratch: {tree}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=tree,
            env=dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu"),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"EXIT={result.returncode}\n")
    return dict(
        revision=revision,
        command=" ".join(command),
        exit_status=result.returncode,
        absent=False,
        log_path=str(log),
    )


def red_arms(arms, absent_ok):
    """The arms that are neither green nor absent where absence is expected."""
    return [
        name
        for name, record in arms.items()
        if record["exit_status"] != 0 and not (record["absent"] and name in absent_ok)
    ]


def run(argv=None):
    """Run both arms, write their logs and the record, and return the verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--tests", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--absent-ok",
        nargs="*",
        default=[],
        choices=("base", "head"),
        help="arms whose whole test set is expected to be absent at its revision",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="symmetry-arms-", dir=os.environ["TMPDIR"]))
    arms = {}
    for name, revision in (("base", args.base), ("head", args.head)):
        tree = scratch / name
        extract(tree, revision, args.repository)
        arms[name] = arm(name, tree, revision, args.output, args.tests)
        print(f"ARM {name} EXIT={arms[name]['exit_status']}", flush=True)
        command = arms[name]["command"] or f"(set absent at {revision})"
        print(f"COMMAND {name} {command}", flush=True)
    (args.output / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    red = red_arms(arms, set(args.absent_ok))
    print(f"SUITE_ARMS red={','.join(red) if red else 'none'}", flush=True)
    return 1 if red else 0


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
