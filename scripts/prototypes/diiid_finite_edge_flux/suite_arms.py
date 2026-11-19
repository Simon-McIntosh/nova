"""Run a named test set against the base revision and the head.

Both arms extract the repository at their own revision, so a test file the
change adds is absent from the base arm's command and a test it edits runs at
the revision it names. Each arm's log opens with the revision, the tree and the
literal command, and closes with the runner's exit status.

``--tests`` names the set, so a caller in another prototype directory records
its own arms through this runner instead of writing a second one. An arm whose
whole set is absent at its revision records ``ABSENT`` with pytest's
file-not-found status and names no command to run; ``--absent-ok`` marks the
revisions where that is the expected result, and any other red arm makes the
runner exit nonzero.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import tempfile

ROOT = Path(__file__).resolve().parents[3]
TESTS = (
    "tests/equilibrium/test_finite_edge_flux_function.py",
    "tests/equilibrium/test_finite_edge_booking.py",
    "tests/test_sampled_profile_core_hold.py",
    "tests/test_sampled_flux_batching.py",
    "tests/test_batched_operator_boundary.py",
)
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
ABSENT = 4


def extract(tree, revision, repository):
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


def arm(name, tree, revision, output, tests=TESTS):
    present = [path for path in tests if (tree / path).is_file()]
    command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", *present]
    log = output / f"{name}-suite.log"
    if not present:
        absent = ", ".join(tests)
        log.write_text(
            f"# revision: {revision}; tree: {name}; cwd: {tree}\n"
            f"# ABSENT: no test file of this set exists at this revision ({absent})\n"
            f"EXIT={ABSENT}\n"
        )
        return {
            "revision": revision,
            "command": None,
            "exit_status": ABSENT,
            "absent": True,
            "log_path": str(log),
            "completed": False,
        }
    with log.open("w") as stream:
        stream.write(
            f"# revision: {revision}; tree: {name}; command: {' '.join(command)}\n"
        )
        stream.write(f"# cwd: {tree}\n")
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
    return {
        "revision": revision,
        "command": " ".join(command),
        "exit_status": result.returncode,
        "absent": False,
        "log_path": str(log),
        "completed": result.returncode in (0, 1),
    }


def verdict(arms, absent_ok):
    """Whether every arm is either green or absent where absence is expected."""
    red = [
        name
        for name, record in arms.items()
        if record["exit_status"] != 0 and not (record["absent"] and name in absent_ok)
    ]
    return red


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", default=list(TESTS))
    parser.add_argument(
        "--absent-ok",
        nargs="*",
        default=[],
        choices=("base", "head"),
        help="arms whose whole test set is expected to be absent at their revision",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="suite-arms-", dir=os.environ["TMPDIR"]))
    arms = {}
    for name, revision in (("base", args.base), ("head", args.head)):
        tree = scratch / name
        extract(tree, revision, args.repository)
        arms[name] = arm(name, tree, revision, args.output, tests=args.tests)
        print(
            f"ARM {name} EXIT={arms[name]['exit_status']} LOG={arms[name]['log_path']}",
            flush=True,
        )
        if arms[name]["command"]:
            print(f"COMMAND {name} {arms[name]['command']}", flush=True)
        else:
            print(f"COMMAND {name} (set absent at {revision})", flush=True)
    (args.output / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    red = verdict(arms, set(args.absent_ok))
    print(f"SUITE_ARMS red={','.join(red) if red else 'none'}", flush=True)
    if red:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
