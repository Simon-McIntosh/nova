"""Run the focused equilibrium and DIII-D tests against the base and the head.

Both arms extract the repository at their own revision, so a test file the
change adds is absent from the base arm's command and a test it edits runs at
the revision it names. Each arm's log opens with the revision, the tree and the
literal command, and closes with the runner's exit status.
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


def arm(name, tree, revision, output):
    present = [path for path in TESTS if (tree / path).is_file()]
    if not present:
        raise SystemExit(f"the {name} arm has no focused test file at {revision}")
    command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", *present]
    log = output / f"{name}-suite.log"
    with log.open("w") as stream:
        stream.write(
            f"# revision: {revision}; tree: {name}; command: {' '.join(command)}\n"
        )
        stream.write(
            f"# module: {tree / 'nova/equilibrium/solve_request.py'}; cwd: {tree}\n"
        )
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
        "log_path": str(log),
        "completed": result.returncode in (0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix="finite-edge-suite-", dir=os.environ["TMPDIR"])
    )
    arms = {}
    for name, revision in (("base", args.base), ("head", args.head)):
        tree = scratch / name
        extract(tree, revision, args.repository)
        arms[name] = arm(name, tree, revision, args.output)
        print(
            f"ARM {name} EXIT={arms[name]['exit_status']} LOG={arms[name]['log_path']}",
            flush=True,
        )
    (args.output / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")


if __name__ == "__main__":
    main()
