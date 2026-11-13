"""Show that the two mesh gates fail on a deliberately broken mesh.

Two mutants are built from one extracted tree of the revision under test:

``under-resolved``
    The accuracy arm is shifted one grading coarser, so the field whose
    accuracy the test asserts is the one it expects to be too coarse: the
    0.5 percent assertion must fail.
``perturbed-uniform``
    The uniform branch of ``_mesh`` steps one part per million longer, so the
    recorded grid digests must fail to reproduce.

    python scripts/iron_graded_refinement/negative_control.py \
        --head-tree <dir> --output <dir>

Both mutants write to one log, so a reader sees which check each one breaks.
The exit status is zero only when both fail for the declared reason.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
ACCURACY = (
    "tests/test_magnetisation.py"
    "::test_graded_shell_reaches_half_percent_where_its_doubled_arm_misses"
)
DIGEST = (
    "tests/test_magnetisation.py::test_uniform_cell_size_reproduces_the_recorded_grid"
)
MUTATIONS = {
    "under-resolved": dict(
        file="tests/test_magnetisation.py",
        old=(
            "    error, count = graded_shell_error(graded_shell_field(), capsys)\n"
            "    doubled, doubled_count = "
            "graded_shell_error(graded_shell_field(2.0), capsys)\n"
        ),
        new=(
            "    error, count = graded_shell_error(graded_shell_field(2.0), capsys)\n"
            "    doubled, doubled_count = "
            "graded_shell_error(graded_shell_field(4.0), capsys)\n"
        ),
        node=ACCURACY,
        platforms="cuda",
        marker="0.5 percent at every probe",
    ),
    "perturbed-uniform": dict(
        file="nova/biot/magnetisation.py",
        old="np.arange(r0, r1, size)",
        new="np.arange(r0, r1, size * 1.000001)",
        node=DIGEST,
        platforms="cpu",
        marker="assert mesh_digest(cells) == digest",
    ),
}


def mutate(tree: Path, spec: dict) -> None:
    """Apply one mutation, refusing unless it lands exactly once."""
    path = tree / spec["file"]
    text = path.read_text()
    count = text.count(spec["old"])
    if count != 1:
        raise SystemExit(f"{path}: mutation matched {count} times, expected one")
    path.write_bytes(text.replace(spec["old"], spec["new"]).encode())


def run(name: str, tree: Path, spec: dict, stream) -> dict:
    env = dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS=spec["platforms"],
        MPLCONFIGDIR=str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
    )
    command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", spec["node"]]
    stream.write(f"# mutation: {name}\n")
    stream.write(f"# tree: {tree}\n")
    stream.write(f"# file: {spec['file']}\n")
    stream.write(f"# change: {spec['old']!r} -> {spec['new']!r}\n")
    stream.write(
        f"# command: PYTHONPATH=. JAX_PLATFORMS={spec['platforms']} "
        f"{' '.join(command)}\n"
    )
    stream.flush()
    done = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True)
    output = done.stdout + done.stderr
    stream.write(output)
    stream.write(f"EXIT={done.returncode}\n\n")
    return dict(
        exit=done.returncode,
        marker=spec["marker"] in output,
        node=spec["node"],
        file=spec["file"],
        change=f"{spec['old']!r} -> {spec['new']!r}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")))
    log = args.output / "negative_control.log"
    record = dict(revision=args.revision, mutants={})
    with log.open("w") as stream:
        stream.write(f"# revision: {args.revision}\n")
        stream.write(f"# head tree: {args.head_tree}\n")
        stream.flush()
        for name, spec in MUTATIONS.items():
            tree = scratch / name
            if tree.exists():
                shutil.rmtree(tree)
            shutil.copytree(args.head_tree, tree, symlinks=True)
            mutate(tree, spec)
            record["mutants"][name] = run(name, tree, spec, stream)
            row = record["mutants"][name]
            print(f"{name}: EXIT={row['exit']} marker={row['marker']}", flush=True)
        broken = [
            name
            for name, row in record["mutants"].items()
            if row["exit"] == 0 or not row["marker"]
        ]
        record["verdict"] = (
            "control holds" if not broken else "control broken: " + ", ".join(broken)
        )
        stream.write(f"VERDICT {record['verdict']}\n")
    (args.output / "negative_control.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    shutil.rmtree(scratch)
    print(record["verdict"], flush=True)
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
