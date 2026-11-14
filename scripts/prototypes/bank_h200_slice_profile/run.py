"""Run archived profiling arms sequentially in one scheduler allocation."""

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
HISTORICAL = "8ce11904123bfebd516bc418416a6ec5eeb10117"


PROFILE_TREE_PATHS = (
    "nova",
    "scripts",
    "tests",
    "benchmarks",
    "conftest.py",
    "pyproject.toml",
)


def measurement_environment(output, *, smoke=False):
    """Configure an isolated process without changing the parent environment."""
    env = dict(
        os.environ,
        TMPDIR="/tmp",
        JAX_PLATFORMS="cpu" if smoke else "cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        JAX_COMPILATION_CACHE_DIR=str(output / "compilation-cache"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
    )
    for key in ("UV_NO_SYNC", "UV_RUN_RECURSION_DEPTH", "XLA_FLAGS"):
        env.pop(key, None)
    return env


def archive_revision(root, revision, tree, paths=PROFILE_TREE_PATHS):
    """Extract only the requested paths from a pinned revision."""
    data = subprocess.check_output(
        ["git", "-C", str(root), "archive", revision, *paths]
    )
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(tree, filter="data")


def execute_arm(
    tree,
    instrument,
    destination,
    *,
    inputs,
    width,
    revision,
    instrument_revision,
    environment,
    smoke=False,
    read_only=False,
):
    """Run one fresh process and retain its exit status and receipt locations.

    The caller owns which exit statuses are valid for its measurement family.
    """
    destination.mkdir(exist_ok=True)
    command = [
        PYTHON,
        str(instrument),
        "--output",
        str(destination),
        "--inputs",
        str(inputs),
        "--width",
        str(width),
        "--revision",
        revision,
        "--instrument",
        instrument_revision,
    ]
    if smoke:
        command += ["--smoke"]
    if read_only:
        command += ["--read-only"]
    env = dict(environment, PYTHONPATH=str(tree))
    prefix = f"PYTHONPATH={shlex.quote(str(tree))} "
    if "TF_GPU_ALLOCATOR" in env:
        prefix += f"TF_GPU_ALLOCATOR={shlex.quote(env['TF_GPU_ALLOCATOR'])} "
    log = destination / "gate.log"
    with log.open("w") as stream:
        stream.write(
            f"# revision: {revision} tree: {tree} "
            f"command: {prefix}{shlex.join(command)}\n"
        )
        stream.flush()
        result = subprocess.run(
            command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
        stream.write(f"EXIT={result.returncode}\n")
    return dict(
        exit_status=result.returncode,
        receipt=str(destination / "receipt.json"),
        log=str(log),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    env = measurement_environment(args.output, smoke=args.smoke)
    common = (args.inputs or args.output.parent / "inputs/common-streams.npz").resolve()
    assert common.is_file()
    summary = dict(
        instrument_revision=args.revision, arms=[], complete=False, smoke=args.smoke
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    save()
    with tempfile.TemporaryDirectory(
        prefix="slice-profile-", dir=env["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        env["TMPDIR"] = str(scratch)
        trees = {}
        for label, revision in (("main", args.revision), ("historical", HISTORICAL)):
            tree = scratch / label
            tree.mkdir()
            paths = PROFILE_TREE_PATHS if label == "main" else ("nova",)
            archive_revision(args.root, revision, tree, paths)
            if label == "historical":
                for name in ("scripts", "tests", "benchmarks"):
                    (tree / name).symlink_to(
                        trees["main"] / name, target_is_directory=True
                    )
            trees[label] = tree
        instrument = (
            trees["main"] / "scripts/prototypes/bank_h200_slice_profile/measure.py"
        )

        def execute(label, width, process, read_only=False):
            tree = trees[label]
            destination = (
                args.output
                / f"{label}-{'read' if read_only else 'slice'}-{width}-{process}"
            )
            revision = args.revision if label == "main" else HISTORICAL
            arm = execute_arm(
                tree,
                instrument,
                destination,
                inputs=common,
                width=width,
                revision=revision,
                instrument_revision=args.revision,
                environment=env,
                smoke=args.smoke,
                read_only=read_only,
            )
            summary["arms"].append(
                dict(
                    label=label,
                    width=width,
                    process=process,
                    read_only=read_only,
                    **arm,
                )
            )
            save()
            print("ARM", destination.name, "EXIT", arm["exit_status"], flush=True)
            if arm["exit_status"] not in (0, 77):
                raise RuntimeError(f"arm failed: {arm['log']}")
            return json.loads(Path(arm["receipt"]).read_text())

        for width in [1] if args.smoke else [1, 16, 64, 256, 1024]:
            records = [
                execute("main", width, p) for p in ([1] if args.smoke else [0, 1, 2])
            ]
            if any(r.get("memory_limited") for r in records):
                summary["memory_limit_width"] = width
                break
            if not args.smoke:
                a, b = records[-2:]
                spread = abs(a["end_to_end_ms"] - b["end_to_end_ms"]) / min(
                    a["end_to_end_ms"], b["end_to_end_ms"]
                )
                summary.setdefault("agreement", {})[width] = spread
                save()
                assert spread <= 0.05, (width, "fresh processes disagree", spread)
        for label in ("historical", "main"):
            for process in [1] if args.smoke else [0, 1, 2]:
                execute(label, 1 if args.smoke else 16, process, read_only=True)
        summary["complete"] = True
        save()
    print("SUMMARY all requested arms complete", flush=True)


if __name__ == "__main__":
    main()
