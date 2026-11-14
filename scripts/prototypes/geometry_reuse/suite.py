"""Run revision-pinned CPU files in fresh processes inside one allocation."""

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
FILES = (
    "tests/test_bank_drive_continuation.py",
    "tests/test_bank_lane_drive.py",
    "tests/test_support_read.py",
    "tests/test_tracked_read.py",
    "tests/test_read_wall_skip.py",
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--mutation", action="store_true")
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(revision=args.revision, complete=False, files=[])

    def save():
        (args.output / "suite.json").write_text(json.dumps(record, indent=2))

    save()
    with tempfile.TemporaryDirectory(
        prefix="geometry-cpu-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary)
        payload = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.root),
                "archive",
                args.revision,
                "nova",
                "tests",
                "scripts",
                "benchmarks",
                "conftest.py",
                "pyproject.toml",
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(tree, filter="data")
        if args.mutation:
            path = tree / "nova/equilibrium/topology.py"
            source = path.read_text()
            marker = "    return changed\n"
            assert source.count(marker) == 1
            path.write_text(
                source.replace(marker, "    return jnp.zeros_like(changed)\n")
            )
        own = "tests/test_geometry_reuse.py"
        files = [own] if args.focused or args.mutation else [own, *FILES]
        for name in files:
            if not (tree / name).exists():
                record["scoped_suite"] = "unarmed: test file absent at this revision"
                save()
                continue
            command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", name]
            if name != own:
                command += ["-rfP"]
            command_text = "PYTHONPATH=. JAX_PLATFORMS=cpu " + shlex.join(command)
            log = args.output / (Path(name).stem + ".log")
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.revision} tree: {tree} "
                    f"command: {command_text}\n"
                )
                stream.flush()
                env = dict(
                    os.environ,
                    TMPDIR=str(tree),
                    PYTHONPATH=str(tree),
                    JAX_PLATFORMS="cpu",
                    OMP_NUM_THREADS="4",
                    OPENBLAS_NUM_THREADS="4",
                )
                subprocess.run(
                    [
                        PYTHON,
                        "-c",
                        "import os; from nova.equilibrium import topology; "
                        "print('MODULE', topology.__file__); "
                        "print('CWD', os.getcwd())",
                    ],
                    cwd=tree,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                result = subprocess.run(
                    command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                stream.write(f"EXIT={result.returncode}\n")
            text = log.read_text()
            failed = [
                line.split(" - ", 1)[0].split(" ", 1)[1]
                for line in text.splitlines()
                if line.startswith("FAILED ") or line.startswith("ERROR ")
            ]
            record["files"].append(
                dict(
                    revision=args.revision,
                    command=command_text,
                    exit_status=result.returncode,
                    completed=result.returncode in (0, 1),
                    failure_count=len(failed),
                    failure_ids=failed,
                    log_path=str(log),
                    measurement_module=str(tree / "nova/equilibrium/topology.py"),
                    measurement_cwd=str(tree),
                )
            )
            save()
            print("FILE", name, "EXIT", result.returncode, flush=True)
        record["complete"] = True
        save()
        print("REMOVING job-owned scratch", tree, flush=True)


if __name__ == "__main__":
    main()
