"""Run pinned correctness arms and an explicit missing-moment mutation."""

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


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--node-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="booking-correctness-", dir=os.environ["TMPDIR"]
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
        env = dict(
            os.environ,
            PYTHONPATH=str(tree),
            JAX_PLATFORMS="cpu",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
        )
        tests = [
            "tests/test_book_primal_buckets.py",
            "tests/test_booking_scalar_tile.py",
        ]
        if not args.baseline:
            tests.insert(0, "tests/test_fused_booking_assembly.py")
        if args.node_only:
            tests = ["tests/test_fused_booking_assembly.py"]

        def execute(label, selected):
            command = [
                PYTHON,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "-rfP",
                "--basetemp",
                str(tree / (label + "-temp")),
                *selected,
            ]
            log = args.output / (label + ".log")
            literal = f"PYTHONPATH={tree} JAX_PLATFORMS=cpu {shlex.join(command)}"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.revision} tree: {tree} command: {literal}\n"
                )
                stream.write(
                    f"# module: {tree}/nova/equilibrium/current.py\n# cwd: {tree}\n"
                )
                stream.flush()
                subprocess.run(
                    [
                        PYTHON,
                        "-c",
                        "from pathlib import Path; "
                        "from nova.equilibrium import current; "
                        "print('IMPORTED_MODULE', current.__file__, flush=True); "
                        "print('RESOLVED_CWD', Path.cwd().resolve(), flush=True)",
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
            receipt = dict(
                revision=args.revision,
                command=literal,
                exit_status=result.returncode,
                log_path=str(log),
                completed=True,
                failure_ids=[],
            )
            lines = log.read_text().splitlines()
            receipt["failure_ids"] = [
                line.split(" - ")[0].removeprefix("FAILED ")
                for line in lines
                if line.startswith("FAILED ")
            ]
            receipt["failure_count"] = len(receipt["failure_ids"])
            (args.output / (label + ".json")).write_text(json.dumps(receipt, indent=2))
            print("ARM", label, "EXIT", result.returncode, flush=True)
            return result.returncode

        label = (
            "baseline" if args.baseline else "after" if args.node_only else "focused"
        )
        if execute(label, tests):
            raise SystemExit(1)
        if not args.baseline:
            if not args.node_only:
                execute("after", ["tests/test_fused_booking_assembly.py"])
            path = tree / "nova/equilibrium/current.py"
            source = path.read_text()
            marker = "response = jnp.concatenate((primal[None], response), axis=0)"
            assert source.count(marker) == 1
            path.write_text(
                source.replace(
                    marker, "primal = primal.at[1].set(0.0)\n            " + marker
                )
            )
            code = execute(
                "negative",
                [
                    "tests/test_fused_booking_assembly.py",
                    "-k",
                    "fused_moments_and_channels_match_separate_passes",
                ],
            )
            assert code != 0, "dropped radial moment escaped the parity check"


if __name__ == "__main__":
    main()
