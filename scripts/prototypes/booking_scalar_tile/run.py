"""Run paired arms in fresh processes and retain import provenance and exit codes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
FILES = (
    "test_bank_step_booking.py",
    "test_support_read.py",
    "test_booking_scalar_tile.py",
)


def capture(command, tree, revision, log, **extra):
    env = dict(os.environ, PYTHONPATH=str(tree), JAX_PLATFORMS="cpu", **extra)
    rendered = "PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command)
    with log.open("w") as stream:
        stream.write(f"# revision: {revision} tree: {tree} command: {rendered}\n")
        stream.flush()
        subprocess.run(
            [
                PYTHON,
                "-P",
                "-c",
                "from pathlib import Path; from nova.equilibrium import current; "
                "print('MODULE', current.__file__, 'CWD', Path.cwd().resolve(), flush=True)",
            ],
            cwd=tree,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
        code = subprocess.run(
            command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT
        ).returncode
        stream.write(f"EXIT={code}\n")
    print(log.name, "EXIT", code, flush=True)
    return code


def archive(root, revision, destination):
    destination.mkdir()
    process = subprocess.Popen(
        [
            "git",
            "-C",
            str(root),
            "archive",
            revision,
            "nova",
            "scripts",
            "tests",
            "benchmarks",
            "pyproject.toml",
            "conftest.py",
        ],
        stdout=subprocess.PIPE,
    )
    with tarfile.open(fileobj=process.stdout, mode="r|") as source:
        source.extractall(destination, filter="data")
    assert process.wait() == 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["suites", "measure", "negative"])
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        "INSTRUMENT",
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        flush=True,
    )
    with tempfile.TemporaryDirectory(
        prefix="scalar-tile-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        if args.mode == "negative":
            mutant = scratch / "mutant"
            archive(root, head, mutant)
            path = mutant / "nova/equilibrium/current.py"
            text = path.read_text()
            before, body = text.split("def _integrate_bucket(", 1)
            body, after = body.split("\ndef book(", 1)
            assert body.count("(count + width - 1) // width") == 1
            body = body.replace("(count + width - 1) // width", "count // width")
            path.write_text(
                before + "def _integrate_bucket(" + body + "\ndef book(" + after
            )
            shutil.copyfile(path, run / "negative-current.py")
            code = capture(
                [
                    PYTHON,
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    "-q",
                    "tests/test_booking_scalar_tile.py",
                    "-k",
                    "independent_population",
                ],
                mutant,
                head + "+drop-partial-tile",
                run / "negative-control.log",
            )
            assert code != 0
            return
        base = scratch / "base"
        archive(root, args.base, base)
        if args.mode == "suites":
            receipts = {}
            for arm, tree, revision in (
                ("base", base, args.base),
                ("head", root, head),
            ):
                for name in FILES:
                    if not (tree / "tests" / name).exists():
                        continue
                    log = run / f"{arm}-{name}.log"
                    command = [
                        PYTHON,
                        "-m",
                        "pytest",
                        "-p",
                        "no:cacheprovider",
                        "-q",
                        "tests/" + name,
                    ]
                    basetemp = scratch / (arm + "-" + name)
                    basetemp.mkdir()
                    code = capture(command, tree, revision, log, TMPDIR=str(basetemp))
                    receipts[arm + ":" + name] = dict(
                        revision=revision,
                        exit_status=code,
                        command="PYTHONPATH=. JAX_PLATFORMS=cpu " + " ".join(command),
                        log_path=str(log),
                    )
                    (run / "suite-receipts.json").write_text(
                        json.dumps(receipts, indent=2)
                    )
            return
        for process in (1, 2):
            for arm, tree, revision in (
                ("base", base, args.base),
                ("head", root, head),
            ):
                command = [
                    PYTHON,
                    str(root / "scripts/prototypes/booking_scalar_tile/measure.py"),
                    str(tree),
                    "--revision",
                    revision,
                    "--output",
                    str(run / f"{arm}-{process}.json"),
                ]
                code = capture(
                    command, tree, revision, run / f"{arm}-measure-{process}.log"
                )
                assert code == 0
    print("SCRATCH_REMOVED", temporary, flush=True)


if __name__ == "__main__":
    main()
