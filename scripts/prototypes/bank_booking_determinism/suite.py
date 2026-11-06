"""Run paired file-isolated CPU suites and retain complete command receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

FILES = (
    "test_bank_step_booking_batch",
    "test_bank_step_booking",
    "test_conic_endpoint_consistency",
    "test_bank_solve_driver",
    "test_support_read",
)
PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    parser.add_argument("--completed-dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="booking-suite-") as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        base.mkdir()
        archive = subprocess.Popen(
            [
                "git",
                "-C",
                str(args.tree),
                "archive",
                args.base,
                "nova",
                "scripts",
                "tests",
                "benchmarks",
                "pyproject.toml",
                "conftest.py",
            ],
            stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(base)], stdin=archive.stdout, check=True)
        assert archive.wait() == 0

        def run_arm(arm, tree, revision):
            def run_file(name):
                if args.completed_dir is not None:
                    previous = args.completed_dir / f"{arm}-{name}.log"
                    text = previous.read_text() if previous.exists() else ""
                    lines = text.splitlines()
                    if (
                        lines
                        and lines[0].startswith(f"# revision: {revision} ")
                        and lines[-1].startswith("EXIT=")
                        and re.search(r"^=+ .* in .* =+$", text, re.MULTILINE)
                    ):
                        ids = re.findall(r"^FAILED\s+(\S+::\S+)", text, re.MULTILINE)
                        return dict(
                            command=lines[0].split(" command: ", 1)[1],
                            log_path=str(previous),
                            exit_status=int(lines[-1].split("=", 1)[1]),
                            completed=True,
                            failure_ids=sorted({i[i.index("tests/") :] for i in ids}),
                        )
                command = [
                    PYTHON,
                    "-P",
                    "-m",
                    "pytest",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={scratch / (arm + '-' + name)}",
                    str(tree / "tests" / (name + ".py")),
                    "-m",
                    "slow or not slow",
                    "-vv",
                    "-s",
                ]
                environment = dict(
                    os.environ,
                    PYTHONPATH=str(tree),
                    TMPDIR=str(scratch),
                    JAX_PLATFORMS="cpu",
                    OMP_NUM_THREADS="2",
                    OPENBLAS_NUM_THREADS="2",
                )
                literal = (
                    "env JAX_PLATFORMS=cpu PYTHONPATH="
                    + shlex.quote(str(tree))
                    + " "
                    + shlex.join(command)
                )
                path = args.run / f"{arm}-{name}.log"
                with path.open("w") as log:
                    log.write(
                        f"# revision: {revision} tree: {tree} command: {literal}\n"
                    )
                    log.flush()
                    subprocess.run(
                        [
                            PYTHON,
                            "-P",
                            "-c",
                            "import nova; import nova.equilibrium.current as m; "
                            "from pathlib import Path; "
                            'print("NOVA", nova.__file__, "MODULE", m.__file__, '
                            '"CWD", Path.cwd().resolve(), flush=True)',
                        ],
                        cwd=tree,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    result = subprocess.run(
                        command,
                        cwd=tree,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    log.write(f"EXIT={result.returncode}\n")
                text = path.read_text()
                ids = sorted(
                    set(re.findall(r"^FAILED\s+(\S+::\S+)", text, re.MULTILINE))
                )
                ids = [identity[identity.index("tests/") :] for identity in ids]
                complete = bool(re.search(r"^=+ .* in .* =+$", text, re.MULTILINE))
                return dict(
                    command=literal,
                    log_path=str(path),
                    exit_status=result.returncode,
                    completed=complete,
                    failure_ids=ids,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                rows = list(pool.map(run_file, FILES))
            failures = sorted({value for row in rows for value in row["failure_ids"]})
            receipt = dict(
                revision=revision,
                command="\n".join(row["command"] for row in rows),
                exit_status=max(row["exit_status"] for row in rows),
                log_path=str(args.run / f"{arm}-suite.log"),
                completed=all(row["completed"] for row in rows),
                failure_count=len(failures),
                failure_ids=failures,
                files=rows,
            )
            with Path(receipt["log_path"]).open("w") as out:
                out.write(
                    f"# revision: {revision} tree: {tree} "
                    f"command: {receipt['command']}\n"
                )
                for row in rows:
                    with Path(row["log_path"]).open() as source:
                        shutil.copyfileobj(source, out)
                out.write(
                    f"SUITE completed={receipt['completed']} "
                    f"failure_count={len(failures)}\n"
                )
                out.write(f"EXIT={receipt['exit_status']}\n")
            (args.run / f"{arm}-suite.json").write_text(json.dumps(receipt, indent=2))
            print("SUITE", arm, json.dumps(receipt), flush=True)

        run_arm("base", base, args.base)
        print("REMOVING_SCRATCH", scratch, flush=True)
    print("EXIT=0", flush=True)


if __name__ == "__main__":
    main()
