"""Run the two scoped suite arms from immutable source extractions."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    command = (
        f"PYTHONPATH=. JAX_PLATFORMS=cpu {python} "
        "-m pytest -p no:cacheprovider -q tests/test_magnetisation.py"
    )
    scratch = Path(tempfile.mkdtemp(prefix="iron-suite-", dir=os.environ["TMPDIR"]))
    try:
        for arm in ("baseline", "after"):
            revision = args.base
            if arm == "after":
                signal = args.run / "suite-head"
                while not signal.exists():
                    time.sleep(10)
                revision = signal.read_text().strip()
            tree = scratch / arm
            tree.mkdir()
            archive = subprocess.Popen(
                [
                    "git",
                    "-C",
                    str(args.repo),
                    "archive",
                    revision,
                    "nova",
                    "tests",
                    "scripts",
                    "pyproject.toml",
                    "conftest.py",
                ],
                stdout=subprocess.PIPE,
            )
            with tarfile.open(fileobj=archive.stdout, mode="r|") as tar:
                tar.extractall(tree, filter="data")
            assert archive.wait() == 0
            env = os.environ.copy()
            env.update(
                PYTHONPATH=".",
                JAX_PLATFORMS="cpu",
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
                TMPDIR=str(scratch),
            )
            log = args.run / f"{arm}-suite.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {revision} tree: {tree} command: {command}\n"
                )
                stream.flush()
                subprocess.run(
                    [
                        python,
                        "-c",
                        (
                            "from pathlib import Path; "
                            "from nova.biot import magnetisation as m; "
                            'print("MODULE="+m.__file__+" CWD="'
                            '+str(Path.cwd().resolve()),flush=True)'
                        ),
                    ],
                    cwd=tree,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=tree,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                stream.write(f"\nEXIT={result.returncode}\n")
            content = log.read_text()
            failures = [
                line.split(" - ")[0][7:]
                for line in content.splitlines()
                if line.startswith("FAILED ")
            ]
            receipt = dict(
                revision=revision,
                command=command,
                exit_status=result.returncode,
                log_path=str(log),
                completed=(" passed" in content or " failed" in content),
                failure_count=len(failures),
                failure_ids=failures,
                measurement_module=str(tree / "nova/biot/magnetisation.py"),
                measurement_cwd=str(tree),
            )
            (args.run / f"{arm}-suite.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            print(arm, json.dumps(receipt), flush=True)
    finally:
        shutil.rmtree(scratch)
        print("Removed own scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
