"""Measure named source revisions in fresh processes within one allocation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-read-only", action="store_true")
    parser.add_argument("revisions", nargs="+")
    arguments = parser.parse_args()
    root = Path.cwd()
    measure = Path(__file__).with_name("measure.py").resolve()
    arguments.output.mkdir(parents=True, exist_ok=True)
    for name in arguments.revisions:
        revision = subprocess.check_output(
            ["git", "rev-parse", name], text=True
        ).strip()
        with tempfile.TemporaryDirectory(
            prefix="seed-history-", dir=os.environ["TMPDIR"]
        ) as temporary:
            tree = Path(temporary)
            archive = tree / "source.tar"
            with archive.open("wb") as stream:
                subprocess.run(
                    ["git", "archive", revision, "nova", "benchmarks"],
                    stdout=stream,
                    check=True,
                )
            with tarfile.open(archive) as source:
                source.extractall(tree, filter="data")
            for directory in ("docs", "data", "input"):
                (tree / directory).symlink_to(
                    root / directory, target_is_directory=True
                )
            command = [
                os.sys.executable,
                str(measure),
                "--revision",
                revision,
                "--measure-only",
                "--output",
                str(arguments.output / f"{revision}.json"),
            ]
            if arguments.source_read_only:
                command.append("--source-read-only")
            environment = {**os.environ, "PYTHONPATH": str(tree)}
            log = arguments.output / f"{revision}.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {revision}\n# cwd: {tree}\n# command: {command!r}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=tree,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                stream.write(f"EXIT={result.returncode}\n")
            print(f"MEASURED {revision} EXIT={result.returncode} LOG={log}", flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
