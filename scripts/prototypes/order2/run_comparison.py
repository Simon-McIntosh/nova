"""Run each cold read measurement in a fresh process inside one allocation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("compare.py").resolve()
    for cells in (132, 550):
        for arm in ("native", "batched"):
            log = args.directory / f"{arm}-{cells}.log"
            command = [
                sys.executable,
                "-u",
                str(script),
                "--arm",
                arm,
                "--cells",
                str(cells),
                "--directory",
                str(args.directory),
            ]
            print("CHILD " + json.dumps(command), flush=True)
            env = dict(
                os.environ,
                JAX_ENABLE_COMPILATION_CACHE="false",
                MEASUREMENT_LOG=str(log),
                TMPDIR="/tmp",
            )
            with log.open("w") as stream:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                stream.write(f"\nEXIT={result.returncode}\n")
            print(
                f"ROW_EXIT arm={arm} cells={cells} code={result.returncode}", flush=True
            )
            if result.returncode:
                raise SystemExit(result.returncode)
    print("COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
