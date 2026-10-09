"""Run cold read rows in fresh processes within one bounded allocation."""

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
    cores = sorted(os.sched_getaffinity(0))
    assert len(cores) >= 2, "two allocated CPUs are needed for the two rung processes"
    for arm in ("native", "batched"):
        running = []
        for core, cells in zip(cores[:2], (132, 550), strict=True):
            log = args.directory / f"{arm}-{cells}.log"
            command = [
                "taskset",
                "--cpu-list",
                str(core),
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
                MEASUREMENT_CONCURRENCY="2",
                MEASUREMENT_CPU=str(core),
            )
            stream = log.open("w")
            process = subprocess.Popen(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            running.append((process, stream, cells))
        codes = []
        for process, stream, cells in running:
            code = process.wait()
            stream.write(f"\nEXIT={code}\n")
            stream.close()
            codes.append(code)
            print(f"ROW_EXIT arm={arm} cells={cells} code={code}", flush=True)
        if any(codes):
            raise SystemExit(1)
    print("COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
