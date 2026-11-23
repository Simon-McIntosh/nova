"""Run isolated census rows sequentially inside one scheduler allocation."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("measure.py").resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = [
        (variant, order, "derivative")
        for order in range(5)
        for variant in ("current", "plain", "without_samples", "masked_samples")
    ]
    rows += [("current", 2, "curvature")]
    records = []
    for variant, order, mode in rows:
        stem = f"{mode}-{variant}-{order}"
        command = [
            sys.executable,
            str(script),
            "--variant",
            variant,
            "--order",
            str(order),
            "--mode",
            mode,
            "--output",
            str(args.output / f"{stem}.json"),
        ]
        log = args.output / f"{stem}.log"
        print(f"START {stem} {time.time()}", flush=True)
        with log.open("w") as stream:
            stream.write(
                f"revision={revision} tree={Path.cwd().resolve()} "
                f"command={shlex.join(command)}\n"
            )
            stream.flush()
            try:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, timeout=900
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
            stream.write(f"\nEXIT={code}\n")
        records.append(
            {
                "variant": variant,
                "order": order,
                "mode": mode,
                "exit": code,
                "log": str(log),
            }
        )
        (args.output / "index.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"END {stem} EXIT={code}", flush=True)
        if code not in (0, 124):
            raise SystemExit(code)
    print("CENSUS_ROWS_FINISHED", flush=True)


if __name__ == "__main__":
    main()
