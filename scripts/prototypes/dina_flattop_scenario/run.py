"""Run independent fresh-process pulse arms inside one H200 allocation.

The arm/log/receipt contract follows bank_h200_slice_profile.run; the archive
and GPU allocation are supplied once by job.sh for the entire measurement.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from scripts.prototypes.bank_h200_slice_profile.run import PYTHON


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(revision=args.revision, arms=[], complete=False, agreement={})

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    for width in (1, 16):
        records = []
        for process in range(3):
            destination = args.output / f"width-{width}-process-{process}"
            destination.mkdir(exist_ok=True)
            command = [
                PYTHON,
                "-m",
                "scripts.prototypes.dina_flattop_scenario.measure",
                "--inputs",
                str(args.inputs),
                "--output",
                str(destination),
                "--width",
                str(width),
                "--revision",
                args.revision,
            ]
            if width == 1 and process == 0:
                command.append("--controls")
            log = destination / "gate.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.revision} tree: {Path.cwd()} command: "
                    f"PYTHONPATH={os.environ['PYTHONPATH']} {shlex.join(command)}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
                stream.write(f"EXIT={result.returncode}\n")
            summary["arms"].append(
                dict(
                    width=width,
                    process=process,
                    log=str(log),
                    exit_status=result.returncode,
                )
            )
            save()
            print("ARM", destination.name, "EXIT", result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f"arm did not return its receipt: {log}")
            records.append(json.loads((destination / "receipt.json").read_text()))
        a, b = records[-2:]
        spread = abs(a["median_warm_ms"] - b["median_warm_ms"]) / min(
            a["median_warm_ms"], b["median_warm_ms"]
        )
        summary["agreement"][width] = dict(
            relative_spread=spread, within_five_percent=spread <= 0.05
        )
        save()
    summary["complete"] = True
    summary["accepted"] = all(
        json.loads((Path(arm["log"]).parent / "receipt.json").read_text())[
            "converged_slices"
        ]
        >= 18
        for arm in summary["arms"]
    ) and all(value["within_five_percent"] for value in summary["agreement"].values())
    save()
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
