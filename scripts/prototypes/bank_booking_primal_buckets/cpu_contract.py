"""Verify host byte pins and challenge the fixed scalar cut-cell tile."""

import argparse
import difflib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_booking_primal_buckets.suite import run_file
from scripts.prototypes.bank_wall_events.suite import archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("host", choices=("fleet", "sirius"))
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    command = shlex.join([sys.executable, "-P", "-m", __spec__.name, *sys.argv[1:]])
    print(f"# revision: {revision} tree: {args.root} command: {command}", flush=True)
    import nova

    print("nova", nova.__file__, "cwd", Path.cwd().resolve(), flush=True)
    print("HOST", platform.node(), flush=True)
    print(subprocess.check_output(["lscpu"], text=True), flush=True)
    names = ("test_bank_step_booking",)
    if args.host == "sirius":
        names = (
            "test_book_primal_buckets",
            "test_bank_step_booking",
            "test_bank_step_booking_batch",
        )
    with tempfile.TemporaryDirectory(
        prefix="booking-contract-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        results = [
            run_file(args.root, revision, name, args.host, args.run, scratch)
            for name in names
        ]
        logs = [args.run / f"{args.host}-{name}.log" for name in names]
        summaries, failures = [], []
        for log in logs:
            text = log.read_text()
            summary = re.findall(r"^=+ .* (?:passed|failed|skipped).* =+$", text, re.M)
            assert summary, log
            summaries.append(summary[-1].strip("= "))
            failures += re.findall(r"^FAILED (\S+)", text, re.M)
        receipt = dict(
            revision=revision,
            command=command,
            exit_status=int(any(results)),
            log_path=str(args.run / f"{args.host}-suite.log"),
            completed=True,
            failure_count=len(failures),
            failure_ids=failures,
        )
        with Path(receipt["log_path"]).open("w") as handle:
            handle.write(
                f"# revision: {revision} tree: {args.root} command: {command}\n"
            )
            for log in logs:
                handle.write(log.read_text() + "\n")
            handle.write("SUMMARY " + json.dumps(receipt) + "\n")
            handle.write(f"EXIT={receipt['exit_status']}\n")
        (args.run / f"{args.host}-suite.json").write_text(json.dumps(receipt, indent=2))
        print("SUITE", args.host, json.dumps(summaries), flush=True)
        negative = 1
        if args.host == "sirius":
            tree = scratch / "narrow-tile"
            archive(args.root, revision, tree)
            path = tree / "nova/equilibrium/current.py"
            source = path.read_text()
            target = "_SCALAR_CUT_BATCH_WIDTH = 32"
            assert source.count(target) == 1
            changed = source.replace(target, "_SCALAR_CUT_BATCH_WIDTH = 16")
            path.write_text(changed)
            (args.run / "tile-mutation.diff").write_text(
                "".join(
                    difflib.unified_diff(
                        source.splitlines(True),
                        changed.splitlines(True),
                        fromfile="current.py",
                        tofile="narrow-tile/current.py",
                    )
                )
            )
            negative = run_file(
                tree,
                revision + "+narrow-tile",
                "test_book_primal_buckets",
                "negative-tile",
                args.run,
                scratch,
                "test_scalar_cut_tile",
            )
            print("NEGATIVE_TILE_EXIT", negative, flush=True)
        print("Removing scratch", scratch, flush=True)
    return int(any(results) or negative == 0)


if __name__ == "__main__":
    raise SystemExit(main())
