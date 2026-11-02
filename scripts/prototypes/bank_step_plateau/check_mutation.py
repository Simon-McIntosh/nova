"""Require the booking control to refuse a missing Jacobian perturbation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


SOURCE = "scripts/prototypes/bank_step_plateau/measure.py"
CHANGES = (
    (
        "fresh_matrix = fresh_matrix.at[0, column].add(1e-3)",
        "fresh_matrix = fresh_matrix",
    ),
    ("held_matrix = held_matrix.at[0, column].add(1e-3)", "held_matrix = held_matrix"),
)
LAUNCHER = """
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
target, trajectory, output = sys.argv[2:]
sys.argv = [
    target, "compare", "--trajectory", trajectory, "--output", output, "--mutation"
]
exec(compile(source, target, "exec"), {"__name__": "__main__", "__file__": target})
"""


def main(revision, trajectory, output):
    root = Path(__file__).resolve().parents[3]
    original = root / SOURCE
    source = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{revision}:{SOURCE}"], text=True
    )
    for before, after in CHANGES:
        if source.count(before) != 1:
            raise AssertionError(f"expected one booking mutation: {before}")
        source = source.replace(before, after, 1)
    with TemporaryDirectory(
        prefix="bank-plateau-control-", dir=os.environ["TMPDIR"]
    ) as scratch:
        muted = Path(scratch) / "measure.py"
        muted.write_text(source)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                LAUNCHER,
                str(muted),
                str(original),
                str(trajectory),
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    print(f"SOURCE_REVISION={revision} CHILD_EXIT={result.returncode}", flush=True)
    if result.returncode == 0:
        raise AssertionError("suppressed booking perturbation passed its guard")
    if "booking perturbation did not exceed its control" not in result.stderr:
        raise AssertionError(
            f"suppressed control failed for another reason: {result.stderr[-1500:]}"
        )
    print("REFUSAL booking perturbation did not exceed its control", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("revision")
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.revision, args.trajectory, args.output)
