"""Pair CPU files in fresh processes and challenge geometry population routing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.suite import archive

FILES = (
    "test_bank_step_booking",
    "test_bank_step_booking_batch",
    "test_conic_endpoint_consistency",
    "test_bank_solve_driver",
    "test_bank_solve_cycle",
)


def run_file(tree, revision, name, label, run, scratch, selection=None):
    command = [
        sys.executable,
        "-P",
        "-c",
        "import nova, nova.equilibrium.current as m; from pathlib import Path; "
        "print('nova', nova.__file__, 'module', m.__file__, "
        "'cwd', Path.cwd().resolve(), flush=True); "
        "import pytest, sys; sys.exit(pytest.main(sys.argv[1:]))",
        "-p",
        "no:cacheprovider",
        f"--basetemp={scratch / (label + '-' + name)}",
        str(tree / "tests" / (name + ".py")),
        "-m",
        "slow or not slow",
        "-vv",
        "-s",
    ]
    if selection:
        command += ["-k", selection]
    environment = dict(
        os.environ,
        PYTHONPATH=str(tree),
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        TMPDIR=str(scratch),
    )
    path = run / f"{label}-{name}.log"
    with path.open("w") as log:
        log.write(
            f"# revision: {revision} tree: {tree} command: {shlex.join(command)}\n"
        )
        log.flush()
        result = subprocess.run(
            command, cwd=tree, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        log.write(f"\nEXIT={result.returncode}\n")
    print("FILE_COMPLETE", label, name, result.returncode, flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="booking-suite-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base, negative = scratch / "base", scratch / "negative"
        archive(args.root, args.base, base)
        archive(args.root, head, negative)
        path = negative / "nova/equilibrium/current.py"
        source = path.read_text()
        target = "    normal = cut & read.normal_form_cells\n"
        assert source.count(target) == 2
        changed = source.replace(
            target,
            target + "    normal = normal.at[jnp.argmax(normal)].set(False)\n",
            1,
        )
        path.write_text(changed)
        (args.run / "negative-mutation.diff").write_text(
            "".join(
                difflib.unified_diff(
                    source.splitlines(True),
                    changed.splitlines(True),
                    fromfile="current.py",
                    tofile="misrouted/current.py",
                )
            )
        )

        def arm(label, tree, revision, names):
            return [
                run_file(tree, revision, name, label, args.run, scratch)
                for name in names
            ]

        # The scheduler allocation isolates the two CPU arms from the fleet.
        with ThreadPoolExecutor(max_workers=2) as pool:
            before = pool.submit(arm, "base", base, args.base, FILES)
            after = pool.submit(
                arm, "head", args.root, head, ("test_book_primal_buckets", *FILES)
            )
            results = {"base": before.result(), "head": after.result()}
        results["negative"] = run_file(
            negative,
            head + "+misrouted-normal",
            "test_book_primal_buckets",
            "negative",
            args.run,
            scratch,
        )
        (args.run / "suite-status.json").write_text(json.dumps(results))
        print("SUITES", results, flush=True)
        print("Removing scratch", scratch, flush=True)
        return int(any(results["base"] + results["head"]) or results["negative"] == 0)


if __name__ == "__main__":
    raise SystemExit(main())
