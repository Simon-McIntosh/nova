"""Check candidate branch selection and compare fresh per-file suite arms."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_solve_cycle.receipts import assemble
from scripts.prototypes.bank_wall_events.device import execute
from scripts.prototypes.bank_wall_events.suite import archive, run_file

FILES = ("test_candidate_inspection_batched", "test_support_read", "test_tracked_read")


def force_full(root, revision, run, scratch, label):
    tree = scratch / label
    archive(root, revision, tree)
    path = tree / "nova/equilibrium/topology.py"
    source = path.read_text()
    target = "any_lane_overflow(jnp.sum(retained) > compact_capacity)"
    assert source.count(target) == 1
    changed = source.replace(target, "any_lane_overflow(jnp.asarray(True))")
    path.write_text(changed)
    (run / (label + "-mutation.diff")).write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(True),
                changed.splitlines(True),
                fromfile="candidate",
                tofile="forced-full",
            )
        )
    )
    return run_file(
        tree,
        revision + "+forced-full",
        FILES[0],
        label,
        run,
        scratch,
        "not certificate",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reproduce", "cpu", "physical"))
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("base")
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="candidate-check-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        if args.mode == "reproduce":
            status = force_full(args.root, args.base, args.run, scratch, "reproduction")
        elif args.mode == "cpu":
            base = scratch / "base"
            archive(args.root, args.base, base)

            def arm(label, tree, revision):
                for name in FILES:
                    run_file(
                        tree,
                        revision,
                        name,
                        label,
                        args.run,
                        scratch,
                        "not certificate" if name == FILES[0] else None,
                    )
                return assemble(
                    args.run,
                    [label + "-" + name for name in FILES],
                    revision,
                    label + "-suite.log",
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                before = pool.submit(arm, "base", base, args.base)
                after = pool.submit(arm, "head", args.root, head)
                suites = {"base": before.result(), "head": after.result()}
            (args.run / "suites.json").write_text(json.dumps(suites, indent=2))
            negative = force_full(args.root, head, args.run, scratch, "negative")
            added = set(suites["head"]["failure_ids"]) - set(
                suites["base"]["failure_ids"]
            )
            print(
                "ADDED_FAILURES", sorted(added), "NEGATIVE_STATUS", negative, flush=True
            )
            status = int(
                bool(added) or negative != 1 or suites["head"]["exit_status"] != 0
            )
        else:
            environment = dict(
                os.environ,
                PYTHONPATH=str(args.root),
                TMPDIR=str(scratch),
                JAX_PLATFORMS="cuda",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="2",
            )
            environment.pop("XLA_FLAGS", None)
            command = [
                sys.executable,
                "-P",
                "-c",
                "from nova.jax.config import configure_dtypes; configure_dtypes(); "
                "import nova, nova.equilibrium.topology as m; "
                "from pathlib import Path; import jax,pytest,sys; "
                "assert jax.config.jax_enable_x64; "
                "print('nova',nova.__file__,'module',m.__file__,"
                "'cwd',Path.cwd().resolve(),'devices',jax.devices(),flush=True); "
                "raise SystemExit(pytest.main(sys.argv[1:]))",
                "-p",
                "no:cacheprovider",
                "--basetemp=" + str(scratch / "physical"),
                str(args.root / "tests/test_candidate_inspection_batched.py"),
                "-m",
                "slow",
                "-vv",
                "-s",
            ]
            execute(args.root, head, command, args.run / "physical.log", environment)
            status = 0
        print("Removing scratch", scratch, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
