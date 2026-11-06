"""Pair warmed revisions and reuse the solve trace and precision instruments."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.device import execute
from scripts.prototypes.bank_wall_events.suite import archive


def main():
    root, run = map(Path, sys.argv[1:3])
    base_revision = sys.argv[3]
    head_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="booking-device-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base = scratch / "base"
        archive(root, base_revision, base)
        for process in range(3):
            for arm, tree, revision in (
                ("base", base, base_revision),
                ("head", root, head_revision),
            ):
                destination = run / f"{arm}-{process}"
                destination.mkdir(exist_ok=True)
                environment = dict(
                    os.environ,
                    PYTHONPATH=str(tree),
                    TMPDIR=str(scratch),
                    JAX_DEBUG_LOG_MODULES="jax._src.compiler",
                )
                command = [
                    "bash",
                    str(
                        root
                        / "scripts/prototypes/bank_booking_determinism/measurement.sh"
                    ),
                    str(run / f"cache-{arm}"),
                    sys.executable,
                    "-P",
                    "-c",
                    "import nova.equilibrium.current as m; "
                    "print('MODULE',m.__file__,flush=True); "
                    "from scripts.prototypes.bank_wall_events.capture import main; "
                    "main()",
                    "--run",
                    str(destination),
                    "--revision",
                    revision,
                ]
                execute(
                    tree,
                    revision,
                    command,
                    run / f"capture-{arm}-{process}.log",
                    environment,
                )
                if process:
                    execute(
                        tree,
                        revision,
                        [
                            sys.executable,
                            "-P",
                            "-m",
                            "scripts.prototypes.bank_solve_trace.analyse",
                            str(destination),
                        ],
                        run / f"attribution-{arm}-{process}.log",
                        environment,
                    )
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
