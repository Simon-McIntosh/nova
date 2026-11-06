"""Pair warm-cache solves and compare four certificate Jacobians on one device."""

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
    run.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="jacobian-device-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        base, head = scratch / "base", scratch / "head"
        archive(root, base_revision, base)
        archive(root, head_revision, head)
        for process in range(3):
            for arm, tree, revision in (
                ("base", base, base_revision),
                ("head", head, head_revision),
            ):
                destination = run / f"{arm}-{process}"
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
                    str(root / "scripts/prototypes/bank_jacobian_launches/capture.py"),
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
                    for module in (
                        "bank_solve_trace/analyse.py",
                        "bank_jacobian_launches/loops.py",
                    ):
                        execute(
                            tree,
                            revision,
                            [
                                sys.executable,
                                "-P",
                                str(root / "scripts/prototypes" / module),
                                str(destination),
                            ],
                            run / f"{Path(module).stem}-{arm}-{process}.log",
                            environment,
                        )
        environment = dict(
            os.environ,
            PYTHONPATH=str(head),
            TMPDIR=str(scratch),
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            BOOKING_REFERENCE_CURRENT=str(base / "nova/equilibrium/current.py"),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            "-P",
            "-c",
            "import nova, nova.equilibrium.current as m; from pathlib import Path; "
            "print('NOVA', nova.__file__, 'MODULE', m.__file__, "
            "'CWD', Path.cwd().resolve(), flush=True); "
            "import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))",
            "-p",
            "no:cacheprovider",
            f"--basetemp={scratch / 'certificate-tests'}",
            str(head / "tests/test_bank_step_booking_batch.py"),
            "-k",
            "certificate_jacobian or conic_channel_jacobian",
            "-vv",
            "-s",
        ]
        execute(
            head, head_revision, command, run / "certificate-tests.log", environment
        )
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
