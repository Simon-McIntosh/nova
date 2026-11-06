"""Compare the certificate matrices against an extracted base source module."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.prototypes.bank_wall_events.device import execute


def main():
    root, run = map(Path, sys.argv[1:3])
    base = sys.argv[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(
        prefix="jacobian-certificates-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        reference = scratch / "reference_current.py"
        reference.write_bytes(
            subprocess.check_output(
                ["git", "-C", str(root), "show", f"{base}:nova/equilibrium/current.py"]
            )
        )
        command = [
            sys.executable,
            "-P",
            "-c",
            "import nova, nova.equilibrium.current as m; from pathlib import Path; "
            "print('NOVA',nova.__file__,'MODULE',m.__file__,"
            "'CWD',Path.cwd().resolve(),flush=True); "
            "import pytest,sys; sys.exit(pytest.main(sys.argv[1:]))",
            "-p",
            "no:cacheprovider",
            f"--basetemp={scratch / 'tests'}",
            str(root / "tests/test_bank_step_booking_batch.py"),
            "-m",
            "slow or not slow",
            "-k",
            "certificate_jacobian or conic_channel_jacobian",
            "-vv",
            "-s",
        ]
        environment = dict(
            os.environ,
            PYTHONPATH=str(root),
            TMPDIR=str(scratch),
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            BOOKING_REFERENCE_CURRENT=str(reference),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
        )
        execute(root, revision, command, run / "certificate-tests.log", environment)
        print("Removing scratch", scratch, flush=True)


if __name__ == "__main__":
    main()
