"""Reject a serial booking-block assembly in an isolated scratch source tree."""

import argparse
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    with tempfile.TemporaryDirectory(
        prefix="bank-booking-negative-", dir=os.environ["TMPDIR"]
    ) as name:
        scratch = Path(name)
        archive = subprocess.Popen(
            [
                "git",
                "-C",
                str(root),
                "archive",
                revision,
                "nova",
                "tests",
                "pyproject.toml",
            ],
            stdout=subprocess.PIPE,
        )
        assert archive.stdout is not None
        with tarfile.open(fileobj=archive.stdout, mode="r|") as stream:
            stream.extractall(scratch, filter="data")
        assert archive.wait() == 0
        source = scratch / "nova/equilibrium/support_read.py"
        original = source.read_text()
        target = "    local = _local_booking_sensitivities(\n"
        replacement = (
            "    def serial_local(*args):\n"
            "        for _ in range(15):\n"
            "            result = _local_booking_sensitivities(*args)\n"
            "        return result\n\n"
            "    local = serial_local(\n"
        )
        assert original.count(target) == 1
        source.write_text(original.replace(target, replacement))
        test = scratch / "tests/test_bank_step_booking.py"
        command = [
            python,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(test)
            + "::test_booking_block_is_assembled_from_one_batched_sensitivity_call",
            "--basetemp=" + str(scratch / "pytest"),
            "-vv",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(scratch)
        environment["JAX_PLATFORMS"] = "cpu"
        with args.log.open("w") as log:
            print(
                f"REVISION={revision} TREE={scratch} CWD={scratch} "
                f"COMMAND={' '.join(command)}",
                file=log,
                flush=True,
            )
            probe = subprocess.run(
                [
                    python,
                    "-c",
                    "import nova, nova.equilibrium.support_read as m; "
                    "print('NOVA='+nova.__file__+' MODULE='+m.__file__)",
                ],
                cwd=scratch,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if probe.returncode != 0:
                raise RuntimeError("mutated module did not import")
            result = subprocess.run(
                command,
                cwd=scratch,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            print(f"EXIT={result.returncode}", file=log)
        if result.returncode == 0:
            raise AssertionError("serial booking mutation escaped its guard")
        print(f"NEGATIVE_CONTROL_EXIT={result.returncode}")


if __name__ == "__main__":
    main()
