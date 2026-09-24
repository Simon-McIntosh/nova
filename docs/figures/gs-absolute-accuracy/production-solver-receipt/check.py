"""Run the receipt contract and its source-reversion negative control."""

import ast
from pathlib import Path
import subprocess
import sys


BASE = "f8c24a819fc821e6549b791892295da30323d2f9"
ROOT = Path(__file__).resolve().parent
TARGET = "tests/test_production_solver_receipt_trip_alignment.py"


def main():
    import pytest

    if "--negative" in sys.argv:
        from benchmarks import solovev_certificate as certificate

        source = subprocess.check_output(
            ["git", "show", f"{BASE}:benchmarks/solovev_certificate.py"], text=True
        )
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_production_solver_receipt"
        )
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                "receipt-before-repair",
                "exec",
            ),
            certificate.__dict__,
        )
        raise SystemExit(pytest.main(["-p", "no:cacheprovider", TARGET, "-q"]))

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name, command in (
        (
            "focused.log",
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", TARGET, "-q"],
        ),
        ("negative-control.log", [sys.executable, str(Path(__file__)), "--negative"]),
    ):
        with (ROOT / name).open("w") as log:
            mutation = (
                "revert the receipt repair; " if name.startswith("negative") else ""
            )
            print(
                f"{mutation}revision={revision} tree={Path.cwd()} "
                f"command={' '.join(command)}",
                file=log,
                flush=True,
            )
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            print(f"EXIT={result.returncode}", file=log)
        print(f"{name} EXIT={result.returncode}", flush=True)
        assert result.returncode == (1 if name.startswith("negative") else 0)


if __name__ == "__main__":
    main()
