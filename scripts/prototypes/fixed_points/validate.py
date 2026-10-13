"""Run identical CPU contracts at a base, head, or explicitly mutated head."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("baseline", "head", "negative"), required=True
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    revision = args.base if args.arm == "baseline" else head
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "scripts.prototypes.fixed_points.validate",
            "--arm",
            args.arm,
            "--base",
            args.base,
            "--out",
            str(args.out.resolve()),
        ]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fixed-point-contract-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = root if args.arm == "head" else Path(temporary)
        if args.arm != "head":
            archive = subprocess.check_output(
                [
                    "git",
                    "archive",
                    revision,
                    "nova",
                    "conftest.py",
                    "pyproject.toml",
                    "tests/conftest.py",
                    "tests/equilibrium/test_topology_read.py",
                    "tests/rotating_equilibrium_references.py",
                    "scripts/analytic_oracle_fixtures",
                    "tests/__init__.py",
                    "scripts/__init__.py",
                ]
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(tree, filter="data")
            test = tree / "tests/test_fixed_points.py"
            test.write_bytes((root / "tests/test_fixed_points.py").read_bytes())
            fixture = tree / "scripts/prototypes/fixed_points/coarse-replay.npz"
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_bytes(
                (
                    root / "scripts/prototypes/fixed_points/coarse-replay.npz"
                ).read_bytes()
            )
        if args.arm == "negative":
            module = tree / "nova/equilibrium/fixed_points.py"
            code = module.read_text()
            original = "return (determinant < 0).at[0].set(axis)"
            assert code.count(original) == 1
            module.write_text(
                code.replace(original, "return jnp.ones(hessian.shape[0], dtype=bool)")
            )
        payload = """import importlib.util, pathlib, nova
print('NOVA_MODULE='+nova.__file__, flush=True)
print('CWD='+str(pathlib.Path.cwd().resolve()), flush=True)
spec = importlib.util.find_spec('nova.equilibrium.fixed_points')
print('MEASUREMENT_MODULE='+str(None if spec is None else spec.origin), flush=True)
import pytest
arguments = ['-p', 'no:cacheprovider', '--basetemp', BASENAME, TESTNAME, '-q']
raise SystemExit(pytest.main(arguments))
""".replace("BASENAME", repr(str(Path(temporary) / "pytest"))).replace(
            "TESTNAME", repr(str(tree / "tests/test_fixed_points.py"))
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(tree),
            JAX_PLATFORMS="cpu",
            TMPDIR=os.environ["TMPDIR"],
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="2",
        )
        log = args.out.resolve() / f"{args.arm}.log"
        with log.open("w") as stream:
            stream.write(f"revision={revision} tree={tree} command={command}\n")
            stream.write(
                "INNER_COMMAND=" + shlex.join([sys.executable, "-c", payload]) + "\n"
            )
            stream.flush()
            result = subprocess.run(
                [sys.executable, "-c", payload],
                cwd=tree,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            stream.write(f"EXIT={result.returncode}\n")
        text = log.read_text()
        failures = sorted(
            set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.MULTILINE))
        )
        completed = bool(re.search(r"\d+ (?:passed|failed|error)", text))
        record = dict(
            revision=revision,
            command=command,
            exit_status=result.returncode,
            log_path=str(log),
            completed=completed,
            failure_count=len(failures),
            failure_ids=failures,
        )
        (args.out / f"{args.arm}.json").write_text(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
