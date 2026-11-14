"""Run each material measurement in one allocation on its declared device lane."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def execute(command, cwd, env, log, revision):
    """Capture a command's complete output and its terminal exit code once."""
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import nova.biot.magnetisation as m; print(m.__file__)",
        ],
        cwd=cwd,
        env=env,
        text=True,
    ).strip()
    with log.open("w") as stream:
        stream.write(f"# revision: {revision} tree: {cwd} command: {command}\n")
        stream.write(f"# module: {probe}\n# cwd: {cwd}\n# command: {command}\n")
        stream.flush()
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.write(f"EXIT={result.returncode}\n")
    print(f"ARM log={log} EXIT={result.returncode}", flush=True)
    text = log.read_text()
    failures = [
        line.split(" - ")[0].removeprefix("FAILED ")
        for line in text.splitlines()
        if line.startswith("FAILED ")
    ]
    completed = any(
        " passed" in line or " failed" in line
        for line in text.splitlines()
        if " in " in line
    ) or any(line.startswith("SUMMARY passed=") for line in text.splitlines())
    return dict(
        revision=revision,
        command=command,
        exit_status=result.returncode,
        log_path=str(log),
        completed=completed,
        failure_count=len(failures),
        failure_ids=failures,
        measurement_module=probe,
        measurement_cwd=str(cwd),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--mode", choices=("production", "suites", "all"), default="production"
    )
    args = parser.parse_args()
    tree = Path.cwd().resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    python = "/home/ITER/mcintos/Code/nova/.venv/bin/python"
    env = dict(
        os.environ,
        PYTHONPATH=".",
        TMPDIR="/tmp",
        OPENBLAS_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "8"),
        OMP_NUM_THREADS=os.environ.get("SLURM_CPUS_PER_TASK", "8"),
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    env.pop("UV_NO_SYNC", None)
    env.pop("UV_RUN_RECURSION_DEPTH", None)
    record = dict(
        base_revision=args.base, head_revision=revision, job_id=env.get("SLURM_JOB_ID")
    )
    receipt = args.run_dir / f"{args.mode}-gate-receipt.json"

    def save():
        receipt.write_text(json.dumps(record, indent=2) + "\n")

    save()
    if args.mode in ("production", "all"):
        command = (
            f"{python} -m scripts.iron_limb_production.measure --output {args.output}"
        )
        record["production"] = execute(
            command, tree, env, args.run_dir / "production.log", revision
        )
        save()
        command = f"{python} scripts/iron_limb_production/negative_control.py"
        record["negative_control"] = execute(
            command, tree, env, args.run_dir / "negative-control.log", revision
        )
        record["passed"] = record["production"]["exit_status"] == 0
        record["passed"] &= record["negative_control"]["exit_status"] != 0 and bool(
            record["negative_control"]["failure_ids"]
        )
        save()
        if args.mode == "production":
            print(f"GATE passed={record['passed']}", flush=True)
            raise SystemExit(0 if record["passed"] else 1)
    with tempfile.TemporaryDirectory(
        prefix="iron-material-", dir=env["TMPDIR"]
    ) as scratch:
        base = Path(scratch) / "base"
        base.mkdir()
        archive = Path(scratch) / "base.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                [
                    "git",
                    "archive",
                    args.base,
                    "nova",
                    "scripts",
                    "tests",
                    "pyproject.toml",
                    "conftest.py",
                ],
                stdout=stream,
                check=True,
            )
        with tarfile.open(archive) as source:
            source.extractall(base, filter="data")
        plugin = base / "scripts/iron_limb_production"
        plugin.mkdir(exist_ok=True)
        shutil.copyfile(
            tree / "scripts/iron_limb_production/compatibility.py",
            plugin / "compatibility.py",
        )
        suite = (
            f"PYTHONPATH=. JAX_PLATFORMS=cpu {python} -m pytest "
            "-p no:cacheprovider -q tests/test_magnetisation.py"
        )
        cpu = dict(
            env,
            JAX_PLATFORMS="cpu",
            PYTEST_PLUGINS="scripts.iron_limb_production.compatibility",
            NOVA_MATERIAL_CAPTURE=str(args.run_dir / "baseline-outputs.npz"),
        )
        record["baseline_suite"] = execute(
            suite, base, cpu, args.run_dir / "baseline-suite.log", args.base
        )
        save()
        cpu["NOVA_MATERIAL_BASELINE"] = str(args.run_dir / "baseline-outputs.npz")
        cpu["NOVA_MATERIAL_CAPTURE"] = str(args.run_dir / "head-outputs.npz")
        record["after_suite"] = execute(
            suite, tree, cpu, args.run_dir / "after-suite.log", revision
        )
        save()
    passed = all(
        record[arm]["exit_status"] == 0 for arm in ("baseline_suite", "after_suite")
    )
    if args.mode == "all":
        passed &= record["passed"]
    record["passed"] = passed
    save()
    print(f"GATE passed={passed}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
