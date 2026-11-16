"""Run each material measurement in one allocation on its declared device lane."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

_CONTROL_RECEIPT = Path(
    "/work/projects/imas_gpu/sophelio/evidence/iron-limb-fill/production/"
    "material-refinement-20261114/summary.json"
)
_CONTROL_REVISION = "8ab69e765a812f1713129852eb65e4b5d886f26d"


def solid_rows(source):
    """Return the solid-outline control rows the receipt carries or names.

    The refinement receipt keeps only the four comparison arms, so the solid
    rows are read once from the source receipt it names and hashes.
    """
    rows = [row for row in source["control"] if row["model"] == "solid"]
    if rows:
        return rows
    origin = source.get("source_receipt")
    assert origin, "control receipt carries no solid-outline control and names none"
    raw = Path(origin).read_bytes()
    recorded = source.get("source_sha256")
    if recorded:
        assert hashlib.sha256(raw).hexdigest() == recorded, origin
    return [row for row in json.loads(raw)["control"] if row["model"] == "solid"]


def validate_control_receipt(path, expected_revision=_CONTROL_REVISION):
    """Assert section-derived material accuracy on the completed fine GPU grid."""
    raw = path.read_bytes()
    source = json.loads(raw)
    revision = source.get("revision")
    assert revision == expected_revision, (
        f"control receipt {path} was measured at revision {revision}, "
        f"the gate expects {expected_revision}"
    )
    assert "H200" in source["device"], "fine-grid acceptance requires its H200 receipt"
    assert source["geometry_factor"] == 0.4
    solid = solid_rows(source)
    assert solid, "control receipt carries no solid-outline control"
    for row in solid:
        assert row["vessel_relative_rms"] > 1.0, (
            f"solid-outline control mu={row['mu']:g} at {row['size_m']:g} m is "
            f"{row['vessel_relative_rms']:.4f} of the reference; the solid arm "
            "must exceed 100 %"
        )
    rows = [row for row in source["control"] if row["size_m"] == 0.032]
    expected = {
        (name, mu)
        for name in ("plain", "smear", "geometry", "fitted")
        for mu in (50.0, 200.0)
    }
    assert len(rows) == len(expected)
    assert {(row["model"], row["mu"]) for row in rows} == expected
    for row in rows:
        assert row["converged"] and row["device"] == source["device"]
        assert math.isfinite(row["vessel_relative_rms"])
        assert row["vessel_relative_rms"] >= 0
        field = row["iron_T"]
        assert len(field) >= 16 and all(len(point) == 2 for point in field)
        assert all(math.isfinite(value) for point in field for value in point)
        assert sum(value**2 for point in field[:16] for value in point) > 0
    acceptance = []
    for mu, limit in ((50.0, 0.10), (200.0, 0.11)):
        geometry = next(
            row for row in rows if row["mu"] == mu and row["model"] == "geometry"
        )
        smear = next(row for row in rows if row["mu"] == mu and row["model"] == "smear")
        assert geometry["cross_field_factor"] == 0.4
        error = geometry["vessel_relative_rms"]
        assert error <= limit, f"mu={mu:g} vessel rms {error} exceeds bound {limit}"
        assert smear["vessel_relative_rms"] > 0
        ratio = error / smear["vessel_relative_rms"]
        assert ratio <= 1 / 3, f"mu={mu:g} smear ratio {ratio} exceeds one third"
        acceptance.append(
            dict(mu=mu, limit=limit, error=error, smear_error_ratio=ratio, passed=True)
        )
    return dict(
        source_receipt=str(path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        device=source["device"],
        size_m=0.032,
        geometry_factor=0.4,
        revision=revision,
        solid=[
            dict(size_m=row["size_m"], mu=row["mu"], vessel_relative_rms=row["vessel_relative_rms"])
            for row in solid
        ],
        acceptance=acceptance,
        comparison=[
            {
                key: value
                for key, value in row.items()
                if key not in ("iron_T", "passed")
            }
            for row in rows
        ],
        passed=True,
    )


def execute(command, cwd, env, log, revision):
    """Capture a command's complete output and its terminal exit code once."""
    if (cwd / ".git").exists():
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
        ).strip()
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
    parser.add_argument("--control-receipt", type=Path, default=_CONTROL_RECEIPT)
    parser.add_argument("--control-revision", default=_CONTROL_REVISION)
    parser.add_argument(
        "--mode",
        choices=("production", "suites", "all", "control"),
        default="production",
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
    if args.mode == "control":
        record["control"] = validate_control_receipt(
            args.control_receipt, args.control_revision
        )
        record["passed"] = True
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "accepted-comparison.json").write_text(
            json.dumps(record["control"], indent=2, allow_nan=False) + "\n"
        )
        save()
        print(
            "CONTROL_ACCEPTANCE " + json.dumps(record["control"]["acceptance"]),
            flush=True,
        )
        print("SUMMARY passed=True control_passed=True", flush=True)
        return
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
