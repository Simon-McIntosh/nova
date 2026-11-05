"""Restore serial lane mapping in scratch and exercise the measured scaling gate."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile


def run(command, tree, log, revision, **variables):
    environment = dict(os.environ, PYTHONPATH=str(tree), **variables)
    with log.open("w") as stream:
        stream.write(
            f"# revision: {revision} tree: {tree} command: {shlex.join(command)}\n"
        )
        stream.flush()
        result = subprocess.run(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT
        )
        stream.write(f"EXIT={result.returncode}\n")
    return result.returncode


def main(run_dir, revision):
    root = Path(__file__).resolve().parents[3]
    base_revision = next(
        line.split(":", 1)[1].strip()
        for line in (run_dir / "manifest.md").read_text().splitlines()
        if line.startswith("orientation_base_sha:")
    )
    for label, tree in (("base", Path(os.environ["TMPDIR"]) / "base"), ("head", root)):
        command = [
            sys.executable,
            str(root / "scripts/prototypes/bank_step_booking_batch/step_profile.py"),
            "--run",
            str(run_dir),
            "--arm",
            label,
            "--revision",
            revision if label == "head" else base_revision,
        ]
        assert (
            run(command, tree, run_dir / f"step-profile-{label}.log", command[-1]) == 0
        )
    test = "tests/test_bank_step_booking_batch.py"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        str(root / test),
        "-k",
        "sixteen_lane_scaling",
        "-vv",
        "-s",
        f"--basetemp={os.environ['TMPDIR']}/head-scaling",
    ]
    head_status = run(
        command,
        root,
        run_dir / "scaling-head.log",
        revision,
        BOOKING_TIMINGS=str(run_dir / "head-timings.json"),
    )
    scratch = Path(os.environ["TMPDIR"]) / "negative"
    scratch.mkdir()
    archive = subprocess.Popen(
        [
            "git",
            "-C",
            str(root),
            "archive",
            "HEAD",
            "nova",
            "scripts",
            "tests",
            "benchmarks",
            "pyproject.toml",
            "conftest.py",
        ],
        stdout=subprocess.PIPE,
    )
    with tarfile.open(fileobj=archive.stdout, mode="r|") as files:
        files.extractall(scratch, filter="data")
    assert archive.wait() == 0
    source = scratch / "nova/equilibrium/support_read.py"
    text = source.read_text()
    original = "lambda _: jax.vmap(one_fresh)((state, step_support))"
    assert text.count(original) == 1
    source.write_text(
        text.replace(
            original, "lambda _: jax.lax.map(one_fresh, (state, step_support))"
        )
    )
    command = [
        sys.executable,
        str(root / "scripts/prototypes/bank_step_booking_batch/measure.py"),
        "--run",
        str(run_dir),
        "--arm",
        "negative",
        "--revision",
        revision,
    ]
    assert run(command, scratch, run_dir / "negative-measure.log", revision) == 0
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        str(scratch / test),
        "-k",
        "sixteen_lane_scaling",
        "-vv",
        "-s",
        f"--basetemp={os.environ['TMPDIR']}/negative-scaling",
    ]
    negative_status = run(
        command,
        scratch,
        run_dir / "negative-control.log",
        revision,
        BOOKING_TIMINGS=str(run_dir / "negative-timings.json"),
    )
    assert head_status == 0, "head scaling assertion failed"
    assert negative_status == 1, "serial lane mutation was not refused"
