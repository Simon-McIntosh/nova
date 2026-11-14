"""Queue unresolved main arms after validating frozen-fixture CPU smoke."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    inventory = run / "repair-jobs.json"
    assert not inventory.exists(), "repair jobs already submitted"
    families = ("decisions", "booking", "candidates")
    for family in families:
        directory = run / f"cpu-frozen-{family}"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["completed"] and summary["smoke"]
        assert (directory / "job.log").read_text().splitlines()[-1] == "EXIT=0"
        for comparison in summary["comparisons"]:
            left, right = comparison["off"]["records"], comparison["on"]["records"]
            assert len(left) == len(right) == 1
            assert left[0]["input_sha256"] == right[0]["input_sha256"]
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "--exit-code",
                summary["revision"],
                args.revision,
                "--",
                "nova",
                "scripts/prototypes/h200_levers/measure.py",
                "scripts/prototypes/h200_levers/run.py",
            ],
            check=True,
        )
    result = dict(revision=args.revision, jobs=[])
    env = dict(os.environ, TMPDIR="/tmp")
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_JOBID", None)
    for family in families:
        output = run / f"measurement-repair-{family}"
        output.mkdir(exist_ok=True)
        source = (
            run
            / (
                "h200-levers-corrected"
                if family == "decisions"
                else f"measurement-{family}"
            )
            / "summary.json"
        )
        command = [
            "sbatch",
            "--parsable",
            "--partition=betelgeuse",
            "--reservation=gpu_0003_grpA",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=96G",
            "--time=04:00:00",
            f"--job-name=nova-{family}",
            f"--output={output / 'job.log'}",
            str(root / "scripts/prototypes/h200_levers/job.sh"),
            str(root),
            str(output),
            args.revision,
            "--measurement",
            family,
            "--reuse-run",
            str(run / "h200-levers-corrected"),
            "--fixture-source",
            str(source),
        ]
        job = subprocess.check_output(command, env=env, text=True).strip()
        assert job.isdigit(), job
        result["jobs"].append(
            dict(measurement=family, job_id=job, output=str(output), command=command)
        )
        inventory.write_text(json.dumps(result, indent=2))
        print(f"QUEUED {family} {job}", flush=True)


if __name__ == "__main__":
    main()
