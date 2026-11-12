"""Queue every unfinished H200 measurement after the CPU smoke gate passes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run.resolve()
    smoke = json.loads((args.smoke / "summary.json").read_text())
    assert smoke["completed"] and smoke["smoke"]
    assert smoke["revision"] == args.revision
    assert (args.smoke / "job.log").read_text().splitlines()[-1] == "EXIT=0"
    prior = run / "h200-levers-corrected"
    existing = json.loads((prior / "summary.json").read_text())
    jobs_file = run / "measurement-jobs.json"
    assert not jobs_file.exists(), f"job inventory already exists: {jobs_file}"
    retained = {}
    for arm in existing["arms"]:
        if arm["exit_status"] == 0:
            for field in ("receipt", "log"):
                path = Path(arm[field])
                retained[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = Path(arm["receipt"]).with_suffix(".npz")
            if receipt.exists():
                retained[str(receipt)] = hashlib.sha256(
                    receipt.read_bytes()
                ).hexdigest()
    (run / "retained-receipts.json").write_text(json.dumps(retained, indent=2))
    decisions = {
        c["width"] for c in existing["comparisons"] if c["name"] == "carried-decisions"
    }
    families = ["booking", "factor", "scalar", "lane", "candidates", "wall", "headline"]
    if decisions != {1, 16}:
        families.append("decisions")
    inventory = dict(revision=args.revision, smoke=str(args.smoke), jobs=[])
    jobs_file.write_text(json.dumps(inventory, indent=2))
    env = dict(os.environ, TMPDIR="/tmp")
    for name in ("SLURM_JOB_ID", "SLURM_JOBID"):
        env.pop(name, None)
    for family in families:
        output = run / ("measurement-" + family)
        output.mkdir(exist_ok=True)
        command = [
            "sbatch",
            "--parsable",
            "--partition=betelgeuse",
            "--reservation=gpu_0003_grpA",
            "--gres=gpu:1",
            "--cpus-per-task=4",
            "--mem=96G",
            "--time=04:00:00",
            "--job-name=nova-" + family,
            "--output=" + str(output / "job.log"),
            str(root / "scripts/prototypes/h200_levers/job.sh"),
            str(root),
            str(output),
            args.revision,
            "--measurement",
            family,
            "--reuse-run",
            str(prior),
        ]
        job = subprocess.check_output(command, env=env, text=True).strip().split(";")[0]
        assert job.isdigit(), job
        inventory["jobs"].append(
            dict(measurement=family, job_id=job, output=str(output), command=command)
        )
        jobs_file.write_text(json.dumps(inventory, indent=2))
        print(f"QUEUED {family} {job}", flush=True)
    for path, digest in retained.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    print(
        "SUMMARY",
        len(inventory["jobs"]),
        "measurement jobs queued; completed receipts preserved",
    )


if __name__ == "__main__":
    main()
