"""Recover an interrupted terminal arm and render it beside its retained base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_logged(command, tree, environment, path, revision):
    with path.open("w") as stream:
        stream.write(
            f"# revision: {revision}; tree: {tree}; command: {shlex.join(command)}\n"
        )
        stream.flush()
        status = subprocess.run(
            command,
            cwd=tree,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        ).returncode
        stream.write(f"EXIT={status}\n")
    if status:
        raise SystemExit(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    records = json.loads((args.previous / "comparison.json").read_text())
    assert len(records) == 7 and all(row["exit_status"] == 0 for row in records)
    expected = {
        (arm, frame, branch)
        for arm in ("base", "head")
        for frame, branch in (
            (137, "diverted"),
            (146, "diverted"),
            (146, "limited"),
            (179, "limited"),
        )
    }
    recovered = ("head", 179, "limited")
    assert {(r["arm"], r["frame"], r["branch"]) for r in records} == expected - {
        recovered
    }
    retained = {}
    for row in records:
        folder = Path(row["log_path"]).parent
        for name in (
            "row.log",
            "receipt.json",
            "solve-receipt.json",
            "terminal.npz",
            "coupling-differences.json",
            "coupling-differences.npz",
        ):
            path = folder / name
            retained[str(path)] = digest(path)
    (args.output / "retained-sha256.json").write_text(
        json.dumps(retained, indent=2) + "\n"
    )
    folder = args.output / "head/frame-179-limited"
    folder.mkdir(parents=True)
    base = args.output / "base/frame-179-limited"
    base.parent.mkdir()
    base.symlink_to(args.previous / "base/frame-179-limited", target_is_directory=True)
    driver = Path(__file__).with_name("measure.py").resolve()
    with tempfile.TemporaryDirectory(
        prefix="coupling-recovery-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary)
        print(f"EXTRACTION={tree}", flush=True)
        with subprocess.Popen(
            ["git", "-C", str(args.repository), "archive", args.revision],
            stdout=subprocess.PIPE,
        ) as archive:
            subprocess.run(
                ["tar", "-x", "-C", str(tree)], stdin=archive.stdout, check=True
            )
            assert archive.wait() == 0
        assert digest(driver) == digest(
            tree / "scripts/prototypes/diiid_single_kernel_coupling/measure.py"
        )
        command = [
            sys.executable,
            str(driver),
            "--tree",
            str(tree),
            "--revision",
            args.revision,
            "--frame",
            "179",
            "--branch",
            "limited",
            "--output",
            str(folder),
        ]
        environment = dict(
            os.environ, PYTHONPATH=str(tree), XLA_PYTHON_CLIENT_PREALLOCATE="false"
        )
        run_logged(command, tree, environment, folder / "row.log", args.revision)
        print(f"REMOVING_EXTRACTION={tree}", flush=True)
    print(f"REMOVED_EXTRACTION={tree}", flush=True)

    import numpy as np

    previous_head = args.previous / "head/frame-179-limited"
    original_log = (previous_head / "row.log").read_text().splitlines()
    recovered_log = (folder / "row.log").read_text().splitlines()
    for prefix in ("REQUEST ", "REQUEST_SEED_SHA256="):
        assert [s for s in original_log if s.startswith(prefix)] == [
            s for s in recovered_log if s.startswith(prefix)
        ]
    for filename, names in (
        ("coupling-differences.npz", None),
        ("warmup-0.npz", ("state",)),
    ):
        with (
            np.load(previous_head / filename) as original,
            np.load(folder / filename) as current,
        ):
            for name in names or original.files:
                np.testing.assert_array_equal(original[name], current[name])
    receipt = json.loads((folder / "receipt.json").read_text())
    solved = json.loads((folder / "solve-receipt.json").read_text())
    item = dict(
        arm="head",
        frame=179,
        branch="limited",
        revision=args.revision,
        exit_status=0,
        log_path=str(folder / "row.log"),
        recovery_job_id=os.environ["SLURM_JOB_ID"],
        converged=solved["converged"],
        residual=solved["residual"],
        terminal_reason=solved["terminal_reason"],
        topology=receipt["terminal_topology"],
        efit_x_point_rz_m=receipt["efit_x_point_rz_m"],
    )
    records.append(item)
    (args.output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
    print("RECOVERED_ROW " + json.dumps(item), flush=True)
    renderer = Path(__file__).with_name("render.py").resolve()
    render_revision = subprocess.check_output(
        ["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True
    ).strip()
    run_logged(
        [
            sys.executable,
            str(renderer),
            "--input",
            str(args.output),
            "--output",
            str(args.figures),
            "--previews",
            str(args.previews),
            "--flux-only",
            "--rows",
            "179-limited",
        ],
        args.repository,
        dict(os.environ, PYTHONPATH=str(args.repository), JAX_PLATFORMS="cpu"),
        args.output / "render.log",
        render_revision,
    )
    assert all(digest(Path(path)) == value for path, value in retained.items())
    (args.output / "verification.json").write_text(
        json.dumps(
            dict(
                retained_files_unchanged=len(retained),
                request_policy_identical=True,
                seed_identical=True,
                initial_state_identical=True,
                coupling_measurement_identical=True,
                completed_arms=len(records),
                recovery_job_id=os.environ["SLURM_JOB_ID"],
            ),
            indent=2,
        )
        + "\n"
    )
    print("RECOVERY_COMPLETE: 8 terminal arms; retained receipts unchanged", flush=True)


if __name__ == "__main__":
    main()
