"""Pair geometry-cache arms on one device using the slice-profile driver."""

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

from scripts.prototypes.bank_h200_slice_profile.run import PYTHON


def compare(off, on):
    """Refuse inert or unstable pairs before applying the decision margin."""
    digests = [
        {row["programs"]["solve"]["lowered_sha256"] for row in arm} for arm in (off, on)
    ]
    if len(digests[0]) != 1 or len(digests[1]) != 1:
        raise ValueError("fresh processes lowered different programs within an arm")
    if digests[0] == digests[1]:
        raise ValueError("inert lever: identical lowered programs")
    walls = [[row["per_slice_ms"] for row in arm] for arm in (off, on)]
    spreads = [(max(values) - min(values)) / min(values) for values in walls]
    if max(spreads) > 0.05:
        raise ValueError(f"fresh processes disagree: {spreads}")
    means = [sum(values) / len(values) for values in walls]
    improvement = 1 - means[1] / means[0]
    margin = max(0.03, *spreads)
    return dict(
        off_ms=means[0],
        on_ms=means[1],
        spread=spreads,
        margin=margin,
        improvement=improvement,
        verdict="on"
        if improvement > margin
        else "off"
        if -improvement > margin
        else "unresolved",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.inputs = args.inputs.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(revision=args.revision, arms=[], decisions={}, complete=False)

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    save()
    env = dict(
        os.environ,
        TMPDIR="/tmp",
        JAX_PLATFORMS="cpu" if args.smoke else "cuda,cpu",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        # Reserve one pool so repeated large solves do not fragment its backing.
        XLA_PYTHON_CLIENT_PREALLOCATE="true",
        XLA_CLIENT_MEM_FRACTION="0.8",
        NOVA_COMPILATION_CACHE_ROOT="/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        JAX_COMPILATION_CACHE_DIR=str(args.output / "compilation-cache"),
    )
    for key in (
        "UV_NO_SYNC",
        "UV_RUN_RECURSION_DEPTH",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
    ):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(
        prefix="geometry-h200-", dir=env["TMPDIR"]
    ) as temporary:
        tree = Path(temporary)
        payload = subprocess.check_output(
            [
                "git",
                "-C",
                str(args.root),
                "archive",
                args.revision,
                "nova",
                "scripts",
                "tests",
                "benchmarks",
                "conftest.py",
                "pyproject.toml",
            ]
        )
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(tree, filter="data")
        env.update(PYTHONPATH=str(tree), TMPDIR=str(tree))

        def execute(width, arm, process):
            destination = args.output / f"{arm}-{width}-{process}"
            destination.mkdir(exist_ok=True)
            command = [
                PYTHON,
                str(tree / "scripts/prototypes/geometry_reuse/measure.py"),
                "--arm",
                arm,
                "--width",
                str(width),
                "--inputs",
                str(args.inputs),
                "--output",
                str(destination),
                "--revision",
                args.revision,
                "--instrument",
                args.revision,
            ]
            if args.smoke:
                command += ["--smoke"]
            log = destination / "gate.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.revision} tree: {tree} "
                    f"command: PYTHONPATH={tree} {shlex.join(command)}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                stream.write(f"EXIT={result.returncode}\n")
            summary["arms"].append(
                dict(
                    width=width,
                    arm=arm,
                    process=process,
                    exit_status=result.returncode,
                    log=str(log),
                    receipt=str(destination / "receipt.json"),
                )
            )
            save()
            print("ARM", destination.name, "EXIT", result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f"arm did not complete: {log}")
            receipt = json.loads((destination / "receipt.json").read_text())
            assert receipt["complete"]
            return receipt

        for width in [1] if args.smoke else [1, 16, 64]:
            paired = {}
            for arm in ("off", "on"):
                rows = [
                    execute(width, arm, process)
                    for process in ([1] if args.smoke else [0, 1, 2])
                ]
                paired[arm] = rows[-2:]
            processes = [1] if args.smoke else [1, 2]
            try:
                decision = compare(paired["off"], paired["on"])
            except ValueError as error:
                if "fresh processes disagree" not in str(error) or args.smoke:
                    raise
                summary.setdefault("rejected_agreement", {})[width] = str(error)
                save()
                print("RERUN_PAIR", width, str(error), flush=True)
                processes = [3, 4]
                paired = {
                    arm: [execute(width, arm, process) for process in processes]
                    for arm in ("off", "on")
                }
                decision = compare(paired["off"], paired["on"])
            decision["processes"] = processes
            summary["decisions"][width] = decision
            print("DECISION", width, json.dumps(decision), flush=True)
            try:
                compare(paired["off"], paired["off"])
            except ValueError as error:
                assert "inert lever" in str(error)
                summary["inert_control"] = str(error)
                print("INERT_CONTROL_REFUSED", str(error), flush=True)
            else:
                raise AssertionError("identical lowered programs escaped refusal")
            save()
        summary["complete"] = True
        save()
        print("REMOVING job-owned scratch", tree, flush=True)


if __name__ == "__main__":
    main()
