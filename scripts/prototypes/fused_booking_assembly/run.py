"""Pair separate and fused booking arms in one H200 allocation."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile

from scripts.prototypes.h200_levers.run import refuse_inert

PYTHON = "/home/ITER/mcintos/Code/nova/.venv/bin/python"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        TMPDIR="/tmp",
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NOVA_COMPILATION_CACHE_ROOT="/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm",
        JAX_COMPILATION_CACHE_DIR=str(args.output / "compilation-cache"),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
    )
    for key in ("UV_NO_SYNC", "UV_RUN_RECURSION_DEPTH", "XLA_FLAGS"):
        env.pop(key, None)
    common = args.output / "common-streams.npz"
    shutil.copyfile(args.inputs, common)
    summary = dict(
        base=args.base,
        head=args.head,
        complete=False,
        arms=[],
        comparisons=[],
        input_sha256=hashlib.sha256(common.read_bytes()).hexdigest(),
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    save()
    with tempfile.TemporaryDirectory(
        prefix="fused-booking-", dir=env["TMPDIR"]
    ) as temp:
        scratch = Path(temp)
        env["TMPDIR"] = str(scratch)
        trees = {}
        for label, revision in (("off", args.base), ("on", args.head)):
            tree = scratch / label
            tree.mkdir()
            data = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.root),
                    "archive",
                    revision,
                    "nova",
                    "scripts",
                    "tests",
                    "benchmarks",
                    "conftest.py",
                    "pyproject.toml",
                ]
            )
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                archive.extractall(tree, filter="data")
            trees[label] = tree
            for name in ("current.py", "support_read.py"):
                durable = (
                    args.output
                    / "measured-source"
                    / revision
                    / "nova/equilibrium"
                    / name
                )
                durable.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tree / "nova/equilibrium" / name, durable)
        instrument = (
            trees["on"] / "scripts/prototypes/fused_booking_assembly/profile.py"
        )

        def execute(label, width, process):
            tree = trees[label]
            revision = args.base if label == "off" else args.head
            destination = args.output / f"{label}-{width}-{process}"
            destination.mkdir(exist_ok=True)
            command = [
                PYTHON,
                str(instrument),
                "--output",
                str(destination),
                "--inputs",
                str(common),
                "--width",
                str(width),
                "--revision",
                revision,
                "--instrument",
                args.head,
            ]
            log = destination / "gate.log"
            literal = f"PYTHONPATH={tree} {shlex.join(command)}"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {revision} tree: {tree} command: {literal}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=tree,
                    env=dict(env, PYTHONPATH=str(tree)),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                stream.write(f"EXIT={result.returncode}\n")
            summary["arms"].append(
                dict(
                    label=label,
                    width=width,
                    process=process,
                    revision=revision,
                    log=str(log),
                    receipt=str(destination / "receipt.json"),
                    exit_status=result.returncode,
                )
            )
            save()
            print("ARM", destination.name, "EXIT", result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(f"measurement failed: {log}")
            path = destination / "receipt.json"
            record = json.loads(path.read_text())
            assert record["complete"]
            assert record["inputs_sha256"] == summary["input_sha256"]
            record["module_original"] = record["module"]
            record["module"] = str(
                args.output
                / "measured-source"
                / revision
                / "nova/equilibrium/support_read.py"
            )
            record["wrapper_sha256"] = hashlib.sha256(
                instrument.read_bytes()
            ).hexdigest()
            path.write_text(json.dumps(record, indent=2))
            return record

        def pair(label, width):
            execute(label, width, "warmup")
            for attempt in (1, 2):
                records = [execute(label, width, f"pair{attempt}-{i}") for i in (1, 2)]
                walls = [r["end_to_end_ms"] for r in records]
                spread = abs(walls[0] - walls[1]) / min(walls)
                if spread <= 0.05:
                    return dict(records=records, spread=spread, wall_ms=sum(walls) / 2)
                summary.setdefault("disagreements", []).append(
                    dict(label=label, width=width, attempt=attempt, spread=spread)
                )
                save()
            raise RuntimeError(f"fresh processes disagree: {label} {width}")

        for width in (1, 16, 64):
            off = pair("off", width)
            on = pair("on", width)

            def digest(record):
                return dict(
                    rows=[
                        dict(
                            lowered_sha256=record["programs"]["solve"]["lowered_sha256"]
                        )
                    ]
                )

            refuse_inert(digest(off["records"][0]), digest(on["records"][0]))
            control = args.output / f"inert-control-{width}.log"
            try:
                refuse_inert(digest(on["records"][0]), digest(on["records"][0]))
            except ValueError as error:
                control.write_text(
                    f"# revision: {args.head} command: refuse_inert(on, on)\n"
                    f"{error}\nEXIT=1\n"
                )
            else:
                raise AssertionError("identical compiled program was admitted")
            margin = max(0.03, off["spread"], on["spread"])
            change = on["wall_ms"] / off["wall_ms"] - 1
            verdict = (
                "faster"
                if change < -margin
                else "slower"
                if change > margin
                else "unresolved"
            )
            summary["comparisons"].append(
                dict(
                    width=width,
                    off=off,
                    on=on,
                    relative_change=change,
                    margin=margin,
                    verdict=verdict,
                )
            )
            save()
        summary["complete"] = True
        save()
        print(
            "SUMMARY",
            json.dumps(
                [
                    {k: r[k] for k in ("width", "relative_change", "margin", "verdict")}
                    for r in summary["comparisons"]
                ]
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
