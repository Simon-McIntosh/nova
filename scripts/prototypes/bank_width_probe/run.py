"""Locate the bank admission boundary using the existing slice profiler."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile

from scripts.prototypes.h200_levers.run import PYTHON, refuse_inert


def classify_outcome(exit_status, record, log_text):
    """Accept complete execution or a specifically witnessed memory refusal."""
    if exit_status == 0 and record.get("complete"):
        return "executed"
    if exit_status == 77 and record.get("memory_limited"):
        assert record["complete"] and record["limiting_program"] == "solve"
        return "compiled-memory-refusal"
    if exit_status == 1 and re.search(
        r"^jax\.errors\.JaxRuntimeError: RESOURCE_EXHAUSTED: "
        r"Out of memory while trying to allocate .+",
        log_text,
        flags=re.MULTILINE,
    ):
        assert "solve" in record["programs"] and not record["complete"]
        return "runtime-oom"
    raise RuntimeError(f"unclassified process failure: exit {exit_status}")


def measure_widths(execute, summary, save):
    """Locate the executing boundary and require agreeing fresh repeats."""
    probes = {}
    # First establish a positive admission and the known oversized refusal.
    for width in (64, 96, 256):
        probes[width] = execute(width, 0)
    assert probes[64]["admitted"] and not probes[96]["admitted"]
    assert probes[256]["outcome"] == "compiled-memory-refusal"
    low, high = 64, 96
    while high - low > 1:
        width = (low + high) // 2
        probes[width] = execute(width, 0)
        if probes[width]["admitted"]:
            low = width
        else:
            high = width
    summary.update(largest_admitted_width=low, first_refused_width=high)
    save()
    # Repeat the reference and boundary in fresh processes after exploration.
    for width in sorted({64, low}):
        measured = [execute(width, p) for p in (1, 2)]
        assert all(a["admitted"] for a in measured), "boundary did not repeat"
        walls = [a["batch_ms"] for a in measured]
        spread = abs(walls[0] - walls[1]) / min(walls)
        summary.setdefault("agreement", {})[width] = spread
        save()
        assert spread <= 0.05, (width, "fresh processes disagree", spread)
    left = {"rows": [{"lowered_sha256": probes[64]["hlo_sha256"]}]}
    right = {"rows": [{"lowered_sha256": probes[96]["hlo_sha256"]}]}
    refuse_inert(left, right)
    try:
        refuse_inert(left, left)
    except ValueError as error:
        summary["inert_control"] = str(error)
        print("INERT_REFUSED", error, flush=True)
    else:
        raise AssertionError("identical programs escaped the inert-lever refusal")
    execute(96, 0, allocator="async")
    summary["complete"] = True
    save()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("revision")
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(exist_ok=True)
    inputs = args.inputs.resolve()
    digest = hashlib.sha256(inputs.read_bytes()).hexdigest()
    assert digest == "52f8d4fb9e60dbdbe1530e8b64a171d3bb6470fb5cf73a1ddcf760895d09a0e9"
    env = dict(
        os.environ,
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NOVA_COMPILATION_CACHE_ROOT="/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm",
        JAX_COMPILATION_CACHE_DIR=str(output / "compilation-cache"),
        JAX_ENABLE_COMPILATION_CACHE="true",
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
    )
    for key in (
        "UV_NO_SYNC",
        "UV_RUN_RECURSION_DEPTH",
        "XLA_FLAGS",
        "TF_GPU_ALLOCATOR",
    ):
        env.pop(key, None)
    summary = dict(
        revision=args.revision,
        input_sha256=digest,
        arms=[],
        complete=False,
        job_id=os.environ.get("SLURM_JOB_ID"),
        admission_fraction=0.90,
    )
    # Record measurement settings, never the caller's credentials.
    summary["environment"] = {
        k: env[k]
        for k in env
        if k.startswith(("JAX_", "XLA_", "NOVA_", "OMP_", "OPENBLAS_"))
    }

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2))

    save()
    with tempfile.TemporaryDirectory(
        prefix="bank-width-", dir=os.environ["TMPDIR"]
    ) as temporary:
        tree = Path(temporary)
        env.update(TMPDIR=str(tree), PYTHONPATH=str(tree))
        data = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
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
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(tree, filter="data")
        retained = output / "measured-source"
        for path in (
            "nova/equilibrium/support_read.py",
            "nova/equilibrium/topology.py",
            "scripts/prototypes/bank_h200_slice_profile/measure.py",
        ):
            destination = retained / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tree / path, destination)
        summary["measurement_cwd"] = str(tree)
        summary["measurement_module"] = str(tree / "nova/equilibrium/support_read.py")
        save()

        def execute(width, process, allocator="default"):
            destination = output / f"{allocator}-width-{width}-process-{process}"
            destination.mkdir()
            process_env = env.copy()
            if allocator == "async":
                process_env["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            command = [
                PYTHON,
                str(tree / "scripts/prototypes/bank_h200_slice_profile/measure.py"),
                "--output",
                str(destination),
                "--inputs",
                str(inputs),
                "--width",
                str(width),
                "--revision",
                args.revision,
                "--instrument",
                args.revision,
            ]
            log = destination / "gate.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {args.revision} tree: {tree} "
                    f"command: PYTHONPATH={tree} "
                    + (
                        "TF_GPU_ALLOCATOR=cuda_malloc_async "
                        if allocator == "async"
                        else ""
                    )
                    + f"{shlex.join(command)}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=tree,
                    env=process_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                stream.write(f"EXIT={result.returncode}\n")
            arm = dict(
                width=width,
                allocator=allocator,
                allocator_setting=process_env.get("TF_GPU_ALLOCATOR", "unset"),
                process=process,
                exit_status=result.returncode,
                receipt=str(destination / "receipt.json"),
                log=str(log),
            )
            summary["arms"].append(arm)
            save()
            record = json.loads(Path(arm["receipt"]).read_text())
            outcome = classify_outcome(result.returncode, record, log.read_text())
            assert record["revision"] == args.revision
            assert record["inputs_sha256"] == digest
            assert record["module"] == summary["measurement_module"]
            assert record["cwd"] == str(tree)
            assert (
                record["module_sha256"]
                == hashlib.sha256(
                    (retained / "nova/equilibrium/support_read.py").read_bytes()
                ).hexdigest()
            )
            memory = record["programs"]["solve"]
            admitted = outcome == "executed"
            arm.update(
                admitted=admitted,
                outcome=outcome,
                working_bytes=memory["working_bytes"],
                device_limit_bytes=memory["device_limit_bytes"],
                batch_ms=record.get("end_to_end_ms") if admitted else None,
                per_slice_ms=record.get("per_slice_ms") if admitted else None,
                compile_seconds=memory["compile_seconds"],
                hlo_sha256=memory["hlo_sha256"],
            )
            if admitted:
                assert memory["working_bytes"] <= 0.90 * memory["device_limit_bytes"]
                assert max(record["fresh_residual"]) <= 1e-10
                for name, limit in (
                    ("axis", 5e-4),
                    ("saddle", 5e-4),
                    ("boundary", 1e-4),
                ):
                    assert len(record["precision"][name]) == width
                    assert max(record["precision"][name]) <= limit
            elif outcome == "compiled-memory-refusal":
                assert record["limiting_program"] == "solve"
                assert memory["working_bytes"] > 0.90 * memory["device_limit_bytes"]
            else:
                arm["refusal"] = next(
                    line
                    for line in log.read_text().splitlines()
                    if line.startswith(
                        "jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED:"
                    )
                )
            save()
            print("ARM", json.dumps(arm), flush=True)
            return arm

        measure_widths(execute, summary, save)
    print(
        "SUMMARY",
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "largest_admitted_width",
                    "first_refused_width",
                    "agreement",
                    "complete",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
