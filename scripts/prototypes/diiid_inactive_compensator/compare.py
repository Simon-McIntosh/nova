"""Compare one production cohort row at two revisions in one GPU allocation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from scripts.prototypes.diiid_core_hold.regressions import extract

ROOT = Path(__file__).resolve().parents[3]
HARNESS = Path("scripts/prototypes/diiid_inactive_compensator")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def reasons(tree):
    module = ast.parse((tree / "nova/equilibrium/fixed_point.py").read_text())
    definition = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "FixedPointTerminationReason"
    )
    return {
        ast.literal_eval(node.value): node.targets[0].id
        for node in definition.body
        if isinstance(node, ast.Assign)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    differences = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "diff",
            "--name-only",
            args.base,
            args.head,
            "--",
            "nova",
            "benchmarks",
            "scripts/prototypes",
        ],
        text=True,
    ).splitlines()
    assert all(
        path in ("nova/equilibrium/constraint.py", "nova/equilibrium/reduced_newton.py")
        or path.startswith(str(HARNESS) + "/")
        for path in differences
    ), differences
    with tempfile.TemporaryDirectory(
        prefix="inactive-cohort-pair-", dir=os.environ["TMPDIR"]
    ) as temporary:
        work = Path(temporary)
        trees = {label: work / label for label in ("base", "head")}
        for label, revision in (("base", args.base), ("head", args.head)):
            extract(trees[label], revision, ROOT)
        shutil.copytree(
            trees["head"] / HARNESS, trees["base"] / HARNESS, dirs_exist_ok=True
        )
        harness_hashes = {
            str(path.relative_to(trees["head"])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (trees["head"] / HARNESS).glob("*.py")
        }
        for path, digest in harness_hashes.items():
            assert (
                hashlib.sha256((trees["base"] / path).read_bytes()).hexdigest()
                == digest
            )
        provenance = dict(
            base=args.base,
            head=args.head,
            product_differences=[p for p in differences if p.startswith("nova/")],
            harness_sha256=harness_hashes,
            harness_policy=(
                "Identical head observer files overlaid into base; "
                "upstream prototype drivers and benchmark inputs unchanged."
            ),
            job_id=os.environ.get("SLURM_JOB_ID"),
            rung=os.environ.get("NOVA_LANE_RUNG"),
        )
        write(args.output / "provenance.json", provenance)
        results = {}
        for label, revision in (("base", args.base), ("head", args.head)):
            tree = trees[label]
            folder = args.output / label / "edge/frame-137-diverted"
            folder.mkdir(parents=True)
            command = [
                sys.executable,
                str(tree / HARNESS / "measure.py"),
                "--tree",
                str(tree),
                "--revision",
                revision,
                "--frame",
                "137",
                "--branch",
                "diverted",
                "--output",
                str(folder),
                "--trace-helper",
                str(tree / "scripts/prototypes/diiid_hollow_current/trace.py"),
                "--arm",
                "edge",
            ]
            environment = dict(
                os.environ,
                PYTHONPATH=str(tree),
                OMP_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="1",
                JAX_ENABLE_COMPILATION_CACHE="0",
                JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="none",
                XLA_PYTHON_CLIENT_MEM_FRACTION="0.9",
                XLA_PYTHON_CLIENT_PREALLOCATE="false",
                CUDA_VISIBLE_DEVICES="0",
            )
            log = folder / "row.log"
            with log.open("w") as stream:
                stream.write(
                    f"# revision: {revision}; tree: {tree}; "
                    f"command: {shlex.join(command)}\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=tree,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                stream.write(f"EXIT={result.returncode}\n")
            results[label] = dict(
                revision=revision, exit_status=result.returncode, log_path=str(log)
            )
            write(args.output / "results.json", results)
            assert result.returncode == 0, results[label]
            for name in (
                "receipt.json",
                "finite-edge.json",
                "trips.json",
                "shadow-image.json",
                "core-analysis.json",
                "solve-receipt.json",
            ):
                assert (folder / name).is_file(), name
            assert "TRIP_READ " in log.read_text()
            receipt = json.loads((folder / "receipt.json").read_text())
            finite = json.loads((folder / "finite-edge.json").read_text())
            assert receipt["base_revision"] == finite["revision"] == revision
            assert Path(receipt["module"]).is_relative_to(tree)
            results[label].update(
                converged=receipt["converged"],
                residual=receipt["residual"],
                terminal_reason=reasons(tree)[receipt["terminal_reason"]],
                lower_x_offset_m=receipt["terminal_topology"][
                    "lower_saddle_distance_m"
                ],
                module=receipt["module"],
                cwd=receipt["cwd"],
                device=receipt["timing"][-1]["device_kind"],
                seed_sha256=receipt["seed_admission"][0]["sha256"],
                source_inputs_sha256=hashlib.sha256(
                    (folder / "source-inputs.json").read_bytes()
                ).hexdigest(),
            )
            write(args.output / "results.json", results)
            print("ARM " + label + " " + json.dumps(results[label]), flush=True)
        for key in ("seed_sha256", "source_inputs_sha256"):
            assert results["base"][key] == results["head"][key], key
        if results["base"]["converged"] and not results["head"]["converged"]:
            attribution = (
                "The inactive-image mask causes the 137D convergence loss. "
                "Stop; do not alter guards or tolerances."
            )
        elif not results["base"]["converged"] and not results["head"]["converged"]:
            attribution = (
                "137D convergence is already lost at the unmasked base; "
                "the mask did not cause that loss. Relative to the earlier "
                "booked-source record, the loss predates this node."
            )
        else:
            attribution = (
                "Read the matched arm outcomes beside the historical record; "
                "the reported convergence loss is not reproduced as a "
                "base-converged/head-unconverged pair."
            )
        write(
            args.output / "comparison.json",
            dict(arms=results, attribution=attribution, **provenance),
        )
        print("ATTRIBUTION " + attribution, flush=True)


if __name__ == "__main__":
    main()
