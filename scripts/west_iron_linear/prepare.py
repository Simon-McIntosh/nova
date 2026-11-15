"""Quote WEST constant receipts and exercise persisted shell interactions on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import numpy as np  # noqa: E402

from nova.biot import magnetisation as iron  # noqa: E402
from scripts.iron_graded_refinement.measure import provenance  # noqa: E402
from scripts.iron_linear_magnetisation.controls import loop, shell, SHELL_PROBES  # noqa: E402
from scripts.west_iron_linear.measure import constant_reference  # noqa: E402


def write(path, row):
    path.write_text(json.dumps(row, indent=2) + "\n")


def solve(region, interaction):
    def refuse(*args, **kwargs):
        raise AssertionError("reassembly attempted during persisted shell solve")

    iron._galerkin_interaction = refuse
    started = time.monotonic()
    result = iron.solve_magnetisation(
        region,
        loop,
        [0, 10],
        [1000, 1000],
        cell_size=0.04,
        basis="linear",
        interaction=interaction,
        tolerance=1e-10,
    )
    wall = time.monotonic() - started
    field = loop(SHELL_PROBES) + result.field(SHELL_PROBES)
    assert result.converged, result.termination
    return dict(
        solve_seconds=wall,
        field_T=field.tolist(),
        residual_history=list(result.residual_history),
        iterations=result.iterations,
        converged=bool(result.converged),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--blob", required=True)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    stamp = provenance(Path(__file__).resolve(), args.revision, args.blob)
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    region = shell()
    path = args.output / "shell-operator.npy"
    row = dict(
        **stamp,
        completed=False,
        device=jax.devices()[0].device_kind,
        x64=True,
        cell_size_m=0.04,
    )
    if args.reload:
        started = time.monotonic()
        loaded = iron.load_magnetisation_interaction(
            path, region=region, cell_size=0.04, basis="linear"
        )
        row["load_seconds"] = time.monotonic() - started
        row["solution"] = solve(region, loaded)
        row["assembly_calls"] = 0
        row["completed"] = True
        write(args.output / "shell-reload.json", row)
        return
    references = {
        label: constant_reference(label)
        for label in ("uniform-0.08", "uniform-0.04", "graded-16-80", "graded-8-40")
    }
    assert [v["cells"] for v in references.values()] == [1695, 6632, 5864, 11708]
    write(args.output / "constant-references.json", references)
    for label, reference in references.items():
        print(
            label,
            "cells",
            reference["cells"],
            "|B| [uT]",
            (np.asarray(reference["magnitude_T"]) * 1e6).tolist(),
            "receipt",
            reference["receipt"],
            "sha256",
            reference["sha256"],
            flush=True,
        )
    cells = iron._mesh(region, 0.04)
    row["cells"] = len(cells.polygons)
    row["unknowns"] = 6 * row["cells"]
    write(args.output / "shell-preparation.json", row)
    started = time.monotonic()
    reference_path = Path(
        "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/"
        "linear-magnetisation/assembly-cost/paired/assembly-0.04.json"
    )
    reference = json.loads(reference_path.read_text())
    matrix_path = Path(reference["operator_path"])
    with matrix_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == reference["operator_sha256"], "shell operator artifact drifted"
    assert reference["cells"] == row["cells"] and reference["cell_size_m"] == 0.04
    matrix = np.load(matrix_path, allow_pickle=False)
    row["source_operator_receipt"] = str(reference_path)
    row["source_operator_sha256"] = digest
    row["source_operator_device"] = reference["devices"]
    row["source_operator_load_seconds"] = time.monotonic() - started
    stored = iron.save_magnetisation_interaction(
        path, matrix, region=region, cell_size=0.04, basis="linear"
    )
    row["geometry_digest"] = stored.geometry_digest
    row["operator_bytes"] = matrix.nbytes
    row["solution"] = solve(region, stored)
    write(args.output / "shell-preparation.json", row)
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--output",
            str(args.output),
            "--revision",
            args.revision,
            "--blob",
            args.blob,
            "--reload",
        ],
        check=True,
    )
    replay = json.loads((args.output / "shell-reload.json").read_text())
    row["fresh_process_bit_identical"] = (
        np.asarray(row["solution"]["field_T"], dtype=np.float64).tobytes()
        == np.asarray(replay["solution"]["field_T"], dtype=np.float64).tobytes()
    )
    assert row["fresh_process_bit_identical"]
    try:
        iron.load_magnetisation_interaction(
            path, region=region, cell_size=0.08, basis="linear"
        )
    except ValueError as error:
        assert "geometry digest" in str(error)
        row["different_mesh_refusal"] = str(error)
        print("different shell mesh refused:", error, flush=True)
    else:
        raise AssertionError("operator from another shell mesh accepted")
    row["completed"] = True
    write(args.output / "shell-preparation.json", row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
