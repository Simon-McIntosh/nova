"""Measure repaired production moments on the attributed cut-cell rows."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from nova.jax.config import configure_dtypes


configure_dtypes()

import jax  # noqa: E402

from benchmarks import exact_cut_cell_moment_stages as stages  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium.forward_operator import set_support_clip_mode  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
ROWS = (
    ("weak-rotation-reactor-static", 110),
    ("weak-rotation-reactor-static", 300),
    ("diverted-single-null", 110),
)


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def _row(case_name: str, requested_cells: int) -> dict[str, object]:
    label = f"{case_name}-cells-{requested_cells}"
    stage_path = stages.OUTPUT / f"{label}.json"
    base_path = stages.BASE_OUTPUT / f"{label}.json"
    base_record = json.loads(base_path.read_text())
    carrier, source, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier, exact, -requested_cells)
    operator = fixture.forward_operator(source, machine)
    with np.load(
        ROOT
        / "docs/figures/plasma-cell-read-fidelity/map-fidelity"
        / f"{label}-exact.npz"
    ) as bank:
        state = bank["analytic"]

    set_support_clip_mode("exact")
    jax.clear_caches()
    incoming, effective, exact_moments, field, _topology, selected, profile = (
        stages._evaluate_path(operator, state)
    )
    base_cells = base_record["modes"]["exact"]["cells"]
    simple_cut = np.asarray(
        [
            item["class"] == "separatrix-cut"
            and not item["non_simple"]
            and selected[item["cell"]]
            for item in base_cells
        ]
    )
    interior = np.asarray(
        [item["class"] == "interior" and selected[item["cell"]] for item in base_cells]
    )
    centres = np.asarray(operator.moment_geometry.atomic_mesh.centroids)
    census = stages._stage_census(
        np.flatnonzero(simple_cut),
        incoming,
        effective,
        exact_moments,
        field,
        profile,
        source.toroidal_current_density,
        centres,
    )
    reduction = float(census["terms"]["moment_reduction"][0])
    base_error = float(
        base_record["modes"]["exact"]["classes"]["separatrix-cut"][
            "simple_moment_integration_error_a"
        ]
    )
    support_fraction = float(
        np.asarray(exact_moments)[:, 0].sum()
        / base_record["modes"]["exact"]["archived_target_current_a"]
    )

    recorded_interior = {
        item["cell"]: np.asarray(item["booked_moments_a_am_am"])
        for item in base_cells
        if item["class"] == "interior"
    }
    interior_delta = np.asarray(
        [
            np.asarray(exact_moments)[cell] - recorded_interior[cell]
            for cell in np.flatnonzero(interior)
        ]
    )
    interior_reference = np.asarray(
        [recorded_interior[cell] for cell in np.flatnonzero(interior)]
    )
    interior_current_relative = np.max(
        np.abs(interior_delta[:, 0])
        / np.maximum(np.abs(interior_reference[:, 0]), np.finfo(float).tiny),
        initial=0.0,
    )
    interior_current_bit_identical = bool(
        np.array_equal(
            np.asarray(exact_moments)[np.flatnonzero(interior), 0],
            interior_reference[:, 0],
        )
    )

    set_support_clip_mode("chord")
    jax.clear_caches()
    _support, _effective, chord_moments, _field, _topology, _selected, _profile = (
        stages._evaluate_path(operator, state)
    )
    recorded_chord = np.zeros_like(np.asarray(chord_moments))
    for item in base_record["modes"]["chord"]["cells"]:
        recorded_chord[item["cell"]] = item["booked_moments_a_am_am"]
    chord_bit_identical = bool(
        np.array_equal(np.asarray(chord_moments), recorded_chord, equal_nan=True)
    )

    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(centres),
        "base_simple_cut_error_a": base_error,
        "moment_reduction_a": reduction,
        "moment_reduction_over_base": abs(reduction / base_error),
        "exact_support_fraction": support_fraction,
        "interior_current_max_relative_change": float(interior_current_relative),
        "interior_current_bit_identical": interior_current_bit_identical,
        "chord_bit_identical": chord_bit_identical,
        "reference_order_doubling_l1_current_a": census[
            "signed_winding_order_doubling_l1_current_a"
        ],
        "stage_record_sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        "base_record_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    output = Path(sys.argv[1])
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
    ).strip()
    print(
        f"revision={revision} tree={ROOT} command={sys.argv!r}",
        flush=True,
    )
    print(
        f"job_id={os.environ.get('SLURM_JOB_ID')} devices={jax.devices()} "
        f"x64={jax.config.jax_enable_x64}",
        flush=True,
    )
    started = time.monotonic()
    rows = []
    for case_name, requested_cells in ROWS:
        row = _row(case_name, requested_cells)
        rows.append(row)
        print("ROW=" + json.dumps(_clean(row), sort_keys=True), flush=True)
    report = {
        "revision": revision,
        "tree": str(ROOT),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "devices": [str(device) for device in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "seconds": time.monotonic() - started,
        "rows": rows,
    }
    output.write_text(json.dumps(_clean(report), indent=2) + "\n")
    assert all(row["moment_reduction_over_base"] < 1e-3 for row in rows)
    assert all(
        row["case"] != "weak-rotation-reactor-static"
        or row["exact_support_fraction"] >= 0.999
        for row in rows
    )
    assert all(row["interior_current_max_relative_change"] <= 1e-11 for row in rows)
    assert all(row["chord_bit_identical"] for row in rows)
    print("REPAIRED_ROWS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
