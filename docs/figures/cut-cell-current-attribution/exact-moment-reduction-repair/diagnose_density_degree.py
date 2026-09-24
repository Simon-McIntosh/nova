"""Measure cut-cell moment error against local density projection degree."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from nova.jax.config import configure_dtypes


configure_dtypes()

import jax  # noqa: E402

from benchmarks import exact_cut_cell_moment_stages as stages  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.equilibrium import clip_quadrature  # noqa: E402
from nova.equilibrium.forward_operator import set_support_clip_mode  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
).strip()
print(
    f"revision={revision} tree={ROOT} command=diagnose-density-projection-degree",
    flush=True,
)

case_name = "weak-rotation-reactor-static"
requested_cells = 110
label = f"{case_name}-cells-{requested_cells}"
base_row = json.loads((stages.BASE_OUTPUT / f"{label}.json").read_text())
stage_row = json.loads((stages.OUTPUT / f"{label}.json").read_text())
simple_cells = np.asarray(
    [item["cell"] for item in stage_row["simple_separatrix_cut"]["cells"]],
    dtype=np.int32,
)
reference = np.asarray(
    [
        item["production_signed_winding_order_16_moments"]
        for item in stage_row["simple_separatrix_cut"]["cells"]
    ]
)
carrier, source, exact = certificate._case(case_name)
machine = certificate._case_machine(case_name, carrier, exact, -requested_cells)
operator = fixture.forward_operator(source, machine)
with np.load(
    ROOT / "docs/figures/plasma-cell-read-fidelity/map-fidelity" / f"{label}-exact.npz"
) as bank:
    state = bank["analytic"]
set_support_clip_mode("exact")

for degree in (4, 6, 8, 9):
    powers = tuple(
        (radial, total - radial)
        for total in range(degree + 1)
        for radial in range(total + 1)
    )
    nodes = np.cos(np.pi * (2.0 * np.arange(degree + 1) + 1.0) / (2.0 * (degree + 1)))
    local = 0.5 * np.stack(np.meshgrid(nodes, nodes, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    design = np.stack(
        [
            local[:, 0] ** radial * local[:, 1] ** vertical
            for radial, vertical in powers
        ],
        axis=1,
    )
    clip_quadrature._DENSITY_POWERS = powers
    clip_quadrature._DENSITY_SAMPLE_LOCAL = local
    clip_quadrature._DENSITY_SAMPLE_INVERSE = np.linalg.pinv(design)
    jax.clear_caches()
    _incoming, _effective, production, _field, _topology, _selected, _profile = (
        stages._evaluate_path(operator, state)
    )
    reduction = np.asarray(production)[simple_cells] - reference
    reduction_current = float(reduction[:, 0].sum())
    base_error = float(
        base_row["modes"]["exact"]["classes"]["separatrix-cut"][
            "simple_moment_integration_error_a"
        ]
    )
    corrected = (
        float(base_row["modes"]["exact"]["booked_current_a"]) - reduction_current
    ) / float(base_row["modes"]["exact"]["archived_target_current_a"])
    print(
        json.dumps(
            {
                "degree": degree,
                "sample_count": len(local),
                "moment_reduction_a": reduction_current,
                "moment_reduction_over_base": abs(reduction_current / base_error),
                "corrected_exact_support_fraction": corrected,
                "cell_108_current_a": float(np.asarray(production)[108, 0]),
                "cell_108_reference_a": float(reference[simple_cells == 108, 0][0]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
