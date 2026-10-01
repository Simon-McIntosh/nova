"""Run the weak-rotation 110-cell moment-stage row on the selected backend."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from nova.jax.config import configure_dtypes


configure_dtypes()

from benchmarks.exact_cut_cell_moment_stages import measure  # noqa: E402


output = Path(sys.argv[1])
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[4]
).strip()
print(
    f"revision={revision} tree={Path(__file__).resolve().parents[4]} "
    f"command={sys.argv!r}",
    flush=True,
)
row = measure("weak-rotation-reactor-static", 110, output)
summary = {
    "case": row["case"],
    "requested_cells": row["requested_cells"],
    "realised_cells": row["realised_cells"],
    "base_simple_integration_error_a": row["base_simple_integration_error_a"],
    "moment_reduction_a": row["simple_separatrix_cut"]["terms"]["moment_reduction"][0],
    "moment_reduction_over_base": abs(
        row["simple_separatrix_cut"]["terms"]["moment_reduction"][0]
        / row["base_simple_integration_error_a"]
    ),
    "exact_support_fraction": row["negative_control"][
        "original_exact_support_fraction"
    ],
    "reference_substitution_fraction": row["negative_control"][
        "corrected_exact_support_fraction"
    ],
    "interior_max_abs_stage_term_over_cell_current": row["interior_positive_control"][
        "max_abs_stage_term_over_cell_current"
    ],
}
print("ROW_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
