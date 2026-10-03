"""Re-judge the five fallback rows against the retained fan's refinement floor.

Decision ``fallback-fence-norm`` is locked to ``moment-relative-l2``: the fence
for the five rows that carry no independent coupling error term is the retained
fan's own refinement floor, expressed in the same moment relative L2 norm the
rows are judged in. This driver measures that directly rather than transcribing
a stored receipt:

* the fence is ``(4/3) * |fan at order 16 - fan at order 8|`` taken as a
  relative L2 on the three moment integrals, measured on the weak 110 row and
  shared, which is the quantity the row-margin table already names as the
  fallback floor's ``other error term``;
* a row's error is the closed-form route's moment relative L2 against the
  retained fan, per arc vertex count (the polyline's only approximation), at
  the clip's native arc resolution (128 segments) for the headline verdict.

A row meets the fence when every one of its three moment series sits at or
below the fence at some swept count; the smallest such count is reported. The
two superseded fences (one tenth of the fan floor, and the closed-form round-off
envelope) are written into the receipt so a reader can see what this replaces.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import jax

from benchmarks.exact_clip_closed_form_floor import CASES, sweep
from nova.jax.config import configure_dtypes

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MOMENTS = ("current", "radial", "vertical")
NATIVE_COUNT = 128
FALLBACK_ROWS = (
    ("weak-rotation-reactor-static", -1000),
    ("moderate-rotation-conventional-static", -1000),
    ("strong-rotation-compact-static", -110),
    ("strong-rotation-compact-static", -300),
    ("strong-rotation-compact-static", -1000),
)
#: The fence the decision retired: one tenth of the fan refinement floor.
SUPERSEDED_FAN_TENTH = {"current": 2.924e-17, "radial": 5.109e-16, "vertical": 5.064e-16}
#: The fence the fallback-floor node measured and the decision retired with it.
SUPERSEDED_CLOSED_FORM_ENVELOPE = {
    "current": 2.038e-16,
    "radial": 2.060e-15,
    "vertical": 4.789e-16,
}


def _fan_refinement_floor() -> dict[str, float]:
    """Measure the retained fan's refinement floor on the weak 110 row.

    The floor is the fan's own order-16 against order-8 difference, scaled by
    Richardson's factor for a halving refinement, taken as a relative L2 on the
    three moment integrals. This is a direct measurement through the module's
    own fan arm, not a transcription.
    """
    from benchmarks.exact_clip_moment_floor import (
        CASES as FAN_CASES,
        CELL_REQUESTS as FAN_CELLS,
        _build,
        _fan_cut_moments,
        _relative_difference,
        FAN_ORDER,
        REFINED_FAN_ORDER,
    )
    import numpy as np

    operator, support, field, _bank, _span = _build(FAN_CASES[0], FAN_CELLS[0])
    boundary = np.asarray(support.included) & np.asarray(support.boundary)
    coarse = _fan_cut_moments(support, field, operator.source.core, FAN_ORDER)
    dense = _fan_cut_moments(support, field, operator.source.core, REFINED_FAN_ORDER)
    delta = _relative_difference(dense[:, boundary], coarse[:, boundary])
    return {name: float((4.0 / 3.0) * value) for name, value in zip(MOMENTS, delta)}


def main() -> None:
    configure_dtypes()
    if not jax.config.jax_enable_x64 or jax.default_backend() != "cpu":
        raise RuntimeError("this measurement requires binary64 on the CPU backend")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    print(
        "HEADER",
        json.dumps(
            {
                "revision": revision,
                "tree": str(ROOT),
                "command": "rejudge_fallback_rows.py",
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "jax_backend": jax.default_backend(),
                "x64": bool(jax.config.jax_enable_x64),
            }
        ),
        flush=True,
    )
    fence = _fan_refinement_floor()
    print("FENCE", fence, flush=True)
    rows = []
    for case, cells in FALLBACK_ROWS:
        receipt = sweep(case, cells)
        counts = sorted(int(key) for key in receipt["counts"])
        series = {
            count: receipt["counts"][str(count)]["moment_relative_l2_against_fan"]
            for count in counts
        }
        smallest = next(
            (
                count
                for count in counts
                if all(series[count][name] <= fence[name] for name in MOMENTS)
            ),
            None,
        )
        native = series[NATIVE_COUNT]
        native_verdict = all(native[name] <= fence[name] for name in MOMENTS)
        rows.append(
            {
                "case": case,
                "requested_cells": cells,
                "cut_cells": receipt["cut_cells"],
                "realised_cells": receipt["realised_cells"],
                "error_by_count": {
                    str(count): {name: series[count][name] for name in MOMENTS}
                    for count in counts
                },
                "error_at_native_count": {
                    name: native[name] for name in MOMENTS
                },
                "error_at_native_count_verdict": {
                    name: bool(native[name] <= fence[name]) for name in MOMENTS
                },
                "smallest_count_meeting_fence": smallest,
                "verdict": "pass" if native_verdict else "fail",
            }
        )
        print(
            "ROW",
            case,
            cells,
            "native",
            {name: native[name] for name in MOMENTS},
            "smallest_meeting",
            smallest,
            "verdict",
            "pass" if native_verdict else "fail",
            flush=True,
        )
    payload = {
        "schema": "nova.exact-clip-fence-in-moment-l2.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "host": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "decision": {"key": "fallback-fence-norm", "choice": "moment-relative-l2"},
        "norm": "moment relative L2 against the retained fan",
        "fence": fence,
        "fence_definition": (
            "(4/3) * |fan at order 16 - fan at order 8| as a relative L2 on the "
            "three moment integrals, measured on the weak 110 row and shared by "
            "every fallback row"
        ),
        "fence_source": "retained fan refinement floor, measured in this run",
        "superseded_fences": {
            "one_tenth_of_the_fan_refinement_floor": SUPERSEDED_FAN_TENTH,
            "closed_form_round_off_envelope": SUPERSEDED_CLOSED_FORM_ENVELOPE,
        },
        "native_arc_count": NATIVE_COUNT,
        "rows": rows,
        "summary": {
            "pass": [f"{row['case']} {abs(row['requested_cells'])}" for row in rows if row["verdict"] == "pass"],
            "fail": [f"{row['case']} {abs(row['requested_cells'])}" for row in rows if row["verdict"] == "fail"],
        },
    }
    (OUT / "receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("PASS", payload["summary"]["pass"], flush=True)
    print("FAIL", payload["summary"]["fail"], flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)