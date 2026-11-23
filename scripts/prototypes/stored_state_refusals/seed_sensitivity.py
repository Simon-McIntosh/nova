"""Measure how far the retained terminal residual moves on a seed nudge.

The record-era fitted conductor currents reproduce the comparison's recorded
TCV residual history to within 1.4 percent for its first eight active-set
iterations, but the retained best iterate lands at 0.004254 today against
0.030364 in the record.  This measures whether that terminal value is a stable
function of the inputs at the precision the fixtures can supply, by running the
same arm twice in one job: once on the stored seed, and once on a seed scaled by
the ratio the record-era and today's stored flux spans differ by.

Both arms carry the comparison's own fitted-current vector, so the only input
that moves between them is the seed scale.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attribute_input_drift as drift  # noqa: E402  (sets OUTPUT from the environment)

import jax  # noqa: E402
import pytest  # noqa: E402

from nova.equilibrium.solve_request import ExplicitSolveSeed  # noqa: E402
from nova.jax.config import configure_dtypes  # noqa: E402

RECORD_FLUX_SPAN = 0.19099370558229956
TODAY_FLUX_SPAN = 0.19093372278049314
SCALE = RECORD_FLUX_SPAN / TODAY_FLUX_SPAN


def _run(scale, label, monkeypatch):
    data = drift._prepare("tcv", monkeypatch, fitted=True)
    columns, provenance = drift._comparison_columns(data, "tcv")
    columns = np.asarray(columns, dtype=float)
    drift._bind_columns(data, columns)
    current = jax.numpy.asarray(columns)
    request = replace(data["request"], current=current)
    data["profile"] = replace(
        data["profile"],
        operator=replace(data["profile"].operator, external_current=current),
    )
    seed = np.asarray(request.seed_policy.state, dtype=float)
    scaled = ExplicitSolveSeed(state=jax.numpy.asarray(seed * scale))
    request = replace(request, seed_policy=scaled)
    record = drift._arm(data, request, label, "tcv", provenance)
    record["seed_span"] = float(np.max(seed) - np.min(seed))
    record["seed_scale"] = scale
    record["seed_scale_reason"] = (
        "record-era stored flux span over today's, the measured drift between the "
        "two fixture sets on this case"
    )
    drift._write(drift.OUTPUT / f"{label}.json", record)
    return record


def main():
    if drift.OUTPUT is None:
        raise SystemExit(
            "STORED_STATE_REFUSALS_OUTPUT must name this job's output directory"
        )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "a GPU lane is required"
    monkeypatch = pytest.MonkeyPatch()
    summary = dict(
        case="tcv",
        device=str(jax.devices()[0]),
        source_revision=os.environ.get("SOURCE_REVISION"),
        record_source=drift.RECORD_SOURCE,
        scale=SCALE,
        arms={},
    )
    drift._write(drift.OUTPUT / "measurement.json", summary)
    summary["arms"]["nominal"] = _run(1.0, "tcv-seed-nominal", monkeypatch)
    drift._write(drift.OUTPUT / "measurement.json", summary)
    summary["arms"]["drifted"] = _run(SCALE, "tcv-seed-drifted", monkeypatch)
    drift._write(drift.OUTPUT / "measurement.json", summary)
    nominal = summary["arms"]["nominal"].get("residual")
    drifted = summary["arms"]["drifted"].get("residual")
    if nominal and drifted:
        summary["retained_residual_ratio"] = drifted / nominal
        print(
            "SEED_SENSITIVITY "
            + json.dumps(
                dict(scale=SCALE, nominal=nominal, drifted=drifted,
                     ratio=drifted / nominal)
            ),
            flush=True,
        )
    summary["finished"] = True
    drift._write(drift.OUTPUT / "measurement.json", summary)
    print("SEED_SENSITIVITY_FINISHED", flush=True)


if __name__ == "__main__":
    main()