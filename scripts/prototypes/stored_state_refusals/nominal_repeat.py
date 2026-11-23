"""Repeat one nominal arm to measure the scatter the record quotes.

The record states the TCV comparison arm's terminal residual twice, from two
independent jobs, and the two values differ in the fourth significant figure. A
reader cannot tell from that whether the solve is bit-reproducible, whether the
two constructions differ, or whether the difference is a transcription slip. So
this job runs the same arm three times in one allocation: the comparison
construction twice, and the seed-sensitivity construction once.

Two runs of one construction that agree to the bit make the across-job
difference a property of the construction; two that differ by the same amount
the across-job pair does make the terminal residual a property of the allocation.
Either answer is the remark the record needs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import attribute_input_drift as drift  # noqa: E402  (sets OUTPUT from the environment)
import seed_sensitivity as sensitivity  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402

from nova.jax.config import configure_dtypes  # noqa: E402


def _solve_construction(name, monkeypatch):
    """Run the comparison arm through the attribution module's own solve path."""
    data = drift._prepare("tcv", monkeypatch, fitted=True)
    columns, provenance = drift._comparison_columns(data, "tcv")
    return drift._solve(data, columns, name, "tcv", provenance)


def _seed_construction(name, monkeypatch):
    """Run the same arm through the seed-sensitivity module's own path."""
    return sensitivity._run(1.0, name, monkeypatch)


def main() -> None:
    if drift.OUTPUT is None:
        raise SystemExit(
            "STORED_STATE_REFUSALS_OUTPUT must name this job's output directory"
        )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "a GPU lane is required"
    monkeypatch = pytest.MonkeyPatch()
    arms = [
        ("solve-first", _solve_construction),
        ("solve-second", _solve_construction),
        ("seed-first", _seed_construction),
    ]
    records = []
    for name, construct in arms:
        record = construct(name, monkeypatch)
        records.append(
            dict(
                name=name,
                residual=record.get("residual"),
                termination_reason=record.get("termination_reason"),
                axis=(record.get("axis") or {}).get("magnetic_axis"),
                applied_current=record.get("applied_current"),
            )
        )
        print("REPEAT_ARM " + json.dumps(records[-1], default=str), flush=True)
    summary = dict(
        case="tcv",
        device=str(jax.devices()[0]),
        source_revision=os.environ.get("SOURCE_REVISION"),
        arms=records,
    )
    (drift.OUTPUT / "repeat.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("NOMINAL_REPEAT_FINISHED", flush=True)


if __name__ == "__main__":
    main()
