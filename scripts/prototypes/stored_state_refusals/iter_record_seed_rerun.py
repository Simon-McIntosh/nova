"""Re-run the ITER arm from the record-era refusal state as its seed.

The earlier comparison run's ITER arms all refused with NoQualifiedAxisError,
and each kept the state it held when the topology read refused.  That state is
on the same 1568-node mesh this branch builds for ITER, so it can be substituted
as the seed while the arm keeps the request this branch would otherwise build
with no fitted-current binding, which is how the record's arm was built.

If the refusal returns, the seed is the input that accounts for it; if an axis
is admitted again, the remaining difference is the fixture rewrite itself.
"""

from __future__ import annotations

import json
import os
import pickle
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

RECORD_HISTORY = Path(
    "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline/"
    "efit-comparison-run-copy/titan-measurement-1318520/"
    "physical-iter/iter-seed-1-rejected-history.pkl"
)
RECORD_INPUTS = RECORD_HISTORY.with_name("iter-inputs.json")
TODAY_INPUTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261121T205621369895-default-policy-refusal-drift/input-drift/"
    "attribution-fixed/iter-inputs.json"
)


def _inputs(path):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return {
        key: payload.get(key)
        for key in ("ip_a", "flux_span_wb", "conductor_count",
                    "active_element_count", "passive_element_count", "time_s")
    }


def _seed_offset(seed, stored):
    seed = np.asarray(seed, dtype=float).ravel()
    stored = np.asarray(stored, dtype=float).ravel()
    if seed.shape != stored.shape:
        return dict(
            comparable=False,
            seed_shape=list(seed.shape),
            stored_shape=list(stored.shape),
        )
    difference = seed - stored
    gauge_free = difference - difference.mean()
    span = float(stored.max() - stored.min())
    return dict(
        comparable=True,
        seed_min=float(seed.min()),
        seed_max=float(seed.max()),
        stored_min=float(stored.min()),
        stored_max=float(stored.max()),
        gauge_free_rms_over_stored_span=float(
            np.sqrt(np.mean(gauge_free * gauge_free)) / span
        ),
        max_abs=float(np.max(np.abs(difference))),
    )


def _relative(record, today):
    if not record or not today:
        return None
    result = {}
    for key in ("ip_a", "flux_span_wb"):
        before, after = record.get(key), today.get(key)
        if before and after:
            result[key] = dict(
                record=before,
                today=after,
                relative_drift=float(abs(after - before) / abs(before)),
            )
    return result


def main():
    if drift.OUTPUT is None:
        raise SystemExit(
            "STORED_STATE_REFUSALS_OUTPUT must name this job's output directory"
        )
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu", "a GPU lane is required"
    monkeypatch = pytest.MonkeyPatch()
    data = drift._prepare("iter", monkeypatch, fitted=False)
    request = data["request"]
    stored_seed = np.asarray(request.seed_policy.state, dtype=float).ravel()
    with open(RECORD_HISTORY, "rb") as handle:
        history = pickle.load(handle)
    seed = np.asarray(history.state, dtype=float).ravel()
    request = replace(
        request, seed_policy=ExplicitSolveSeed(state=jax.numpy.asarray(seed))
    )
    record = drift._arm(
        data,
        request,
        "iter-record-seed",
        "iter",
        "measured currents + record-era seed state",
    )
    record["seed_source"] = str(RECORD_HISTORY)
    record["record_seed_reason"] = int(np.asarray(history.termination_reason))
    record["record_active_set_residuals"] = [
        None if not np.isfinite(value) else float(value)
        for value in np.asarray(history.active_set_residuals).ravel()
    ]
    record["seed_difference"] = _seed_offset(seed, stored_seed)
    record["input_drift"] = _relative(_inputs(RECORD_INPUTS), _inputs(TODAY_INPUTS))
    record["finished"] = True
    drift._write(drift.OUTPUT / "iter-record-seed.json", record)
    drift._write(
        drift.OUTPUT / "measurement.json",
        dict(
            case="iter",
            arm="iter-record-seed",
            device=str(jax.devices()[0]),
            source_revision=os.environ.get("SOURCE_REVISION"),
            record_source=drift.RECORD_SOURCE,
            seed_source=str(RECORD_HISTORY),
            residual=record.get("residual"),
            termination_reason=record.get("termination_reason"),
            seed_difference=record["seed_difference"],
            input_drift=record["input_drift"],
            finished=True,
        ),
    )
    print(
        "ITER_RECORD_SEED "
        + json.dumps(
            dict(
                residual=record.get("residual"),
                termination_reason=record.get("termination_reason"),
                axis=record.get("axis"),
                recorded=record.get("recorded"),
            )
        ),
        flush=True,
    )
    print("ITER_RECORD_SEED_FINISHED", flush=True)


if __name__ == "__main__":
    main()